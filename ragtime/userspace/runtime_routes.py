from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import re
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response, StreamingResponse

from ragtime.config.settings import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.auth import get_browser_matched_origin
from ragtime.core.rate_limit import SHARE_AUTH_RATE_LIMIT, limiter
from ragtime.core.security import get_current_user, get_current_user_optional
from ragtime.userspace.models import (
    UserSpaceBrowserAuthorization,
    UserSpaceBrowserAuthRequest,
    UserSpaceBrowserAuthResponse,
    UserSpaceBrowserSurface,
    UserSpaceCapabilityTokenResponse,
    UserSpaceFileResponse,
    UserSpacePreviewLaunchRequest,
    UserSpacePreviewLaunchResponse,
    UserSpaceRuntimeActionResponse,
    UserSpaceRuntimeSessionResponse,
    UserSpaceRuntimeStatusResponse,
    UserSpaceWorkspaceTabStateResponse,
)
from ragtime.userspace.runtime_service import (
    RuntimeVersionConflictError,
    userspace_runtime_service,
)
from ragtime.userspace.service import userspace_service
from ragtime.userspace.share_auth import (
    set_share_auth_cookie,
    share_auth_token_from_request,
)

router = APIRouter(prefix="/indexes/userspace", tags=["User Space Runtime"])


_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
_COLLAB_CAPABILITY_COOKIE = "userspace_collab_capability"
_RUNTIME_PTY_CAPABILITY_COOKIE = "userspace_runtime_pty_capability"
_PROXY_TIMEOUT_FLOOR = 300.0  # seconds — minimum proxy read/write timeout
_PROXY_TIMEOUT_BUFFER = 20.0  # seconds — headroom above max tool timeout
_USERSPACE_SURFACE_HEADER = "X-Ragtime-Userspace-Surface"
_USERSPACE_PREVIEW_PROXY_HEADER = "X-Ragtime-Userspace-Preview-Proxy"


def _root_share_target_url(base_url: str, share_path: str, target_path: str) -> str:
    normalized_base = (base_url or "").rstrip("/")
    normalized_share_path = "/" + share_path.lstrip("/")
    normalized_target_path = "/" + (target_path or "/").lstrip("/")
    if normalized_target_path == "/":
        return f"{normalized_base}{normalized_share_path}"
    return (
        f"{normalized_base}{normalized_share_path.rstrip('/')}{normalized_target_path}"
    )


def _root_share_launch_response(
    *,
    workspace_id: str,
    base_url: str,
    share_path: str,
    target_path: str,
) -> UserSpacePreviewLaunchResponse:
    preview_origin = (base_url or "").rstrip("/")
    return UserSpacePreviewLaunchResponse(
        workspace_id=workspace_id,
        preview_origin=preview_origin,
        preview_url=_root_share_target_url(base_url, share_path, target_path),
        expires_at=datetime.now(timezone.utc),
    )


def _collab_cookie_path(workspace_id: str) -> str:
    return f"/indexes/userspace/collab/workspaces/{workspace_id}/files"


def _runtime_pty_cookie_path(workspace_id: str) -> str:
    return f"/indexes/userspace/runtime/workspaces/{workspace_id}/pty"


_BROWSER_SURFACE_COOKIE_CONFIG: dict[str, dict[str, Any]] = {
    "collab": {
        "cookie_name": _COLLAB_CAPABILITY_COOKIE,
        "capabilities": ["userspace.collab_connect"],
        "path_builder": _collab_cookie_path,
    },
    "runtime_pty": {
        "cookie_name": _RUNTIME_PTY_CAPABILITY_COOKIE,
        "capabilities": ["userspace.runtime_pty"],
        "path_builder": _runtime_pty_cookie_path,
    },
}


def _default_browser_surfaces() -> list[UserSpaceBrowserSurface]:
    return ["collab", "runtime_pty"]


def _normalize_browser_surfaces(
    raw_surfaces: Sequence[UserSpaceBrowserSurface] | None,
) -> list[UserSpaceBrowserSurface]:
    surfaces = raw_surfaces or _default_browser_surfaces()
    ordered: list[UserSpaceBrowserSurface] = []
    seen: set[UserSpaceBrowserSurface] = set()
    for surface in surfaces:
        if surface in _BROWSER_SURFACE_COOKIE_CONFIG and surface not in seen:
            ordered.append(surface)
            seen.add(surface)
    if raw_surfaces is None:
        return ordered or _default_browser_surfaces()
    return ordered


def _get_max_proxy_timeout() -> float:
    """Return the proxy read/write timeout derived from max tool timeout_max_seconds."""
    try:
        from ragtime.core.app_settings import SettingsCache

        cached_configs = SettingsCache.get_instance()._tool_configs
        if cached_configs:
            max_t = max(
                (cfg.get("timeout_max_seconds", 300) or 300 for cfg in cached_configs),
                default=300,
            )
            return max(float(max_t), _PROXY_TIMEOUT_FLOOR) + _PROXY_TIMEOUT_BUFFER
    except Exception:
        pass
    return _PROXY_TIMEOUT_FLOOR + _PROXY_TIMEOUT_BUFFER


async def _safe_close_websocket(websocket: WebSocket, code: int) -> None:
    with contextlib.suppress(Exception):
        await websocket.close(code=code)


async def _broadcast_collab_message(
    workspace_id: str,
    file_path: str,
    message: dict[str, Any],
) -> None:
    clients = await userspace_runtime_service.get_collab_clients(
        workspace_id, file_path
    )
    payload = json.dumps(message)
    for client in clients:
        with contextlib.suppress(Exception):
            await client.send_text(payload)


def _proxy_request_headers(request: Request) -> dict[str, str]:
    """Build headers to forward to the workspace devserver.

    Sensitive credentials from the *Ragtime* session must not leak to
    the untrusted user-controlled devserver process.
    """
    _blocked = {
        # Hop-by-hop
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        # Credentials - must not reach the untrusted devserver
        "authorization",
        "cookie",
        "x-api-key",
        "x-userspace-share-password",
    }
    forwarded_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _blocked
    }
    # Preserve request context for framework URL generation and redirects.
    forwarded_headers.setdefault("x-forwarded-proto", request.url.scheme)
    forwarded_headers.setdefault("x-forwarded-host", request.headers.get("host", ""))
    client_host = request.client.host if request.client else ""
    if client_host:
        forwarded_headers.setdefault("x-forwarded-for", client_host)
    return forwarded_headers


def _proxy_response_headers(
    headers: httpx.Headers,
) -> dict[str, str]:
    blocked = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        # Untrusted preview apps must not set first-party cookies on ragtime origin.
        "set-cookie",
        # Devserver frameworks (Express+helmet, Django, etc.) may emit these
        # headers which prevent the content from rendering inside the platform
        # iframe.  The iframe sandbox attribute is the real security boundary;
        # devserver-originated policies are not trustworthy anyway.
        "x-frame-options",
        "content-security-policy",
        "content-security-policy-report-only",
    }
    out = {key: value for key, value in headers.items() if key.lower() not in blocked}
    return out


def _make_proxy_response_uncacheable(
    headers: dict[str, str],
    *,
    clear_site_cache: bool = False,
) -> None:
    headers.pop("etag", None)
    headers.pop("ETag", None)
    headers.pop("last-modified", None)
    headers.pop("Last-Modified", None)
    headers.pop("expires", None)
    headers.pop("Expires", None)
    headers.pop("age", None)
    headers.pop("Age", None)
    headers["cache-control"] = "no-store, no-cache, must-revalidate, max-age=0"
    headers["pragma"] = "no-cache"
    if clear_site_cache:
        headers["clear-site-data"] = '"cache"'


def _is_html_media_type(media_type: str) -> bool:
    return "text/html" in (media_type or "").lower()


def _extract_capability_token_from_request(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    explicit = request.headers.get("x-userspace-capability-token", "").strip()
    if explicit:
        return explicit
    query_token = request.query_params.get("cap_token", "").strip()
    if query_token:
        return query_token
    return None


def _extract_capability_token_from_websocket(websocket: WebSocket) -> str | None:
    query_token = websocket.query_params.get("cap_token", "").strip()
    if query_token:
        return query_token
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    explicit = websocket.headers.get("x-userspace-capability-token", "").strip()
    if explicit:
        return explicit
    return None


def _extract_cookie_token_from_request(
    request: Request,
    cookie_name: str,
) -> str | None:
    token = request.cookies.get(cookie_name, "").strip()
    return token or None


def _extract_cookie_token_from_websocket(
    websocket: WebSocket,
    cookie_name: str,
) -> str | None:
    token = websocket.cookies.get(cookie_name, "").strip()
    return token or None


def _require_workspace_capability(
    token: str | None,
    workspace_id: str,
    capability: str,
) -> tuple[dict[str, Any], str]:
    if not token:
        raise HTTPException(status_code=401, detail="Capability token required")
    claims = userspace_runtime_service.verify_capability_token(
        token,
        workspace_id,
        capability,
    )
    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Capability token missing user")
    return claims, user_id


def _to_websocket_url(http_url: str) -> str:
    parsed = urlsplit(http_url)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme
    return urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _sanitize_preview_query(query: str | None) -> str | None:
    if not query:
        return None
    cleaned_pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key != "cap_token"
    ]
    if not cleaned_pairs:
        return None
    return urlencode(cleaned_pairs, doseq=True)


async def _proxy_websocket_request(
    websocket: WebSocket,
    upstream_url: str,
    *,
    read_only: bool = False,
) -> None:
    try:
        websockets_module = importlib.import_module("websockets")
    except Exception:
        await _safe_close_websocket(websocket, code=1011)
        return

    requested_subprotocols = [
        str(protocol).strip()
        for protocol in (websocket.scope.get("subprotocols") or [])
        if str(protocol).strip()
    ]

    # Mirror the worker auth token that _proxy_http_request sends so the
    # upstream runtime worker accepts the WebSocket connection.
    extra_headers: dict[str, str] = {}
    worker_token = getattr(settings, "userspace_runtime_worker_auth_token", "") or ""
    if worker_token:
        extra_headers["authorization"] = f"Bearer {worker_token}"

    try:
        async with websockets_module.connect(
            upstream_url,
            max_size=None,
            open_timeout=20,
            subprotocols=requested_subprotocols or None,
            additional_headers=extra_headers or None,
        ) as upstream:
            await websocket.accept(
                subprotocol=getattr(upstream, "subprotocol", None) or None
            )

            if read_only:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "read_only": True,
                            "message": "Read-only terminal session",
                        }
                    )
                )

            async def downstream_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get("type")
                    if message_type == "websocket.disconnect":
                        break
                    text = message.get("text")
                    data = message.get("bytes")
                    if read_only and text is not None:
                        try:
                            payload = json.loads(text)
                            if payload.get("type") == "input":
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if text is not None:
                        await upstream.send(text)
                    elif data is not None:
                        await upstream.send(data)

            async def upstream_to_downstream() -> None:
                while True:
                    message = await upstream.recv()
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(str(message))

            down_task = asyncio.create_task(downstream_to_upstream())
            up_task = asyncio.create_task(upstream_to_downstream())
            done, pending = await asyncio.wait(
                {down_task, up_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        return
    except TimeoutError:
        with contextlib.suppress(Exception):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "status",
                        "read_only": read_only,
                        "message": "Runtime terminal is still starting. Reconnecting...",
                    }
                )
            )
        await _safe_close_websocket(websocket, code=1011)
    except Exception:
        await _safe_close_websocket(websocket, code=1011)


async def _proxy_http_request(
    request: Request,
    upstream_url: str,
    *,
    bridge_workspace_id: str | None = None,
    bridge_context: dict[str, Any] | None = None,
    bridge_script_src: str | None = None,
) -> Response:
    if request.headers.get("upgrade", "").lower() == "websocket":
        raise HTTPException(
            status_code=501,
            detail="WebSocket proxy upgrades are not available on this route",
        )

    body = await request.body()
    headers = _proxy_request_headers(request)

    # Inject runtime worker auth token for upstream worker requests
    worker_token = getattr(settings, "userspace_runtime_worker_auth_token", "") or ""
    if worker_token:
        headers["authorization"] = f"Bearer {worker_token}"

    # Derive proxy read/write timeout from the maximum tool timeout across
    # all configured tools so the proxy never cuts off a legitimate query.
    # Falls back to 320 s if no tools are configured.
    proxy_read_timeout = _get_max_proxy_timeout()
    timeout = httpx.Timeout(
        connect=2.0, read=proxy_read_timeout, write=proxy_read_timeout, pool=5.0
    )
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            content=body if body else None,
            headers=headers,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Runtime preview upstream unavailable: {exc}",
        ) from exc

    media_type = upstream_response.headers.get("content-type", "")
    resp_headers = _proxy_response_headers(upstream_response.headers)
    resp_headers[_USERSPACE_SURFACE_HEADER] = "preview-proxy"
    resp_headers[_USERSPACE_PREVIEW_PROXY_HEADER] = "true"
    _make_proxy_response_uncacheable(
        resp_headers,
        clear_site_cache=_is_html_media_type(media_type),
    )

    if _is_html_media_type(media_type):
        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
            await client.aclose()

        sandbox_flags: list[str] | None = None
        try:
            app_settings = await get_app_settings()
            sandbox_flags = list(
                app_settings.get("userspace_preview_sandbox_flags") or []
            )
        except Exception:
            sandbox_flags = None
        content = _inject_bridge_script(
            content,
            sandbox_flags,
            workspace_id=bridge_workspace_id,
            bridge_context=bridge_context,
            bridge_script_src=bridge_script_src,
        )
        # Content length changed after injection; drop stale header so
        # Starlette re-calculates it from the actual body.
        resp_headers.pop("content-length", None)

        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=resp_headers,
            media_type=media_type or None,
        )

    async def _iter_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_bytes():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        _iter_stream(),
        status_code=upstream_response.status_code,
        headers=resp_headers,
        media_type=media_type or None,
    )


_BRIDGE_CONFIG_MARKER = b"__ragtime_preview_sandbox_flags"
_BRIDGE_CONTEXT_MARKER = b"__ragtime_preview_bridge"
_BRIDGE_DETECT_RE = re.compile(rb"bridge\.js", re.IGNORECASE)
_HEAD_CLOSE_RE = re.compile(rb"(</head\s*>)", re.IGNORECASE)
_FIRST_SCRIPT_RE = re.compile(rb"(<script[\s>])", re.IGNORECASE)


def _build_bridge_script_tag(
    workspace_id: str | None = None,
    *,
    bridge_script_src: str | None = None,
) -> bytes:
    params = {"workspace_id": workspace_id} if workspace_id else {}
    query = urlencode(params)
    src = (
        bridge_script_src or "/__ragtime/bridge.js"
    ).strip() or "/__ragtime/bridge.js"
    if query:
        joiner = "&" if "?" in src else "?"
        src = f"{src}{joiner}{query}"
    return f'<script src="{src}"></script>'.encode("utf-8")


def _build_bridge_config_tag(sandbox_flags: list[str]) -> bytes:
    normalized_flags = [
        flag.strip() for flag in sandbox_flags if isinstance(flag, str) and flag.strip()
    ]
    serialized_flags = json.dumps(normalized_flags).encode("utf-8")
    return (
        b"<script>window.__ragtime_preview_sandbox_flags="
        + serialized_flags
        + b";</script>"
    )


def _build_bridge_context_tag(bridge_context: dict[str, Any]) -> bytes:
    serialized_context = json.dumps(bridge_context, separators=(",", ":")).encode(
        "utf-8"
    )
    return (
        b"<script>window.__ragtime_preview_bridge=" + serialized_context + b";</script>"
    )


def _inject_bridge_script(
    html: bytes,
    sandbox_flags: list[str] | None = None,
    *,
    workspace_id: str | None = None,
    bridge_context: dict[str, Any] | None = None,
    bridge_script_src: str | None = None,
) -> bytes:
    """Inject the platform data-bridge script into HTML responses.

    Inserts ``<script src="/__ragtime/bridge.js?...">`` before
    ``</head>`` or the first ``<script`` tag so ``window.__ragtime_context``
    and platform visualization libraries are available to workspace code.
    Skips injection if bridge.js is already referenced.
    """
    injected = b""
    if sandbox_flags is not None and _BRIDGE_CONFIG_MARKER not in html:
        injected += _build_bridge_config_tag(sandbox_flags) + b"\n"
    if bridge_context is not None and _BRIDGE_CONTEXT_MARKER not in html:
        injected += _build_bridge_context_tag(bridge_context) + b"\n"
    if not _BRIDGE_DETECT_RE.search(html):
        injected += (
            _build_bridge_script_tag(
                workspace_id,
                bridge_script_src=bridge_script_src,
            )
            + b"\n"
        )
    if not injected:
        return html
    m = _HEAD_CLOSE_RE.search(html)
    if m:
        return html[: m.start()] + injected + html[m.start() :]
    m = _FIRST_SCRIPT_RE.search(html)
    if m:
        return html[: m.start()] + injected + html[m.start() :]
    return injected + html


@router.get(
    "/runtime/workspaces/{workspace_id}/session",
    response_model=UserSpaceRuntimeSessionResponse,
)
async def get_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.get_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/session/start",
    response_model=UserSpaceRuntimeActionResponse,
)
async def start_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.start_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/session/stop",
    response_model=UserSpaceRuntimeActionResponse,
)
async def stop_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.stop_runtime_session(workspace_id, user.id)


@router.get(
    "/runtime/workspaces/{workspace_id}/events",
)
async def workspace_events_sse(
    workspace_id: str,
    request: Request,
    _user: Any = Depends(get_current_user),
):
    """SSE stream that emits a message whenever the workspace generation advances."""

    await userspace_runtime_service.track_workspace_runtime_events(workspace_id)

    async def _stream():
        # Always emit the current generation so the client knows the
        # connection is live and what the baseline is.
        generation = await userspace_runtime_service.get_workspace_generation(
            workspace_id
        )
        initial_payload = userspace_runtime_service.get_workspace_event_payload(
            workspace_id
        )
        yield f"data: {json.dumps(initial_payload)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            new_gen = await userspace_runtime_service.wait_workspace_generation(
                workspace_id, generation, timeout=25.0
            )
            if await request.is_disconnected():
                break
            if new_gen > generation:
                generation = new_gen
                payload = userspace_runtime_service.get_workspace_event_payload(
                    workspace_id
                )
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                # Keepalive – SSE comment to prevent proxy/browser timeouts
                yield ": keepalive\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/runtime/workspaces/{workspace_id}/devserver/status",
    response_model=UserSpaceRuntimeStatusResponse,
)
async def get_devserver_status(
    workspace_id: str,
    request: Request,
    user: Any = Depends(get_current_user),
):
    status = await userspace_runtime_service.get_devserver_status(workspace_id, user.id)
    status.preview_url = userspace_runtime_service.get_preview_origin(
        workspace_id,
        control_plane_origin=get_browser_matched_origin(request),
    )
    return status


@router.get(
    "/runtime/workspaces/{workspace_id}/tab-state",
    response_model=UserSpaceWorkspaceTabStateResponse,
)
async def get_workspace_tab_state(
    workspace_id: str,
    request: Request,
    selected_conversation_id: str | None = None,
    user: Any = Depends(get_current_user),
):
    tab_state = await userspace_runtime_service.get_workspace_tab_state(
        workspace_id,
        user.id,
        selected_conversation_id=selected_conversation_id,
        is_admin=bool(getattr(user, "role", None) == "admin"),
    )
    tab_state.runtime_status.preview_url = userspace_runtime_service.get_preview_origin(
        workspace_id,
        control_plane_origin=get_browser_matched_origin(request),
    )
    return tab_state


@router.post(
    "/runtime/workspaces/{workspace_id}/devserver/restart",
    response_model=UserSpaceRuntimeActionResponse,
)
async def restart_devserver(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.restart_devserver(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/mounts/{mount_id}/refresh",
    response_model=UserSpaceRuntimeActionResponse,
)
async def refresh_runtime_mount(
    workspace_id: str,
    mount_id: str,
    user: Any = Depends(get_current_user),
):
    result = await userspace_runtime_service.refresh_workspace_mount(
        workspace_id,
        user.id,
        mount_id,
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id,
        "mount_refresh",
    )
    return result


@router.get("/runtime/workspaces/{workspace_id}/screenshots/{filename}")
async def get_runtime_screenshot(
    workspace_id: str,
    filename: str,
    user: Any = Depends(get_current_user),
):
    await userspace_service.enforce_workspace_role(workspace_id, user.id, "viewer")

    normalized_name = (filename or "").strip().replace("\\", "/")
    basename = Path(normalized_name).name
    if (
        not basename
        or basename != normalized_name
        or not basename.lower().endswith(".png")
    ):
        raise HTTPException(status_code=400, detail="Invalid screenshot filename")

    index_data_root = Path(os.getenv("INDEX_DATA_PATH", "/data"))
    screenshot_dir = (
        index_data_root
        / "_userspace"
        / "workspaces"
        / workspace_id
        / "runtime-artifacts"
        / "screenshots"
    ).resolve()
    screenshot_path = (screenshot_dir / basename).resolve()

    try:
        screenshot_path.relative_to(screenshot_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid screenshot path") from exc

    if not screenshot_path.exists() or not screenshot_path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return FileResponse(
        path=str(screenshot_path),
        media_type="image/png",
        filename=basename,
    )


@router.post(
    "/runtime/workspaces/{workspace_id}/capability-token",
    response_model=UserSpaceCapabilityTokenResponse,
)
async def issue_capability_token(
    workspace_id: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else []
    if not isinstance(capabilities, list):
        capabilities = []
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    return await userspace_runtime_service.issue_capability_token(
        workspace_id,
        user.id,
        [str(value) for value in capabilities],
        str(session_id) if session_id else None,
    )


@router.post(
    "/runtime/workspaces/{workspace_id}/browser-auth",
    response_model=UserSpaceBrowserAuthResponse,
)
async def authorize_browser_surfaces(
    workspace_id: str,
    payload: UserSpaceBrowserAuthRequest,
    request: Request,
    response: Response,
    user: Any = Depends(get_current_user),
):
    surfaces = _normalize_browser_surfaces(payload.surfaces)
    authorizations: list[UserSpaceBrowserAuthorization] = []

    for surface in surfaces:
        config = _BROWSER_SURFACE_COOKIE_CONFIG[surface]
        token_response = await userspace_runtime_service.issue_capability_token(
            workspace_id,
            user.id,
            list(config["capabilities"]),
        )
        max_age = max(
            60,
            int(
                (token_response.expires_at - datetime.now(timezone.utc)).total_seconds()
            ),
        )
        response.set_cookie(
            key=str(config["cookie_name"]),
            value=token_response.token,
            max_age=max_age,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            path=str(config["path_builder"](workspace_id)),
        )
        authorizations.append(
            UserSpaceBrowserAuthorization(
                surface=surface,
                expires_at=token_response.expires_at,
            )
        )

    return UserSpaceBrowserAuthResponse(
        workspace_id=workspace_id,
        authorizations=authorizations,
    )


@router.post(
    "/runtime/workspaces/{workspace_id}/preview-launch",
    response_model=UserSpacePreviewLaunchResponse,
)
async def issue_workspace_preview_launch(
    workspace_id: str,
    payload: UserSpacePreviewLaunchRequest,
    request: Request,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.issue_workspace_preview_launch(
        workspace_id,
        user.id,
        control_plane_origin=get_browser_matched_origin(
            request,
            browser_origin=payload.parent_origin,
        ),
        path=payload.path,
        parent_origin=payload.parent_origin,
    )


@router.post(
    "/shared/{share_token}/preview-launch",
    response_model=UserSpacePreviewLaunchResponse,
)
@limiter.limit(SHARE_AUTH_RATE_LIMIT)
async def issue_shared_preview_launch(
    share_token: str,
    payload: UserSpacePreviewLaunchRequest,
    request: Request,
    response: Response,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    share_auth_token = share_auth_token_from_request(
        request.headers,
        request.cookies,
        share_token=share_token,
    )
    authorization = await userspace_service.authorize_shared_workspace_access(
        share_token,
        current_user=user,
        password=share_password,
        share_auth_token=share_auth_token,
    )
    workspace_id = authorization["workspace_id"]
    expires_at = authorization["expires_at"]
    if authorization["share_auth_token"] and expires_at is not None:
        max_age = max(
            60,
            int((expires_at - datetime.now(timezone.utc)).total_seconds()),
        )
        set_share_auth_cookie(
            response,
            authorization["share_auth_token"],
            max_age=max_age,
            secure=request.url.scheme == "https"
            or bool(getattr(settings, "session_cookie_secure", False)),
            share_token=share_token,
        )
    external_origin = get_browser_matched_origin(
        request,
        browser_origin=payload.parent_origin,
    )
    if payload.prefer_root_proxy:
        return _root_share_launch_response(
            workspace_id=workspace_id,
            base_url=external_origin,
            share_path=f"/shared/{quote(share_token, safe='')}",
            target_path=payload.path,
        )
    return await userspace_runtime_service.issue_shared_preview_launch(
        workspace_id,
        control_plane_origin=external_origin,
        path=payload.path,
        parent_origin=payload.parent_origin,
        share_token=share_token,
        subject_user_id=str(getattr(user, "id", "") or "") or None,
    )


@router.post(
    "/shared/{owner_username}/{share_slug}/preview-launch",
    response_model=UserSpacePreviewLaunchResponse,
)
@limiter.limit(SHARE_AUTH_RATE_LIMIT)
async def issue_shared_preview_launch_by_slug(
    owner_username: str,
    share_slug: str,
    payload: UserSpacePreviewLaunchRequest,
    request: Request,
    response: Response,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    share_auth_token = share_auth_token_from_request(
        request.headers,
        request.cookies,
        owner_username=owner_username,
        share_slug=share_slug,
    )
    authorization = await userspace_service.authorize_shared_workspace_access_by_slug(
        owner_username,
        share_slug,
        current_user=user,
        password=share_password,
        share_auth_token=share_auth_token,
    )
    workspace_id = authorization["workspace_id"]
    expires_at = authorization["expires_at"]
    if authorization["share_auth_token"] and expires_at is not None:
        max_age = max(
            60,
            int((expires_at - datetime.now(timezone.utc)).total_seconds()),
        )
        set_share_auth_cookie(
            response,
            authorization["share_auth_token"],
            max_age=max_age,
            secure=request.url.scheme == "https"
            or bool(getattr(settings, "session_cookie_secure", False)),
            owner_username=owner_username,
            share_slug=share_slug,
        )
    external_origin = get_browser_matched_origin(
        request,
        browser_origin=payload.parent_origin,
    )
    if payload.prefer_root_proxy:
        return _root_share_launch_response(
            workspace_id=workspace_id,
            base_url=external_origin,
            share_path=(
                f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}"
            ),
            target_path=payload.path,
        )
    return await userspace_runtime_service.issue_shared_preview_launch(
        workspace_id,
        control_plane_origin=external_origin,
        path=payload.path,
        parent_origin=payload.parent_origin,
        owner_username=owner_username,
        share_slug=share_slug,
        subject_user_id=str(getattr(user, "id", "") or "") or None,
    )


@router.get(
    "/runtime/workspaces/{workspace_id}/fs/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def runtime_fs_read(
    workspace_id: str,
    file_path: str,
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.runtime_fs_read",
    )
    return await userspace_runtime_service.runtime_fs_read(
        workspace_id,
        file_path,
        user_id,
    )


@router.put(
    "/runtime/workspaces/{workspace_id}/fs/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def runtime_fs_write(
    workspace_id: str,
    file_path: str,
    payload: dict[str, Any],
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.runtime_fs_write",
    )
    return await userspace_runtime_service.runtime_fs_write(
        workspace_id,
        file_path,
        str(payload.get("content", "")),
        user_id,
    )


@router.delete("/runtime/workspaces/{workspace_id}/fs/{file_path:path}")
async def runtime_fs_delete(
    workspace_id: str,
    file_path: str,
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.runtime_fs_write",
    )
    return await userspace_runtime_service.runtime_fs_delete(
        workspace_id,
        file_path,
        user_id,
    )


@router.websocket("/runtime/workspaces/{workspace_id}/pty")
async def runtime_pty(workspace_id: str, websocket: WebSocket):
    token = _extract_cookie_token_from_websocket(
        websocket,
        _RUNTIME_PTY_CAPABILITY_COOKIE,
    )
    try:
        _, user_id = _require_workspace_capability(
            token,
            workspace_id,
            "userspace.runtime_pty",
        )
    except HTTPException:
        await websocket.close(code=4401)
        return

    try:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
    except HTTPException:
        await websocket.close(code=4403)
        return

    can_write = True
    try:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
    except HTTPException:
        can_write = False

    upstream_ws_url = (
        await userspace_runtime_service.build_workspace_pty_upstream_ws_url(
            workspace_id,
            user_id,
        )
    )
    if not can_write:
        await _proxy_websocket_request(
            websocket,
            upstream_ws_url,
            read_only=True,
        )
        return

    await _proxy_websocket_request(websocket, upstream_ws_url)


@router.websocket("/collab/workspaces/{workspace_id}/files/{file_path:path}")
async def collab_file_socket(workspace_id: str, file_path: str, websocket: WebSocket):
    token = _extract_cookie_token_from_websocket(
        websocket,
        _COLLAB_CAPABILITY_COOKIE,
    )
    try:
        _, user_id = _require_workspace_capability(
            token,
            workspace_id,
            "userspace.collab_connect",
        )
    except HTTPException:
        await websocket.close(code=4401)
        return

    try:
        snapshot = await userspace_runtime_service.get_collab_snapshot(
            workspace_id,
            file_path,
            user_id,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    can_edit = not snapshot.read_only

    await websocket.accept()
    await userspace_runtime_service.register_collab_client(
        workspace_id, snapshot.file_path, websocket, user_id
    )

    async def _cleanup_collab() -> None:
        users = await userspace_runtime_service.clear_collab_presence(
            workspace_id,
            snapshot.file_path,
            user_id,
        )
        await userspace_runtime_service.unregister_collab_client(
            workspace_id,
            snapshot.file_path,
            websocket,
        )
        await _broadcast_collab_message(
            workspace_id,
            snapshot.file_path,
            {
                "type": "presence",
                "workspace_id": workspace_id,
                "file_path": snapshot.file_path,
                "users": users,
            },
        )

    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "workspace_id": workspace_id,
                    "file_path": snapshot.file_path,
                    "version": snapshot.version,
                    "content": snapshot.content,
                    "read_only": snapshot.read_only,
                }
            )
        )
        current_presence = await userspace_runtime_service.get_collab_presence(
            workspace_id,
            snapshot.file_path,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "presence",
                    "workspace_id": workspace_id,
                    "file_path": snapshot.file_path,
                    "users": current_presence,
                }
            )
        )

        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message_type = payload.get("type")
            if message_type == "presence":
                users = await userspace_runtime_service.update_collab_presence(
                    workspace_id,
                    snapshot.file_path,
                    user_id,
                    payload if isinstance(payload, dict) else {},
                )
                await _broadcast_collab_message(
                    workspace_id,
                    snapshot.file_path,
                    {
                        "type": "presence",
                        "workspace_id": workspace_id,
                        "file_path": snapshot.file_path,
                        "users": users,
                    },
                )
                continue
            if message_type != "update":
                continue
            if not can_edit:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "Read-only collaboration session"}
                    )
                )
                continue

            content = str(payload.get("content", ""))
            client_version_raw = payload.get("version")
            client_version: int | None = None
            if isinstance(client_version_raw, (int, float, str)):
                version_str = str(client_version_raw).strip()
                if version_str.isdigit():
                    parsed = int(version_str)
                    if parsed > 0:
                        client_version = parsed
            try:
                updated = await userspace_runtime_service.apply_collab_update(
                    workspace_id,
                    snapshot.file_path,
                    content,
                    user_id,
                    expected_version=client_version,
                )
            except RuntimeVersionConflictError as conflict:
                latest = await userspace_runtime_service.get_collab_snapshot(
                    workspace_id,
                    snapshot.file_path,
                    user_id,
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": (
                                f"Version conflict: expected {conflict.expected_version}, "
                                f"current {conflict.actual_version}"
                            ),
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "snapshot",
                            "workspace_id": latest.workspace_id,
                            "file_path": latest.file_path,
                            "version": latest.version,
                            "content": latest.content,
                            "read_only": latest.read_only,
                        }
                    )
                )
                continue
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ack",
                        "workspace_id": updated.workspace_id,
                        "file_path": updated.file_path,
                        "version": updated.version,
                    }
                )
            )

    except WebSocketDisconnect:
        await _cleanup_collab()
    except Exception:
        # Client disconnected during send (e.g. initial snapshot/presence).
        # Treat identically to WebSocketDisconnect for cleanup purposes.
        await _cleanup_collab()


@router.post("/collab/workspaces/{workspace_id}/files/create")
async def collab_create_file(
    workspace_id: str,
    payload: dict[str, Any],
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.collab_mutate",
    )
    file_path = str(payload.get("file_path", "")).strip()
    content = str(payload.get("content", ""))
    created = await userspace_runtime_service.create_collab_file(
        workspace_id,
        file_path,
        user_id,
        content=content,
    )
    await _broadcast_collab_message(
        workspace_id,
        created.file_path,
        {
            "type": "file_created",
            "workspace_id": workspace_id,
            "file_path": created.file_path,
            "version": created.version,
        },
    )
    return {
        "success": True,
        "workspace_id": workspace_id,
        "file_path": created.file_path,
        "version": created.version,
    }


@router.post("/collab/workspaces/{workspace_id}/files/rename")
async def collab_rename_file(
    workspace_id: str,
    payload: dict[str, Any],
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.collab_mutate",
    )
    old_path = str(payload.get("old_path", "")).strip()
    new_path = str(payload.get("new_path", "")).strip()
    result = await userspace_runtime_service.rename_collab_file(
        workspace_id,
        old_path,
        new_path,
        user_id,
    )
    await _broadcast_collab_message(
        workspace_id,
        result["new_path"],
        {
            "type": "file_renamed",
            "workspace_id": workspace_id,
            "old_path": result["old_path"],
            "new_path": result["new_path"],
        },
    )
    return result


@router.post("/collab/workspaces/{workspace_id}/files/delete")
async def collab_delete_file(
    workspace_id: str,
    payload: dict[str, Any],
    request: Request,
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.collab_mutate",
    )
    file_path = str(payload.get("file_path", "")).strip()
    result = await userspace_runtime_service.delete_collab_file(
        workspace_id,
        file_path,
        user_id,
    )
    return result
    return result
