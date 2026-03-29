from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import (APIRouter, Depends, Header, HTTPException, Request,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, Response, StreamingResponse

from ragtime.config.settings import settings
from ragtime.core.auth import validate_session
from ragtime.core.database import get_db
from ragtime.core.security import get_current_user, get_current_user_optional
from ragtime.userspace.models import (UserSpaceCapabilityTokenResponse,
                                      UserSpaceFileResponse,
                                      UserSpaceRuntimeActionResponse,
                                      UserSpaceRuntimeSessionResponse,
                                      UserSpaceRuntimeStatusResponse)
from ragtime.userspace.runtime_service import (RuntimeVersionConflictError,
                                               userspace_runtime_service)
from ragtime.userspace.service import (_RUNTIME_BRIDGE_CONTENT,
                                       userspace_service)

router = APIRouter(prefix="/indexes/userspace", tags=["User Space Runtime"])


_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
_PREVIEW_CAPABILITY_COOKIE = "userspace_preview_capability"


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
    *,
    proxy_base_path: str | None = None,
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
    out = {
        key: value
        for key, value in headers.items()
        if key.lower() not in blocked
    }
    # Rewrite root-relative Location headers so browser redirects stay inside
    # the proxy chain instead of escaping to the outer Ragtime origin.
    if proxy_base_path:
        location = out.get("location") or out.get("Location") or ""
        if location.startswith("/") and not location.startswith("//"):
            base = proxy_base_path.rstrip("/")
            out["location"] = base + location
    return out


def _is_html_media_type(media_type: str) -> bool:
    return "text/html" in (media_type or "").lower()


def _should_rewrite_proxy_content(media_type: str) -> bool:
    """Return True for response content-types that may contain root-relative URLs."""
    mt = (media_type or "").lower()
    return any(
        t in mt
        for t in (
            "text/html",
            "text/css",
        )
    )


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
    cookie_token = request.cookies.get(_PREVIEW_CAPABILITY_COOKIE, "").strip()
    return cookie_token or None


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
    cookie_token = websocket.cookies.get(_PREVIEW_CAPABILITY_COOKIE, "").strip()
    return cookie_token or None


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

    await websocket.accept()

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

    try:
        async with websockets_module.connect(
            upstream_url,
            max_size=None,
            open_timeout=20,
        ) as upstream:

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
    proxy_base_path: str | None = None,
) -> Response:
    if request.headers.get("upgrade", "").lower() == "websocket":
        raise HTTPException(
            status_code=501,
            detail="WebSocket proxy upgrades are not available on this route",
        )

    body = await request.body()
    headers = _proxy_request_headers(request)
    if proxy_base_path:
        headers.setdefault("x-forwarded-prefix", proxy_base_path)

    # Inject runtime worker auth token for upstream worker requests
    worker_token = getattr(settings, "userspace_runtime_worker_auth_token", "") or ""
    if worker_token:
        headers["authorization"] = f"Bearer {worker_token}"

    timeout = httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=5.0)
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
    resp_headers = _proxy_response_headers(
        upstream_response.headers, proxy_base_path=proxy_base_path
    )

    rewrite = proxy_base_path and _should_rewrite_proxy_content(media_type)
    if rewrite:
        base_path: str = proxy_base_path  # type: ignore[assignment]
        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
            await client.aclose()

        content = _rewrite_root_relative_urls(content, base_path)
        if _is_html_media_type(media_type):
            content = _inject_bridge_script(content)
        # Content length changed after rewriting; drop stale header so
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


_BRIDGE_SCRIPT_TAG = b'<script src="/indexes/userspace/runtime-bridge.js"></script>'
_BRIDGE_DETECT_RE = re.compile(rb"bridge\.js", re.IGNORECASE)
_HEAD_CLOSE_RE = re.compile(rb"(</head\s*>)", re.IGNORECASE)
_FIRST_SCRIPT_RE = re.compile(rb"(<script[\s>])", re.IGNORECASE)


def _inject_bridge_script(html: bytes) -> bytes:
    """Inject the platform data-bridge script into HTML responses.

    Inserts ``<script src="/indexes/userspace/runtime-bridge.js">`` before
    ``</head>`` or the first ``<script`` tag so ``window.__ragtime_context``
    and platform visualization libraries are available to workspace code.
    Skips injection if bridge.js is already referenced.
    """
    if _BRIDGE_DETECT_RE.search(html):
        return html
    tag = _BRIDGE_SCRIPT_TAG + b"\n"
    m = _HEAD_CLOSE_RE.search(html)
    if m:
        return html[: m.start()] + tag + html[m.start() :]
    m = _FIRST_SCRIPT_RE.search(html)
    if m:
        return html[: m.start()] + tag + html[m.start() :]
    return html


@router.get("/runtime-bridge.js", include_in_schema=False)
async def runtime_bridge_script() -> Response:
    return Response(
        content=_RUNTIME_BRIDGE_CONTENT,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=3600",
        },
    )


def _rewrite_root_relative_urls(content: bytes, proxy_base_path: str) -> bytes:
    """Rewrite root-relative URLs in text content so they route through the
    preview proxy.

    Matches *any* single-quoted, double-quoted, or backtick-quoted string
    literal that begins with ``/`` (excluding protocol-relative ``//``).
    This covers HTML attributes (``src``, ``href``, …), ES module ``import``
    statements, CSS ``url()`` references, ``fetch()`` calls, and every other
    context where a root-relative path appears as a string literal.

    A negative-lookahead prevents double-rewriting when the path already
    starts with the proxy base prefix.
    """
    base = proxy_base_path.rstrip("/").encode()
    base_no_leading_slash = proxy_base_path.lstrip("/").rstrip("/").encode()
    pattern = re.compile(
        rb'(["' + rb"'`" + rb'])(/)(?!/)(?!' + re.escape(base_no_leading_slash) + rb')'
    )

    def _replace(m: re.Match[bytes]) -> bytes:
        return m.group(1) + base + m.group(2)

    return pattern.sub(_replace, content)


async def _websocket_user(websocket: WebSocket) -> Any | None:
    session_token = websocket.cookies.get("ragtime_session")
    if not session_token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header[7:]

    if not session_token:
        return None

    token_data = await validate_session(session_token)
    if not token_data:
        return None

    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


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
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.get_devserver_status(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/devserver/start",
    response_model=UserSpaceRuntimeActionResponse,
    deprecated=True,
    description="Alias for session/start. Prefer POST .../session/start instead.",
)
async def start_devserver(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.start_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/devserver/restart",
    response_model=UserSpaceRuntimeActionResponse,
)
async def restart_devserver(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.restart_devserver(workspace_id, user.id)


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
    screenshot_dir = (index_data_root / "_tmp" / workspace_id).resolve()
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
    token = _extract_capability_token_from_websocket(websocket)
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
    token = _extract_capability_token_from_websocket(websocket)
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


@router.api_route("/workspaces/{workspace_id}/preview", methods=_PROXY_METHODS)
@router.api_route(
    "/workspaces/{workspace_id}/preview/{path:path}", methods=_PROXY_METHODS
)
async def workspace_preview_proxy(
    workspace_id: str,
    request: Request,
    path: str = "",
):
    token = _extract_capability_token_from_request(request)
    _, user_id = _require_workspace_capability(
        token,
        workspace_id,
        "userspace.preview_http",
    )
    capability_token = token or ""
    upstream_url = await userspace_runtime_service.build_workspace_preview_upstream_url(
        workspace_id,
        user_id,
        path,
        query=_sanitize_preview_query(request.url.query),
    )
    base = f"/indexes/userspace/workspaces/{workspace_id}/preview"
    response = await _proxy_http_request(request, upstream_url, proxy_base_path=base)
    response.set_cookie(
        key=_PREVIEW_CAPABILITY_COOKIE,
        value=capability_token,
        max_age=900,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path=base,
    )
    return response


@router.api_route(
    "/shared/{owner_username}/{share_slug}/preview", methods=_PROXY_METHODS
)
@router.api_route(
    "/shared/{owner_username}/{share_slug}/preview/{path:path}",
    methods=_PROXY_METHODS,
)
async def shared_preview_proxy(
    owner_username: str,
    share_slug: str,
    request: Request,
    path: str = "",
    share_password: str | None = Header(
        default=None, alias="X-UserSpace-Share-Password"
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
        owner_username,
        share_slug,
        current_user=user,
        password=share_password,
    )
    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=request.url.query or None,
    )
    base = f"/indexes/userspace/shared/{owner_username}/{share_slug}/preview"
    return await _proxy_http_request(request, upstream_url, proxy_base_path=base)


@router.api_route("/shared/{share_token}/preview", methods=_PROXY_METHODS)
@router.api_route("/shared/{share_token}/preview/{path:path}", methods=_PROXY_METHODS)
async def shared_token_preview_proxy(
    share_token: str,
    request: Request,
    path: str = "",
    share_password: str | None = Header(
        default=None, alias="X-UserSpace-Share-Password"
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    workspace_id = await userspace_service.resolve_shared_workspace_id(
        share_token,
        current_user=user,
        password=share_password,
    )
    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=request.url.query or None,
    )
    base = f"/indexes/userspace/shared/{share_token}/preview"
    return await _proxy_http_request(request, upstream_url, proxy_base_path=base)


@router.websocket("/workspaces/{workspace_id}/preview")
@router.websocket("/workspaces/{workspace_id}/preview/{path:path}")
async def workspace_preview_proxy_websocket(
    workspace_id: str,
    websocket: WebSocket,
    path: str = "",
):
    token = _extract_capability_token_from_websocket(websocket)
    try:
        _, user_id = _require_workspace_capability(
            token,
            workspace_id,
            "userspace.preview_ws",
        )
    except HTTPException:
        await websocket.close(code=4401)
        return

    try:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
    except HTTPException:
        await websocket.close(code=4403)
        return

    upstream_url = await userspace_runtime_service.build_workspace_preview_upstream_url(
        workspace_id,
        user_id,
        path,
        query=_sanitize_preview_query(websocket.url.query),
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


@router.websocket("/shared/{owner_username}/{share_slug}")
@router.websocket("/shared/{owner_username}/{share_slug}/{path:path}")
async def shared_preview_proxy_websocket(
    owner_username: str,
    share_slug: str,
    websocket: WebSocket,
    path: str = "",
):
    user = await _websocket_user(websocket)
    share_password = websocket.headers.get("x-userspace-share-password")

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
            owner_username,
            share_slug,
            current_user=user,
            password=share_password,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=websocket.url.query or None,
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


@router.websocket("/shared/{share_token}/preview")
@router.websocket("/shared/{share_token}/preview/{path:path}")
async def shared_token_preview_proxy_websocket(
    share_token: str,
    websocket: WebSocket,
    path: str = "",
):
    user = await _websocket_user(websocket)
    share_password = websocket.headers.get("x-userspace-share-password")

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id(
            share_token,
            current_user=user,
            password=share_password,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=websocket.url.query or None,
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))
