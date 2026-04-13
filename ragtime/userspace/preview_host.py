from __future__ import annotations

import asyncio
import heapq
from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import RedirectResponse
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse

from ragtime.config import settings
from ragtime.userspace.models import (ExecuteComponentRequest,
                                      ExecuteComponentResponse)
from ragtime.userspace.runtime_routes import (_PROXY_METHODS,
                                              _proxy_http_request,
                                              _proxy_websocket_request,
                                              _sanitize_preview_query,
                                              _to_websocket_url)
from ragtime.userspace.runtime_service import (
    _DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN, _RUNTIME_PREVIEW_GRANT_KIND,
    _RUNTIME_PREVIEW_SESSION_COOKIE_NAME, _RUNTIME_PREVIEW_SESSION_KIND,
    userspace_runtime_service)
from ragtime.userspace.service import userspace_service

preview_host_app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@preview_host_app.exception_handler(HTTPException)
async def _handle_preview_auth_error(request: StarletteRequest, exc: HTTPException):
    """Redirect unauthenticated visitors to the share link password/login page
    instead of showing a raw JSON 401."""
    if exc.status_code != 401:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if getattr(request, "scope", {}).get("type") != "http":
        return JSONResponse(status_code=401, content={"detail": exc.detail})
    if request.method != "GET" or request.url.path.startswith("/__ragtime/"):
        return JSONResponse(status_code=401, content={"detail": exc.detail})
    workspace_id = _workspace_id_from_preview_host(request.headers.get("host"))
    if workspace_id:
        share_token = await userspace_service.get_share_token(workspace_id)
        if share_token:
            # Derive control-plane origin from the subdomain host header:
            # strip the workspace label to get "base_domain:port".
            hostname, port = _split_host(request.headers.get("host"))
            base_domain = hostname.split(".", 1)[1] if "." in hostname else hostname
            scheme = request.url.scheme or "http"
            port_suffix = f":{port}" if port else ""
            share_url = f"{scheme}://{base_domain}{port_suffix}/shared/{quote(share_token, safe='')}"
            return RedirectResponse(url=share_url, status_code=302)
    return JSONResponse(status_code=401, content={"detail": exc.detail})


# ---------------------------------------------------------------------------
# In-memory preview host session registry
#
# Internal workspace preview surfaces may run in embedded browser contexts
# where SameSite=Lax cookies are not sent reliably. The bootstrap endpoint
# registers those sessions here so the preview host can still resolve them by
# host label when the cookie is absent. Shared-preview auth does not use this
# fallback anymore; protected shared subdomains are cookie-backed only.
# ---------------------------------------------------------------------------

_SESSION_REGISTRY_MAX = 500  # prevent unbounded growth


@dataclass
class _PreviewHostSessionEntry:
    claims: dict[str, Any]
    expires_at: datetime
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_preview_host_sessions: dict[str, _PreviewHostSessionEntry] = {}
_preview_host_session_order: OrderedDict[str, None] = OrderedDict()
_preview_host_expiry_heap: list[tuple[float, int, str]] = []
_registry_lock = asyncio.Lock()
_registry_sequence = count()


def _discard_preview_session_locked(label: str) -> None:
    _preview_host_sessions.pop(label, None)
    _preview_host_session_order.pop(label, None)


def _evict_preview_sessions_locked(now: datetime) -> None:
    now_ts = now.timestamp()
    while _preview_host_expiry_heap and _preview_host_expiry_heap[0][0] <= now_ts:
        expires_ts, _, label = heapq.heappop(_preview_host_expiry_heap)
        entry = _preview_host_sessions.get(label)
        if entry is None or entry.expires_at.timestamp() != expires_ts:
            continue
        _discard_preview_session_locked(label)

    while len(_preview_host_sessions) > _SESSION_REGISTRY_MAX:
        oldest_label, _ = _preview_host_session_order.popitem(last=False)
        _preview_host_sessions.pop(oldest_label, None)


async def invalidate_preview_sessions_for_workspace(workspace_id: str) -> None:
    """Remove all in-memory preview session registry entries for *workspace_id*."""
    label = (workspace_id or "").strip().lower()
    if not label:
        return
    async with _registry_lock:
        _discard_preview_session_locked(label)


def _host_label_from_hostname(hostname: str) -> str:
    """Extract the first sub-domain label (e.g. 'ws-abc' from 'ws-abc.localhost')."""
    return hostname.split(".", 1)[0] if "." in hostname else hostname


def _workspace_id_from_preview_host(host_header: str | None) -> str | None:
    hostname, _ = _split_host(host_header)
    label = _host_label_from_hostname(hostname)
    workspace_id = label.strip().lower()
    return workspace_id or None


async def _register_preview_session(
    host_header: str,
    claims: dict[str, Any],
    expires_at: datetime,
) -> None:
    hostname, _ = _split_host(host_header)
    label = _host_label_from_hostname(hostname)
    if not label:
        return
    now = datetime.now(timezone.utc)
    async with _registry_lock:
        _preview_host_sessions[label] = _PreviewHostSessionEntry(
            claims=claims,
            expires_at=expires_at,
        )
        _preview_host_session_order.pop(label, None)
        _preview_host_session_order[label] = None
        heapq.heappush(
            _preview_host_expiry_heap,
            (expires_at.timestamp(), next(_registry_sequence), label),
        )
        _evict_preview_sessions_locked(now)


async def _lookup_preview_session(host_header: str) -> dict[str, Any] | None:
    hostname, _ = _split_host(host_header)
    label = _host_label_from_hostname(hostname)
    if not label:
        return None
    now = datetime.now(timezone.utc)
    async with _registry_lock:
        _evict_preview_sessions_locked(now)
        entry = _preview_host_sessions.get(label)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _discard_preview_session_locked(label)
            return None
        _preview_host_session_order.move_to_end(label)
    return entry.claims


def _split_host(value: str | None) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    if not raw:
        return "", None
    if raw.startswith("["):
        end = raw.find("]")
        if end != -1:
            host = raw[: end + 1]
            port = raw[end + 2 :] if len(raw) > end + 2 and raw[end + 1] == ":" else None
            return host.lower(), port or None
    if ":" not in raw:
        return raw.lower(), None
    host, port = raw.rsplit(":", 1)
    if port.isdigit():
        return host.lower(), port
    return raw.lower(), None


def _preview_base_domains() -> set[str]:
    base_domain = str(getattr(settings, "userspace_preview_base_domain", "") or "").strip().strip(".").lower()
    domains = {base_domain} if base_domain else set()
    if base_domain == _DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN:
        domains.add("localhost")
    return domains


def is_preview_host(host_header: str | None) -> bool:
    hostname, _ = _split_host(host_header)
    base_domains = _preview_base_domains()
    if not hostname or not base_domains:
        return False
    return any(hostname.endswith(f".{base_domain}") for base_domain in base_domains)


def _is_equivalent_preview_host(actual_host: str, expected_host: str) -> bool:
    actual_hostname, actual_port = _split_host(actual_host)
    expected_hostname, expected_port = _split_host(expected_host)
    if not actual_hostname or not expected_hostname:
        return False
    if actual_port != expected_port:
        return False

    actual_label = actual_hostname.split(".", 1)[0]
    expected_label = expected_hostname.split(".", 1)[0]
    if not actual_label or actual_label != expected_label:
        return False

    base_domains = _preview_base_domains()
    return any(actual_hostname.endswith(f".{base_domain}") for base_domain in base_domains) and any(
        expected_hostname.endswith(f".{base_domain}") for base_domain in base_domains
    )


def _scope_host(scope: MutableMapping[str, Any]) -> str:
    for key, value in scope.get("headers", []):
        if key == b"host":
            return value.decode("latin-1")
    return ""


def _ensure_preview_host_matches_workspace(
    host_header: str | None,
    workspace_id: str,
    expected_host: str | None = None,
) -> None:
    expected = str(expected_host or "").strip().lower()
    if not expected:
        expected = urlsplit(userspace_runtime_service.get_preview_origin(workspace_id)).netloc.lower()
    actual_host = str(host_header or "").strip().lower()
    if actual_host != expected and not _is_equivalent_preview_host(actual_host, expected):
        raise HTTPException(status_code=404, detail="Preview host mismatch")


def _bridge_context_from_claims(claims: dict[str, Any]) -> dict[str, Any]:
    parent_origin = str(claims.get("parent_origin") or "").strip() or None
    return {
        "parent_origin": parent_origin,
        "execute_url": "/__ragtime/execute-component",
    }


def _verify_preview_session_token(token: str | None) -> dict[str, Any]:
    if not token:
        raise HTTPException(status_code=401, detail="Preview session required")
    return userspace_runtime_service.verify_preview_token(
        token,
        expected_kind=_RUNTIME_PREVIEW_SESSION_KIND,
    )


async def _resolve_public_preview_session(host_header: str | None) -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_preview_host(host_header)
    if not workspace_id:
        return None
    if not await userspace_service.is_public_direct_share_host_enabled(workspace_id):
        return None
    return {
        "workspace_id": workspace_id,
        "preview_mode": "shared_public_host",
        "share_access_mode": "token",
        "preview_host": str(host_header or "").strip().lower() or None,
        "parent_origin": None,
    }


async def _enforce_shared_subdomain_allowed(claims: dict[str, Any]) -> None:
    workspace_id = str(claims.get("workspace_id") or "").strip()
    if not workspace_id:
        raise HTTPException(status_code=401, detail="Preview session required")
    if _preview_mode(claims) == "workspace":
        return
    if not await userspace_service.has_active_share_link(workspace_id):
        raise HTTPException(status_code=404, detail="Preview host unavailable")
    # When the workspace has protection enabled (anything other than plain
    # "token" mode), the session must carry a matching share_access_mode
    # proving it was created through the proper auth flow (password entry,
    # group check, etc.).  Sessions without share_access_mode (legacy or
    # stale) are rejected so visitors are forced back through the share URL.
    current_mode = await userspace_service.get_share_access_mode(workspace_id)
    if current_mode and current_mode != "token":
        session_access_mode = str(claims.get("share_access_mode") or "").strip()
        if session_access_mode != current_mode:
            raise HTTPException(status_code=401, detail="Share access changed \u2014 please use the share link again")


async def _resolve_preview_session(
    host_header: str | None,
    cookie_token: str | None,
) -> dict[str, Any]:
    """Resolve preview session claims from cookie or in-memory registry.

    Tries the cookie first. Falls back to the host-label registry only for
    internal workspace preview sessions, where embedded browser contexts may
    suppress SameSite=Lax cookies.
    """
    # 1. Try cookie-based auth
    if cookie_token:
        try:
            claims = _verify_preview_session_token(cookie_token)
            await _enforce_shared_subdomain_allowed(claims)
            _ensure_preview_host_matches_workspace(
                host_header,
                str(claims.get("workspace_id") or ""),
                str(claims.get("preview_host") or "").strip() or None,
            )
            return claims
        except HTTPException:
            pass  # fall through to registry

    # 2. Fall back to in-memory host-label registry
    registry_claims = await _lookup_preview_session(host_header or "")
    if registry_claims is not None and _preview_mode(registry_claims) == "workspace":
        await _enforce_shared_subdomain_allowed(registry_claims)
        _ensure_preview_host_matches_workspace(
            host_header,
            str(registry_claims.get("workspace_id") or ""),
            str(registry_claims.get("preview_host") or "").strip() or None,
        )
        return registry_claims

    public_claims = await _resolve_public_preview_session(host_header)
    if public_claims is not None:
        _ensure_preview_host_matches_workspace(
            host_header,
            str(public_claims.get("workspace_id") or ""),
            str(public_claims.get("preview_host") or "").strip() or None,
        )
        return public_claims

    raise HTTPException(status_code=401, detail="Preview session required")


async def _verify_preview_session_cookie(request: Request) -> dict[str, Any]:
    token = request.cookies.get(_RUNTIME_PREVIEW_SESSION_COOKIE_NAME, "").strip()
    return await _resolve_preview_session(request.headers.get("host"), token or None)


async def _verify_preview_session_websocket(websocket: WebSocket) -> dict[str, Any]:
    token = websocket.cookies.get(_RUNTIME_PREVIEW_SESSION_COOKIE_NAME, "").strip()
    return await _resolve_preview_session(websocket.headers.get("host"), token or None)


def _preview_mode(claims: dict[str, Any]) -> str:
    return str(claims.get("preview_mode") or "workspace").strip() or "workspace"


async def _build_upstream_url(
    claims: dict[str, Any],
    *,
    path: str,
    query: str | None,
) -> str:
    workspace_id = str(claims.get("workspace_id") or "").strip()
    mode = _preview_mode(claims)
    if mode == "workspace":
        user_id = str(claims.get("sub") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="Preview session missing user")
        return await userspace_runtime_service.build_workspace_preview_upstream_url(
            workspace_id,
            user_id,
            path,
            query=query,
        )

    return await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=query,
    )


@preview_host_app.get("/__ragtime/bootstrap")
async def preview_bootstrap(request: Request, grant: str):
    claims = userspace_runtime_service.verify_preview_token(
        grant,
        expected_kind=_RUNTIME_PREVIEW_GRANT_KIND,
    )
    await _enforce_shared_subdomain_allowed(claims)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    _ensure_preview_host_matches_workspace(
        request.headers.get("host"),
        workspace_id,
        str(claims.get("preview_host") or "").strip() or None,
    )

    session_claims = {
        "workspace_id": workspace_id,
        "sub": str(claims.get("sub") or "").strip() or None,
        "preview_mode": _preview_mode(claims),
        "preview_host": str(claims.get("preview_host") or "").strip() or None,
        "parent_origin": claims.get("parent_origin"),
        "share_token": claims.get("share_token"),
        "owner_username": claims.get("owner_username"),
        "share_slug": claims.get("share_slug"),
        "share_access_mode": claims.get("share_access_mode"),
    }
    session_token, expires_at = userspace_runtime_service.build_preview_session_token(
        session_claims
    )

    # Register session in the in-memory host registry so the proxy handler
    # can authenticate subsequent requests even when the browser refuses to
    # send the SameSite cookie (cross-site iframe context).
    await _register_preview_session(
        request.headers.get("host", ""),
        session_claims,
        expires_at,
    )

    target_path = str(claims.get("target_path") or "/").strip() or "/"
    response = RedirectResponse(url=target_path, status_code=307)
    max_age = max(60, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key=_RUNTIME_PREVIEW_SESSION_COOKIE_NAME,
        value=session_token,
        max_age=max_age,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return response


@preview_host_app.get("/__ragtime/bridge.js")
async def preview_bridge_script(request: Request):
    claims = await _verify_preview_session_cookie(request)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    return Response(
        content=await userspace_service.build_runtime_bridge_content(workspace_id),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-store",
        },
    )


@preview_host_app.post(
    "/__ragtime/execute-component",
    response_model=ExecuteComponentResponse,
)
async def preview_execute_component(
    request: Request,
    payload: ExecuteComponentRequest,
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    mode = _preview_mode(claims)
    if mode == "workspace":
        user_id = str(claims.get("sub") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="Preview session missing user")
        return await userspace_service.execute_component(workspace_id, payload, user_id)

    return await userspace_service.execute_component_from_authorized_shared_preview(
        workspace_id,
        payload,
    )


@preview_host_app.api_route("/", methods=_PROXY_METHODS)
@preview_host_app.api_route("/{path:path}", methods=_PROXY_METHODS)
async def preview_proxy(request: Request, path: str = ""):
    claims = await _verify_preview_session_cookie(request)
    upstream_url = await _build_upstream_url(
        claims,
        path=path,
        query=_sanitize_preview_query(request.url.query),
    )
    return await _proxy_http_request(
        request,
        upstream_url,
        bridge_workspace_id=str(claims.get("workspace_id") or "").strip(),
        bridge_context=_bridge_context_from_claims(claims),
    )


@preview_host_app.websocket("/")
@preview_host_app.websocket("/{path:path}")
async def preview_proxy_websocket(websocket: WebSocket, path: str = ""):
    claims = await _verify_preview_session_websocket(websocket)
    upstream_url = await _build_upstream_url(
        claims,
        path=path,
        query=_sanitize_preview_query(websocket.url.query),
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


class PreviewHostDispatchMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in {"http", "websocket"} and is_preview_host(
            _scope_host(scope)
        ):
            await preview_host_app(scope, receive, send)
            return
        await self.app(scope, receive, send)
