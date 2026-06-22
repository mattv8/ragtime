from __future__ import annotations

import asyncio
import heapq
import json
from collections import OrderedDict
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import count
from typing import Any, cast
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.responses import RedirectResponse
from starlette.requests import Request as StarletteRequest
from starlette.responses import HTMLResponse, JSONResponse

from ragtime.config import settings
from ragtime.core.auth import authenticate, create_access_token, create_session, validate_session_and_fetch_user
from ragtime.core.auth_methods import AuthMethodStatusPayload, build_auth_method_statuses
from ragtime.core.database import get_db
from ragtime.userspace.html_templates import render_browser_auth_start_page_html
from ragtime.userspace.models import (
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    UserSpaceBrowserAuthRequest,
    UserSpaceBrowserAuthResponse,
    UserSpaceBrowserSurface,
)
from ragtime.userspace.preview_probe import PREVIEW_HOST_PROBE_HEADER, PREVIEW_HOST_PROBE_PATH, PREVIEW_HOST_PROBE_VALUE
from ragtime.userspace.runtime_routes import (
    _PREVIEW_SESSION_CAPABILITY,
    _PROXY_METHODS,
    _authorize_browser_surfaces_for_user_id,
    _clear_browser_surface_cookies,
    _normalize_browser_surfaces,
    _normalize_uploaded_table_upload,
    _parse_uploaded_document_upload,
    _parse_uploaded_spreadsheet_upload,
    _primitive_archive_extract,
    _primitive_capabilities,
    _primitive_job_create,
    _primitive_job_get,
    _primitive_job_put,
    _primitive_object_response,
    _primitive_object_write,
    _primitive_object_write_request,
    _primitive_progress_get,
    _primitive_progress_put,
    _primitive_session_payload,
    _primitive_upload_target,
    _primitive_workspace_file_response,
    _primitive_workspace_file_write,
    _primitive_workspace_file_write_request,
    _proxy_http_request,
    _proxy_websocket_request,
    _render_uploaded_document_preview_upload,
    _sanitize_preview_query,
    _to_websocket_url,
)

_RUNTIME_PREVIEW_GRANT_KIND = "userspace_preview_grant"
_RUNTIME_PREVIEW_SESSION_KIND = "userspace_preview_session"
_RUNTIME_PREVIEW_SESSION_COOKIE_NAME = "userspace_preview_session"
_RUNTIME_PREVIEW_HANDOFF_QUERY_PARAM = "__ragtime_preview_handoff"
_RUNTIME_PREVIEW_HANDOFF_TTL_SECONDS = 60


def _runtime_service() -> Any:
    # Lazy import avoids import cycle: runtime_service -> service -> preview_host.
    from ragtime.userspace.runtime_service import userspace_runtime_service

    return userspace_runtime_service


def _userspace_service() -> Any:
    # Lazy import avoids import cycle: service -> preview_host.
    from ragtime.userspace.service import userspace_service

    return userspace_service


async def _preview_user_from_id(user_id: str | None) -> Any | None:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return None
    try:
        db = await get_db()
        return await db.user.find_unique(where={"id": normalized_user_id})
    except Exception:
        return None


preview_host_app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@preview_host_app.exception_handler(HTTPException)
async def _handle_preview_auth_error(request: StarletteRequest, exc: HTTPException):
    """Redirect unauthenticated visitors to the share link password/login page
    instead of showing a raw JSON 401."""
    if exc.status_code != 401:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if getattr(request, "scope", {}).get("type") != "http":
        return JSONResponse(status_code=401, content={"detail": exc.detail})

    if request.method != "GET":
        return JSONResponse(status_code=401, content={"detail": exc.detail})

    is_bootstrap = request.url.path.startswith("/__ragtime/bootstrap")
    workspace_id = _workspace_id_from_preview_host(request.headers.get("host"))
    # Sec-Fetch-Dest == "document" identifies a top-level navigation (new tab,
    # bookmark, reload of the bootstrap URL). In that context the
    # postMessage() recovery below is useless because there is no parent
    # frame, which is why users see the subdomain URL stuck on
    # /__ragtime/bootstrap?grant=... after an expired or already-consumed
    # grant. For that case we bounce back to the main-origin /shared/{token}
    # so it can mint a fresh grant and re-enter the subdomain bootstrap.
    is_top_level_nav = str(request.headers.get("sec-fetch-dest", "")).lower() == "document"

    should_redirect_to_share = workspace_id and (not request.url.path.startswith("/__ragtime/") or (is_bootstrap and is_top_level_nav))
    if should_redirect_to_share:
        share_token = await _userspace_service().get_share_token(workspace_id)
        if share_token:
            # Derive control-plane origin from the subdomain host header:
            # strip the workspace label to get "base_domain:port".
            hostname, port = _split_host(request.headers.get("host"))
            base_domain = hostname.split(".", 1)[1] if "." in hostname else hostname
            scheme = request.url.scheme or "http"
            port_suffix = f":{port}" if port else ""
            share_url = f"{scheme}://{base_domain}{port_suffix}/shared/{quote(share_token, safe='')}"
            return RedirectResponse(url=share_url, status_code=302)

    wants_html = "text/html" in str(request.headers.get("accept", "")).lower()

    if is_bootstrap or wants_html:
        safe_detail = json.dumps({"detail": exc.detail})
        script = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script>
window.parent.postMessage({{
    "bridge": "userspace-exec-v1",
    "type": "ragtime-preview-session-expired",
    "error": {safe_detail}
}}, "*");
</script>
</body>
</html>"""
        return HTMLResponse(content=script, status_code=401)

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
_preview_bootstrap_handoffs: dict[str, _PreviewHostSessionEntry] = {}
_registry_lock = asyncio.Lock()
_registry_sequence = count()


def _discard_preview_session_locked(label: str) -> None:
    _preview_host_sessions.pop(label, None)
    _preview_host_session_order.pop(label, None)
    prefix = f"{label}:"
    for key in [key for key in _preview_bootstrap_handoffs if key.startswith(prefix)]:
        _preview_bootstrap_handoffs.pop(key, None)


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

    expired_handoffs = [key for key, entry in _preview_bootstrap_handoffs.items() if entry.expires_at <= now]
    for key in expired_handoffs:
        _preview_bootstrap_handoffs.pop(key, None)

    while len(_preview_bootstrap_handoffs) > _SESSION_REGISTRY_MAX:
        oldest_key = min(
            _preview_bootstrap_handoffs,
            key=lambda key: _preview_bootstrap_handoffs[key].registered_at,
        )
        _preview_bootstrap_handoffs.pop(oldest_key, None)


async def invalidate_preview_sessions_for_workspace(workspace_id: str) -> None:
    """Remove all in-memory preview session registry entries for *workspace_id*."""
    label = (workspace_id or "").strip().lower()
    if not label:
        return
    async with _registry_lock:
        _discard_preview_session_locked(label)


async def _invalidate_preview_session_for_host(host_header: str | None) -> None:
    hostname, _ = _split_host(host_header)
    label = _host_label_from_hostname(hostname)
    if not label:
        return
    async with _registry_lock:
        _discard_preview_session_locked(label)


def _host_label_from_hostname(hostname: str) -> str:
    """Extract the first sub-domain label (e.g. 'ws-abc' from 'ws-abc.localhost')."""
    return hostname.split(".", 1)[0] if "." in hostname else hostname


def _preview_handoff_key(host_header: str | None, nonce: str | None) -> str | None:
    normalized_nonce = str(nonce or "").strip()
    if not normalized_nonce:
        return None
    hostname, _ = _split_host(host_header)
    label = _host_label_from_hostname(hostname)
    if not label:
        return None
    return f"{label}:{normalized_nonce}"


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


async def _register_preview_handoff(
    host_header: str,
    nonce: str,
    claims: dict[str, Any],
) -> None:
    key = _preview_handoff_key(host_header, nonce)
    if key is None:
        return
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=_RUNTIME_PREVIEW_HANDOFF_TTL_SECONDS)
    async with _registry_lock:
        _preview_bootstrap_handoffs[key] = _PreviewHostSessionEntry(
            claims=claims,
            expires_at=expires_at,
        )
        _evict_preview_sessions_locked(now)


async def _consume_preview_handoff(host_header: str | None, nonce: str | None) -> dict[str, Any] | None:
    key = _preview_handoff_key(host_header, nonce)
    if key is None:
        return None
    now = datetime.now(timezone.utc)
    async with _registry_lock:
        _evict_preview_sessions_locked(now)
        entry = _preview_bootstrap_handoffs.pop(key, None)
    if entry is None or entry.expires_at <= now:
        return None
    return entry.claims


def _add_preview_handoff_to_target_path(target_path: str, nonce: str | None) -> str:
    normalized_nonce = str(nonce or "").strip()
    if not normalized_nonce:
        return target_path
    parsed = urlsplit(target_path or "/")
    if parsed.scheme or parsed.netloc:
        return "/"
    path = parsed.path or "/"
    if not path.startswith("/") or path.startswith("//"):
        path = "/"
    pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != _RUNTIME_PREVIEW_HANDOFF_QUERY_PARAM]
    pairs.append((_RUNTIME_PREVIEW_HANDOFF_QUERY_PARAM, normalized_nonce))
    return urlunsplit(("", "", path, urlencode(pairs, doseq=True), parsed.fragment))


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


def is_preview_host(host_header: str | None) -> bool:
    hostname, _ = _split_host(host_header)
    base_domains = _runtime_service().get_preview_base_domains()
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

    base_domains = _runtime_service().get_preview_base_domains()
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
        expected = urlsplit(_runtime_service().get_preview_origin(workspace_id)).netloc.lower()
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
    return _runtime_service().verify_preview_token(
        token,
        expected_kind=_RUNTIME_PREVIEW_SESSION_KIND,
    )


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    return None


def _extract_direct_preview_capability_token(request: Request) -> str | None:
    for header_name in ("x-userspace-preview-capability-token", "x-userspace-capability-token"):
        token = request.headers.get(header_name, "").strip()
        if token:
            return token
    return _extract_bearer_token(request)


def _preview_user_response(user: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(user, "id", "") or ""),
        "username": str(getattr(user, "username", "") or ""),
        "display_name": getattr(user, "displayName", None),
        "email": getattr(user, "email", None),
        "role": str(getattr(user, "role", "user") or "user"),
        "auth_provider": str(getattr(user, "authProvider", None) or "local"),
        "role_manually_set": bool(getattr(user, "roleManuallySet", False)),
        "source_provider": getattr(user, "sourceProvider", None),
        "source_synced_at": getattr(user, "sourceSyncedAt", None),
        "source_expires_at": getattr(user, "sourceExpiresAt", None),
        "cached_groups": list(getattr(user, "cachedGroups", None) or []),
        "manual_group_ids": [],
        "ldap_group_ids": [],
        "local_group_ids": [],
    }


def _safe_browser_auth_return_to(value: str | None) -> str:
    raw = str(value or "").strip() or "/"
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


def _browser_auth_start_url(
    *,
    auth_method_key: str | None = None,
    surfaces: Sequence[UserSpaceBrowserSurface] | None = None,
    return_to: str | None = None,
) -> str:
    params: list[tuple[str, str]] = []
    for surface in _normalize_browser_surfaces(surfaces or None):
        params.append(("surfaces", surface))
    normalized_method = str(auth_method_key or "").strip()
    if normalized_method:
        params.append(("auth_method_key", normalized_method))
    params.append(("return_to", _safe_browser_auth_return_to(return_to)))
    return "/__ragtime/browser-auth/start?" + urlencode(params)


def _browser_auth_request_from_values(
    *,
    surfaces: Sequence[UserSpaceBrowserSurface] | None,
    auth_method_key: str | None,
    username: str | None = None,
    password: str | None = None,
) -> UserSpaceBrowserAuthRequest:
    return UserSpaceBrowserAuthRequest(
        surfaces=_normalize_browser_surfaces(surfaces or None),
        auth_method_key=str(auth_method_key or "").strip() or None,
        username=username,
        password=password,
    )


def _coerce_browser_surfaces(raw_surfaces: Sequence[str] | None) -> list[UserSpaceBrowserSurface]:
    if raw_surfaces is None:
        return _normalize_browser_surfaces(None)
    surfaces: list[UserSpaceBrowserSurface] = []
    for surface in raw_surfaces:
        normalized = str(surface).strip()
        if normalized in {"collab", "runtime_pty"}:
            surfaces.append(cast(UserSpaceBrowserSurface, normalized))
    return _normalize_browser_surfaces(surfaces)


def _browser_auth_query_values(request: Request) -> tuple[list[UserSpaceBrowserSurface], str | None, str]:
    raw_surfaces: list[str] = []
    for value in request.query_params.getlist("surfaces"):
        raw_surfaces.extend(part.strip() for part in str(value).split(",") if part.strip())
    auth_method_key = str(request.query_params.get("auth_method_key") or "").strip() or None
    return_to = _safe_browser_auth_return_to(request.query_params.get("return_to"))
    return _coerce_browser_surfaces(raw_surfaces or None), auth_method_key, return_to


def _browser_auth_method_label(auth_method_key: str | None, methods: Sequence[AuthMethodStatusPayload]) -> str:
    normalized = str(auth_method_key or "").strip()
    for method in methods:
        if str(method.get("key") or "").strip() == normalized:
            return str(method.get("label") or normalized or "Ragtime")
    return "Ragtime"


async def _render_browser_auth_start_page(
    request: Request,
    *,
    surfaces: Sequence[UserSpaceBrowserSurface],
    auth_method_key: str | None,
    return_to: str,
    parent_origin: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    methods = await build_auth_method_statuses()
    method_label = _browser_auth_method_label(auth_method_key, methods)
    surface_value = ",".join(_normalize_browser_surfaces(surfaces or None))
    template_methods: list[dict[str, Any]] = [dict(method) for method in methods]
    username_value = settings.local_admin_user if settings.debug_mode else ""
    password_value = settings.local_admin_password if settings.debug_mode else ""
    html_body = render_browser_auth_start_page_html(
        method_label=method_label,
        methods=template_methods,
        surface_value=surface_value,
        auth_method_key=auth_method_key,
        return_to=return_to,
        username_value=username_value,
        password_value=password_value,
        parent_origin=parent_origin,
        error=error,
    )
    return HTMLResponse(content=html_body, status_code=status_code, headers={"Cache-Control": "no-store"})


async def _complete_preview_browser_auth_redirect(
    request: Request,
    claims: dict[str, Any],
    payload: UserSpaceBrowserAuthRequest,
    return_to: str,
) -> RedirectResponse:
    safe_return_to = _safe_browser_auth_return_to(return_to)
    redirect = RedirectResponse(url=safe_return_to, status_code=303)
    workspace_id, user = await _workspace_user_for_preview_auth(request, redirect, claims, payload)
    handoff_nonce = str(uuid4())
    await _register_preview_handoff(
        request.headers.get("host", ""),
        handoff_nonce,
        _workspace_preview_session_claims(request, workspace_id, str(user.id)),
    )
    redirect.headers["location"] = _add_preview_handoff_to_target_path(safe_return_to, handoff_nonce)
    await _authorize_browser_surfaces_for_user_id(
        workspace_id,
        payload,
        request,
        redirect,
        user_id=str(user.id),
    )
    redirect.headers["Cache-Control"] = "no-store"
    return redirect


async def _workspace_user_from_request_session(request: Request, workspace_id: str) -> Any | None:
    token = request.cookies.get(settings.session_cookie_name, "").strip() or _extract_bearer_token(request)
    if not token:
        return None
    _, user = await validate_session_and_fetch_user(token)
    if user is None:
        return None
    await _userspace_service().enforce_workspace_role(workspace_id, user.id, "viewer")
    return user


async def _workspace_user_from_request_capability(request: Request, workspace_id: str) -> Any | None:
    token = _extract_direct_preview_capability_token(request)
    if not token:
        return None
    try:
        claims = _runtime_service().verify_capability_token(
            token,
            workspace_id,
            _PREVIEW_SESSION_CAPABILITY,
        )
    except HTTPException:
        return None
    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        return None
    await _userspace_service().enforce_workspace_role(workspace_id, user_id, "viewer")
    return await _preview_user_from_id(user_id) or type("PreviewUser", (), {"id": user_id})()


async def _authenticate_preview_workspace_user(
    request: Request,
    response: Response,
    workspace_id: str,
    *,
    username: str | None,
    password: str | None,
) -> Any | None:
    normalized_username = str(username or "").strip()
    raw_password = str(password or "")
    if not normalized_username or not raw_password:
        return None

    result = await authenticate(normalized_username, raw_password)
    if not result.success:
        raise HTTPException(status_code=401, detail=result.error or "Authentication failed")
    if not result.user_id or not result.username:
        raise HTTPException(status_code=500, detail="Authentication succeeded but user data is missing")

    user = await _preview_user_from_id(result.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    await _userspace_service().enforce_workspace_role(workspace_id, user.id, "viewer")

    session_token = create_access_token(result.user_id, result.username, result.role)
    await create_session(
        user_id=result.user_id,
        token=session_token,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_token,
        httponly=settings.session_cookie_httponly,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        max_age=settings.jwt_expire_hours * 3600,
        path="/",
    )
    return user


def _workspace_preview_session_claims(request: Request, workspace_id: str, user_id: str) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "sub": user_id,
        "preview_mode": "workspace",
        "preview_host": str(request.headers.get("host") or "").strip().lower() or None,
        "parent_origin": None,
    }


async def _set_preview_session_cookie(
    request: Request,
    response: Response,
    claims: dict[str, Any],
) -> None:
    session_token, expires_at = _runtime_service().build_preview_session_token(claims)
    await _register_preview_session(
        request.headers.get("host", ""),
        claims,
        expires_at,
    )
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


async def _set_workspace_preview_session_for_user(
    request: Request,
    response: Response,
    workspace_id: str,
    user_id: str,
) -> None:
    session_claims = _workspace_preview_session_claims(request, workspace_id, user_id)
    _ensure_preview_host_matches_workspace(
        request.headers.get("host"),
        workspace_id,
        str(session_claims.get("preview_host") or "").strip() or None,
    )
    await _set_preview_session_cookie(request, response, session_claims)


async def _workspace_user_for_preview_auth(
    request: Request,
    response: Response,
    claims: dict[str, Any],
    payload: UserSpaceBrowserAuthRequest | None = None,
) -> tuple[str, Any]:
    workspace_id = str(claims.get("workspace_id") or "").strip()
    if not workspace_id:
        workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "").strip()
    if not workspace_id:
        raise HTTPException(status_code=401, detail="Workspace preview session required")

    if _preview_mode(claims) == "workspace":
        user_id = str(claims.get("sub") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="Preview session missing user")
        return workspace_id, await _preview_user_from_id(user_id) or type("PreviewUser", (), {"id": user_id})()

    user = await _workspace_user_from_request_session(request, workspace_id)
    if user is None:
        user = await _workspace_user_from_request_capability(request, workspace_id)
    if user is None and payload is not None:
        user = await _authenticate_preview_workspace_user(
            request,
            response,
            workspace_id,
            username=payload.username,
            password=payload.password,
        )
    if user is None:
        raise HTTPException(status_code=401, detail="Workspace preview session required")

    await _set_workspace_preview_session_for_user(request, response, workspace_id, str(user.id))
    return workspace_id, user


async def _resolve_public_preview_session(
    host_header: str | None,
) -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_preview_host(host_header)
    if not workspace_id:
        return None
    if not await _userspace_service().is_public_direct_share_host_enabled(workspace_id):
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
    if not await _userspace_service().has_active_share_link(workspace_id):
        raise HTTPException(status_code=404, detail="Preview host unavailable")
    # When the workspace has protection enabled (anything other than plain
    # "token" mode), the session must carry a matching share_access_mode
    # proving it was created through the proper auth flow (password entry,
    # group check, etc.).  Sessions without share_access_mode (legacy or
    # stale) are rejected so visitors are forced back through the share URL.
    current_mode = await _userspace_service().get_share_access_mode(workspace_id)
    if current_mode and current_mode != "token":
        session_access_mode = str(claims.get("share_access_mode") or "").strip()
        if session_access_mode != current_mode:
            raise HTTPException(
                status_code=401,
                detail="Share access changed \u2014 please use the share link again",
            )


async def _resolve_preview_session(
    host_header: str | None,
    cookie_token: str | None,
    *,
    allow_registry_fallback: bool = True,
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
    registry_claims = await _lookup_preview_session(host_header or "") if allow_registry_fallback else None
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


def _is_direct_preview_document_request(request: Request) -> bool:
    if request.method not in {"GET", "HEAD"}:
        return False
    if request.url.path.startswith("/__ragtime/"):
        return False
    last_path_segment = (request.url.path or "/").rsplit("/", 1)[-1].lower()
    if "." in last_path_segment and not last_path_segment.endswith((".html", ".htm")):
        return False
    accept = str(request.headers.get("accept", "")).lower()
    sec_fetch_dest = str(request.headers.get("sec-fetch-dest", "")).lower()
    sec_fetch_site = str(request.headers.get("sec-fetch-site", "")).lower()
    if sec_fetch_dest:
        if sec_fetch_dest != "document":
            return False
        return sec_fetch_site in {"", "none"}
    return not accept or "text/html" in accept or "*/*" in accept


async def _verify_preview_session_cookie(request: Request) -> dict[str, Any]:
    token = request.cookies.get(_RUNTIME_PREVIEW_SESSION_COOKIE_NAME, "").strip()
    handoff_claims = await _consume_preview_handoff(
        request.headers.get("host"),
        request.query_params.get(_RUNTIME_PREVIEW_HANDOFF_QUERY_PARAM),
    )
    if handoff_claims is not None:
        await _enforce_shared_subdomain_allowed(handoff_claims)
        _ensure_preview_host_matches_workspace(
            request.headers.get("host"),
            str(handoff_claims.get("workspace_id") or ""),
            str(handoff_claims.get("preview_host") or "").strip() or None,
        )
        return handoff_claims
    is_direct_document = _is_direct_preview_document_request(request)
    if is_direct_document and not token:
        # A top-level no-cookie direct visit must establish fresh public/shared
        # context. If an earlier authenticated workspace preview registered the
        # same host, leaving it in the process-wide fallback registry lets the
        # page's subsequent same-origin primitive calls inherit that unrelated
        # user session. Clear it before the document response is proxied.
        await _invalidate_preview_session_for_host(request.headers.get("host"))
    return await _resolve_preview_session(
        request.headers.get("host"),
        token or None,
        allow_registry_fallback=not is_direct_document,
    )


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
        return await _runtime_service().build_workspace_preview_upstream_url(
            workspace_id,
            user_id,
            path,
            query=query,
        )

    return await _runtime_service().build_shared_preview_upstream_url(
        workspace_id,
        path,
        query=query,
    )


@preview_host_app.get("/__ragtime/bootstrap")
async def preview_bootstrap(request: Request, grant: str):
    claims = _runtime_service().verify_preview_token(
        grant,
        expected_kind=_RUNTIME_PREVIEW_GRANT_KIND,
    )
    # Enforce single-use semantics: a leaked bootstrap URL must not be
    # replayable even within the grant TTL.
    await _runtime_service().consume_preview_grant(claims)
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
    target_path = _add_preview_handoff_to_target_path(
        str(claims.get("target_path") or "/").strip() or "/",
        str(claims.get("jti") or "").strip() or None,
    )
    response = RedirectResponse(url=target_path, status_code=307)
    await _set_preview_session_cookie(request, response, session_claims)
    await _register_preview_handoff(
        request.headers.get("host", ""),
        str(claims.get("jti") or "").strip(),
        session_claims,
    )
    # Prevent caching of the bootstrap redirect (which carried the grant
    # token in its URL) and block Referer propagation to the target page.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@preview_host_app.get(PREVIEW_HOST_PROBE_PATH)
async def preview_probe() -> Response:
    return Response(
        status_code=204,
        headers={
            PREVIEW_HOST_PROBE_HEADER: PREVIEW_HOST_PROBE_VALUE,
            "Cache-Control": "no-store",
        },
    )


@preview_host_app.get("/__ragtime/bridge.js")
async def preview_bridge_script(request: Request):
    claims = await _verify_preview_session_cookie(request)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    return Response(
        content=await _userspace_service().build_runtime_bridge_content(workspace_id),
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
        return await _userspace_service().execute_component(workspace_id, payload, user_id)

    return await _userspace_service().execute_component_from_authorized_shared_preview(
        workspace_id,
        payload,
    )


@preview_host_app.get("/__ragtime/capabilities")
async def preview_capabilities(request: Request):
    claims = await _verify_preview_session_cookie(request)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    user_id = str(claims.get("sub") or "").strip() or None
    return await _primitive_capabilities(workspace_id, user_id, preview_mode=_preview_mode(claims))


@preview_host_app.get("/__ragtime/session")
async def preview_session(request: Request):
    claims = await _verify_preview_session_cookie(request)
    workspace_id = str(claims.get("workspace_id") or "").strip()
    user_id = str(claims.get("sub") or "").strip() or None
    mode = _preview_mode(claims)
    return await _primitive_session_payload(
        workspace_id,
        user_id,
        mode=mode,
        share_access_mode=claims.get("share_access_mode"),
        user=await _preview_user_from_id(user_id),
        same_origin_auth_endpoints=True,
    )


@preview_host_app.get("/auth/status")
async def preview_auth_status(request: Request):
    user = await _workspace_user_from_request_session(
        request,
        str(_workspace_id_from_preview_host(request.headers.get("host")) or ""),
    )
    return {
        "authenticated": user is not None,
        "ldap_configured": any(str(method.get("key") or "") == "ldap" and bool(method.get("configured")) for method in await build_auth_method_statuses()),
        "local_admin_enabled": bool(settings.local_admin_password),
        "debug_mode": False,
        "debug_username": settings.local_admin_user if settings.debug_mode else None,
        "debug_password": settings.local_admin_password if settings.debug_mode else None,
        "cookie_warning": None,
        "api_key_configured": False,
        "session_cookie_secure": False,
        "allowed_origins_open": False,
        "auth_methods": await build_auth_method_statuses(),
        "server_name": "Ragtime",
    }


@preview_host_app.get("/auth/me")
async def preview_auth_me(request: Request):
    workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "")
    user = await _workspace_user_from_request_session(request, workspace_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _preview_user_response(user)


@preview_host_app.post("/auth/login")
async def preview_auth_login(request: Request, response: Response, payload: dict[str, Any]):
    workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "")
    if not workspace_id:
        raise HTTPException(status_code=401, detail="Workspace preview session required")
    user = await _authenticate_preview_workspace_user(
        request,
        response,
        workspace_id,
        username=payload.get("username"),
        password=payload.get("password"),
    )
    if user is None:
        raise HTTPException(status_code=400, detail="Username and password are required")
    await _set_workspace_preview_session_for_user(request, response, workspace_id, str(user.id))
    return {
        "success": True,
        "user_id": str(user.id),
        "username": str(getattr(user, "username", "") or ""),
        "display_name": getattr(user, "displayName", None),
        "email": getattr(user, "email", None),
        "role": str(getattr(user, "role", "user") or "user"),
    }


@preview_host_app.get("/__ragtime/browser-auth/start")
async def preview_browser_auth_start(request: Request):
    surfaces, auth_method_key, return_to = _browser_auth_query_values(request)
    payload = _browser_auth_request_from_values(surfaces=surfaces, auth_method_key=auth_method_key)
    claims = await _verify_preview_session_cookie(request)
    try:
        return await _complete_preview_browser_auth_redirect(request, claims, payload, return_to)
    except HTTPException as exc:
        if exc.status_code != 401:
            return await _render_browser_auth_start_page(
                request,
                surfaces=surfaces,
                auth_method_key=auth_method_key,
                return_to=return_to,
                parent_origin=claims.get("parent_origin"),
                error=str(exc.detail),
                status_code=exc.status_code,
            )
        return await _render_browser_auth_start_page(
            request,
            surfaces=surfaces,
            auth_method_key=auth_method_key,
            return_to=return_to,
            parent_origin=claims.get("parent_origin"),
        )


@preview_host_app.post("/__ragtime/browser-auth/start")
async def preview_browser_auth_start_submit(request: Request):
    form = await request.form()
    raw_surfaces: list[str] = []
    for value in form.getlist("surfaces"):
        raw_surfaces.extend(part.strip() for part in str(value).split(",") if part.strip())
    surfaces = _coerce_browser_surfaces(raw_surfaces or None)
    auth_method_key = str(form.get("auth_method_key") or "").strip() or None
    return_to = _safe_browser_auth_return_to(str(form.get("return_to") or ""))
    payload = _browser_auth_request_from_values(
        surfaces=surfaces,
        auth_method_key=auth_method_key,
        username=str(form.get("username") or ""),
        password=str(form.get("password") or ""),
    )
    claims = await _verify_preview_session_cookie(request)
    try:
        return await _complete_preview_browser_auth_redirect(request, claims, payload, return_to)
    except HTTPException as exc:
        return await _render_browser_auth_start_page(
            request,
            surfaces=surfaces,
            auth_method_key=auth_method_key,
            return_to=return_to,
            parent_origin=claims.get("parent_origin"),
            error=str(exc.detail),
            status_code=exc.status_code,
        )


@preview_host_app.post(
    "/__ragtime/browser-auth",
    response_model=UserSpaceBrowserAuthResponse,
)
async def preview_browser_auth(
    request: Request,
    response: Response,
    payload: UserSpaceBrowserAuthRequest,
):
    claims = await _verify_preview_session_cookie(request)
    try:
        workspace_id, user = await _workspace_user_for_preview_auth(request, response, claims, payload)
    except HTTPException as exc:
        if exc.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "authentication_required",
                    "message": "Authentication required",
                    "interactive_auth_url": _browser_auth_start_url(
                        auth_method_key=payload.auth_method_key,
                        surfaces=list(payload.surfaces or []),
                        return_to="/",
                    ),
                },
            ) from exc
        raise
    return await _authorize_browser_surfaces_for_user_id(
        workspace_id,
        payload,
        request,
        response,
        user_id=str(user.id),
    )


@preview_host_app.post(
    "/indexes/userspace/runtime/workspaces/{workspace_id}/browser-auth",
    response_model=UserSpaceBrowserAuthResponse,
)
async def preview_workspace_browser_auth_alias(
    workspace_id: str,
    request: Request,
    response: Response,
    payload: UserSpaceBrowserAuthRequest,
):
    host_workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "").strip()
    if host_workspace_id and host_workspace_id != str(workspace_id or "").strip():
        raise HTTPException(status_code=404, detail="Preview host mismatch")
    return await preview_browser_auth(request, response, payload)


async def _clear_preview_browser_auth(request: Request, response: Response) -> None:
    workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "").strip()
    if workspace_id:
        _clear_browser_surface_cookies(workspace_id, response)
    response.delete_cookie(key=_RUNTIME_PREVIEW_SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    await _invalidate_preview_session_for_host(request.headers.get("host"))


@preview_host_app.post("/__ragtime/browser-auth/logout")
async def preview_browser_auth_logout(request: Request, response: Response):
    await _clear_preview_browser_auth(request, response)
    return {"success": True}


@preview_host_app.get("/__ragtime/browser-auth/logout")
async def preview_browser_auth_logout_nav(request: Request):
    return_to = _safe_browser_auth_return_to(request.query_params.get("return_to"))
    response = RedirectResponse(url=return_to, status_code=303)
    await _clear_preview_browser_auth(request, response)
    response.headers["Cache-Control"] = "no-store"
    return response


@preview_host_app.post("/indexes/userspace/runtime/workspaces/{workspace_id}/browser-auth/logout")
async def preview_workspace_browser_auth_logout_alias(
    workspace_id: str,
    request: Request,
    response: Response,
):
    host_workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "").strip()
    if host_workspace_id and host_workspace_id != str(workspace_id or "").strip():
        raise HTTPException(status_code=404, detail="Preview host mismatch")
    return await preview_browser_auth_logout(request, response)


@preview_host_app.get("/indexes/userspace/runtime/workspaces/{workspace_id}/browser-auth/logout")
async def preview_workspace_browser_auth_logout_nav_alias(
    workspace_id: str,
    request: Request,
):
    host_workspace_id = str(_workspace_id_from_preview_host(request.headers.get("host")) or "").strip()
    if host_workspace_id and host_workspace_id != str(workspace_id or "").strip():
        raise HTTPException(status_code=404, detail="Preview host mismatch")
    return await preview_browser_auth_logout_nav(request)


@preview_host_app.post("/__ragtime/parse-document")
async def preview_parse_document(
    request: Request,
    file: UploadFile = File(...),
):
    await _verify_preview_session_cookie(request)
    return await _parse_uploaded_document_upload(file)


@preview_host_app.post("/__ragtime/parse-spreadsheet")
async def preview_parse_spreadsheet(
    request: Request,
    file: UploadFile = File(...),
):
    await _verify_preview_session_cookie(request)
    return await _parse_uploaded_spreadsheet_upload(file)


@preview_host_app.post("/__ragtime/tables/normalize")
async def preview_table_normalize(
    request: Request,
    file: UploadFile = File(...),
):
    await _verify_preview_session_cookie(request)
    return await _normalize_uploaded_table_upload(file)


@preview_host_app.post("/__ragtime/documents/render-preview")
async def preview_document_render_preview(
    request: Request,
    file: UploadFile = File(...),
):
    await _verify_preview_session_cookie(request)
    return await _render_uploaded_document_preview_upload(file)


def _workspace_user_from_preview_claims(claims: dict[str, Any]) -> tuple[str, str]:
    if _preview_mode(claims) != "workspace":
        raise HTTPException(status_code=403, detail="Workspace preview session required")
    workspace_id = str(claims.get("workspace_id") or "").strip()
    user_id = str(claims.get("sub") or "").strip()
    if not workspace_id or not user_id:
        raise HTTPException(status_code=401, detail="Preview session missing user")
    return workspace_id, user_id


@preview_host_app.get("/__ragtime/files/{file_path:path}")
async def preview_file_read(request: Request, file_path: str):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_workspace_file_response(workspace_id, file_path, user_id)


@preview_host_app.put("/__ragtime/files/{file_path:path}")
async def preview_file_write(
    request: Request,
    file_path: str,
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_workspace_file_write_request(workspace_id, file_path, request, user_id)


@preview_host_app.post("/__ragtime/upload-target")
async def preview_upload_target(
    request: Request,
    payload: dict[str, Any],
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_upload_target(workspace_id, payload, user_id, preview_origin=True)


@preview_host_app.post("/__ragtime/archives/extract")
async def preview_archive_extract(
    request: Request,
    target: str = "files",
    destination_path: str = "",
    bucket: str | None = None,
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_archive_extract(
        workspace_id,
        request,
        user_id,
        target=target,
        destination_path=destination_path,
        bucket_name=bucket,
    )


@preview_host_app.get("/__ragtime/objects/{bucket_name}/{object_path:path}")
async def preview_object_read(
    request: Request,
    bucket_name: str,
    object_path: str,
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_object_response(workspace_id, bucket_name, object_path, user_id)


@preview_host_app.put("/__ragtime/objects/{bucket_name}/{object_path:path}")
async def preview_object_write(
    request: Request,
    bucket_name: str,
    object_path: str,
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_object_write_request(workspace_id, bucket_name, object_path, request, user_id)


@preview_host_app.get("/__ragtime/progress/{task_id}")
async def preview_progress_get(request: Request, task_id: str):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_progress_get(workspace_id, task_id, user_id)


@preview_host_app.put("/__ragtime/progress/{task_id}")
async def preview_progress_put(
    request: Request,
    task_id: str,
    payload: dict[str, Any],
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_progress_put(workspace_id, task_id, payload, user_id)


@preview_host_app.post("/__ragtime/jobs")
async def preview_job_create(
    request: Request,
    payload: dict[str, Any],
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_job_create(workspace_id, payload, user_id)


@preview_host_app.get("/__ragtime/jobs/{job_id}")
async def preview_job_get(request: Request, job_id: str):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_job_get(workspace_id, job_id, user_id)


@preview_host_app.put("/__ragtime/jobs/{job_id}")
async def preview_job_put(
    request: Request,
    job_id: str,
    payload: dict[str, Any],
):
    claims = await _verify_preview_session_cookie(request)
    workspace_id, user_id = _workspace_user_from_preview_claims(claims)
    return await _primitive_job_put(workspace_id, job_id, payload, user_id)


@preview_host_app.api_route("/", methods=_PROXY_METHODS)
@preview_host_app.api_route("/{path:path}", methods=_PROXY_METHODS)
async def preview_proxy(request: Request, path: str = ""):
    claims = await _verify_preview_session_cookie(request)

    async def primitive_session_factory() -> dict[str, Any]:
        workspace_id = str(claims.get("workspace_id") or "").strip()
        user_id = str(claims.get("sub") or "").strip() or None
        return await _primitive_session_payload(
            workspace_id,
            user_id,
            mode=_preview_mode(claims),
            share_access_mode=claims.get("share_access_mode"),
            user=await _preview_user_from_id(user_id),
            same_origin_auth_endpoints=True,
        )

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
        primitive_session_factory=primitive_session_factory,
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
        if scope.get("type") in {"http", "websocket"} and is_preview_host(_scope_host(scope)):
            await preview_host_app(scope, receive, send)
            return
        await self.app(scope, receive, send)
