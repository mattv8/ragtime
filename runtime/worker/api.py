from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import json
import logging
import math
import os
import pty as pty_module
import struct
import termios
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from websockets.typing import Subprotocol

from runtime.auth import OptionalWorkerAuth, WorkerAuth, get_runtime_auth_token
from runtime.manager.models import (
    BridgeCredentialRefreshRequest,
    RuntimeAppRestartRequest,
    RuntimeBridgeCredentialMetadata,
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecRequest,
    RuntimeExecResponse,
    RuntimeExternalBrowseRequest,
    RuntimeExternalBrowseResponse,
    RuntimeFileReadResponse,
    RuntimeMcpToolCallRequest,
    RuntimeMcpToolCallResponse,
    RuntimeMcpToolListResponse,
    RuntimePdfReadRequest,
    RuntimePdfReadResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeWorkspaceMaintenanceRequest,
    RuntimeWorkspaceMaintenanceResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.http_client import close_preview_http_clients, get_preview_http_client
from runtime.worker.sandbox import (
    SandboxSpec,
    ensure_sandbox_ready,
    sandbox_env,
    spawn_sandboxed,
    terminate_process_group,
)
from runtime.worker.service import get_worker_service

router = APIRouter(tags=["Runtime Worker"])

_SANDBOX_BASHRC_TEMPLATE_PATH = Path(__file__).parent / "templates" / "sandbox_bashrc.sh"
_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
logger = logging.getLogger(__name__)

# Namespace prefix applied to user-app cookies as they cross the preview proxy
# boundary. Every upstream devserver ``Set-Cookie`` name is rewritten to
# ``{_USER_APP_COOKIE_PREFIX}{base64url(original_name)}`` before reaching the
# browser; the ragtime control plane decodes only prefixed cookies back to the
# app on the way upstream. This keeps platform cookies (preview session,
# capability, share-auth) unforwardable to untrusted app code and stops an app
# from shadowing a platform cookie by name, without a platform-cookie blocklist.
#
# This is a COPY of the same constant in
# ``ragtime/userspace/runtime_routes.py``. The runtime worker and ragtime app
# containers cannot cross-import, so the two definitions must be kept
# byte-for-byte in sync; changing only one side silently breaks user-app
# session persistence in previews.
_USER_APP_COOKIE_PREFIX = "__ragtime_app_cookie_"
_AUTHENTICATED_IDENTITY_HEADER_MAP = {
    "x-ragtime-authenticated-username": "x-ragtime-internal-authenticated-username",
    "x-ragtime-authenticated-display-name": "x-ragtime-internal-authenticated-display-name",
    "x-ragtime-user-fingerprint": "x-ragtime-internal-user-fingerprint",
}
_SERVICE_AUTHENTICATED_IDENTITY_HEADER_MAP = {
    "x-ragtime-authenticated-actor-type": "x-ragtime-internal-authenticated-actor-type",
    "x-ragtime-service-credential-id": "x-ragtime-internal-service-credential-id",
    "x-ragtime-service-credential-label": "x-ragtime-internal-service-credential-label",
    "x-ragtime-published-endpoint-key": "x-ragtime-internal-published-endpoint-key",
}
_PUBLIC_AUTHENTICATED_ENTITLEMENTS_HEADER = "x-ragtime-authenticated-entitlements"
# The private entitlement header is injected by the ragtime control plane and
# must reach the workspace backend unchanged (backends authorize against it).
# It is therefore intentionally NOT added to the blocked set in
# ``_preview_request_headers``; only the public spoof alias is blocked there.
_PRIVATE_AUTHENTICATED_ENTITLEMENTS_HEADER = "x-ragtime-internal-authenticated-entitlements"
# This wire header is intentionally duplicated in ragtime/userspace/runtime_routes.py.
# The deployed worker image does not contain the Ragtime application package.
_INTERNAL_HTTP_BUDGET_HEADER = "x-ragtime-internal-http-budget-ms"
_DEFAULT_HTTP_PROXY_BUDGET_MS = 90_000
_MAX_HTTP_PROXY_BUDGET_MS = 3_600_000
_PROXY_STREAM_CHUNK_BYTES = 64 * 1024


class _ProxyDeadlineExceeded(Exception):
    """The request-wide preview HTTP deadline elapsed."""


class _InvalidProxyRequestBody(Exception):
    """Inbound bytes do not match their declared HTTP framing."""


class _DeadlineStreamingResponse(StreamingResponse):
    """Ensure ASGI downstream backpressure is inside the proxy deadline."""

    def __init__(self, *args: Any, deadline: float, on_close: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline = deadline
        self._on_close = on_close

    async def stream_response(self, send: Any) -> None:
        try:
            if _remaining_proxy_seconds(self._deadline) <= 0:
                await asyncio.wait_for(
                    send({"type": "http.response.start", "status": 504, "headers": [(b"content-type", b"application/json")]}),
                    timeout=0.01,
                )
                await asyncio.wait_for(
                    send({"type": "http.response.body", "body": b'{"detail":"Runtime dev server request timed out"}'}),
                    timeout=0.01,
                )
                return
            await _within_proxy_deadline(
                send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}),
                self._deadline,
            )
            async for chunk in self.body_iterator:
                if not isinstance(chunk, bytes):
                    chunk = bytes(chunk) if isinstance(chunk, memoryview) else chunk.encode(self.charset)
                await _within_proxy_deadline(send({"type": "http.response.body", "body": chunk, "more_body": True}), self._deadline)
            await _within_proxy_deadline(send({"type": "http.response.body", "body": b"", "more_body": False}), self._deadline)
        except _ProxyDeadlineExceeded:
            return
        finally:
            if self._on_close is not None:
                with contextlib.suppress(Exception):
                    await self._on_close()
            closer = getattr(self.body_iterator, "aclose", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()


def _worker_proxy_budget_seconds(request: Request) -> float:
    """Read only the trusted control-plane budget, with a bounded fallback."""
    raw = request.headers.get(_INTERNAL_HTTP_BUDGET_HEADER)
    if raw is None:
        return _DEFAULT_HTTP_PROXY_BUDGET_MS / 1000
    try:
        milliseconds = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_HTTP_PROXY_BUDGET_MS / 1000
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        return _DEFAULT_HTTP_PROXY_BUDGET_MS / 1000
    # Leave the control plane a little time to produce its own structured 504.
    return min(milliseconds, _MAX_HTTP_PROXY_BUDGET_MS) / 1000 * 0.95


def _remaining_proxy_seconds(deadline: float) -> float:
    return deadline - asyncio.get_running_loop().time()


async def _within_proxy_deadline(awaitable: Any, deadline: float) -> Any:
    remaining = _remaining_proxy_seconds(deadline)
    if remaining <= 0:
        closer = getattr(awaitable, "close", None)
        if closer is not None:
            closer()
        raise _ProxyDeadlineExceeded
    try:
        return await asyncio.wait_for(awaitable, timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise _ProxyDeadlineExceeded from exc


async def _bounded_request_stream(request: Request, deadline: float) -> AsyncIterator[bytes]:
    raw_length = request.headers.get("content-length")
    if raw_length is not None and not raw_length.isdecimal():
        raise _InvalidProxyRequestBody("Invalid Content-Length")
    expected = int(raw_length) if raw_length is not None else None
    sent = 0
    async for incoming in request.stream():
        if not incoming:
            continue
        for offset in range(0, len(incoming), _PROXY_STREAM_CHUNK_BYTES):
            if _remaining_proxy_seconds(deadline) <= 0:
                raise _ProxyDeadlineExceeded
            chunk = incoming[offset : offset + _PROXY_STREAM_CHUNK_BYTES]
            sent += len(chunk)
            if expected is not None and sent > expected:
                raise _InvalidProxyRequestBody("Request body exceeds Content-Length")
            yield chunk
    if expected is not None and sent != expected:
        raise _InvalidProxyRequestBody("Request body does not match Content-Length")


def _encode_user_app_cookie_name(cookie_name: str) -> str | None:
    normalized = cookie_name.strip()
    if not normalized or "=" in normalized or ";" in normalized:
        return None
    encoded = base64.urlsafe_b64encode(normalized.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_USER_APP_COOKIE_PREFIX}{encoded}"


def _rewrite_user_app_set_cookie(raw_set_cookie: str | None) -> str | None:
    if not raw_set_cookie or "=" not in raw_set_cookie:
        return None
    name_value, separator, raw_attributes = raw_set_cookie.partition(";")
    name, value = name_value.split("=", 1)
    encoded_name = _encode_user_app_cookie_name(name)
    if not encoded_name:
        return None
    attributes = [item.strip() for item in raw_attributes.split(";") if item.strip()]
    attributes = [item for item in attributes if not item.lower().startswith("domain=")]
    rewritten = f"{encoded_name}={value}"
    if separator and attributes:
        rewritten += "; " + "; ".join(attributes)
    return rewritten


def _is_html_document_request(request: Request) -> bool:
    if request.method.upper() not in {"GET", "HEAD"}:
        return False
    if request.headers.get("range"):
        return False
    sec_fetch_dest = request.headers.get("sec-fetch-dest", "").strip().lower()
    if sec_fetch_dest == "document":
        return True
    accept = request.headers.get("accept", "").strip().lower()
    return "text/html" in accept or "application/xhtml+xml" in accept


def _preview_request_headers(request: Request) -> dict[str, str]:
    raw_headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in request.scope.get("headers", [])}
    blocked = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        _INTERNAL_HTTP_BUDGET_HEADER,
        "authorization",
        "cookie",
        _PUBLIC_AUTHENTICATED_ENTITLEMENTS_HEADER,
        *set(_AUTHENTICATED_IDENTITY_HEADER_MAP),
        *set(_AUTHENTICATED_IDENTITY_HEADER_MAP.values()),
        *set(_SERVICE_AUTHENTICATED_IDENTITY_HEADER_MAP),
        *set(_SERVICE_AUTHENTICATED_IDENTITY_HEADER_MAP.values()),
    }
    forwarded_headers = {key: value for key, value in request.headers.items() if key.lower() not in blocked}
    # The inbound ``Cookie`` header is forwarded verbatim to the devserver. It
    # is trusted because the ragtime control-plane proxy already stripped its
    # own platform/session cookies, decoded only the user-app cookies from the
    # ``__ragtime_app_cookie_`` namespace, and replaced any browser-supplied
    # identity headers with verified private preview-session headers before
    # sending the request here (see ragtime/userspace/runtime_routes.py).
    # The worker must not be reached directly by browsers; only via that proxy.
    cookie_header = request.headers.get("cookie")
    if cookie_header:
        forwarded_headers["cookie"] = cookie_header
    for public_name, private_name in _AUTHENTICATED_IDENTITY_HEADER_MAP.items():
        value = str(raw_headers.get(private_name, "") or "").strip()
        if value:
            forwarded_headers[public_name] = value
    for public_name, private_name in _SERVICE_AUTHENTICATED_IDENTITY_HEADER_MAP.items():
        value = str(raw_headers.get(private_name, "") or "").strip()
        if value:
            forwarded_headers[public_name] = value
    forwarded_headers.setdefault("x-forwarded-proto", request.url.scheme)
    forwarded_headers.setdefault("x-forwarded-host", request.headers.get("host", ""))
    client_host = request.client.host if request.client else ""
    if client_host:
        forwarded_headers.setdefault("x-forwarded-for", client_host)
    # HTML documents are rewritten later in the preview pipeline, so asking the
    # devserver for an identity-encoded body avoids avoidable decode/re-encode
    # work while leaving assets and downloads untouched.
    if _is_html_document_request(request):
        forwarded_headers["accept-encoding"] = "identity"
    return forwarded_headers


def _preview_response_headers(headers: httpx.Headers) -> tuple[dict[str, str], list[str]]:
    blocked = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "set-cookie",
    }
    out = {key: value for key, value in headers.items() if key.lower() not in blocked}
    set_cookies = [rewritten for value in headers.get_list("set-cookie") if (rewritten := _rewrite_user_app_set_cookie(value))]
    return out, set_cookies


def _append_set_cookie_headers(response: Response, set_cookie_headers: list[str]) -> Response:
    # Each value is appended as its own ``Set-Cookie`` header line via the
    # public MutableHeaders API; cookies must never be comma-joined into a
    # single header because cookie attribute values can contain commas.
    for value in set_cookie_headers:
        response.headers.append("set-cookie", value)
    return response


def _is_html_media_type(media_type: str) -> bool:
    return "text/html" in (media_type or "").lower()


async def _proxy_preview_request(request: Request, upstream_url: str) -> Response:
    deadline = asyncio.get_running_loop().time() + _worker_proxy_budget_seconds(request)
    headers = _preview_request_headers(request)
    client = get_preview_http_client()
    try:
        remaining = _remaining_proxy_seconds(deadline)
        if remaining <= 0:
            raise _ProxyDeadlineExceeded
        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            content=_bounded_request_stream(request, deadline),
            headers=headers,
            timeout=httpx.Timeout(
                connect=min(2.0, remaining),
                read=remaining,
                write=remaining,
                pool=min(5.0, remaining),
            ),
        )
        upstream_response = await _within_proxy_deadline(client.send(upstream_request, stream=True), deadline)
    except _ProxyDeadlineExceeded:
        return Response(
            content=json.dumps({"detail": "Runtime dev server request timed out"}),
            status_code=504,
            media_type="application/json",
        )
    except _InvalidProxyRequestBody as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Runtime dev server unavailable: {exc}",
        ) from exc

    media_type = upstream_response.headers.get("content-type") or "application/octet-stream"
    response_headers, set_cookie_headers = _preview_response_headers(upstream_response.headers)

    async def _iter_stream() -> AsyncIterator[bytes]:
        try:
            iterator = upstream_response.aiter_raw().__aiter__()
            while True:
                try:
                    chunk = await _within_proxy_deadline(iterator.__anext__(), deadline)
                except StopAsyncIteration:
                    break
                except _ProxyDeadlineExceeded:
                    # The downstream response already has app headers.  Ending
                    # the stream preserves protocol and byte fidelity.
                    break
                yield chunk
        finally:
            await upstream_response.aclose()

    return _append_set_cookie_headers(
        _DeadlineStreamingResponse(
            _iter_stream(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=media_type or None,
            deadline=deadline,
            on_close=upstream_response.aclose,
        ),
        set_cookie_headers,
    )


def _write_sandbox_init_file(spec: SandboxSpec) -> None:
    """Write bash init file into the sandbox rootfs.

    Uses ``\\044`` (octal for ``$``) so PS1 renders a literal dollar sign
    regardless of UID. The prompt is derived from ``$PWD`` each render so
    directory changes in the interactive PTY are reflected immediately.
    """
    init_file = spec.rootfs_path / "tmp" / ".sandbox_bashrc"
    init_file.parent.mkdir(parents=True, exist_ok=True)
    init_file.write_text(
        _SANDBOX_BASHRC_TEMPLATE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


@router.get("/worker/health", response_model=WorkerHealthResponse)
async def health(
    is_authenticated: bool = OptionalWorkerAuth,
) -> WorkerHealthResponse:
    full = await get_worker_service().health()
    # Unauthenticated container healthchecks only receive liveness;
    # session counts and worker metadata require a valid worker bearer token.
    if not is_authenticated:
        return WorkerHealthResponse(
            status=full.status,
            service_mode=full.service_mode,
            active_sessions=0,
            metadata={},
        )
    return full


@router.post("/worker/sessions/start", response_model=WorkerSessionResponse)
async def start_session(
    payload: WorkerStartSessionRequest,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().start_session(payload)


@router.post("/worker/workspaces/{workspace_id}/sqlite-maintenance", response_model=RuntimeWorkspaceMaintenanceResponse)
async def acquire_sqlite_workspace_maintenance(
    workspace_id: str, payload: RuntimeWorkspaceMaintenanceRequest, _auth: None = WorkerAuth
) -> RuntimeWorkspaceMaintenanceResponse:
    result = await get_worker_service().acquire_sqlite_workspace_access(
        workspace_id, payload.lease_id, maintenance=payload.maintenance
    )
    return RuntimeWorkspaceMaintenanceResponse.model_validate(result)


@router.delete("/worker/workspaces/{workspace_id}/sqlite-maintenance/{lease_id}", status_code=204)
async def release_sqlite_workspace_maintenance(
    workspace_id: str, lease_id: str, _auth: None = WorkerAuth
) -> None:
    await get_worker_service().release_sqlite_workspace_access(workspace_id, lease_id)


@router.get("/worker/sessions/{worker_session_id}", response_model=WorkerSessionResponse)
async def get_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().get_session(worker_session_id)


@router.post("/worker/sessions/{worker_session_id}/stop", response_model=WorkerSessionResponse)
async def stop_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().stop_session(worker_session_id)


@router.post("/worker/sessions/{worker_session_id}/restart", response_model=WorkerSessionResponse)
async def restart_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().restart_session(worker_session_id)


@router.post("/worker/sessions/{worker_session_id}/app/restart", response_model=WorkerSessionResponse)
async def restart_app(
    worker_session_id: str,
    payload: RuntimeAppRestartRequest,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().restart_app(worker_session_id, payload.request_id)


@router.post(
    "/worker/sessions/{worker_session_id}/bridge-credential/refresh",
    response_model=RuntimeBridgeCredentialMetadata,
)
async def refresh_bridge_credential(
    worker_session_id: str,
    payload: BridgeCredentialRefreshRequest,
    _auth: None = WorkerAuth,
) -> RuntimeBridgeCredentialMetadata:
    return await get_worker_service().refresh_bridge_credential(
        worker_session_id,
        token=payload.token,
        expected_session_id=payload.expected_session_id,
        expected_revision=payload.expected_revision,
        request_id=payload.request_id,
    )


@router.get(
    "/worker/sessions/{worker_session_id}/fs/{file_path:path}",
    response_model=RuntimeFileReadResponse,
)
async def read_file(
    worker_session_id: str,
    file_path: str,
    _auth: None = WorkerAuth,
) -> RuntimeFileReadResponse:
    return await get_worker_service().read_file(worker_session_id, file_path)


@router.put(
    "/worker/sessions/{worker_session_id}/fs/{file_path:path}",
    response_model=RuntimeFileReadResponse,
)
async def write_file(
    worker_session_id: str,
    file_path: str,
    payload: dict[str, Any],
    _auth: None = WorkerAuth,
) -> RuntimeFileReadResponse:
    return await get_worker_service().write_file(
        worker_session_id,
        file_path,
        str(payload.get("content", "")),
    )


@router.delete("/worker/sessions/{worker_session_id}/fs/{file_path:path}")
async def delete_file(
    worker_session_id: str,
    file_path: str,
    _auth: None = WorkerAuth,
) -> dict[str, Any]:
    return await get_worker_service().delete_file(worker_session_id, file_path)


@router.post(
    "/worker/sessions/{worker_session_id}/screenshot",
    response_model=RuntimeScreenshotResponse,
)
async def capture_screenshot(
    worker_session_id: str,
    payload: RuntimeScreenshotRequest,
    _auth: None = WorkerAuth,
) -> RuntimeScreenshotResponse:
    service = get_worker_service()
    capture_method = getattr(service, "capture_screenshot", None)
    if capture_method is None:
        raise HTTPException(status_code=503, detail="Runtime screenshot not available")
    return await capture_method(worker_session_id, payload)


@router.post(
    "/worker/sessions/{worker_session_id}/content-probe",
    response_model=RuntimeContentProbeResponse,
)
async def content_probe(
    worker_session_id: str,
    payload: RuntimeContentProbeRequest,
    _auth: None = WorkerAuth,
) -> RuntimeContentProbeResponse:
    service = get_worker_service()
    probe_method = getattr(service, "content_probe", None)
    if probe_method is None:
        raise HTTPException(status_code=503, detail="Runtime content probe not available")
    return await probe_method(worker_session_id, payload)


@router.post(
    "/worker/sessions/{worker_session_id}/exec",
    response_model=RuntimeExecResponse,
)
async def exec_command(
    worker_session_id: str,
    payload: RuntimeExecRequest,
    _auth: None = WorkerAuth,
) -> RuntimeExecResponse:
    return await get_worker_service().exec_command(
        worker_session_id,
        payload.command,
        timeout_seconds=payload.timeout_seconds,
        cwd=payload.cwd,
    )


@router.post(
    "/worker/sessions/{worker_session_id}/external-browse",
    response_model=RuntimeExternalBrowseResponse,
)
async def external_browse_for_session(
    worker_session_id: str,
    payload: RuntimeExternalBrowseRequest,
    _auth: None = WorkerAuth,
) -> RuntimeExternalBrowseResponse:
    return await get_worker_service().external_browse(
        payload,
        worker_session_id=worker_session_id,
    )


@router.post(
    "/worker/external-browse",
    response_model=RuntimeExternalBrowseResponse,
)
async def external_browse(
    payload: RuntimeExternalBrowseRequest,
    _auth: None = WorkerAuth,
) -> RuntimeExternalBrowseResponse:
    """Drive Playwright against an arbitrary http/https URL.

    URL safety is enforced upstream (control plane); this endpoint trusts the
    manager-issued bearer token and only exposes the broker capability.
    """
    service = get_worker_service()
    method = getattr(service, "external_browse", None)
    if method is None:
        raise HTTPException(status_code=503, detail="Runtime external browse not available")
    return await method(payload)


@router.post(
    "/worker/sessions/{worker_session_id}/mcp/tools/call",
    response_model=RuntimeMcpToolCallResponse,
)
async def call_mcp_tool(
    worker_session_id: str,
    payload: RuntimeMcpToolCallRequest,
    _auth: None = WorkerAuth,
) -> RuntimeMcpToolCallResponse:
    return await get_worker_service().call_mcp_tool(worker_session_id, payload)


@router.get(
    "/worker/mcp/tools",
    response_model=RuntimeMcpToolListResponse,
)
async def list_global_mcp_tools(
    _auth: None = WorkerAuth,
) -> RuntimeMcpToolListResponse:
    return await get_worker_service().list_global_mcp_tools()


@router.get(
    "/worker/sessions/{worker_session_id}/mcp/tools",
    response_model=RuntimeMcpToolListResponse,
)
async def list_mcp_tools(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> RuntimeMcpToolListResponse:
    return await get_worker_service().list_mcp_tools(worker_session_id)


@router.post(
    "/worker/sessions/{worker_session_id}/pdf-read",
    response_model=RuntimePdfReadResponse,
)
async def read_pdf_for_session(
    worker_session_id: str,
    payload: RuntimePdfReadRequest,
    _auth: None = WorkerAuth,
) -> RuntimePdfReadResponse:
    return await get_worker_service().read_pdf(
        payload,
        worker_session_id=worker_session_id,
    )


@router.post("/worker/pdf-read", response_model=RuntimePdfReadResponse)
async def read_pdf(
    payload: RuntimePdfReadRequest,
    _auth: None = WorkerAuth,
) -> RuntimePdfReadResponse:
    """Fetch and extract bounded text from an arbitrary PDF URL."""
    return await get_worker_service().read_pdf(payload)


@router.api_route("/worker/sessions/{worker_session_id}/preview", methods=_PROXY_METHODS)
@router.api_route("/worker/sessions/{worker_session_id}/preview/{path:path}", methods=_PROXY_METHODS)
async def preview(
    worker_session_id: str,
    request: Request,
    path: str = "",
    _auth: None = WorkerAuth,
) -> Response:
    upstream_url = await get_worker_service().build_preview_upstream_url(
        worker_session_id,
        path,
        query=request.url.query or None,
    )
    return await _proxy_preview_request(request, upstream_url)


def _verify_worker_auth_from_websocket(websocket: WebSocket) -> None:
    """Validate worker Bearer token from the WebSocket handshake headers."""
    cached_token = get_runtime_auth_token()
    if not cached_token:
        raise HTTPException(status_code=503, detail="Runtime auth not configured")
    auth_header = ""
    for key, value in websocket.scope.get("headers", []):
        if key == b"authorization":
            auth_header = value.decode("latin-1")
            break
    if not auth_header.startswith("Bearer ") or auth_header[7:] != cached_token:
        raise HTTPException(status_code=403, detail="Invalid runtime auth token")


@router.websocket("/worker/sessions/{worker_session_id}/preview/{path:path}")
async def preview_websocket(
    worker_session_id: str,
    path: str,
    websocket: WebSocket,
) -> None:
    try:
        _verify_worker_auth_from_websocket(websocket)
    except HTTPException as exc:
        logger.warning("WS preview auth rejected for %s: %s", worker_session_id, exc.detail)
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    try:
        upstream_url = await get_worker_service().build_preview_upstream_url(
            worker_session_id,
            path,
            query=websocket.url.query or None,
        )
    except HTTPException as exc:
        logger.warning(
            "WS preview upstream lookup failed for %s/%s: %s",
            worker_session_id,
            path,
            exc.detail,
        )
        await websocket.close(code=4000 + exc.status_code)
        return

    # Convert http:// to ws://
    ws_url = upstream_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)

    requested_subprotocols = [Subprotocol(protocol) for p in (websocket.scope.get("subprotocols") or []) if (protocol := str(p).strip())]

    try:
        import websockets as _ws_mod

        async with _ws_mod.connect(
            ws_url,
            max_size=None,
            open_timeout=10,
            subprotocols=requested_subprotocols or None,
        ) as upstream:
            await websocket.accept(
                subprotocol=getattr(upstream, "subprotocol", None) or None,
            )

            async def _down_to_up() -> None:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    text = msg.get("text")
                    data = msg.get("bytes")
                    if text is not None:
                        await upstream.send(text)
                    elif data is not None:
                        await upstream.send(data)

            async def _up_to_down() -> None:
                while True:
                    msg = await upstream.recv()
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_text(str(msg))

            down_task = asyncio.create_task(_down_to_up())
            up_task = asyncio.create_task(_up_to_down())
            done, pending = await asyncio.wait(
                {down_task, up_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in done:
                t.result()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.debug("WS preview proxy error for %s/%s: %s", worker_session_id, path, exc)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
        return


# ---------------------------------------------------------------------------
# PTY session tracker – one PTY per worker session at a time
# ---------------------------------------------------------------------------
_pty_processes: dict[str, asyncio.subprocess.Process] = {}
_pty_master_fds: dict[str, int] = {}
_pty_workspace_ids: dict[str, str] = {}
_pty_lock = asyncio.Lock()


async def _terminate_pty_process(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 2,
) -> None:
    await terminate_process_group(process, timeout=timeout)


async def _evict_pty(session_id: str) -> None:
    """Terminate any existing PTY process for *session_id*."""
    process = _pty_processes.pop(session_id, None)
    master_fd = _pty_master_fds.pop(session_id, None)
    _pty_workspace_ids.pop(session_id, None)
    if process is not None and process.returncode is None:
        await _terminate_pty_process(process)
    if master_fd is not None:
        with contextlib.suppress(Exception):
            os.close(master_fd)


async def evict_workspace_ptys(workspace_id: str) -> None:
    """Atomically close every PTY for a workspace before maintenance continues."""
    async with _pty_lock:
        for session_id in tuple(_pty_processes):
            session_workspace_id = _pty_workspace_ids.get(session_id)
            if session_workspace_id is None:
                session_workspace_id = await get_worker_service().workspace_id_for_session(session_id)
            if session_workspace_id == workspace_id:
                await _evict_pty(session_id)


@router.websocket("/worker/sessions/{worker_session_id}/pty")
async def pty(worker_session_id: str, websocket: WebSocket):
    # Prefer the X-PTY-Token header so the per-session token never appears
    # in URL-based access logs. The query-string fallback is retained for
    # backward compatibility with callers that still embed it in the URL.
    token = websocket.headers.get("x-pty-token", "") or websocket.query_params.get("token", "")
    try:
        session = await get_worker_service().verify_pty_token(worker_session_id, token)
        await get_worker_service().assert_pty_available(worker_session_id)
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    await websocket.accept()

    # Hold the PTY admission lock from the final availability check through
    # registration. Maintenance takes this same lock before it drains PTYs, so
    # no shell can slip through the former check-to-spawn window.
    shell = "/bin/bash"
    master_fd: int | None = None
    process: asyncio.subprocess.Process | None = None
    sandbox_spec = session.sandbox_spec
    async with _pty_lock:
        try:
            await get_worker_service().assert_pty_available(worker_session_id)
            await _evict_pty(worker_session_id)
            master_fd, slave_fd = pty_module.openpty()
            await asyncio.to_thread(ensure_sandbox_ready, sandbox_spec)

    # Write bash init file inside the sandbox rootfs so that PS1 renders
    # a literal "$" regardless of UID, and updates based on the current
    # working directory after each command.
            _write_sandbox_init_file(sandbox_spec)
            shell_command = [shell, "--noprofile", "--init-file", "/tmp/.sandbox_bashrc", "-i"]

            service = get_worker_service()
            environment = service.build_agent_process_environment(session)
            environment = sandbox_env(sandbox_spec, environment)
            environment["TERM"] = "xterm-256color"
    # PS1 is set by the init file; PROMPT_COMMAND cleared to prevent
    # any inherited prompt logic from overriding it.
            environment["PROMPT_COMMAND"] = ""
            process = await spawn_sandboxed(
                sandbox_spec, shell_command, stdin=slave_fd, stdout=slave_fd,
                stderr=slave_fd, env=environment, pty=True, ensure_ready=False,
            )
            _pty_processes[worker_session_id] = process
            _pty_master_fds[worker_session_id] = master_fd
            _pty_workspace_ids[worker_session_id] = session.workspace_id
        except Exception:
            if master_fd is not None:
                with contextlib.suppress(Exception):
                    os.close(master_fd)
            raise
        finally:
            if 'slave_fd' in locals():
                with contextlib.suppress(Exception):
                    os.close(slave_fd)
    if process is None or master_fd is None:
        return
    logger.debug(
        "PTY spawned: worker_session_id=%s workspace_id=%s pid=%s mode=%s",
        worker_session_id,
        session.workspace_id,
        process.pid,
        sandbox_spec.mode,
    )

    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "message": "Runtime PTY bridge online",
                "read_only": False,
            }
        )
    )

    def _resize_pty(fd: int, cols: int, rows: int) -> None:
        """Send TIOCSWINSZ to resize the PTY."""
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    # Harmless bash warnings emitted when the shell starts inside a
    # user-namespace sandbox (tcgetpgrp returns -1 after unshare).
    _STARTUP_NOISE = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
    )
    redaction_carry = ""
    buffered_for_redaction_logged = False
    loop = asyncio.get_running_loop()
    os.set_blocking(master_fd, False)
    pty_output_queue: asyncio.Queue[bytes | OSError | None] = asyncio.Queue()
    startup_reads = 4  # only filter the first N reads

    def _queue_pty_output() -> None:
        try:
            chunk = os.read(master_fd, 1024)
        except BlockingIOError:
            return
        except OSError as exc:
            with contextlib.suppress(Exception):
                pty_output_queue.put_nowait(exc)
            loop.remove_reader(master_fd)
            return
        if not chunk:
            with contextlib.suppress(Exception):
                pty_output_queue.put_nowait(None)
            loop.remove_reader(master_fd)
            return
        with contextlib.suppress(Exception):
            pty_output_queue.put_nowait(chunk)

    async def _flush_output_chunk(chunk: bytes) -> None:
        nonlocal buffered_for_redaction_logged, redaction_carry, startup_reads
        text = chunk.decode("utf-8", errors="replace")
        if startup_reads > 0:
            startup_reads -= 1
            lines = text.splitlines(keepends=True)
            lines = [ln for ln in lines if not any(noise in ln for noise in _STARTUP_NOISE)]
            text = "".join(lines)
            if not text:
                return
        output_text, redaction_carry = service.split_workspace_secret_output(
            session,
            text,
            carry=redaction_carry,
        )
        if text and not output_text and redaction_carry and not buffered_for_redaction_logged:
            buffered_for_redaction_logged = True
            logger.debug(
                "PTY output buffered for redaction overlap: worker_session_id=%s chunk_len=%s carry_len=%s",
                worker_session_id,
                len(text),
                len(redaction_carry),
            )
        if output_text:
            await websocket.send_text(json.dumps({"type": "output", "data": output_text}))

    loop.add_reader(master_fd, _queue_pty_output)
    output_task = asyncio.create_task(pty_output_queue.get())
    input_task = asyncio.create_task(websocket.receive())
    try:
        while True:
            done, _ = await asyncio.wait(
                {input_task, output_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if output_task in done:
                output_result = output_task.result()
                output_task = asyncio.create_task(pty_output_queue.get())
                if isinstance(output_result, OSError):
                    logger.debug(
                        "PTY stream read failed: worker_session_id=%s errno=%s returncode=%s carry_len=%s",
                        worker_session_id,
                        getattr(output_result, "errno", None),
                        process.returncode,
                        len(redaction_carry),
                    )
                    break
                if output_result is None:
                    logger.debug(
                        "PTY stream EOF: worker_session_id=%s returncode=%s carry_len=%s",
                        worker_session_id,
                        process.returncode,
                        len(redaction_carry),
                    )
                    break
                await _flush_output_chunk(output_result)

            if input_task in done:
                message = input_task.result()
                input_task = asyncio.create_task(websocket.receive())
                if message.get("type") == "websocket.disconnect":
                    break
                text_payload = message.get("text")
                if text_payload is None:
                    continue
                try:
                    payload = json.loads(text_payload)
                except (json.JSONDecodeError, TypeError):
                    continue
                msg_type = payload.get("type")
                if msg_type == "resize":
                    cols = int(payload.get("cols", 80))
                    rows = int(payload.get("rows", 24))
                    _resize_pty(master_fd, cols, rows)
                    continue
                if msg_type != "input":
                    continue
                line = str(payload.get("data", ""))
                try:
                    os.write(master_fd, line.encode("utf-8", errors="ignore"))
                except OSError:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            loop.remove_reader(master_fd)
        for task in (input_task, output_task):
            task.cancel()
        # Unregister from PTY tracker only if this process is still the
        # registered one (a newer connection may have already replaced it)
        if _pty_processes.get(worker_session_id) is process:
            _pty_processes.pop(worker_session_id, None)
            _pty_workspace_ids.pop(worker_session_id, None)
        if _pty_master_fds.get(worker_session_id) == master_fd:
            _pty_master_fds.pop(worker_session_id, None)
        if redaction_carry:
            logger.debug(
                "PTY flushing buffered carry on stream end: worker_session_id=%s carry_len=%s",
                worker_session_id,
                len(redaction_carry),
            )
            output_text = service.redact_workspace_secret_output(
                session,
                redaction_carry,
            )
            if output_text:
                with contextlib.suppress(Exception):
                    await websocket.send_text(json.dumps({"type": "output", "data": output_text}))
        if process.returncode is None:
            await _terminate_pty_process(process)
        with contextlib.suppress(Exception):
            os.close(master_fd)


def include_worker_routes(application: FastAPI) -> None:
    application.include_router(router)


async def shutdown_worker_resources() -> None:
    """Close worker-owned processes and reusable HTTP resources."""
    try:
        await get_worker_service().shutdown()
    finally:
        async with _pty_lock:
            for sid in list(_pty_processes.keys()):
                await _evict_pty(sid)
        await close_preview_http_clients()


@contextlib.asynccontextmanager
async def _worker_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    # On shutdown (including WatchFiles reload), terminate all devserver
    # processes so orphaned children don't accumulate across reloads.
    await shutdown_worker_resources()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Ragtime User Space Runtime Worker",
        version="0.1.0",
        lifespan=_worker_lifespan,
    )
    include_worker_routes(application)

    @application.get("/health", response_model=WorkerHealthResponse)
    async def standalone_health(
        is_authenticated: bool = OptionalWorkerAuth,
    ) -> WorkerHealthResponse:
        full = await get_worker_service().health()
        if not is_authenticated:
            return WorkerHealthResponse(
                status=full.status,
                service_mode=full.service_mode,
                active_sessions=0,
                metadata={},
            )
        return full

    return application


app = create_app()
