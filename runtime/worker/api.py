from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
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

from runtime.auth import OptionalWorkerAuth, WorkerAuth
from runtime.manager.models import (
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecRequest,
    RuntimeExecResponse,
    RuntimeExternalBrowseRequest,
    RuntimeExternalBrowseResponse,
    RuntimeFileReadResponse,
    RuntimePdfReadRequest,
    RuntimePdfReadResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.sandbox import (
    SandboxSpec,
    ensure_sandbox_ready,
    make_sandbox_preexec,
    sandbox_env,
)
from runtime.worker.service import get_worker_service

router = APIRouter(tags=["Runtime Worker"])

_SANDBOX_BASHRC_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "sandbox_bashrc.sh"
)
_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
logger = logging.getLogger(__name__)


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
        "content-length",
        "authorization",
        "cookie",
    }
    forwarded_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked
    }
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


def _preview_response_headers(headers: httpx.Headers) -> dict[str, str]:
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
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _is_html_media_type(media_type: str) -> bool:
    return "text/html" in (media_type or "").lower()


async def _proxy_preview_request(request: Request, upstream_url: str) -> Response:
    body = await request.body()
    headers = _preview_request_headers(request)
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
            detail=f"Runtime dev server unavailable: {exc}",
        ) from exc

    media_type = (
        upstream_response.headers.get("content-type") or "application/octet-stream"
    )
    response_headers = _preview_response_headers(upstream_response.headers)

    if _is_html_media_type(media_type):
        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
            await client.aclose()
        # httpx decodes encoded bodies during aread(), so the upstream
        # encoding and byte length metadata no longer applies.
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=media_type or None,
        )

    async def _iter_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        _iter_stream(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=media_type or None,
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


@router.get(
    "/worker/sessions/{worker_session_id}", response_model=WorkerSessionResponse
)
async def get_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().get_session(worker_session_id)


@router.post(
    "/worker/sessions/{worker_session_id}/stop", response_model=WorkerSessionResponse
)
async def stop_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().stop_session(worker_session_id)


@router.post(
    "/worker/sessions/{worker_session_id}/restart", response_model=WorkerSessionResponse
)
async def restart_session(
    worker_session_id: str,
    _auth: None = WorkerAuth,
) -> WorkerSessionResponse:
    return await get_worker_service().restart_session(worker_session_id)


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
        raise HTTPException(
            status_code=503, detail="Runtime content probe not available"
        )
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
        raise HTTPException(
            status_code=503, detail="Runtime external browse not available"
        )
    return await method(payload)


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


@router.api_route(
    "/worker/sessions/{worker_session_id}/preview", methods=_PROXY_METHODS
)
@router.api_route(
    "/worker/sessions/{worker_session_id}/preview/{path:path}", methods=_PROXY_METHODS
)
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
    cached_token = os.getenv("RUNTIME_WORKER_AUTH_TOKEN", "").strip()
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
        logger.warning(
            "WS preview auth rejected for %s: %s", worker_session_id, exc.detail
        )
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
    ws_url = upstream_url.replace("http://", "ws://", 1).replace(
        "https://", "wss://", 1
    )

    requested_subprotocols = [
        str(p).strip()
        for p in (websocket.scope.get("subprotocols") or [])
        if str(p).strip()
    ]

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
        logger.debug(
            "WS preview proxy error for %s/%s: %s", worker_session_id, path, exc
        )
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
        return


# ---------------------------------------------------------------------------
# PTY session tracker – one PTY per worker session at a time
# ---------------------------------------------------------------------------
_pty_processes: dict[str, asyncio.subprocess.Process] = {}
_pty_master_fds: dict[str, int] = {}
_pty_lock = asyncio.Lock()


async def _evict_pty(session_id: str) -> None:
    """Terminate any existing PTY process for *session_id*."""
    process = _pty_processes.pop(session_id, None)
    master_fd = _pty_master_fds.pop(session_id, None)
    if process is not None and process.returncode is None:
        process.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=2)
        if process.returncode is None:
            process.kill()
    if master_fd is not None:
        with contextlib.suppress(Exception):
            os.close(master_fd)


@router.websocket("/worker/sessions/{worker_session_id}/pty")
async def pty(worker_session_id: str, websocket: WebSocket):
    # Prefer the X-PTY-Token header so the per-session token never appears
    # in URL-based access logs. The query-string fallback is retained for
    # backward compatibility with callers that still embed it in the URL.
    token = websocket.headers.get("x-pty-token", "") or websocket.query_params.get(
        "token", ""
    )
    try:
        session = await get_worker_service().verify_pty_token(worker_session_id, token)
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    await websocket.accept()

    # Evict any previous PTY for this session before spawning a new one
    async with _pty_lock:
        await _evict_pty(worker_session_id)

    # PTY shell runs inside the workspace sandbox
    shell = "/bin/bash"
    master_fd, slave_fd = pty_module.openpty()

    # Build sandbox environment for the PTY session
    sandbox_spec = session.sandbox_spec
    ensure_sandbox_ready(sandbox_spec)

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

    sandbox_preexec = make_sandbox_preexec(sandbox_spec, target_cwd=None)

    def _pty_preexec() -> None:
        # Create a new session and acquire the controlling terminal
        # BEFORE entering the sandbox.  The PTY slave fd was opened in
        # the original namespace, so TIOCSCTTY must run here — after
        # unshare(CLONE_NEWUSER) the new user namespace no longer owns
        # the device and the ioctl would silently fail.
        os.setsid()
        try:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        except OSError:
            pass
        try:
            os.tcsetpgrp(0, os.getpid())
        except OSError:
            pass
        # Now enter the sandbox (unshare/chroot/pivot_root/chdir).
        # The controlling-terminal association is stored in the kernel's
        # session struct and survives chroot/pivot_root.
        sandbox_preexec()

    try:
        process = await asyncio.create_subprocess_exec(
            *shell_command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=None,  # preexec_fn will chdir to /workspace
            env=environment,
            preexec_fn=_pty_preexec,
            start_new_session=False,
        )
        logger.debug(
            "PTY spawned: worker_session_id=%s workspace_id=%s pid=%s mode=%s",
            worker_session_id,
            session.workspace_id,
            process.pid,
            sandbox_spec.mode,
        )
    finally:
        with contextlib.suppress(Exception):
            os.close(slave_fd)

    # Register in PTY tracker so future connections can evict this one
    _pty_processes[worker_session_id] = process
    _pty_master_fds[worker_session_id] = master_fd

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
            lines = [
                ln for ln in lines if not any(noise in ln for noise in _STARTUP_NOISE)
            ]
            text = "".join(lines)
            if not text:
                return
        output_text, redaction_carry = service.split_workspace_secret_output(
            session,
            text,
            carry=redaction_carry,
        )
        if (
            text
            and not output_text
            and redaction_carry
            and not buffered_for_redaction_logged
        ):
            buffered_for_redaction_logged = True
            logger.debug(
                "PTY output buffered for redaction overlap: worker_session_id=%s chunk_len=%s carry_len=%s",
                worker_session_id,
                len(text),
                len(redaction_carry),
            )
        if output_text:
            await websocket.send_text(
                json.dumps({"type": "output", "data": output_text})
            )

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
                    await websocket.send_text(
                        json.dumps({"type": "output", "data": output_text})
                    )
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2)
        with contextlib.suppress(Exception):
            os.close(master_fd)


def include_worker_routes(application: FastAPI) -> None:
    application.include_router(router)


@contextlib.asynccontextmanager
async def _worker_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    # On shutdown (including WatchFiles reload), terminate all devserver
    # processes so orphaned children don't accumulate across reloads.
    service = get_worker_service()
    await service.shutdown()
    # Also terminate any active PTY sessions
    async with _pty_lock:
        for sid in list(_pty_processes.keys()):
            await _evict_pty(sid)


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
