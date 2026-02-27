from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import pty as pty_module
import struct
import termios
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response

from runtime.auth import WorkerAuth
from runtime.manager.models import (
    RuntimeFileReadResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.service import get_worker_service

router = APIRouter(tags=["Runtime Worker"])

# Safe environment variables to forward into PTY sessions.
_PTY_ENV_ALLOWLIST = {
    "HOME",
    "PATH",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SHELL",
    "USER",
    "LOGNAME",
    "HOSTNAME",
    "PWD",
    "RUNTIME_WORKSPACE_ROOT",
    "INDEX_DATA_PATH",
}


@router.get("/worker/health", response_model=WorkerHealthResponse)
async def health() -> WorkerHealthResponse:
    return await get_worker_service().health()


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


@router.get("/worker/sessions/{worker_session_id}/preview")
@router.get("/worker/sessions/{worker_session_id}/preview/{path:path}")
async def preview(
    worker_session_id: str,
    request: Request,
    path: str = "",
) -> Response:
    content, media_type, status_code = await get_worker_service().preview_response(
        worker_session_id,
        path,
        query=request.url.query or None,
    )
    return Response(content=content, media_type=media_type, status_code=status_code)


@router.websocket("/worker/sessions/{worker_session_id}/pty")
async def pty(worker_session_id: str, websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    try:
        session = await get_worker_service().verify_pty_token(worker_session_id, token)
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    await websocket.accept()
    shell = os.environ.get("SHELL", "/bin/bash")
    shell_command = [shell, "-i"]
    if os.path.basename(shell).endswith("bash"):
        shell_command = [shell, "--noprofile", "--norc", "-i"]
    master_fd, slave_fd = pty_module.openpty()
    # Only forward safe env vars into the PTY to avoid leaking secrets
    # (e.g. RUNTIME_WORKER_AUTH_TOKEN, API keys)
    environment = {k: v for k, v in os.environ.items() if k in _PTY_ENV_ALLOWLIST}
    environment.setdefault("TERM", "xterm-256color")
    user = os.environ.get("USER", "user")
    environment["PS1"] = f"{user}$ "
    environment["PROMPT_COMMAND"] = ""

    try:
        process = await asyncio.create_subprocess_exec(
            *shell_command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(session.workspace_root),
            env=environment,
            start_new_session=True,
        )
    finally:
        with contextlib.suppress(Exception):
            os.close(slave_fd)

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

    async def _pump_stream() -> None:
        while True:
            try:
                chunk = await asyncio.to_thread(os.read, master_fd, 1024)
            except OSError:
                break
            if not chunk:
                break
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "output",
                        "data": chunk.decode("utf-8", errors="replace"),
                    }
                )
            )

    stream_task = asyncio.create_task(_pump_stream())
    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")
            if msg_type == "resize":
                cols = int(payload.get("cols", 80))
                rows = int(payload.get("rows", 24))
                await asyncio.to_thread(_resize_pty, master_fd, cols, rows)
                continue
            if msg_type != "input":
                continue
            line = str(payload.get("data", ""))
            try:
                await asyncio.to_thread(
                    os.write,
                    master_fd,
                    line.encode("utf-8", errors="ignore"),
                )
            except OSError:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2)
        stream_task.cancel()
        with contextlib.suppress(Exception):
            await stream_task
        with contextlib.suppress(Exception):
            os.close(master_fd)


def include_worker_routes(application: FastAPI) -> None:
    application.include_router(router)


def create_app() -> FastAPI:
    application = FastAPI(title="Ragtime User Space Runtime Worker", version="0.1.0")
    include_worker_routes(application)

    @application.get("/health", response_model=WorkerHealthResponse)
    async def standalone_health() -> WorkerHealthResponse:
        return await get_worker_service().health()

    return application


app = create_app()
