from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response

from runtime.manager.models import (
    RuntimeFileReadResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.service import get_worker_service

router = APIRouter(tags=["Runtime Worker"])


def _authorize_worker_call(authorization: str | None) -> None:
    worker_auth_token = os.getenv("RUNTIME_WORKER_AUTH_TOKEN", "").strip()
    if not worker_auth_token:
        return
    value = (authorization or "").strip()
    if not value.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing runtime worker auth")
    if value[7:] != worker_auth_token:
        raise HTTPException(status_code=403, detail="Invalid runtime worker auth")


@router.get("/worker/health", response_model=WorkerHealthResponse)
async def health() -> WorkerHealthResponse:
    return await get_worker_service().health()


@router.post("/worker/sessions/start", response_model=WorkerSessionResponse)
async def start_session(
    payload: WorkerStartSessionRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> WorkerSessionResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().start_session(payload)


@router.get(
    "/worker/sessions/{worker_session_id}", response_model=WorkerSessionResponse
)
async def get_session(
    worker_session_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> WorkerSessionResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().get_session(worker_session_id)


@router.post(
    "/worker/sessions/{worker_session_id}/stop", response_model=WorkerSessionResponse
)
async def stop_session(
    worker_session_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> WorkerSessionResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().stop_session(worker_session_id)


@router.post(
    "/worker/sessions/{worker_session_id}/restart", response_model=WorkerSessionResponse
)
async def restart_session(
    worker_session_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> WorkerSessionResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().restart_session(worker_session_id)


@router.get(
    "/worker/sessions/{worker_session_id}/fs/{file_path:path}",
    response_model=RuntimeFileReadResponse,
)
async def read_file(
    worker_session_id: str,
    file_path: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> RuntimeFileReadResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().read_file(worker_session_id, file_path)


@router.put(
    "/worker/sessions/{worker_session_id}/fs/{file_path:path}",
    response_model=RuntimeFileReadResponse,
)
async def write_file(
    worker_session_id: str,
    file_path: str,
    payload: dict[str, Any],
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> RuntimeFileReadResponse:
    _authorize_worker_call(authorization)
    return await get_worker_service().write_file(
        worker_session_id,
        file_path,
        str(payload.get("content", "")),
    )


@router.delete("/worker/sessions/{worker_session_id}/fs/{file_path:path}")
async def delete_file(
    worker_session_id: str,
    file_path: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    _authorize_worker_call(authorization)
    return await get_worker_service().delete_file(worker_session_id, file_path)


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
    process = await asyncio.create_subprocess_exec(
        shell,
        "-i",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(session.workspace_root),
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

    async def _pump_stream(reader: asyncio.StreamReader) -> None:
        while True:
            chunk = await reader.read(1024)
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

    if process.stdout is None or process.stderr is None:
        await websocket.close(code=1011)
        return

    stdout_task = asyncio.create_task(_pump_stream(process.stdout))
    stderr_task = asyncio.create_task(_pump_stream(process.stderr))
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "input":
                continue
            if process.stdin is None:
                continue
            line = str(payload.get("data", ""))
            process.stdin.write(line.encode("utf-8", errors="ignore"))
            await process.stdin.drain()
    except WebSocketDisconnect:
        pass
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2)
        for task in (stdout_task, stderr_task):
            task.cancel()
            with contextlib.suppress(Exception):
                await task


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
