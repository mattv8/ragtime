from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from runtime.auth import ManagerAuth
from runtime.manager.models import (
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecRequest,
    RuntimeExecResponse,
    RuntimeFileReadResponse,
    RuntimeFileWriteRequest,
    RuntimeManagerHealthResponse,
    RuntimeMountRefreshRequest,
    RuntimePtyUrlResponse,
    RuntimeRestartRequest,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeSessionResponse,
    StartSessionRequest,
)
from runtime.manager.service import SessionManager


def create_app() -> FastAPI:
    manager = SessionManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.startup()
        yield
        await manager.shutdown()
        # Also clean up local worker devserver processes (relevant when
        # manager+worker routes are combined in the same app).
        try:
            from runtime.worker.service import get_worker_service

            svc = get_worker_service()
            async with svc._lock:
                for sid in list(svc._devserver_processes.keys()):
                    await svc._terminate_devserver_locked(sid)
        except Exception:
            pass

    application = FastAPI(
        title="Ragtime User Space Runtime Manager",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=RuntimeManagerHealthResponse)
    async def health() -> RuntimeManagerHealthResponse:
        pool = await manager.pool_status()
        return RuntimeManagerHealthResponse(
            status="ok",
            workers_total=pool["workers_total"],
            workers_leased=pool["workers_leased"],
            active_sessions=pool["active_sessions"],
            max_sessions=pool["max_sessions"],
            sessions=pool["sessions"],
        )

    @application.post("/sessions/start", response_model=RuntimeSessionResponse)
    async def start_session(
        request: StartSessionRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.start_session(request)

    @application.get(
        "/sessions/{provider_session_id}", response_model=RuntimeSessionResponse
    )
    async def get_session(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.get_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/stop", response_model=RuntimeSessionResponse
    )
    async def stop_session(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.stop_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/restart", response_model=RuntimeSessionResponse
    )
    async def restart_session(
        provider_session_id: str,
        payload: RuntimeRestartRequest | None = None,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.restart_devserver(
            provider_session_id,
            workspace_env=payload.workspace_env if payload else None,
            workspace_mounts=payload.workspace_mounts if payload else None,
        )

    @application.post(
        "/sessions/{provider_session_id}/mounts/refresh",
        response_model=RuntimeSessionResponse,
    )
    async def refresh_mounts(
        provider_session_id: str,
        payload: RuntimeMountRefreshRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeSessionResponse:
        return await manager.refresh_mounts(
            provider_session_id,
            workspace_mounts=payload.workspace_mounts,
        )

    @application.get(
        "/sessions/{provider_session_id}/pty/ws-url",
        response_model=RuntimePtyUrlResponse,
    )
    async def get_pty_ws_url(
        provider_session_id: str,
        _auth: None = ManagerAuth,
    ) -> RuntimePtyUrlResponse:
        return await manager.get_pty_websocket_url(provider_session_id)

    @application.get(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def read_file(
        provider_session_id: str,
        file_path: str,
        _auth: None = ManagerAuth,
    ) -> RuntimeFileReadResponse:
        return await manager.read_file(provider_session_id, file_path)

    @application.put(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def write_file(
        provider_session_id: str,
        file_path: str,
        payload: RuntimeFileWriteRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeFileReadResponse:
        return await manager.write_file(
            provider_session_id,
            file_path,
            payload.content,
        )

    @application.delete("/sessions/{provider_session_id}/fs/{file_path:path}")
    async def delete_file(
        provider_session_id: str,
        file_path: str,
        _auth: None = ManagerAuth,
    ) -> dict[str, Any]:
        return await manager.delete_file(provider_session_id, file_path)

    @application.post(
        "/sessions/{provider_session_id}/screenshot",
        response_model=RuntimeScreenshotResponse,
    )
    async def capture_screenshot(
        provider_session_id: str,
        payload: RuntimeScreenshotRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeScreenshotResponse:
        return await manager.capture_screenshot(provider_session_id, payload)

    @application.post(
        "/sessions/{provider_session_id}/content-probe",
        response_model=RuntimeContentProbeResponse,
    )
    async def content_probe(
        provider_session_id: str,
        payload: RuntimeContentProbeRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeContentProbeResponse:
        return await manager.content_probe(provider_session_id, payload)

    @application.post(
        "/sessions/{provider_session_id}/exec",
        response_model=RuntimeExecResponse,
    )
    async def exec_command(
        provider_session_id: str,
        payload: RuntimeExecRequest,
        _auth: None = ManagerAuth,
    ) -> RuntimeExecResponse:
        return await manager.exec_command(
            provider_session_id,
            payload.command,
            timeout_seconds=payload.timeout_seconds,
            cwd=payload.cwd,
        )

    return application


app = create_app()
