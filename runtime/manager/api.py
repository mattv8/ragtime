from __future__ import annotations

import os
from typing import Any, cast

from fastapi import FastAPI, Header, HTTPException

from runtime.manager.models import (
    RuntimeFileReadResponse,
    RuntimeFileWriteRequest,
    RuntimeManagerHealthResponse,
    RuntimePtyUrlResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeSessionResponse,
    StartSessionRequest,
)
from runtime.manager.service import SessionManager


def create_app() -> FastAPI:
    application = FastAPI(title="Ragtime User Space Runtime Manager", version="0.1.0")
    manager = SessionManager()
    manager_auth_token = os.getenv("RUNTIME_MANAGER_AUTH_TOKEN", "").strip()

    def _authorize_manager_call(authorization: str | None) -> None:
        if not manager_auth_token:
            return
        value = (authorization or "").strip()
        if not value.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing runtime manager auth")
        if value[7:] != manager_auth_token:
            raise HTTPException(status_code=403, detail="Invalid runtime manager auth")

    @application.on_event("startup")
    async def on_startup() -> None:
        await manager.startup()

    @application.on_event("shutdown")
    async def on_shutdown() -> None:
        await manager.shutdown()

    @application.get("/health", response_model=RuntimeManagerHealthResponse)
    async def health() -> RuntimeManagerHealthResponse:
        pool = await manager.pool_status()
        return RuntimeManagerHealthResponse(
            status="ok",
            workers_total=pool["workers_total"],
            workers_leased=pool["workers_leased"],
            active_sessions=pool["active_sessions"],
        )

    @application.post("/sessions/start", response_model=RuntimeSessionResponse)
    async def start_session(
        request: StartSessionRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeSessionResponse:
        _authorize_manager_call(authorization)
        return await manager.start_session(request)

    @application.get(
        "/sessions/{provider_session_id}", response_model=RuntimeSessionResponse
    )
    async def get_session(
        provider_session_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeSessionResponse:
        _authorize_manager_call(authorization)
        return await manager.get_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/stop", response_model=RuntimeSessionResponse
    )
    async def stop_session(
        provider_session_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeSessionResponse:
        _authorize_manager_call(authorization)
        return await manager.stop_session(provider_session_id)

    @application.post(
        "/sessions/{provider_session_id}/restart", response_model=RuntimeSessionResponse
    )
    async def restart_session(
        provider_session_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeSessionResponse:
        _authorize_manager_call(authorization)
        return await manager.restart_devserver(provider_session_id)

    @application.get(
        "/sessions/{provider_session_id}/pty/ws-url",
        response_model=RuntimePtyUrlResponse,
    )
    async def get_pty_ws_url(
        provider_session_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimePtyUrlResponse:
        _authorize_manager_call(authorization)
        return await manager.get_pty_websocket_url(provider_session_id)

    @application.get(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def read_file(
        provider_session_id: str,
        file_path: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeFileReadResponse:
        _authorize_manager_call(authorization)
        return await manager.read_file(provider_session_id, file_path)

    @application.put(
        "/sessions/{provider_session_id}/fs/{file_path:path}",
        response_model=RuntimeFileReadResponse,
    )
    async def write_file(
        provider_session_id: str,
        file_path: str,
        payload: RuntimeFileWriteRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeFileReadResponse:
        _authorize_manager_call(authorization)
        return await manager.write_file(
            provider_session_id,
            file_path,
            payload.content,
        )

    @application.delete("/sessions/{provider_session_id}/fs/{file_path:path}")
    async def delete_file(
        provider_session_id: str,
        file_path: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        _authorize_manager_call(authorization)
        return await manager.delete_file(provider_session_id, file_path)

    @application.post(
        "/sessions/{provider_session_id}/screenshot",
        response_model=RuntimeScreenshotResponse,
    )
    async def capture_screenshot(
        provider_session_id: str,
        payload: RuntimeScreenshotRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> RuntimeScreenshotResponse:
        _authorize_manager_call(authorization)
        return await cast(Any, manager).capture_screenshot(provider_session_id, payload)

    return application


app = create_app()
