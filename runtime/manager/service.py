from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException

from runtime.manager.models import (
    ManagerSession,
    RuntimeFileReadResponse,
    RuntimePtyUrlResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeSessionResponse,
    StartSessionRequest,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.service import get_worker_service


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ManagerSession] = {}
        self._workspace_index: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._worker_service = get_worker_service()
        self._max_sessions = self._get_positive_int_env(
            "RUNTIME_MAX_SESSIONS",
            12,
        )
        self._lease_ttl_seconds = self._get_positive_int_env(
            "RUNTIME_LEASE_TTL_SECONDS",
            3600,
        )
        self._reconcile_interval_seconds = self._get_positive_int_env(
            "RUNTIME_RECONCILE_INTERVAL_SECONDS",
            15,
        )
        self._reconcile_task: asyncio.Task[None] | None = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _get_positive_int_env(name: str, default_value: int) -> int:
        raw_value = os.getenv(name, str(default_value)).strip()
        try:
            parsed = int(raw_value)
            return parsed if parsed > 0 else default_value
        except Exception:
            return default_value

    async def startup(self) -> None:
        if self._reconcile_task is None:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def shutdown(self) -> None:
        task = self._reconcile_task
        self._reconcile_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    @staticmethod
    def _as_response(session: ManagerSession) -> RuntimeSessionResponse:
        return RuntimeSessionResponse(
            provider_session_id=session.provider_session_id,
            workspace_id=session.workspace_id,
            state=session.state,
            preview_internal_url=session.preview_internal_url,
            launch_framework=session.launch_framework,
            launch_command=session.launch_command,
            launch_cwd=session.launch_cwd,
            launch_port=session.launch_port,
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            updated_at=session.updated_at,
        )

    def _ensure_capacity_locked(self) -> None:
        active_sessions = sum(
            1
            for session in self._sessions.values()
            if session.state in {"running", "starting"}
        )
        if active_sessions >= self._max_sessions:
            raise HTTPException(
                status_code=503,
                detail="Runtime capacity exhausted. Retry after an active session stops.",
            )

    def _create_session(
        self,
        provider_session_id: str,
        workspace_id: str,
        leased_by_user_id: str,
        worker_data: WorkerSessionResponse,
        pty_access_token: str,
    ) -> ManagerSession:
        now = self._utc_now()
        return ManagerSession(
            provider_session_id=provider_session_id,
            workspace_id=workspace_id,
            leased_by_user_id=leased_by_user_id,
            worker_id="runtime-internal",
            worker_base_url="internal",
            worker_session_id=worker_data.worker_session_id,
            pty_access_token=pty_access_token,
            preview_internal_url=worker_data.preview_internal_url,
            launch_framework=worker_data.launch_framework,
            launch_command=worker_data.launch_command,
            launch_cwd=worker_data.launch_cwd,
            launch_port=worker_data.launch_port,
            state=worker_data.state,
            devserver_running=worker_data.devserver_running,
            last_error=worker_data.last_error,
            updated_at=now,
            lease_expires_at=now + timedelta(seconds=self._lease_ttl_seconds),
        )

    async def _sync_session_from_worker(self, session: ManagerSession) -> None:
        parsed = await self._worker_service.get_session(session.worker_session_id)
        session.state = parsed.state
        session.preview_internal_url = parsed.preview_internal_url
        session.launch_framework = parsed.launch_framework
        session.launch_command = parsed.launch_command
        session.launch_cwd = parsed.launch_cwd
        session.launch_port = parsed.launch_port
        session.devserver_running = parsed.devserver_running
        session.last_error = parsed.last_error
        session.updated_at = self._utc_now()
        session.lease_expires_at = self._utc_now() + timedelta(
            seconds=self._lease_ttl_seconds
        )

    async def _cleanup_expired_sessions_locked(self, now: datetime) -> None:
        for provider_session_id, session in list(self._sessions.items()):
            if (
                session.state in {"starting", "running"}
                and session.lease_expires_at <= now
            ):
                with suppress(Exception):
                    await self._worker_service.stop_session(session.worker_session_id)
                session.state = "stopped"
                session.devserver_running = False
                session.last_error = "Lease expired"
                session.updated_at = now

            if session.state in {"stopped", "error"}:
                workspace_provider_session = self._workspace_index.get(
                    session.workspace_id
                )
                if workspace_provider_session == provider_session_id:
                    self._workspace_index.pop(session.workspace_id, None)
                self._sessions.pop(provider_session_id, None)

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reconcile_interval_seconds)
            now = self._utc_now()
            async with self._lock:
                await self._cleanup_expired_sessions_locked(now)
                for session in list(self._sessions.values()):
                    if session.state not in {"starting", "running"}:
                        continue
                    try:
                        await self._sync_session_from_worker(session)
                    except Exception:
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = "Runtime worker heartbeat failed"
                        session.updated_at = now

    async def start_session(
        self,
        request: StartSessionRequest,
    ) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            await self._cleanup_expired_sessions_locked(now)

            provider_session_id = request.provider_session_id
            if provider_session_id and provider_session_id in self._sessions:
                session = self._sessions[provider_session_id]
                await self._sync_session_from_worker(session)
            else:
                existing_id = self._workspace_index.get(request.workspace_id)
                if existing_id and existing_id in self._sessions:
                    session = self._sessions[existing_id]
                    await self._sync_session_from_worker(session)
                else:
                    self._ensure_capacity_locked()
                    provider_session_id = provider_session_id or (
                        f"mgr-{request.workspace_id[:8]}-{uuid4().hex[:8]}"
                    )
                    pty_access_token = uuid4().hex
                    worker_request = WorkerStartSessionRequest(
                        workspace_id=request.workspace_id,
                        provider_session_id=provider_session_id,
                        pty_access_token=pty_access_token,
                    )
                    worker_session = await self._worker_service.start_session(
                        worker_request
                    )
                    session = self._create_session(
                        provider_session_id,
                        request.workspace_id,
                        request.leased_by_user_id,
                        worker_session,
                        pty_access_token,
                    )
                    self._sessions[provider_session_id] = session
                    self._workspace_index[session.workspace_id] = (
                        session.provider_session_id
                    )

            session.leased_by_user_id = request.leased_by_user_id
            session.lease_expires_at = now + timedelta(seconds=self._lease_ttl_seconds)
            session.updated_at = now
            return self._as_response(session)

    async def get_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            await self._cleanup_expired_sessions_locked(now)
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            await self._sync_session_from_worker(session)
            return self._as_response(session)

    async def stop_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            parsed = await self._worker_service.stop_session(session.worker_session_id)
            session.state = parsed.state
            session.devserver_running = parsed.devserver_running
            session.last_error = parsed.last_error
            session.updated_at = self._utc_now()
            self._workspace_index.pop(session.workspace_id, None)
            self._sessions.pop(provider_session_id, None)
            return self._as_response(session)

    async def restart_devserver(
        self,
        provider_session_id: str,
    ) -> RuntimeSessionResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            parsed = await self._worker_service.restart_session(
                session.worker_session_id
            )
            session.state = parsed.state
            session.preview_internal_url = parsed.preview_internal_url
            session.devserver_running = parsed.devserver_running
            session.last_error = parsed.last_error
            session.updated_at = self._utc_now()
            session.lease_expires_at = self._utc_now() + timedelta(
                seconds=self._lease_ttl_seconds
            )
            return self._as_response(session)

    async def get_pty_websocket_url(
        self,
        provider_session_id: str,
    ) -> RuntimePtyUrlResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            base_url = session.preview_internal_url
            if base_url.startswith("https://"):
                ws_base = "wss://" + base_url[8:].split("/worker/sessions/")[0]
            elif base_url.startswith("http://"):
                ws_base = "ws://" + base_url[7:].split("/worker/sessions/")[0]
            else:
                raise HTTPException(
                    status_code=502, detail="Runtime preview URL invalid"
                )
            ws_url = (
                f"{ws_base}/worker/sessions/{quote(session.worker_session_id, safe='')}/pty"
                f"?token={quote(session.pty_access_token, safe='')}"
            )
            return RuntimePtyUrlResponse(ws_url=ws_url)

    async def read_file(
        self,
        provider_session_id: str,
        file_path: str,
    ) -> RuntimeFileReadResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            return await self._worker_service.read_file(
                session.worker_session_id, file_path
            )

    async def write_file(
        self,
        provider_session_id: str,
        file_path: str,
        content: str,
    ) -> RuntimeFileReadResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            return await self._worker_service.write_file(
                session.worker_session_id,
                file_path,
                content,
            )

    async def delete_file(
        self,
        provider_session_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            return await self._worker_service.delete_file(
                session.worker_session_id, file_path
            )

    async def capture_screenshot(
        self,
        provider_session_id: str,
        payload: RuntimeScreenshotRequest,
    ) -> RuntimeScreenshotResponse:
        async with self._lock:
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            return await cast(Any, self._worker_service).capture_screenshot(
                session.worker_session_id,
                payload,
            )

    async def pool_status(self) -> dict[str, int]:
        async with self._lock:
            active_sessions = sum(
                1
                for session in self._sessions.values()
                if session.state in {"running", "starting"}
            )
            return {
                "workers_total": 1,
                "workers_leased": active_sessions,
                "active_sessions": active_sessions,
            }
