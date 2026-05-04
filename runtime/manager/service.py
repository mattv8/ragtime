from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import HTTPException

from runtime.manager.models import (
    ManagerSession,
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecResponse,
    RuntimeExternalBrowseRequest,
    RuntimeExternalBrowseResponse,
    RuntimeFileReadResponse,
    RuntimePtyUrlResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeSessionResponse,
    RuntimeWorkspaceFileListResponse,
    RuntimeWorkspaceGitCommandResponse,
    RuntimeWorkspaceScmStatusResponse,
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
        self._worker_service: Any = get_worker_service()
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
        self._worker_call_timeout = self._get_positive_int_env(
            "RUNTIME_WORKER_CALL_TIMEOUT_SECONDS",
            10,
        )
        self._workspace_start_locks: dict[str, asyncio.Lock] = {}
        self._provider_locks: dict[str, asyncio.Lock] = {}
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
            runtime_capabilities=session.runtime_capabilities,
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            runtime_operation_id=session.runtime_operation_id,
            runtime_operation_phase=session.runtime_operation_phase,
            runtime_operation_started_at=session.runtime_operation_started_at,
            runtime_operation_updated_at=session.runtime_operation_updated_at,
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

    def _workspace_start_lock(self, workspace_id: str) -> asyncio.Lock:
        return self._workspace_start_locks.setdefault(workspace_id, asyncio.Lock())

    def _provider_lock(self, provider_session_id: str) -> asyncio.Lock:
        return self._provider_locks.setdefault(provider_session_id, asyncio.Lock())

    @staticmethod
    def _apply_worker_state(
        session: ManagerSession,
        worker_data: WorkerSessionResponse,
    ) -> None:
        """Sync mutable session fields from worker response (single source of truth)."""
        session.state = worker_data.state
        session.preview_internal_url = worker_data.preview_internal_url
        session.launch_framework = worker_data.launch_framework
        session.launch_command = worker_data.launch_command
        session.launch_cwd = worker_data.launch_cwd
        session.launch_port = worker_data.launch_port
        session.runtime_capabilities = worker_data.runtime_capabilities
        session.devserver_running = worker_data.devserver_running
        session.last_error = worker_data.last_error
        session.runtime_operation_id = worker_data.runtime_operation_id
        session.runtime_operation_phase = worker_data.runtime_operation_phase
        session.runtime_operation_started_at = worker_data.runtime_operation_started_at
        session.runtime_operation_updated_at = worker_data.runtime_operation_updated_at

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
            worker_session_id=worker_data.worker_session_id,
            pty_access_token=pty_access_token,
            preview_internal_url=worker_data.preview_internal_url,
            launch_framework=worker_data.launch_framework,
            launch_command=worker_data.launch_command,
            launch_cwd=worker_data.launch_cwd,
            launch_port=worker_data.launch_port,
            runtime_capabilities=worker_data.runtime_capabilities,
            state=worker_data.state,
            devserver_running=worker_data.devserver_running,
            last_error=worker_data.last_error,
            runtime_operation_id=worker_data.runtime_operation_id,
            runtime_operation_phase=worker_data.runtime_operation_phase,
            runtime_operation_started_at=worker_data.runtime_operation_started_at,
            runtime_operation_updated_at=worker_data.runtime_operation_updated_at,
            updated_at=now,
            lease_expires_at=now + timedelta(seconds=self._lease_ttl_seconds),
        )

    async def _sync_session_from_worker_by_provider_id(
        self,
        provider_session_id: str,
    ) -> ManagerSession | None:
        async with self._provider_lock(provider_session_id):
            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session:
                    return None
                worker_session_id = session.worker_session_id

            worker_data = await asyncio.wait_for(
                self._worker_service.get_session(worker_session_id),
                timeout=self._worker_call_timeout,
            )

            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session or session.worker_session_id != worker_session_id:
                    return None
                self._apply_worker_state(session, worker_data)
                now = self._utc_now()
                session.updated_at = now
                session.lease_expires_at = now + timedelta(
                    seconds=self._lease_ttl_seconds
                )
                return session

    def _purge_terminal_sessions_locked(self, now: datetime) -> None:
        for provider_session_id, session in list(self._sessions.items()):
            if session.state not in {"stopped", "error"}:
                continue
            if (
                session.state == "error"
                and (now - session.updated_at).total_seconds() < 60
            ):
                continue
            workspace_provider_session = self._workspace_index.get(session.workspace_id)
            if workspace_provider_session == provider_session_id:
                self._workspace_index.pop(session.workspace_id, None)
            self._sessions.pop(provider_session_id, None)

    async def _cleanup_expired_sessions(self, now: datetime) -> None:
        async with self._lock:
            expired_provider_ids = [
                provider_session_id
                for provider_session_id, session in self._sessions.items()
                if session.state in {"starting", "running"}
                and session.lease_expires_at <= now
            ]

        for provider_session_id in expired_provider_ids:
            async with self._provider_lock(provider_session_id):
                async with self._lock:
                    session = self._sessions.get(provider_session_id)
                    if (
                        not session
                        or session.state not in {"starting", "running"}
                        or session.lease_expires_at > now
                    ):
                        continue
                    worker_session_id = session.worker_session_id

                with suppress(Exception):
                    await asyncio.wait_for(
                        self._worker_service.stop_session(worker_session_id),
                        timeout=self._worker_call_timeout,
                    )

                async with self._lock:
                    session = self._sessions.get(provider_session_id)
                    if not session or session.worker_session_id != worker_session_id:
                        continue
                    session.state = "stopped"
                    session.devserver_running = False
                    session.last_error = "Lease expired"
                    session.updated_at = now

        async with self._lock:
            self._purge_terminal_sessions_locked(now)

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reconcile_interval_seconds)
            now = self._utc_now()
            await self._cleanup_expired_sessions(now)
            async with self._lock:
                active_session_ids = [
                    session.provider_session_id
                    for session in self._sessions.values()
                    if session.state in {"starting", "running"}
                ]

            for provider_session_id in active_session_ids:
                try:
                    await self._sync_session_from_worker_by_provider_id(
                        provider_session_id
                    )
                except Exception:
                    async with self._lock:
                        session = self._sessions.get(provider_session_id)
                        if not session:
                            continue
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = "Runtime worker heartbeat failed"
                        session.updated_at = now

    async def start_session(
        self,
        request: StartSessionRequest,
    ) -> RuntimeSessionResponse:
        await self._cleanup_expired_sessions(self._utc_now())

        async with self._workspace_start_lock(request.workspace_id):
            existing_provider_id: str | None = None
            provider_session_id: str | None = None
            pty_access_token: str | None = None

            async with self._lock:
                provider_session_id = request.provider_session_id
                if provider_session_id and provider_session_id in self._sessions:
                    existing_provider_id = provider_session_id
                else:
                    existing_id = self._workspace_index.get(request.workspace_id)
                    if existing_id and existing_id in self._sessions:
                        existing_provider_id = existing_id
                    else:
                        self._ensure_capacity_locked()
                        provider_session_id = provider_session_id or (
                            f"mgr-{request.workspace_id[:8]}-{uuid4().hex[:8]}"
                        )
                        pty_access_token = uuid4().hex

            if existing_provider_id:
                session = await self._sync_session_from_worker_by_provider_id(
                    existing_provider_id
                )
                if session is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Runtime session not found",
                    )
                async with self._lock:
                    session = self._sessions.get(existing_provider_id)
                    if not session:
                        raise HTTPException(
                            status_code=404,
                            detail="Runtime session not found",
                        )
                    now = self._utc_now()
                    session.leased_by_user_id = request.leased_by_user_id
                    session.lease_expires_at = now + timedelta(
                        seconds=self._lease_ttl_seconds
                    )
                    session.updated_at = now
                    return self._as_response(session)

            if not provider_session_id or pty_access_token is None:
                raise HTTPException(
                    status_code=500, detail="Runtime session start failed"
                )

            worker_request = WorkerStartSessionRequest(
                workspace_id=request.workspace_id,
                provider_session_id=provider_session_id,
                pty_access_token=pty_access_token,
                workspace_env=request.workspace_env,
                workspace_env_visibility=request.workspace_env_visibility,
                workspace_mounts=request.workspace_mounts,
            )
            worker_session = await asyncio.wait_for(
                self._worker_service.start_session(worker_request),
                timeout=self._worker_call_timeout,
            )
            session = self._create_session(
                provider_session_id,
                request.workspace_id,
                request.leased_by_user_id,
                worker_session,
                pty_access_token,
            )

            duplicate_worker_session_id: str | None = None
            async with self._lock:
                existing_id = self._workspace_index.get(request.workspace_id)
                if existing_id and existing_id in self._sessions:
                    session = self._sessions[existing_id]
                    duplicate_worker_session_id = worker_session.worker_session_id
                else:
                    self._sessions[provider_session_id] = session
                    self._workspace_index[session.workspace_id] = (
                        session.provider_session_id
                    )
                now = self._utc_now()
                session.leased_by_user_id = request.leased_by_user_id
                session.lease_expires_at = now + timedelta(
                    seconds=self._lease_ttl_seconds
                )
                session.updated_at = now
                response = self._as_response(session)

            if duplicate_worker_session_id:
                with suppress(Exception):
                    await asyncio.wait_for(
                        self._worker_service.stop_session(duplicate_worker_session_id),
                        timeout=self._worker_call_timeout,
                    )

            return response

    async def get_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        await self._cleanup_expired_sessions(self._utc_now())
        async with self._lock:
            if provider_session_id not in self._sessions:
                raise HTTPException(status_code=404, detail="Runtime session not found")

        session = await self._sync_session_from_worker_by_provider_id(
            provider_session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Runtime session not found")
        return self._as_response(session)

    async def stop_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        async with self._provider_lock(provider_session_id):
            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session:
                    raise HTTPException(
                        status_code=404, detail="Runtime session not found"
                    )
                worker_session_id = session.worker_session_id
                response_session = replace(session)
            try:
                worker_data = await asyncio.wait_for(
                    self._worker_service.stop_session(worker_session_id),
                    timeout=self._worker_call_timeout,
                )
                self._apply_worker_state(response_session, worker_data)
            except TimeoutError:
                response_session.state = "stopped"
                response_session.devserver_running = False
                response_session.last_error = "Worker stop timed out"
            response_session.updated_at = self._utc_now()

            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if session and session.worker_session_id == worker_session_id:
                    self._workspace_index.pop(session.workspace_id, None)
                    self._sessions.pop(provider_session_id, None)
            return self._as_response(response_session)

    async def restart_devserver(
        self,
        provider_session_id: str,
        workspace_env: dict[str, str] | None = None,
        workspace_env_visibility: dict[str, bool] | None = None,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> RuntimeSessionResponse:
        async with self._provider_lock(provider_session_id):
            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session:
                    raise HTTPException(
                        status_code=404, detail="Runtime session not found"
                    )
                worker_session_id = session.worker_session_id
            worker_data = await asyncio.wait_for(
                self._worker_service.restart_session(
                    worker_session_id,
                    workspace_env,
                    workspace_env_visibility,
                    workspace_mounts,
                ),
                timeout=self._worker_call_timeout,
            )

            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session or session.worker_session_id != worker_session_id:
                    raise HTTPException(
                        status_code=404, detail="Runtime session not found"
                    )
                self._apply_worker_state(session, worker_data)
                now = self._utc_now()
                session.updated_at = now
                session.lease_expires_at = now + timedelta(
                    seconds=self._lease_ttl_seconds
                )
                return self._as_response(session)

    async def refresh_mounts(
        self,
        provider_session_id: str,
        workspace_mounts: list[dict[str, Any]],
    ) -> RuntimeSessionResponse:
        async with self._provider_lock(provider_session_id):
            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session:
                    raise HTTPException(
                        status_code=404, detail="Runtime session not found"
                    )
                worker_session_id = session.worker_session_id
            worker_data = await asyncio.wait_for(
                self._worker_service.refresh_mounts(
                    worker_session_id,
                    workspace_mounts=workspace_mounts,
                ),
                timeout=self._worker_call_timeout,
            )

            async with self._lock:
                session = self._sessions.get(provider_session_id)
                if not session or session.worker_session_id != worker_session_id:
                    raise HTTPException(
                        status_code=404, detail="Runtime session not found"
                    )
                self._apply_worker_state(session, worker_data)
                now = self._utc_now()
                session.updated_at = now
                session.lease_expires_at = now + timedelta(
                    seconds=self._lease_ttl_seconds
                )
                return self._as_response(session)

    def _get_session_or_raise(self, provider_session_id: str) -> ManagerSession:
        """Look up a session without acquiring the lock (caller must hold it or accept race)."""
        session = self._sessions.get(provider_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Runtime session not found")
        return session

    async def get_pty_websocket_url(
        self,
        provider_session_id: str,
    ) -> RuntimePtyUrlResponse:
        session = self._get_session_or_raise(provider_session_id)
        parsed = urlparse(session.preview_internal_url)
        if parsed.scheme == "https":
            ws_scheme = "wss"
        elif parsed.scheme == "http":
            ws_scheme = "ws"
        else:
            raise HTTPException(status_code=502, detail="Runtime preview URL invalid")
        ws_url = (
            f"{ws_scheme}://{parsed.hostname}"
            f"{':%d' % parsed.port if parsed.port else ''}"
            f"/worker/sessions/{quote(session.worker_session_id, safe='')}/pty"
            f"?token={quote(session.pty_access_token, safe='')}"
        )
        return RuntimePtyUrlResponse(ws_url=ws_url)

    async def read_file(
        self,
        provider_session_id: str,
        file_path: str,
    ) -> RuntimeFileReadResponse:
        session = self._get_session_or_raise(provider_session_id)
        return await self._worker_service.read_file(
            session.worker_session_id, file_path
        )

    async def write_file(
        self,
        provider_session_id: str,
        file_path: str,
        content: str,
    ) -> RuntimeFileReadResponse:
        session = self._get_session_or_raise(provider_session_id)
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
        session = self._get_session_or_raise(provider_session_id)
        return await self._worker_service.delete_file(
            session.worker_session_id, file_path
        )

    async def capture_screenshot(
        self,
        provider_session_id: str,
        payload: RuntimeScreenshotRequest,
    ) -> RuntimeScreenshotResponse:
        session = self._get_session_or_raise(provider_session_id)
        return await self._worker_service.capture_screenshot(
            session.worker_session_id,
            payload,
        )

    async def content_probe(
        self,
        provider_session_id: str,
        payload: RuntimeContentProbeRequest,
    ) -> RuntimeContentProbeResponse:
        session = self._get_session_or_raise(provider_session_id)
        return await self._worker_service.content_probe(
            session.worker_session_id,
            payload,
        )

    async def exec_command(
        self,
        provider_session_id: str,
        command: str,
        timeout_seconds: int = 30,
        cwd: str | None = None,
    ) -> RuntimeExecResponse:
        session = self._get_session_or_raise(provider_session_id)
        return await self._worker_service.exec_command(
            session.worker_session_id,
            command,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )

    async def external_browse(
        self,
        payload: "RuntimeExternalBrowseRequest",
    ) -> "RuntimeExternalBrowseResponse":
        method = getattr(self._worker_service, "external_browse", None)
        if method is None:
            raise HTTPException(
                status_code=503,
                detail="Runtime external browse capability not available",
            )
        return await asyncio.wait_for(
            method(payload),
            timeout=max(self._worker_call_timeout, int(payload.timeout_ms / 1000) + 5),
        )

    async def external_browse_for_session(
        self,
        provider_session_id: str,
        payload: RuntimeExternalBrowseRequest,
    ) -> RuntimeExternalBrowseResponse:
        session = self._get_session_or_raise(provider_session_id)
        return await asyncio.wait_for(
            self._worker_service.external_browse(
                payload,
                worker_session_id=session.worker_session_id,
            ),
            timeout=max(self._worker_call_timeout, int(payload.timeout_ms / 1000) + 5),
        )

    async def list_workspace_files(
        self,
        workspace_id: str,
        *,
        include_dirs: bool = False,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> RuntimeWorkspaceFileListResponse:
        return await self._worker_service.list_workspace_files(
            workspace_id,
            include_dirs=include_dirs,
            workspace_mounts=workspace_mounts or [],
        )

    async def run_workspace_git_command(
        self,
        workspace_id: str,
        *,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> RuntimeWorkspaceGitCommandResponse:
        return await self._worker_service.run_workspace_git_command(
            workspace_id,
            args=args,
            env=env,
        )

    async def get_workspace_scm_status(
        self,
        workspace_id: str,
    ) -> RuntimeWorkspaceScmStatusResponse:
        return await self._worker_service.get_workspace_scm_status(workspace_id)

    async def pool_status(self) -> dict[str, Any]:
        async def _collect() -> dict[str, Any]:
            async with self._lock:
                active = [
                    s
                    for s in self._sessions.values()
                    if s.state in {"running", "starting"}
                ]
                return {
                    "workers_total": 1,
                    "workers_leased": len(active),
                    "active_sessions": len(active),
                    "max_sessions": self._max_sessions,
                    "sessions": [
                        {
                            "provider_session_id": s.provider_session_id,
                            "workspace_id": s.workspace_id,
                            "state": s.state,
                            "devserver_running": s.devserver_running,
                        }
                        for s in active
                    ],
                }

        try:
            return await asyncio.wait_for(_collect(), timeout=2)
        except TimeoutError:
            # Lock held by reconcile loop; return minimal healthy response
            return {
                "workers_total": 1,
                "workers_leased": -1,
                "active_sessions": -1,
                "max_sessions": self._max_sessions,
                "sessions": [],
            }
