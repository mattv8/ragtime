from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from runtime.manager.models import (
    RuntimeFileReadResponse,
    RuntimeSessionState,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)


@dataclass
class WorkerSession:
    id: str
    workspace_id: str
    provider_session_id: str
    workspace_root: Path
    pty_access_token: str
    state: RuntimeSessionState
    devserver_running: bool
    last_error: str | None
    updated_at: datetime


class WorkerService:
    def __init__(self) -> None:
        self._sessions: dict[str, WorkerSession] = {}
        self._provider_to_session: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._worker_name = os.getenv("RUNTIME_WORKER_NAME", "runtime-worker").strip()
        self._base_url = (
            os.getenv("RUNTIME_WORKER_BASE_URL", "http://runtime:8090")
            .strip()
            .rstrip("/")
        )
        self._root = Path(
            os.getenv("RUNTIME_WORKSPACE_ROOT", "/data/_userspace")
        ).resolve()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_file_path(self, file_path: str) -> str:
        normalized = file_path.replace("\\", "/").strip().lstrip("/")
        path = Path(normalized)
        if not normalized or any(part == ".." for part in path.parts):
            raise HTTPException(status_code=400, detail="Invalid file path")
        return "/".join(path.parts)

    def _session_response(self, session: WorkerSession) -> WorkerSessionResponse:
        return WorkerSessionResponse(
            worker_session_id=session.id,
            workspace_id=session.workspace_id,
            state=session.state,
            preview_internal_url=f"{self._base_url}/worker/sessions/{session.id}/preview",
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            updated_at=session.updated_at,
        )

    def _runtime_file_response(
        self,
        session: WorkerSession,
        rel_path: str,
        content: str,
    ) -> RuntimeFileReadResponse:
        return RuntimeFileReadResponse(
            path=rel_path,
            content=content,
            updated_at=session.updated_at,
        )

    async def start_session(
        self,
        request: WorkerStartSessionRequest,
    ) -> WorkerSessionResponse:
        async with self._lock:
            existing_session_id = self._provider_to_session.get(
                request.provider_session_id
            )
            if existing_session_id and existing_session_id in self._sessions:
                session = self._sessions[existing_session_id]
                session.state = "running"
                session.devserver_running = True
                session.last_error = None
                session.pty_access_token = request.pty_access_token
                session.updated_at = self._utc_now()
                return self._session_response(session)

            session_id = f"wkr-{request.workspace_id[:8]}-{os.urandom(4).hex()}"
            workspace_root = self._root / request.workspace_id
            workspace_root.mkdir(parents=True, exist_ok=True)
            session = WorkerSession(
                id=session_id,
                workspace_id=request.workspace_id,
                provider_session_id=request.provider_session_id,
                workspace_root=workspace_root,
                pty_access_token=request.pty_access_token,
                state="running",
                devserver_running=True,
                last_error=None,
                updated_at=self._utc_now(),
            )
            self._sessions[session_id] = session
            self._provider_to_session[request.provider_session_id] = session_id
            return self._session_response(session)

    async def get_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def stop_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            session.state = "stopped"
            session.devserver_running = False
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def restart_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            session.state = "running"
            session.devserver_running = True
            session.last_error = None
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def read_file(
        self,
        worker_session_id: str,
        file_path: str,
    ) -> RuntimeFileReadResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            rel_path = self._normalize_file_path(file_path)
            target = session.workspace_root / rel_path
            if not target.exists() or not target.is_file():
                return self._runtime_file_response(session, rel_path, "")
            content = target.read_text(encoding="utf-8")
            session.updated_at = self._utc_now()
            return self._runtime_file_response(session, rel_path, content)

    async def write_file(
        self,
        worker_session_id: str,
        file_path: str,
        content: str,
    ) -> RuntimeFileReadResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            rel_path = self._normalize_file_path(file_path)
            target = session.workspace_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            session.updated_at = self._utc_now()
            return self._runtime_file_response(session, rel_path, content)

    async def delete_file(
        self, worker_session_id: str, file_path: str
    ) -> dict[str, str | bool]:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            rel_path = self._normalize_file_path(file_path)
            target = session.workspace_root / rel_path
            if target.exists() and target.is_file():
                target.unlink()
            session.updated_at = self._utc_now()
            return {"success": True, "path": rel_path}

    async def preview_html(self, worker_session_id: str, path: str) -> str:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")
            normalized = self._normalize_file_path(path) if path else ""
            if normalized:
                candidate = session.workspace_root / normalized
                if candidate.exists() and candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            return (
                "<!doctype html><html><head><meta charset='utf-8'><title>Runtime Preview</title></head>"
                "<body><main><h2>Runtime session active</h2><p>Runtime preview is running.</p></main></body></html>"
            )

    async def verify_pty_token(
        self, worker_session_id: str, token: str
    ) -> WorkerSession:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if token != session.pty_access_token:
                raise HTTPException(status_code=403, detail="Invalid PTY token")
            return session

    async def health(self) -> WorkerHealthResponse:
        async with self._lock:
            active_sessions = sum(
                1
                for session in self._sessions.values()
                if session.state in {"running", "starting"}
            )
            return WorkerHealthResponse(
                status="ok",
                service_mode="worker",
                active_sessions=active_sessions,
                metadata={"worker_name": self._worker_name},
            )


@lru_cache(maxsize=1)
def get_worker_service() -> WorkerService:
    return WorkerService()
