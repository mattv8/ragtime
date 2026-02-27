from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import httpx
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
    devserver_port: int | None
    devserver_command: list[str] | None
    launch_framework: str | None
    launch_cwd: str | None
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
        self._devserver_processes: dict[str, asyncio.subprocess.Process] = {}
        self._devserver_start_timeout_seconds = int(
            os.getenv("RUNTIME_DEVSERVER_START_TIMEOUT_SECONDS", "25")
        )
        self._runtime_config_file = ".ragtime/runtime-entrypoint.json"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_file_path(self, file_path: str) -> str:
        normalized = file_path.replace("\\", "/").strip().lstrip("/")
        path = Path(normalized)
        if not normalized or any(part == ".." for part in path.parts):
            raise HTTPException(status_code=400, detail="Invalid file path")
        return "/".join(path.parts)

    def _resolve_workspace_root(self, workspace_id: str) -> Path:
        canonical_root = self._root / "workspaces" / workspace_id / "files"
        if canonical_root.exists() and canonical_root.is_dir():
            return canonical_root
        legacy_root = self._root / workspace_id
        return legacy_root

    def _resolve_launch_cwd(self, session: WorkerSession) -> Path:
        relative = (session.launch_cwd or ".").strip().replace("\\", "/")
        if relative in {"", "."}:
            return session.workspace_root
        normalized = Path(relative)
        if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
            return session.workspace_root
        return session.workspace_root / normalized

    def _session_response(self, session: WorkerSession) -> WorkerSessionResponse:
        return WorkerSessionResponse(
            worker_session_id=session.id,
            workspace_id=session.workspace_id,
            state=session.state,
            preview_internal_url=f"{self._base_url}/worker/sessions/{session.id}/preview",
            launch_framework=session.launch_framework,
            launch_command=(
                " ".join(session.devserver_command)
                if session.devserver_command
                else None
            ),
            launch_cwd=session.launch_cwd,
            launch_port=session.devserver_port,
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

    def _pick_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _read_runtime_entrypoint_config(self, workspace_root: Path) -> dict[str, str]:
        config_path = workspace_root / self._runtime_config_file
        if not config_path.exists() or not config_path.is_file():
            return {}
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        command = str(raw.get("command") or "").strip()
        cwd = str(raw.get("cwd") or "").strip().replace("\\", "/")
        framework = str(raw.get("framework") or "").strip().lower()
        return {
            "command": command,
            "cwd": cwd,
            "framework": framework,
        }

    def _resolve_python_launch(
        self,
        workspace_root: Path,
        port: int,
    ) -> tuple[list[str] | None, str | None, str | None]:
        if (workspace_root / "manage.py").exists():
            return (
                ["python3", "manage.py", "runserver", f"0.0.0.0:{port}"],
                "django",
                ".",
            )

        for filename in ("main.py", "app.py"):
            target = workspace_root / filename
            if not target.exists() or not target.is_file():
                continue
            try:
                content = target.read_text(encoding="utf-8")
            except Exception:
                content = ""
            if "FastAPI" in content:
                return (
                    [
                        "python3",
                        "-m",
                        "uvicorn",
                        f"{filename[:-3]}:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(port),
                    ],
                    "fastapi",
                    ".",
                )
            if "Flask" in content:
                return (
                    [
                        "python3",
                        "-m",
                        "flask",
                        "--app",
                        filename,
                        "run",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(port),
                    ],
                    "flask",
                    ".",
                )

        return (None, None, None)

    def _resolve_devserver_command(
        self,
        workspace_root: Path,
        port: int,
    ) -> tuple[list[str] | None, str | None, str | None, str | None]:
        config = self._read_runtime_entrypoint_config(workspace_root)
        config_command = config.get("command", "")
        if config_command:
            config_cwd = config.get("cwd") or "."
            config_framework = config.get("framework") or "custom"
            return (
                [
                    "sh",
                    "-lc",
                    f"PORT={port} {config_command}",
                ],
                None,
                config_framework,
                config_cwd,
            )

        package_json = workspace_root / "package.json"
        if package_json.exists() and package_json.is_file():
            npm_path = shutil.which("npm")
            if npm_path:
                return (
                    [
                        npm_path,
                        "run",
                        "dev",
                        "--",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(port),
                    ],
                    None,
                    "node",
                    ".",
                )

        python_command, python_framework, python_cwd = self._resolve_python_launch(
            workspace_root,
            port,
        )
        if python_command:
            return (
                python_command,
                None,
                python_framework,
                python_cwd,
            )

        index_html = workspace_root / "index.html"
        if index_html.exists() and index_html.is_file():
            return (
                [
                    "python3",
                    "-m",
                    "http.server",
                    str(port),
                    "--bind",
                    "0.0.0.0",
                    "--directory",
                    str(workspace_root),
                ],
                None,
                "static",
                ".",
            )

        return (
            None,
            (
                "No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json "
                "with a command/cwd or provide package.json dev script, Python app.py/main.py, or index.html. "
                "If package.json exists, ensure npm is installed in the runtime environment."
            ),
            None,
            None,
        )

    async def _wait_devserver_ready(self, port: int) -> bool:
        deadline = (
            asyncio.get_event_loop().time() + self._devserver_start_timeout_seconds
        )
        probe_url = f"http://127.0.0.1:{port}/"
        timeout = httpx.Timeout(connect=0.5, read=1.0, write=1.0, pool=0.5)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    response = await client.get(probe_url)
                    if response.status_code < 500:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.25)
        return False

    async def _terminate_devserver_locked(self, session_id: str) -> None:
        process = self._devserver_processes.pop(session_id, None)
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except Exception:
                process.kill()
                await process.wait()

    async def _sync_devserver_state_locked(self, session: WorkerSession) -> None:
        process = self._devserver_processes.get(session.id)
        if process is None:
            session.devserver_running = False
            return
        if process.returncode is None:
            session.devserver_running = True
            return

        self._devserver_processes.pop(session.id, None)
        session.devserver_running = False
        if process.returncode != 0:
            session.last_error = f"Dev server exited with code {process.returncode}"
        session.updated_at = self._utc_now()

    async def _start_devserver_locked(self, session: WorkerSession) -> None:
        await self._sync_devserver_state_locked(session)
        if session.devserver_running:
            return

        port = session.devserver_port or self._pick_free_port()
        command, command_error, framework, launch_cwd = self._resolve_devserver_command(
            session.workspace_root, port
        )
        session.devserver_port = port
        session.launch_framework = framework
        session.launch_cwd = launch_cwd

        if not command:
            session.devserver_running = False
            session.last_error = command_error
            session.updated_at = self._utc_now()
            return

        await self._terminate_devserver_locked(session.id)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._resolve_launch_cwd(session)),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            session.devserver_running = False
            session.last_error = (
                "Dev server command not found: "
                f"{command[0]}. Install the required runtime dependency or "
                "set .ragtime/runtime-entrypoint.json command to an available executable."
            )
            session.updated_at = self._utc_now()
            return
        except Exception as exc:
            session.devserver_running = False
            session.last_error = f"Failed to launch dev server: {exc}"
            session.updated_at = self._utc_now()
            return
        self._devserver_processes[session.id] = process
        session.devserver_command = command

        ready = await self._wait_devserver_ready(port)
        if not ready:
            await self._terminate_devserver_locked(session.id)
            session.devserver_running = False
            session.last_error = "Dev server failed to become ready"
            session.updated_at = self._utc_now()
            return

        session.devserver_running = True
        session.last_error = None
        session.updated_at = self._utc_now()

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
                session.pty_access_token = request.pty_access_token
                await self._start_devserver_locked(session)
                session.updated_at = self._utc_now()
                return self._session_response(session)

            session_id = f"wkr-{request.workspace_id[:8]}-{os.urandom(4).hex()}"
            workspace_root = self._resolve_workspace_root(request.workspace_id)
            workspace_root.mkdir(parents=True, exist_ok=True)
            session = WorkerSession(
                id=session_id,
                workspace_id=request.workspace_id,
                provider_session_id=request.provider_session_id,
                workspace_root=workspace_root,
                pty_access_token=request.pty_access_token,
                state="running",
                devserver_running=False,
                devserver_port=None,
                devserver_command=None,
                launch_framework=None,
                launch_cwd=None,
                last_error=None,
                updated_at=self._utc_now(),
            )
            self._sessions[session_id] = session
            self._provider_to_session[request.provider_session_id] = session_id
            await self._start_devserver_locked(session)
            return self._session_response(session)

    async def get_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            await self._sync_devserver_state_locked(session)
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def stop_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            await self._terminate_devserver_locked(session.id)
            session.state = "stopped"
            session.devserver_running = False
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def restart_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            await self._terminate_devserver_locked(session.id)
            session.state = "running"
            await self._start_devserver_locked(session)
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

    async def preview_response(
        self,
        worker_session_id: str,
        path: str,
        query: str | None = None,
    ) -> tuple[bytes | str, str, int]:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")
            await self._sync_devserver_state_locked(session)
            if not session.devserver_running:
                await self._start_devserver_locked(session)

            if not session.devserver_running or not session.devserver_port:
                error_message = session.last_error or "Dev server is not running"
                raise HTTPException(status_code=502, detail=error_message)

            normalized = self._normalize_file_path(path) if path else ""
            upstream_base = f"http://127.0.0.1:{session.devserver_port}"
            upstream_url = (
                f"{upstream_base}/{normalized}" if normalized else f"{upstream_base}/"
            )
            if query:
                upstream_url = f"{upstream_url}?{query}"

        timeout = httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            try:
                upstream_response = await client.get(upstream_url)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Runtime dev server unavailable: {exc}",
                ) from exc

        media_type = (
            upstream_response.headers.get("content-type") or "application/octet-stream"
        )
        return upstream_response.content, media_type, upstream_response.status_code

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
            for session in self._sessions.values():
                await self._sync_devserver_state_locked(session)
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
