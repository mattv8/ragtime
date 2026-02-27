from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import HTTPException

from runtime.manager.models import (RuntimeFileReadResponse,
                                    RuntimeSessionState, WorkerHealthResponse,
                                    WorkerSessionResponse,
                                    WorkerStartSessionRequest)

_PORT_PATTERNS = (
    re.compile(r"(?:^|\s)--port(?:=|\s+)(\d{2,5})(?:\s|$)"),
    re.compile(r"(?:^|\s)-p\s+(\d{2,5})(?:\s|$)"),
    re.compile(r"(?:^|\s)PORT=(\d{2,5})(?:\s|$)"),
)
_COMMAND_TOKEN_PATTERNS = {
    "npx": re.compile(r"(?:^|\s)npx(?:\s|$)"),
    "npm": re.compile(r"(?:^|\s)npm(?:\s|$)"),
}
_WORKSPACE_BOOTSTRAP_GUIDANCE = (
    "Initialize the workspace with required runtime dependencies "
    "(for example a package manager install step) or update "
    ".ragtime/runtime-entrypoint.json to use executables available in this runtime image."
)
_RUNTIME_BOOTSTRAP_CONFIG_PATH = ".ragtime/runtime-bootstrap.json"
_RUNTIME_BOOTSTRAP_STAMP_PATH = ".ragtime/.runtime-bootstrap.done"


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
            os.getenv("RUNTIME_DEVSERVER_START_TIMEOUT_SECONDS", "90")
        )
        self._runtime_bootstrap_timeout_seconds = int(
            os.getenv("RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS", "180")
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

    def _read_runtime_bootstrap_config(self, workspace_root: Path) -> list[dict[str, str]]:
        config_path = workspace_root / _RUNTIME_BOOTSTRAP_CONFIG_PATH
        if not config_path.exists() or not config_path.is_file():
            return []
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, dict):
            return []
        commands = raw.get("commands")
        if not isinstance(commands, list):
            return []
        normalized: list[dict[str, str]] = []
        for item in commands:
            if not isinstance(item, dict):
                continue
            run = str(item.get("run") or "").strip()
            if not run:
                continue
            normalized.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "run": run,
                    "when_exists": str(item.get("when_exists") or "").strip(),
                    "unless_exists": str(item.get("unless_exists") or "").strip(),
                    "cwd": str(item.get("cwd") or ".").strip(),
                }
            )
        return normalized

    def _runtime_bootstrap_config_digest(self, workspace_root: Path) -> str | None:
        config_path = workspace_root / _RUNTIME_BOOTSTRAP_CONFIG_PATH
        if not config_path.exists() or not config_path.is_file():
            return None
        try:
            payload = config_path.read_bytes()
        except Exception:
            return None
        if not payload:
            return None
        return hashlib.sha256(payload).hexdigest()

    def _resolve_bootstrap_relative_path(
        self,
        workspace_root: Path,
        relative_path: str,
    ) -> Path | None:
        normalized = (relative_path or "").strip().replace("\\", "/")
        if not normalized or normalized in {".", "./"}:
            return workspace_root
        candidate = Path(normalized)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            return None
        return workspace_root / candidate

    async def _run_workspace_bootstrap_if_needed(
        self,
        session: WorkerSession,
    ) -> str | None:
        workspace_root = session.workspace_root
        stamp_path = workspace_root / _RUNTIME_BOOTSTRAP_STAMP_PATH
        config_digest = self._runtime_bootstrap_config_digest(workspace_root)
        existing_digest = ""
        if stamp_path.exists() and stamp_path.is_file():
            try:
                existing_digest = stamp_path.read_text(encoding="utf-8").strip()
            except Exception:
                existing_digest = ""
            if config_digest and existing_digest == config_digest:
                return None
            if not config_digest and existing_digest:
                return None

        commands = self._read_runtime_bootstrap_config(workspace_root)
        if not commands:
            return None

        for command_cfg in commands:
            when_exists = command_cfg.get("when_exists", "")
            unless_exists = command_cfg.get("unless_exists", "")
            cwd_value = command_cfg.get("cwd", ".")
            command_name = command_cfg.get("name") or "bootstrap"
            run = command_cfg.get("run", "")

            when_path = self._resolve_bootstrap_relative_path(workspace_root, when_exists)
            unless_path = self._resolve_bootstrap_relative_path(
                workspace_root,
                unless_exists,
            )
            cwd_path = self._resolve_bootstrap_relative_path(workspace_root, cwd_value)

            if cwd_path is None:
                return (
                    "Runtime bootstrap config has invalid cwd path. "
                    "Use workspace-relative paths only."
                )
            if when_exists and (when_path is None or not when_path.exists()):
                continue
            if unless_exists and unless_path is not None and unless_path.exists():
                continue

            try:
                process = await asyncio.create_subprocess_exec(
                    "sh",
                    "-lc",
                    run,
                    cwd=str(cwd_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._runtime_bootstrap_timeout_seconds,
                )
            except asyncio.TimeoutError:
                return (
                    f"Runtime bootstrap command '{command_name}' timed out after "
                    f"{self._runtime_bootstrap_timeout_seconds}s."
                )
            except Exception as exc:
                return f"Runtime bootstrap command '{command_name}' failed to launch: {exc}"

            returncode = process.returncode or 0
            if returncode != 0:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
                output = stderr_text or stdout_text or "unknown error"
                return (
                    f"Runtime bootstrap command '{command_name}' failed with code {returncode}: "
                    f"{output[:300]}. {_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                )

        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_value = config_digest or self._utc_now().isoformat()
        stamp_path.write_text(stamp_value, encoding="utf-8")
        return None

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

    def _index_requires_typescript_runtime(self, workspace_root: Path) -> bool:
        index_html = workspace_root / "index.html"
        if not index_html.exists() or not index_html.is_file():
            return False
        try:
            content = index_html.read_text(encoding="utf-8")
        except Exception:
            return False
        script_src_pattern = re.compile(
            r"<script[^>]+src=[\"'][^\"']+\.(ts|tsx)(\?[^\"']*)?[\"']",
            re.IGNORECASE,
        )
        module_import_pattern = re.compile(
            r"import\s+[^;]*['\"][^'\"]+\.(ts|tsx)(\?[^'\"]*)?['\"]",
            re.IGNORECASE,
        )
        return bool(
            script_src_pattern.search(content) or module_import_pattern.search(content)
        )

    def _extract_explicit_port(self, command: str) -> int | None:
        for pattern in _PORT_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            try:
                candidate = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 1 <= candidate <= 65535:
                return candidate
        return None

    def _command_uses_tool(self, command: str, tool: str) -> bool:
        pattern = _COMMAND_TOKEN_PATTERNS.get(tool)
        if not pattern:
            return False
        return bool(pattern.search(command))

    def _missing_tool_error(self, tool: str) -> str:
        return (
            f"Runtime entrypoint uses '{tool}' but it is not installed in this isolated runtime container. "
            f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
        )

    def _resolve_devserver_command(
        self,
        workspace_root: Path,
        port: int,
    ) -> tuple[list[str] | None, str | None, str | None, str | None, int | None]:
        config = self._read_runtime_entrypoint_config(workspace_root)
        config_command = config.get("command", "")
        if config_command:
            config_cwd = config.get("cwd") or "."
            config_framework = config.get("framework") or "custom"
            configured_port = self._extract_explicit_port(config_command)
            effective_port = configured_port or port
            if self._command_uses_tool(config_command, "npx") and not shutil.which("npx"):
                return (
                    None,
                    self._missing_tool_error("npx"),
                    config_framework,
                    config_cwd,
                    effective_port,
                )
            if self._command_uses_tool(config_command, "npm") and not shutil.which("npm"):
                return (
                    None,
                    self._missing_tool_error("npm"),
                    config_framework,
                    config_cwd,
                    effective_port,
                )
            return (
                [
                    "sh",
                    "-lc",
                    f"PORT={effective_port} {config_command}",
                ],
                None,
                config_framework,
                config_cwd,
                effective_port,
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
                    port,
                )
            if self._index_requires_typescript_runtime(workspace_root):
                return (
                    None,
                    (
                        "This workspace references TypeScript modules in index.html "
                        "but npm is unavailable in the isolated runtime container, so "
                        "no transpilation can occur. Add a runtime entrypoint that builds/serves JS, "
                        "or use JavaScript assets for static preview. "
                        f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                    ),
                    None,
                    None,
                    None,
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
                port,
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
                port,
            )

        return (
            None,
            (
                "No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json "
                "with a command/cwd or provide package.json dev script, Python app.py/main.py, or index.html. "
                "If package.json exists, ensure npm is installed in the runtime environment. "
                f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
            ),
            None,
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

        bootstrap_error = await self._run_workspace_bootstrap_if_needed(session)
        if bootstrap_error:
            session.devserver_running = False
            session.last_error = bootstrap_error
            session.updated_at = self._utc_now()
            return

        port = session.devserver_port or self._pick_free_port()
        (
            command,
            command_error,
            framework,
            launch_cwd,
            resolved_port,
        ) = self._resolve_devserver_command(session.workspace_root, port)
        session.launch_framework = framework
        session.launch_cwd = launch_cwd

        if not command:
            session.devserver_port = resolved_port
            session.devserver_command = None
            session.devserver_running = False
            session.last_error = command_error
            session.updated_at = self._utc_now()
            return

        session.devserver_port = resolved_port or port

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
                "set .ragtime/runtime-entrypoint.json command to an available executable. "
                f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
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

        ready = await self._wait_devserver_ready(session.devserver_port)
        if not ready:
            await self._sync_devserver_state_locked(session)
            await self._terminate_devserver_locked(session.id)
            session.devserver_running = False
            if not session.last_error:
                session.last_error = (
                    "Dev server failed to become ready on "
                    f"port {session.devserver_port} within "
                    f"{self._devserver_start_timeout_seconds}s. "
                    "Ensure the runtime-entrypoint command serves HTTP on PORT."
                )
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
