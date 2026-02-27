from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from runtime.manager.models import (
    RuntimeFileReadResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.shared import (
    RUNTIME_BOOTSTRAP_CONFIG_PATH,
    RUNTIME_BOOTSTRAP_STAMP_PATH,
    RuntimeSessionState,
    normalize_file_path,
)

_PORT_PATTERNS = (
    re.compile(r"(?:^|\s)--port(?:=|\s+)(\d{2,5})(?:\s|$)"),
    re.compile(r"(?:^|\s)-p\s+(\d{2,5})(?:\s|$)"),
    re.compile(r"(?:^|\s)PORT=(\d{2,5})(?:\s|$)"),
)
_PORT_REWRITE_PATTERNS = (
    (re.compile(r"(^|\s)--port=(\d{2,5})(?=\s|$)"), r"\1--port={port}"),
    (re.compile(r"(^|\s)--port\s+(\d{2,5})(?=\s|$)"), r"\1--port {port}"),
    (re.compile(r"(^|\s)-p\s+(\d{2,5})(?=\s|$)"), r"\1-p {port}"),
    (re.compile(r"(^|\s)PORT=(\d{2,5})(?=\s|$)"), r"\1PORT={port}"),
)
_PY_HTTP_SERVER_PORT_PATTERN = re.compile(
    r"(^|\s)(python3?\s+-m\s+http\.server\s+)(\d{2,5})(?=\s|$)"
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

_RUNTIME_DEVSERVER_LOG_DIR = "/tmp/ragtime-runtime-devserver"
_RUNTIME_DEVSERVER_LOG_TAIL_CHARS = 400
MAX_USERSPACE_SCREENSHOT_WIDTH = 1600
MAX_USERSPACE_SCREENSHOT_HEIGHT = 1200
MAX_USERSPACE_SCREENSHOT_PIXELS = 1_440_000
_SCREENSHOT_WAIT_AFTER_LOAD_FLOOR_MS = 900
_SCREENSHOT_WAIT_AFTER_LOAD_HMR_FLOOR_MS = 1800

_SCREENSHOT_JS_PATH = Path(__file__).with_name("screenshot.js")


def _load_screenshot_script() -> str:
    """Load the Playwright screenshot script from the adjacent JS file."""
    return _SCREENSHOT_JS_PATH.read_text(encoding="utf-8")


@dataclass
class DevserverResolution:
    """Result of resolving a devserver launch command for a workspace."""

    command: list[str] | None = None
    error: str | None = None
    framework: str | None = None
    cwd: str | None = None
    port: int | None = None


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
        self._devserver_log_paths: dict[str, Path] = {}
        self._devserver_log_handles: dict[str, Any] = {}
        self._bootstrap_retry_flags: dict[str, bool] = {}
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
        return normalize_file_path(file_path)

    def _resolve_workspace_root(self, workspace_id: str) -> Path:
        canonical_root = self._root / "workspaces" / workspace_id / "files"
        canonical_root.mkdir(parents=True, exist_ok=True)
        return canonical_root

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
        exists: bool,
    ) -> RuntimeFileReadResponse:
        return RuntimeFileReadResponse(
            path=rel_path,
            content=content,
            exists=exists,
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

    def _read_package_dev_script(self, workspace_root: Path) -> str:
        package_json = workspace_root / "package.json"
        if not package_json.exists() or not package_json.is_file():
            return ""
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        scripts = payload.get("scripts")
        if not isinstance(scripts, dict):
            return ""
        return str(scripts.get("dev") or "").strip()

    def _read_runtime_bootstrap_config(
        self, workspace_root: Path
    ) -> list[dict[str, str]]:
        config_path = workspace_root / RUNTIME_BOOTSTRAP_CONFIG_PATH
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
        config_path = workspace_root / RUNTIME_BOOTSTRAP_CONFIG_PATH
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
        stamp_path = workspace_root / RUNTIME_BOOTSTRAP_STAMP_PATH
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

            when_path = self._resolve_bootstrap_relative_path(
                workspace_root, when_exists
            )
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

    def _is_module_dashboard_workspace(self, workspace_root: Path) -> bool:
        return (workspace_root / "dashboard" / "main.ts").exists()

    def _command_uses_runtime_port(self, command: str) -> bool:
        if self._extract_explicit_port(command) is not None:
            return True
        return bool(re.search(r"\$\{?PORT\}?", command))

    def _resolve_devserver_log_path(self, session_id: str) -> Path:
        log_dir = Path(
            os.getenv("RUNTIME_DEVSERVER_LOG_DIR", _RUNTIME_DEVSERVER_LOG_DIR)
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{session_id}.log"

    def _read_devserver_log_tail(self, session_id: str) -> str:
        log_path = self._devserver_log_paths.get(session_id)
        if not log_path:
            return ""
        try:
            if not log_path.exists() or not log_path.is_file():
                return ""
            content = log_path.read_text(encoding="utf-8", errors="replace").strip()
            # PostgreSQL text columns reject null bytes (0x00)
            content = content.replace("\x00", "")
        except Exception:
            return ""
        if not content:
            return ""
        compact = " ".join(content.split())
        if len(compact) > _RUNTIME_DEVSERVER_LOG_TAIL_CHARS:
            return compact[-_RUNTIME_DEVSERVER_LOG_TAIL_CHARS:]
        return compact

    def _should_retry_bootstrap_after_exit(
        self,
        returncode: int,
        log_tail: str,
    ) -> bool:
        lowered = (log_tail or "").lower()
        if returncode == 127:
            return True
        return (
            "command not found" in lowered
            or "not found" in lowered
            or "cannot find module" in lowered
        )

    _ESBUILD_SERVEDIR_PATTERN: re.Pattern[str] = re.compile(r"--servedir=(\S+)")
    _ESBUILD_SERVE_PATTERN: re.Pattern[str] = re.compile(r"--serve=\S+")
    _ESBUILD_OUTFILE_PATTERN: re.Pattern[str] = re.compile(r"--outfile=(\S+)")
    _INDEX_HTML_SEARCH_DIRS: list[str] = ["public", "static", "src", "www"]

    def _fix_esbuild_servedir(self, command: str, workspace_root: Path) -> str:
        """If esbuild --servedir points to a dir without index.html, try to fix it.

        When the servedir changes, also move --outfile inside the new servedir so
        esbuild doesn't reject the command (output dir must be inside serve dir).
        """
        match = self._ESBUILD_SERVEDIR_PATTERN.search(command)
        if not match:
            return command
        servedir = match.group(1)
        servedir_path = workspace_root / servedir
        if (servedir_path / "index.html").exists():
            return command
        for candidate in self._INDEX_HTML_SEARCH_DIRS:
            candidate_path = workspace_root / candidate
            if (candidate_path / "index.html").exists():
                result = command.replace(
                    f"--servedir={servedir}", f"--servedir={candidate}"
                )
                # Relocate --outfile inside new servedir so esbuild accepts it.
                outfile_match = self._ESBUILD_OUTFILE_PATTERN.search(result)
                if outfile_match:
                    outfile = outfile_match.group(1)
                    # Only relocate if outfile is not already under the new servedir
                    if not outfile.startswith(f"{candidate}/"):
                        new_outfile = f"{candidate}/{outfile}"
                        result = result.replace(
                            f"--outfile={outfile}", f"--outfile={new_outfile}"
                        )
                return result
        return command

    def _inject_esbuild_serve_flag(self, command: str, port: int) -> str:
        """Ensure --serve=0.0.0.0:PORT is present, replacing any existing --serve flag."""
        serve_flag = f"--serve=0.0.0.0:{port}"
        if self._ESBUILD_SERVE_PATTERN.search(command):
            return self._ESBUILD_SERVE_PATTERN.sub(serve_flag, command)
        return f"{command} {serve_flag}"

    def _invalidate_bootstrap_stamp(self, session: WorkerSession) -> None:
        stamp_path = session.workspace_root / RUNTIME_BOOTSTRAP_STAMP_PATH
        try:
            if stamp_path.exists() and stamp_path.is_file():
                stamp_path.unlink()
        except Exception:
            pass

    def _rewrite_command_port(self, command: str, port: int) -> str:
        python_http_server_match = _PY_HTTP_SERVER_PORT_PATTERN.search(command)
        if python_http_server_match:
            prefix = python_http_server_match.group(1)
            command_prefix = python_http_server_match.group(2)
            replacement = f"{prefix}{command_prefix}{port}"
            return (
                f"{command[:python_http_server_match.start()]}"
                f"{replacement}"
                f"{command[python_http_server_match.end():]}"
            )

        rewritten = command
        for pattern, template in _PORT_REWRITE_PATTERNS:
            match = pattern.search(rewritten)
            if not match:
                continue
            replacement = template.format(port=port).replace("\\1", match.group(1))
            rewritten = (
                f"{rewritten[:match.start()]}{replacement}{rewritten[match.end():]}"
            )
            if replacement:
                return rewritten
        return rewritten

    def _resolve_devserver_command(
        self,
        workspace_root: Path,
        port: int,
    ) -> DevserverResolution:
        package_json = workspace_root / "package.json"
        module_dashboard = self._is_module_dashboard_workspace(workspace_root)
        config = self._read_runtime_entrypoint_config(workspace_root)
        config_command = config.get("command", "")
        if (
            config_command
            and module_dashboard
            and package_json.exists()
            and package_json.is_file()
            and not self._command_uses_runtime_port(config_command)
        ):
            config_command = ""

        if config_command:
            config_cwd = config.get("cwd") or "."
            config_framework = config.get("framework") or "custom"
            effective_port = port
            final_command = self._rewrite_command_port(config_command, effective_port)

            if self._command_uses_tool(config_command, "npx") and not shutil.which(
                "npx"
            ):
                return DevserverResolution(
                    error=self._missing_tool_error("npx"),
                    framework=config_framework,
                    cwd=config_cwd,
                    port=effective_port,
                )
            if self._command_uses_tool(config_command, "npm") and not shutil.which(
                "npm"
            ):
                return DevserverResolution(
                    error=self._missing_tool_error("npm"),
                    framework=config_framework,
                    cwd=config_cwd,
                    port=effective_port,
                )
            return DevserverResolution(
                command=[
                    "sh",
                    "-lc",
                    f"export PATH=./node_modules/.bin:$PATH && PORT={effective_port} {final_command}",
                ],
                framework=config_framework,
                cwd=config_cwd,
                port=effective_port,
            )

        if package_json.exists() and package_json.is_file():
            npm_path = shutil.which("npm")
            if npm_path:
                dev_script_raw = self._read_package_dev_script(workspace_root)
                dev_script = dev_script_raw.lower()
                if "esbuild" in dev_script:
                    # Build corrected esbuild command instead of blindly appending
                    # args, so we can fix --servedir when index.html is missing.
                    cmd_text = dev_script_raw
                    cmd_text = self._fix_esbuild_servedir(cmd_text, workspace_root)
                    cmd_text = self._inject_esbuild_serve_flag(cmd_text, port)
                    if "--watch" in dev_script and "--watch=forever" not in dev_script:
                        cmd_text = cmd_text.replace("--watch", "--watch=forever")
                    # npm adds node_modules/.bin to PATH when running scripts;
                    # replicate that here since we run the raw command directly.
                    return DevserverResolution(
                        command=[
                            "sh",
                            "-lc",
                            f"export PATH=./node_modules/.bin:$PATH && {cmd_text}",
                        ],
                        framework="node",
                        cwd=".",
                        port=port,
                    )
                return DevserverResolution(
                    command=[
                        npm_path,
                        "run",
                        "dev",
                        "--",
                        "--port",
                        str(port),
                    ],
                    framework="node",
                    cwd=".",
                    port=port,
                )
            if self._index_requires_typescript_runtime(workspace_root):
                return DevserverResolution(
                    error=(
                        "This workspace references TypeScript modules in index.html "
                        "but npm is unavailable in the isolated runtime container, so "
                        "no transpilation can occur. Add a runtime entrypoint that builds/serves JS, "
                        "or use JavaScript assets for static preview. "
                        f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                    ),
                )

        python_command, python_framework, python_cwd = self._resolve_python_launch(
            workspace_root,
            port,
        )
        if python_command:
            return DevserverResolution(
                command=python_command,
                framework=python_framework,
                cwd=python_cwd,
                port=port,
            )

        index_html = workspace_root / "index.html"
        if index_html.exists() and index_html.is_file():
            return DevserverResolution(
                command=[
                    "python3",
                    "-m",
                    "http.server",
                    str(port),
                    "--bind",
                    "0.0.0.0",
                    "--directory",
                    str(workspace_root),
                ],
                framework="static",
                cwd=".",
                port=port,
            )

        return DevserverResolution(
            error=(
                "No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json "
                "with a command/cwd or provide package.json dev script, Python app.py/main.py, or index.html. "
                "If package.json exists, ensure npm is installed in the runtime environment. "
                f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
            ),
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
        log_handle = self._devserver_log_handles.pop(session_id, None)
        if process is None:
            if log_handle:
                try:
                    log_handle.close()
                except Exception:
                    pass
            return
        if process.returncode is None:
            # Kill the entire process group so that grandchild processes
            # (e.g. esbuild spawned via sh -lc) are also terminated.
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except Exception:
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    process.kill()
                await process.wait()
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass

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
        log_handle = self._devserver_log_handles.pop(session.id, None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass
        if process.returncode != 0:
            log_tail = self._read_devserver_log_tail(session.id)
            if log_tail:
                session.last_error = (
                    f"Dev server exited with code {process.returncode}: {log_tail}"
                )
            else:
                session.last_error = f"Dev server exited with code {process.returncode}"

            if not self._bootstrap_retry_flags.get(
                session.id, False
            ) and self._should_retry_bootstrap_after_exit(
                process.returncode,
                log_tail,
            ):
                self._bootstrap_retry_flags[session.id] = True
                self._invalidate_bootstrap_stamp(session)
                session.last_error = (
                    f"{session.last_error} Retrying workspace bootstrap on next start."
                )
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
        resolution = self._resolve_devserver_command(session.workspace_root, port)
        session.launch_framework = resolution.framework
        session.launch_cwd = resolution.cwd

        if not resolution.command:
            session.devserver_port = resolution.port
            session.devserver_command = None
            session.devserver_running = False
            session.last_error = resolution.error
            session.updated_at = self._utc_now()
            return

        session.devserver_port = resolution.port or port

        await self._terminate_devserver_locked(session.id)
        log_path = self._resolve_devserver_log_path(session.id)
        try:
            log_handle = open(log_path, "wb", buffering=0)
        except Exception as exc:
            session.devserver_running = False
            session.last_error = f"Failed to initialize devserver log file: {exc}"
            session.updated_at = self._utc_now()
            return
        self._devserver_log_paths[session.id] = log_path
        self._devserver_log_handles[session.id] = log_handle
        try:
            process = await asyncio.create_subprocess_exec(
                *resolution.command,
                cwd=str(self._resolve_launch_cwd(session)),
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            try:
                log_handle.close()
            except Exception:
                pass
            self._devserver_log_handles.pop(session.id, None)
            session.devserver_running = False
            session.last_error = (
                "Dev server command not found: "
                f"{resolution.command[0]}. Install the required runtime dependency or "
                "set .ragtime/runtime-entrypoint.json command to an available executable. "
                f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
            )
            session.updated_at = self._utc_now()
            return
        except Exception as exc:
            try:
                log_handle.close()
            except Exception:
                pass
            self._devserver_log_handles.pop(session.id, None)
            session.devserver_running = False
            session.last_error = f"Failed to launch dev server: {exc}"
            session.updated_at = self._utc_now()
            return
        self._devserver_processes[session.id] = process
        session.devserver_command = resolution.command

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
        self._bootstrap_retry_flags.pop(session.id, None)
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
            session.last_error = None
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def restart_session(self, worker_session_id: str) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            await self._terminate_devserver_locked(session.id)
            # Pick a fresh port to avoid TIME_WAIT "address already in use"
            session.devserver_port = self._pick_free_port()
            session.last_error = None
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
                return self._runtime_file_response(session, rel_path, "", False)
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
            session.updated_at = self._utc_now()
            return self._runtime_file_response(session, rel_path, content, True)

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
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            session.updated_at = self._utc_now()
            return self._runtime_file_response(session, rel_path, content, True)

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

    async def capture_screenshot(
        self,
        worker_session_id: str,
        payload: RuntimeScreenshotRequest,
    ) -> RuntimeScreenshotResponse:
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
                raise HTTPException(
                    status_code=502,
                    detail=session.last_error or "Dev server is not running",
                )

            requested_width = max(320, int(payload.width))
            requested_height = max(240, int(payload.height))
            requested_wait_after_load_ms = max(0, int(payload.wait_after_load_ms))
            wait_after_load_floor_ms = (
                _SCREENSHOT_WAIT_AFTER_LOAD_HMR_FLOOR_MS
                if bool(payload.refresh_before_capture)
                else _SCREENSHOT_WAIT_AFTER_LOAD_FLOOR_MS
            )
            effective_wait_after_load_ms = max(
                requested_wait_after_load_ms,
                wait_after_load_floor_ms,
            )
            width = min(requested_width, MAX_USERSPACE_SCREENSHOT_WIDTH)
            height = min(requested_height, MAX_USERSPACE_SCREENSHOT_HEIGHT)
            requested_pixels = width * height
            if requested_pixels > MAX_USERSPACE_SCREENSHOT_PIXELS:
                scale = (MAX_USERSPACE_SCREENSHOT_PIXELS / requested_pixels) ** 0.5
                width = max(320, int(width * scale))
                height = max(240, int(height * scale))

            normalized_preview_path = (payload.path or "").strip().lstrip("/")
            if normalized_preview_path:
                normalized_preview_path = self._normalize_file_path(
                    normalized_preview_path
                )
            upstream_base = f"http://127.0.0.1:{session.devserver_port}"
            upstream_url = (
                f"{upstream_base}/{normalized_preview_path}"
                if normalized_preview_path
                else f"{upstream_base}/"
            )
            cache_busted_url = (
                f"{upstream_url}{'&' if '?' in upstream_url else '?'}"
                f"_ragtime_screenshot_ts={int(time.time() * 1000)}"
            )

            index_data_root = Path(os.getenv("INDEX_DATA_PATH", "/data"))
            output_dir = index_data_root / "_tmp" / session.workspace_id
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            if payload.filename and str(payload.filename).strip():
                candidate = (
                    str(payload.filename).strip().replace("\\", "/").split("/")[-1]
                )
            else:
                path_slug = (
                    normalized_preview_path.replace("/", "_").replace(" ", "_")
                    or "root"
                )
                candidate = f"preview_{path_slug}_{timestamp}.png"

            safe_candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)[:200]
            if not safe_candidate:
                safe_candidate = f"preview_{timestamp}.png"
            if not safe_candidate.lower().endswith(".png"):
                safe_candidate += ".png"

            output_path = output_dir / safe_candidate

        node_binary = shutil.which("node")
        if not node_binary:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Runtime screenshot capture requires Node.js in runtime container "
                    "but it is not installed."
                ),
            )

        # Ensure Node.js can resolve globally-installed packages (e.g. playwright)
        node_env = {**os.environ, "NODE_PATH": "/usr/local/lib/node_modules"}

        process = await asyncio.create_subprocess_exec(
            node_binary,
            "-e",
            _load_screenshot_script(),
            cache_busted_url,
            str(output_path),
            str(width),
            str(height),
            "true" if payload.full_page else "false",
            str(payload.timeout_ms),
            str(payload.wait_for_selector or "").strip(),
            str(effective_wait_after_load_ms),
            "true" if payload.refresh_before_capture else "false",
            str(MAX_USERSPACE_SCREENSHOT_PIXELS),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=node_env,
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Runtime screenshot capture failed. Ensure Playwright + Chromium "
                    f"are available in runtime container. {stderr_text or stdout_text or 'unknown error'}"
                ),
            )

        probe: dict[str, Any]
        try:
            probe = json.loads(stdout_text) if stdout_text else {}
        except Exception:
            probe = {}

        if not output_path.exists() or not output_path.is_file():
            raise HTTPException(
                status_code=502,
                detail=(
                    "Runtime screenshot capture reported success but no file was written"
                ),
            )

        return RuntimeScreenshotResponse(
            ok=True,
            workspace_id=session.workspace_id,
            preview_path=normalized_preview_path,
            screenshot_path=str(output_path),
            screenshot_size_bytes=int(output_path.stat().st_size),
            render={
                "requested_width": requested_width,
                "requested_height": requested_height,
                "width": width,
                "height": height,
                "full_page": bool(payload.full_page),
                "max_pixels": MAX_USERSPACE_SCREENSHOT_PIXELS,
                "wait_for_selector": str(payload.wait_for_selector or "").strip()
                or None,
                "wait_after_load_ms": requested_wait_after_load_ms,
                "effective_wait_after_load_ms": effective_wait_after_load_ms,
                "refresh_before_capture": bool(payload.refresh_before_capture),
            },
            probe=probe if isinstance(probe, dict) else {},
        )

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
