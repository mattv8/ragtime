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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from runtime.manager.models import (
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecResponse,
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
    EntrypointStatus,
    RuntimeSessionState,
    normalize_file_path,
    parse_entrypoint_config,
)
from runtime.worker.sandbox import (
    SANDBOX_WORKSPACE_MOUNT,
    SandboxSpec,
    cleanup_sandbox,
    ensure_sandbox_ready,
    get_sandbox_spec,
    sandbox_diagnostics,
    spawn_sandboxed,
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
_COMMAND_TOKEN_PATTERNS = {
    "npx": re.compile(r"(?:^|\s)npx(?:\s|$)"),
    "npm": re.compile(r"(?:^|\s)npm(?:\s|$)"),
}
_WORKSPACE_BOOTSTRAP_GUIDANCE = (
    "Initialize the workspace with required runtime dependencies "
    "(for example a package manager install step) or update "
    ".ragtime/runtime-entrypoint.json to use executables available in this runtime image."
)

# Maps runtime-entrypoint framework names to pip packages that should be
# auto-installed before the devserver starts.  pip is invoked with
# ``--quiet`` and will no-op if the package is already present.
_FRAMEWORK_PIP_PACKAGES: dict[str, list[str]] = {
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi", "uvicorn"],
    "streamlit": ["streamlit"],
    "dash": ["dash"],
    "gradio": ["gradio"],
}

_RUNTIME_DEVSERVER_LOG_DIR = "/tmp/ragtime-runtime-devserver"
_RUNTIME_DEVSERVER_LOG_TAIL_CHARS = 400
MAX_USERSPACE_SCREENSHOT_WIDTH = 1600
MAX_USERSPACE_SCREENSHOT_HEIGHT = 1200
MAX_USERSPACE_SCREENSHOT_PIXELS = 1_440_000
_SCREENSHOT_WAIT_AFTER_LOAD_FLOOR_MS = 900
_SCREENSHOT_WAIT_AFTER_LOAD_HMR_FLOOR_MS = 1800

_SCREENSHOT_JS_PATH = Path(__file__).with_name("screenshot.js")
_CONTENT_PROBE_JS_PATH = Path(__file__).with_name("content_probe.js")


def _load_screenshot_script() -> str:
    """Load the Playwright screenshot script from the adjacent JS file."""
    return _SCREENSHOT_JS_PATH.read_text(encoding="utf-8")


def _load_content_probe_script() -> str:
    """Load the Playwright content probe script from the adjacent JS file."""
    return _CONTENT_PROBE_JS_PATH.read_text(encoding="utf-8")


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
    workspace_files_path: Path
    sandbox_spec: SandboxSpec
    pty_access_token: str
    workspace_env: dict[str, str]
    state: RuntimeSessionState
    devserver_running: bool
    devserver_port: int | None
    devserver_command: list[str] | None
    launch_framework: str | None
    launch_cwd: str | None
    last_error: str | None
    runtime_operation_id: str | None
    runtime_operation_phase: str | None
    runtime_operation_started_at: datetime | None
    runtime_operation_updated_at: datetime | None
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
        self._startup_tasks: dict[str, asyncio.Task[None]] = {}
        self._workspace_startup_locks: dict[str, asyncio.Lock] = {}
        self._startup_semaphore = asyncio.Semaphore(
            self._get_positive_int_env("RUNTIME_STARTUP_CONCURRENCY", 2)
        )

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

    def _normalize_file_path(
        self,
        file_path: str,
        *,
        enforce_sqlite_managed: bool = False,
    ) -> str:
        return normalize_file_path(
            file_path,
            enforce_sqlite_managed=enforce_sqlite_managed,
        )

    def _resolve_workspace_root(
        self, workspace_id: str
    ) -> tuple[Path, Path, SandboxSpec]:
        """Resolve workspace paths and build a SandboxSpec.

        Returns (workspace_root, workspace_files_path, sandbox_spec).
        workspace_root = .../workspaces/<id>
        workspace_files_path = .../workspaces/<id>/files
        """
        workspace_root = self._root / "workspaces" / workspace_id
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace_files = workspace_root / "files"
        workspace_files.mkdir(parents=True, exist_ok=True)
        spec = get_sandbox_spec(workspace_id, workspace_root, workspace_files)
        ensure_sandbox_ready(spec)
        return workspace_root, workspace_files, spec

    def _resolve_launch_cwd(self, session: WorkerSession) -> str:
        """Resolve the launch cwd as a sandbox-internal absolute path."""
        relative = (session.launch_cwd or ".").strip().replace("\\", "/")
        if relative in {"", "."}:
            return SANDBOX_WORKSPACE_MOUNT
        normalized = Path(relative)
        if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
            return SANDBOX_WORKSPACE_MOUNT
        return f"{SANDBOX_WORKSPACE_MOUNT}/{normalized}"

    def _resolve_host_launch_cwd(self, session: WorkerSession) -> Path:
        """Resolve the launch cwd as a host-side path (for file reads, etc.)."""
        relative = (session.launch_cwd or ".").strip().replace("\\", "/")
        if relative in {"", "."}:
            return session.workspace_files_path
        normalized = Path(relative)
        if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
            return session.workspace_files_path
        return session.workspace_files_path / normalized

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
            runtime_capabilities=sandbox_diagnostics(),
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            runtime_operation_id=session.runtime_operation_id,
            runtime_operation_phase=session.runtime_operation_phase,
            runtime_operation_started_at=session.runtime_operation_started_at,
            runtime_operation_updated_at=session.runtime_operation_updated_at,
            updated_at=session.updated_at,
        )

    def _workspace_startup_lock(self, workspace_id: str) -> asyncio.Lock:
        lock = self._workspace_startup_locks.get(workspace_id)
        if lock is None:
            lock = asyncio.Lock()
            self._workspace_startup_locks[workspace_id] = lock
        return lock

    def _begin_operation(self, session: WorkerSession, phase: str) -> None:
        now = self._utc_now()
        session.runtime_operation_id = os.urandom(12).hex()
        session.runtime_operation_phase = phase
        session.runtime_operation_started_at = now
        session.runtime_operation_updated_at = now

    def _set_operation_phase(self, session: WorkerSession, phase: str) -> None:
        session.runtime_operation_phase = phase
        session.runtime_operation_updated_at = self._utc_now()

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
        """Read entrypoint config via the shared canonical parser.

        Returns the same dict shape as the old ad-hoc reader for backward
        compatibility with callers that expect ``{command, cwd, framework}``.
        """
        status = parse_entrypoint_config(workspace_root)
        if status.state == "missing":
            return {}
        return dict(status.raw) if status.raw else {}

    def _get_entrypoint_status(self, workspace_root: Path) -> EntrypointStatus:
        """Return canonical entrypoint status for a workspace."""
        return parse_entrypoint_config(workspace_root)

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

        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception:
            return hashlib.sha256(payload).hexdigest()
        if not isinstance(parsed, dict):
            return hashlib.sha256(payload).hexdigest()

        watch_paths = parsed.get("watch_paths")
        if not isinstance(watch_paths, list):
            return hashlib.sha256(payload).hexdigest()

        digest = hashlib.sha256(payload)
        for watch_item in watch_paths:
            relative = str(watch_item or "").strip().replace("\\", "/")
            if not relative:
                continue
            resolved = self._resolve_bootstrap_relative_path(workspace_root, relative)
            if resolved is None:
                continue

            digest.update(relative.encode("utf-8", errors="ignore"))
            if not resolved.exists():
                digest.update(b"::missing")
                continue

            if resolved.is_file():
                digest.update(b"::file")
                digest.update(resolved.read_bytes())
                continue

            if resolved.is_dir():
                digest.update(b"::dir")
                for child in sorted(
                    path for path in resolved.rglob("*") if path.is_file()
                ):
                    rel_child = str(child.relative_to(workspace_root)).replace(
                        "\\", "/"
                    )
                    digest.update(rel_child.encode("utf-8", errors="ignore"))
                    digest.update(child.read_bytes())

        return digest.hexdigest()

    def _resolve_bootstrap_relative_path(
        self,
        workspace_files: Path,
        relative_path: str,
    ) -> Path | None:
        normalized = (relative_path or "").strip().replace("\\", "/")
        if not normalized or normalized in {".", "./"}:
            return workspace_files
        candidate = Path(normalized)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            return None
        return workspace_files / candidate

    async def _run_workspace_bootstrap_if_needed(
        self,
        session: WorkerSession,
    ) -> str | None:
        workspace_root = session.workspace_files_path
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
                # Resolve cwd relative to sandbox workspace mount
                sandbox_cwd = SANDBOX_WORKSPACE_MOUNT
                if cwd_value and cwd_value not in {".", "./"}:
                    sandbox_cwd = f"{SANDBOX_WORKSPACE_MOUNT}/{cwd_value}"
                # Embed an explicit ``cd`` so cwd is reliable even when
                # preexec_fn's chdir is lost after exec.
                process = await spawn_sandboxed(
                    session.sandbox_spec,
                    ["sh", "-lc", f"cd {sandbox_cwd} && {run}"],
                    cwd=sandbox_cwd,
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

    async def _ensure_entrypoint_dependencies(
        self,
        session: WorkerSession,
    ) -> str | None:
        """Auto-install pip packages required by the runtime entrypoint framework.

        Reads the ``framework`` field from ``.ragtime/runtime-entrypoint.json``
        and pip-installs missing packages listed in :data:`_FRAMEWORK_PIP_PACKAGES`.
        Returns an error string if installation fails, else ``None``.
        """
        config = self._read_runtime_entrypoint_config(session.workspace_files_path)
        framework = (config.get("framework") or "").strip().lower()
        packages = _FRAMEWORK_PIP_PACKAGES.get(framework)
        if not packages:
            return None

        pkg_list = " ".join(packages)
        try:
            process = await spawn_sandboxed(
                session.sandbox_spec,
                [
                    "sh",
                    "-lc",
                    f"cd {SANDBOX_WORKSPACE_MOUNT} && python3 -m pip install --quiet {pkg_list}",
                ],
                cwd=SANDBOX_WORKSPACE_MOUNT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._runtime_bootstrap_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return (
                f"Auto-install of {framework} dependencies timed out after "
                f"{self._runtime_bootstrap_timeout_seconds}s."
            )
        except Exception as exc:
            return f"Auto-install of {framework} dependencies failed to launch: {exc}"

        returncode = process.returncode or 0
        if returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            output = stderr_text or stdout_text or "unknown error"
            return (
                f"Auto-install of {framework} dependencies failed with code {returncode}: "
                f"{output[:300]}"
            )
        return None

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
            or "no module named" in lowered
            or "modulenotfounderror" in lowered
        )

    def _invalidate_bootstrap_stamp(self, session: WorkerSession) -> None:
        stamp_path = session.workspace_files_path / RUNTIME_BOOTSTRAP_STAMP_PATH
        try:
            if stamp_path.exists() and stamp_path.is_file():
                stamp_path.unlink()
        except Exception:
            pass

    def _rewrite_command_port(self, command: str, port: int) -> str:
        # Explicit $PORT / ${PORT} placeholder substitution (entrypoint-first path)
        if re.search(r"\$\{PORT\}|\$PORT(?![A-Za-z0-9_])", command):
            rewritten = re.sub(r"\$\{PORT\}", str(port), command)
            rewritten = re.sub(r"\$PORT(?![A-Za-z0-9_])", str(port), rewritten)
            return rewritten

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
        config = self._read_runtime_entrypoint_config(workspace_root)
        config_command = config.get("command", "")

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
            # Resolve the sandbox-internal cwd for the command.  We embed
            # an explicit ``cd`` because ``os.chdir`` in preexec_fn can be
            # lost after exec under certain uvicorn/event-loop contexts.
            _sandbox_cwd = SANDBOX_WORKSPACE_MOUNT
            if config_cwd and config_cwd not in {".", "./"}:
                _sandbox_cwd = f"{SANDBOX_WORKSPACE_MOUNT}/{config_cwd}"
            return DevserverResolution(
                command=[
                    "sh",
                    "-lc",
                    f"cd {_sandbox_cwd} && export PATH=./node_modules/.bin:$PATH PORT={effective_port} && {final_command}",
                ],
                framework=config_framework,
                cwd=config_cwd,
                port=effective_port,
            )

        return DevserverResolution(
            error=(
                "No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json "
                "with a command/cwd/framework. Runtime no longer falls back to package.json, Python entrypoints, or index.html. "
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
            if session.runtime_operation_phase in {
                "queued",
                "bootstrapping",
                "deps_install",
                "launching",
                "probing",
            }:
                session.devserver_running = False
            else:
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
            session.runtime_operation_phase = "failed"
            session.runtime_operation_updated_at = self._utc_now()
        session.updated_at = self._utc_now()

    async def _mark_operation_failed(
        self,
        session_id: str,
        operation_id: str,
        error: str,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.runtime_operation_id != operation_id:
                return
            session.state = "error"
            session.devserver_running = False
            session.last_error = error
            self._set_operation_phase(session, "failed")
            session.updated_at = self._utc_now()

    async def _run_startup_pipeline(
        self,
        session_id: str,
        operation_id: str,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.runtime_operation_id != operation_id:
                return
            workspace_id = session.workspace_id

        workspace_lock = self._workspace_startup_lock(workspace_id)
        async with workspace_lock:
            async with self._startup_semaphore:
                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    self._set_operation_phase(session, "bootstrapping")
                    session.updated_at = self._utc_now()

                bootstrap_error = await self._run_workspace_bootstrap_if_needed(session)
                if bootstrap_error:
                    await self._mark_operation_failed(
                        session_id,
                        operation_id,
                        bootstrap_error,
                    )
                    return

                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    self._set_operation_phase(session, "deps_install")
                    session.updated_at = self._utc_now()

                deps_error = await self._ensure_entrypoint_dependencies(session)
                if deps_error:
                    await self._mark_operation_failed(
                        session_id,
                        operation_id,
                        deps_error,
                    )
                    return

                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    self._set_operation_phase(session, "launching")
                    session.updated_at = self._utc_now()
                    port = session.devserver_port or self._pick_free_port()
                    resolution = self._resolve_devserver_command(
                        session.workspace_files_path, port
                    )
                    session.launch_framework = resolution.framework
                    session.launch_cwd = resolution.cwd
                    if not resolution.command:
                        session.devserver_port = resolution.port
                        session.devserver_command = None
                        error = resolution.error or "Invalid runtime entrypoint"
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = error
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return

                    session.devserver_port = resolution.port or port
                    await self._terminate_devserver_locked(session.id)
                    log_path = self._resolve_devserver_log_path(session.id)
                    try:
                        log_handle = open(log_path, "wb", buffering=0)
                    except Exception as exc:
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = (
                            f"Failed to initialize devserver log file: {exc}"
                        )
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return

                    self._devserver_log_paths[session.id] = log_path
                    self._devserver_log_handles[session.id] = log_handle
                    try:
                        process = await spawn_sandboxed(
                            session.sandbox_spec,
                            resolution.command,
                            cwd=self._resolve_launch_cwd(session),
                            env=session.workspace_env,
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
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = (
                            "Dev server command not found: "
                            f"{resolution.command[0]}. Install the required runtime dependency or "
                            "set .ragtime/runtime-entrypoint.json command to an available executable. "
                            f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                        )
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return
                    except Exception as exc:
                        try:
                            log_handle.close()
                        except Exception:
                            pass
                        self._devserver_log_handles.pop(session.id, None)
                        session.state = "error"
                        session.devserver_running = False
                        session.last_error = f"Failed to launch dev server: {exc}"
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return

                    self._devserver_processes[session.id] = process
                    session.devserver_command = resolution.command
                    self._set_operation_phase(session, "probing")
                    session.updated_at = self._utc_now()
                    target_port = session.devserver_port

                ready = await self._wait_devserver_ready(target_port or 0)
                if not ready:
                    async with self._lock:
                        session = self._sessions.get(session_id)
                        if not session or session.runtime_operation_id != operation_id:
                            return
                        await self._sync_devserver_state_locked(session)
                        await self._terminate_devserver_locked(session.id)
                        session.state = "error"
                        session.devserver_running = False
                        if not session.last_error:
                            session.last_error = (
                                "Dev server failed to become ready on "
                                f"port {session.devserver_port} within "
                                f"{self._devserver_start_timeout_seconds}s. "
                                "Ensure the runtime-entrypoint command serves HTTP on PORT."
                            )
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                    return

                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    session.state = "running"
                    session.devserver_running = True
                    session.last_error = None
                    self._bootstrap_retry_flags.pop(session.id, None)
                    self._set_operation_phase(session, "ready")
                    session.updated_at = self._utc_now()

    def _schedule_startup_locked(self, session: WorkerSession) -> None:
        existing = self._startup_tasks.get(session.id)
        if existing and not existing.done():
            existing.cancel()
        self._begin_operation(session, "queued")
        session.state = "starting"
        session.devserver_running = False
        session.last_error = None
        session.updated_at = self._utc_now()
        op_id = session.runtime_operation_id or ""
        task = asyncio.create_task(self._run_startup_pipeline(session.id, op_id))
        self._startup_tasks[session.id] = task

        def _cleanup(done_task: asyncio.Task[None]) -> None:
            current = self._startup_tasks.get(session.id)
            if current is done_task:
                self._startup_tasks.pop(session.id, None)

        task.add_done_callback(_cleanup)

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
                session.pty_access_token = request.pty_access_token
                session.workspace_env = {
                    str(key): str(value)
                    for key, value in (request.workspace_env or {}).items()
                    if str(key).strip()
                }
                self._schedule_startup_locked(session)
                session.updated_at = self._utc_now()
                return self._session_response(session)

            session_id = f"wkr-{request.workspace_id[:8]}-{os.urandom(4).hex()}"
            workspace_root, workspace_files, sandbox_spec = (
                self._resolve_workspace_root(request.workspace_id)
            )
            session = WorkerSession(
                id=session_id,
                workspace_id=request.workspace_id,
                provider_session_id=request.provider_session_id,
                workspace_root=workspace_root,
                workspace_files_path=workspace_files,
                sandbox_spec=sandbox_spec,
                pty_access_token=request.pty_access_token,
                workspace_env={
                    str(key): str(value)
                    for key, value in (request.workspace_env or {}).items()
                    if str(key).strip()
                },
                state="running",
                devserver_running=False,
                devserver_port=None,
                devserver_command=None,
                launch_framework=None,
                launch_cwd=None,
                last_error=None,
                runtime_operation_id=None,
                runtime_operation_phase=None,
                runtime_operation_started_at=None,
                runtime_operation_updated_at=None,
                updated_at=self._utc_now(),
            )
            self._sessions[session_id] = session
            self._provider_to_session[request.provider_session_id] = session_id
            self._schedule_startup_locked(session)
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
            startup_task = self._startup_tasks.pop(session.id, None)
            if startup_task and not startup_task.done():
                startup_task.cancel()
            await self._terminate_devserver_locked(session.id)
            cleanup_sandbox(session.sandbox_spec)
            session.state = "stopped"
            session.devserver_running = False
            session.last_error = None
            self._set_operation_phase(session, "stopped")
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def restart_session(
        self,
        worker_session_id: str,
        workspace_env: dict[str, str] | None = None,
    ) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if workspace_env is not None:
                session.workspace_env = {
                    str(key): str(value)
                    for key, value in workspace_env.items()
                    if str(key).strip()
                }
            # Pick a fresh port to avoid TIME_WAIT "address already in use"
            session.devserver_port = self._pick_free_port()
            self._schedule_startup_locked(session)
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
            rel_path = self._normalize_file_path(
                file_path,
                enforce_sqlite_managed=True,
            )
            target = session.workspace_files_path / rel_path
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
            rel_path = self._normalize_file_path(
                file_path,
                enforce_sqlite_managed=True,
            )
            target = session.workspace_files_path / rel_path
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
            rel_path = self._normalize_file_path(
                file_path,
                enforce_sqlite_managed=True,
            )
            target = session.workspace_files_path / rel_path
            if target.exists() and target.is_file():
                target.unlink()
            session.updated_at = self._utc_now()
            return {"success": True, "path": rel_path}

    _EXEC_MAX_OUTPUT_BYTES = 60_000

    async def exec_command(
        self,
        worker_session_id: str,
        command: str,
        timeout_seconds: int = 30,
        cwd: str | None = None,
    ) -> RuntimeExecResponse:
        """Execute a shell command in the workspace sandbox."""
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")

            # Resolve cwd as a sandbox-internal path
            if cwd:
                # Validate cwd doesn't escape workspace
                normalized_cwd = Path(cwd.replace("\\", "/"))
                if normalized_cwd.is_absolute() or any(
                    part == ".." for part in normalized_cwd.parts
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="cwd must be within the workspace root",
                    )
                sandbox_cwd = f"{SANDBOX_WORKSPACE_MOUNT}/{normalized_cwd}"
            else:
                sandbox_cwd = SANDBOX_WORKSPACE_MOUNT

            sandbox_spec = session.sandbox_spec

        timeout_seconds = max(1, min(timeout_seconds, 120))
        timed_out = False
        truncated = False

        try:
            process = await spawn_sandboxed(
                sandbox_spec,
                ["sh", "-lc", command],
                cwd=sandbox_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                process.kill()  # type: ignore[union-attr]
                await process.wait()  # type: ignore[union-attr]
            except Exception:
                pass
            stdout_bytes = b""
            stderr_bytes = f"Command timed out after {timeout_seconds}s".encode()
        except Exception as exc:
            return RuntimeExecResponse(
                exit_code=-1,
                stdout="",
                stderr=f"Failed to execute command: {exc}",
                timed_out=False,
                truncated=False,
            )

        exit_code = process.returncode if process.returncode is not None else -1

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        max_out = self._EXEC_MAX_OUTPUT_BYTES
        if len(stdout_text) > max_out or len(stderr_text) > max_out:
            truncated = True
            stdout_text = stdout_text[:max_out]
            stderr_text = stderr_text[:max_out]

        return RuntimeExecResponse(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            timed_out=timed_out,
            truncated=truncated,
        )

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
                self._schedule_startup_locked(session)
            if not session.devserver_running or not session.devserver_port:
                raise HTTPException(
                    status_code=503,
                    detail=session.last_error
                    or "Dev server is starting. Retry screenshot when runtime operation is ready.",
                )

            requested_width = max(320, int(payload.width))
            requested_height = max(240, int(payload.height))
            requested_wait_after_load_ms = max(0, int(payload.wait_after_load_ms))
            requested_clip_padding_px = min(max(0, int(payload.clip_padding_px)), 256)
            capture_element = bool(payload.capture_element)
            wait_selector = str(payload.wait_for_selector or "").strip()
            if capture_element and not wait_selector:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "capture_element=true requires wait_for_selector to target "
                        "a unique visible element"
                    ),
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

            safe_candidate = f"{uuid.uuid4().hex}.png"

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
            wait_selector,
            "true" if capture_element else "false",
            str(requested_clip_padding_px),
            str(requested_wait_after_load_ms),
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
                "wait_for_selector": wait_selector or None,
                "capture_element": capture_element,
                "clip_padding_px": requested_clip_padding_px,
                "wait_after_load_ms": requested_wait_after_load_ms,
                "effective_wait_after_load_ms": int(
                    (
                        probe.get("effective_wait_after_load_ms")
                        or requested_wait_after_load_ms
                    )
                ),
                "refresh_before_capture": bool(payload.refresh_before_capture),
            },
            probe=probe if isinstance(probe, dict) else {},
        )

    async def content_probe(
        self,
        worker_session_id: str,
        payload: RuntimeContentProbeRequest,
    ) -> RuntimeContentProbeResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")

            await self._sync_devserver_state_locked(session)
            if not session.devserver_running:
                self._schedule_startup_locked(session)
            if not session.devserver_running or not session.devserver_port:
                raise HTTPException(
                    status_code=503,
                    detail=session.last_error
                    or "Dev server is starting. Retry when runtime is ready.",
                )

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

        node_binary = shutil.which("node")
        if not node_binary:
            raise HTTPException(
                status_code=503,
                detail="Content probe requires Node.js but it is not installed.",
            )

        node_env = {**os.environ, "NODE_PATH": "/usr/local/lib/node_modules"}

        process = await asyncio.create_subprocess_exec(
            node_binary,
            "-e",
            _load_content_probe_script(),
            upstream_url,
            str(payload.timeout_ms),
            str(payload.wait_after_load_ms),
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
                    "Content probe failed. Ensure Playwright + Chromium "
                    f"are available. {stderr_text or stdout_text or 'unknown error'}"
                ),
            )

        probe: dict[str, Any]
        try:
            probe = json.loads(stdout_text) if stdout_text else {}
        except Exception:
            probe = {}

        return RuntimeContentProbeResponse(
            ok=probe.get("ok", False),
            workspace_id=session.workspace_id,
            preview_path=normalized_preview_path,
            status_code=probe.get("status_code"),
            body_text_length=probe.get("body_text_length", 0),
            body_text_preview=probe.get("body_text_preview", ""),
            body_html_length=probe.get("body_html_length", 0),
            title=probe.get("title", ""),
            has_error_indicator=probe.get("has_error_indicator", False),
            console_errors=probe.get("console_errors", []),
        )

    async def build_preview_upstream_url(
        self,
        worker_session_id: str,
        path: str,
        query: str | None = None,
    ) -> str:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")
            await self._sync_devserver_state_locked(session)
            if not session.devserver_running:
                self._schedule_startup_locked(session)

            if not session.devserver_running or not session.devserver_port:
                if session.runtime_operation_phase in {
                    "queued",
                    "bootstrapping",
                    "deps_install",
                    "launching",
                    "probing",
                }:
                    raise HTTPException(
                        status_code=503,
                        detail="Dev server is starting. Retry preview shortly.",
                    )
                error_message = session.last_error or "Dev server is not running"
                raise HTTPException(status_code=502, detail=error_message)

            normalized = self._normalize_file_path(path) if path else ""
            upstream_base = f"http://127.0.0.1:{session.devserver_port}"
            upstream_url = (
                f"{upstream_base}/{normalized}" if normalized else f"{upstream_base}/"
            )
            if query:
                upstream_url = f"{upstream_url}?{query}"
        return upstream_url

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

    async def shutdown(self) -> None:
        async with self._lock:
            for sid in list(self._startup_tasks.keys()):
                task = self._startup_tasks.pop(sid, None)
                if task and not task.done():
                    task.cancel()
            for sid in list(self._devserver_processes.keys()):
                await self._terminate_devserver_locked(sid)

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
                metadata={
                    "worker_name": self._worker_name,
                    **sandbox_diagnostics(),
                },
            )


@lru_cache(maxsize=1)
def get_worker_service() -> WorkerService:
    return WorkerService()
