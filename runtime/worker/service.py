from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
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
    RuntimeExternalBrowseLink,
    RuntimeExternalBrowseRequest,
    RuntimeExternalBrowseResponse,
    RuntimeFileReadResponse,
    RuntimeScreenshotRequest,
    RuntimeScreenshotResponse,
    RuntimeWorkspaceFileInfo,
    RuntimeWorkspaceFileListResponse,
    RuntimeWorkspaceGitCommandResponse,
    RuntimeWorkspaceScmStatusResponse,
    WorkerHealthResponse,
    WorkerSessionResponse,
    WorkerStartSessionRequest,
)
from runtime.worker.sandbox import (
    SANDBOX_WORKSPACE_MOUNT,
    SandboxSpec,
    cleanup_sandbox,
    ensure_sandbox_ready,
    get_sandbox_spec,
    materialize_mounts,
    sandbox_diagnostics,
    spawn_sandboxed,
)

from ..core.shared import (
    RUNTIME_BOOTSTRAP_CONFIG_PATH,
    RUNTIME_BOOTSTRAP_STAMP_PATH,
    EntrypointStatus,
    RuntimeSessionState,
    normalize_file_path,
    parse_entrypoint_config,
)
from ..core.workspace_ops import (
    PLATFORM_MANAGED_GITIGNORE_PATTERNS,
    deduplicate_ancestor_paths,
    list_mount_source_tree_entries,
    list_workspace_tree_entries,
    sync_scope_relative_paths,
    workspace_mount_target_repo_relative_path,
    workspace_path_matches_mount_prefix,
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
    "bun": re.compile(r"(?:^|\s)bun(?:\s|$)"),
    "npx": re.compile(r"(?:^|\s)npx(?:\s|$)"),
    "npm": re.compile(r"(?:^|\s)npm(?:\s|$)"),
    "pipenv": re.compile(r"(?:^|\s)pipenv(?:\s|$)"),
    "pnpm": re.compile(r"(?:^|\s)pnpm(?:\s|$)"),
    "poetry": re.compile(r"(?:^|\s)poetry(?:\s|$)"),
    "uv": re.compile(r"(?:^|\s)uv(?:\s|$)"),
    "yarn": re.compile(r"(?:^|\s)yarn(?:\s|$)"),
}
_ENTRYPOINT_REQUIRED_TOOLS = (
    "bun",
    "npx",
    "npm",
    "pipenv",
    "pnpm",
    "poetry",
    "uv",
    "yarn",
)
_WORKSPACE_BOOTSTRAP_GUIDANCE = (
    "Initialize the workspace with required runtime dependencies "
    "(for example a package manager install step) or update "
    ".ragtime/runtime-entrypoint.json to use executables available in this runtime image."
)
_NPM_DEBUG_LOG_PATH_RE = re.compile(
    r"A complete log of this run can be found in:\s*(?P<path>/[^\s]+)"
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
_PLAYWRIGHT_BROKER_READY_TIMEOUT_SECONDS = 20.0
_PLAYWRIGHT_BROKER_STDERR_MAX_LINES = 40
_PLAYWRIGHT_BROKER_POOL_SIZE = 2
_OBJECT_STORAGE_READY_TIMEOUT_SECONDS = 10.0
_OBJECT_STORAGE_CONFIG_DIRNAME = "s3"
_OBJECT_STORAGE_CONFIG_NAME = "config.json"
_OBJECT_STORAGE_ENDPOINT_ENV_KEY = "RAGTIME_OBJECT_STORAGE_ENDPOINT"
_OBJECT_STORAGE_PORT_ENV_KEY = "RAGTIME_OBJECT_STORAGE_PORT"
_AGENT_SHELL_ROOT = Path("/tmp/.ragtime-agent-shell")
_AGENT_SHELL_BIN_DIR = _AGENT_SHELL_ROOT / "bin"
_AGENT_SHELL_ENV_METADATA_NAME = "workspace-env.json"
_AGENT_SHELL_ENV_VIEW_NAME = "redacted_env_view.py"
_AGENT_SHELL_INTERNAL_ENV_KEYS = ("RAGTIME_REDACTED_ENV_FILE",)
_RAGTIME_REDACTED_ENV_FILE_VAR = "RAGTIME_REDACTED_ENV_FILE"
_RAGTIME_REDACTED_ENV_SENTINEL_SET = "*****"
_RAGTIME_REDACTED_ENV_SENTINEL_MISSING = "__RAGTIME_SECRET_MISSING__"

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PLAYWRIGHT_BROKER_JS_PATH = _TEMPLATES_DIR / "playwright_broker.js"
_S3RVER_RUNNER_JS_PATH = _TEMPLATES_DIR / "s3rver_runner.js"
_REDACTED_ENV_VIEW_TEMPLATE_PATH = _TEMPLATES_DIR / "redacted_env_view.py"


@dataclass
class DevserverResolution:
    """Result of resolving a devserver launch command for a workspace."""

    command: list[str] | None = None
    error: str | None = None
    framework: str | None = None
    cwd: str | None = None
    port: int | None = None


@dataclass
class PlaywrightBrokerSlot:
    slot_id: int
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task[None] | None = None
    stderr_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=_PLAYWRIGHT_BROKER_STDERR_MAX_LINES)
    )


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
    workspace_env_visibility: dict[str, bool]
    workspace_mounts: list[dict[str, Any]]
    mount_targets_to_clear: set[str]
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
        self._object_storage_processes: dict[str, asyncio.subprocess.Process] = {}
        self._object_storage_env_overrides: dict[str, dict[str, str]] = {}
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
        self._mount_materialization_semaphore = asyncio.Semaphore(
            self._get_positive_int_env(
                "RUNTIME_MOUNT_MATERIALIZATION_CONCURRENCY",
                2,
            )
        )
        self._playwright_pool_size = self._get_positive_int_env(
            "RUNTIME_PLAYWRIGHT_BROKER_POOL_SIZE",
            _PLAYWRIGHT_BROKER_POOL_SIZE,
        )
        self._playwright_slots = [
            PlaywrightBrokerSlot(slot_id=index)
            for index in range(self._playwright_pool_size)
        ]
        self._playwright_available_slots: asyncio.Queue[int] = asyncio.Queue(
            maxsize=self._playwright_pool_size
        )
        for index in range(self._playwright_pool_size):
            self._playwright_available_slots.put_nowait(index)
        self._playwright_request_counter = 0

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
        return workspace_root, workspace_files, spec

    @staticmethod
    def _workspace_file_info(entry: Any) -> RuntimeWorkspaceFileInfo:
        return RuntimeWorkspaceFileInfo(
            path=str(entry.path),
            size_bytes=int(entry.size_bytes),
            updated_at=entry.updated_at,
            entry_type=str(entry.entry_type),
        )

    async def _run_git_in_workspace_raw(
        self,
        workspace_id: str,
        *,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        _, workspace_files_path, _ = self._resolve_workspace_root(workspace_id)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(workspace_files_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            return (
                process.returncode if process.returncode is not None else 1,
                stdout_bytes,
                stderr_bytes,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="Git binary not available in runtime worker",
            ) from exc

    async def list_workspace_files(
        self,
        workspace_id: str,
        *,
        include_dirs: bool = False,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> RuntimeWorkspaceFileListResponse:
        _, workspace_files_path, _ = self._resolve_workspace_root(workspace_id)
        mount_specs = list(workspace_mounts or [])
        active_workspace_path = await self._active_runtime_workspace_path(workspace_id)
        tree_root = active_workspace_path or workspace_files_path

        base_entries = await asyncio.to_thread(
            list_workspace_tree_entries,
            tree_root,
            include_dirs=include_dirs,
        )
        mount_prefixes = deduplicate_ancestor_paths(
            [
                repo_rel
                for spec in mount_specs
                if (
                    repo_rel := workspace_mount_target_repo_relative_path(
                        str(spec.get("target_path", "") or "")
                    )
                )
            ]
        )
        if mount_prefixes and active_workspace_path is None:
            base_entries = [
                entry
                for entry in base_entries
                if not any(
                    workspace_path_matches_mount_prefix(entry.path, prefix)
                    for prefix in mount_prefixes
                )
            ]

        mount_entries = await asyncio.to_thread(
            list_mount_source_tree_entries,
            mount_specs,
            include_dirs=include_dirs,
        )
        entries_by_path = {entry.path: entry for entry in base_entries}
        for entry in mount_entries:
            entries_by_path.setdefault(entry.path, entry)

        return RuntimeWorkspaceFileListResponse(
            files=[
                self._workspace_file_info(entry)
                for entry in sorted(
                    entries_by_path.values(), key=lambda item: item.path
                )
            ]
        )

    async def _active_runtime_workspace_path(self, workspace_id: str) -> Path | None:
        async with self._lock:
            active_sessions = [
                session
                for session in self._sessions.values()
                if session.workspace_id == workspace_id
                and session.state in {"running", "starting"}
            ]
        if not active_sessions:
            return None
        session = max(active_sessions, key=lambda item: item.updated_at)
        workspace_path = (
            session.sandbox_spec.rootfs_path
            / session.sandbox_spec.sandbox_workspace.lstrip("/")
        )
        return workspace_path if workspace_path.is_dir() else None

    async def run_workspace_git_command(
        self,
        workspace_id: str,
        *,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> RuntimeWorkspaceGitCommandResponse:
        returncode, stdout_bytes, stderr_bytes = await self._run_git_in_workspace_raw(
            workspace_id,
            args=args,
            env=env,
        )
        return RuntimeWorkspaceGitCommandResponse(
            returncode=returncode,
            stdout_b64=base64.b64encode(stdout_bytes).decode("ascii"),
            stderr_b64=base64.b64encode(stderr_bytes).decode("ascii"),
        )

    async def get_workspace_scm_status(
        self,
        workspace_id: str,
    ) -> RuntimeWorkspaceScmStatusResponse:
        _, workspace_files_path, _ = self._resolve_workspace_root(workspace_id)
        sync_scope_paths = await asyncio.to_thread(
            sync_scope_relative_paths,
            workspace_files_path,
            ignored_relative_paths=PLATFORM_MANAGED_GITIGNORE_PATTERNS,
        )
        commit_result = await self._run_git_in_workspace_raw(
            workspace_id,
            args=["rev-parse", "HEAD"],
        )
        status_result = await self._run_git_in_workspace_raw(
            workspace_id,
            args=["status", "--porcelain", "--untracked-files=all"],
        )
        current_commit_hash = (
            commit_result[1].decode("utf-8", errors="replace").strip()
            if commit_result[0] == 0
            else ""
        )
        return RuntimeWorkspaceScmStatusResponse(
            has_sync_scope_files=bool(sync_scope_paths),
            has_uncommitted_changes=bool(
                status_result[1].decode("utf-8", errors="replace").strip()
            ),
            current_commit_hash=current_commit_hash or None,
        )

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

    @staticmethod
    def _normalize_workspace_env(raw_env: dict[str, Any] | None) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in (raw_env or {}).items()
            if str(key).strip()
        }

    @staticmethod
    def _normalize_workspace_env_visibility(
        raw_visibility: dict[str, Any] | None,
        workspace_env: dict[str, str],
    ) -> dict[str, bool]:
        visibility: dict[str, bool] = {}
        for key, has_value in (raw_visibility or {}).items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            visibility[normalized_key] = bool(has_value)
        for key in workspace_env:
            visibility.setdefault(key, True)
        return dict(sorted(visibility.items()))

    @staticmethod
    def _agent_shell_host_root(spec: SandboxSpec) -> Path:
        return spec.rootfs_path / _AGENT_SHELL_ROOT.relative_to("/")

    @staticmethod
    def _agent_shell_host_bin_dir(spec: SandboxSpec) -> Path:
        return WorkerService._agent_shell_host_root(spec) / "bin"

    @staticmethod
    def _agent_shell_host_metadata_path(spec: SandboxSpec) -> Path:
        return (
            WorkerService._agent_shell_host_root(spec) / _AGENT_SHELL_ENV_METADATA_NAME
        )

    @staticmethod
    def _agent_shell_host_viewer_path(spec: SandboxSpec) -> Path:
        return WorkerService._agent_shell_host_root(spec) / _AGENT_SHELL_ENV_VIEW_NAME

    def _build_agent_shell_metadata(self, session: WorkerSession) -> dict[str, Any]:
        items: list[dict[str, str | bool]] = []
        for key, has_value in sorted(session.workspace_env_visibility.items()):
            items.append(
                {
                    "key": key,
                    "has_value": has_value,
                    "sentinel": (
                        _RAGTIME_REDACTED_ENV_SENTINEL_SET
                        if has_value
                        else _RAGTIME_REDACTED_ENV_SENTINEL_MISSING
                    ),
                }
            )
        return {"items": items}

    def _write_agent_shell_artifacts(self, session: WorkerSession) -> None:
        shell_root = self._agent_shell_host_root(session.sandbox_spec)
        bin_dir = self._agent_shell_host_bin_dir(session.sandbox_spec)
        shell_root.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self._agent_shell_host_metadata_path(session.sandbox_spec)
        metadata_path.write_text(
            json.dumps(self._build_agent_shell_metadata(session), indent=2),
            encoding="utf-8",
        )
        metadata_path.chmod(0o600)

        viewer_path = self._agent_shell_host_viewer_path(session.sandbox_spec)
        viewer_path.write_text(
            _REDACTED_ENV_VIEW_TEMPLATE_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        viewer_path.chmod(0o755)

        wrapper_target = _AGENT_SHELL_ROOT / _AGENT_SHELL_ENV_VIEW_NAME
        for wrapper_name in ("printenv", "env"):
            wrapper_path = bin_dir / wrapper_name
            wrapper_path.write_text(
                "#!/bin/sh\n" f'exec /usr/bin/python3 {wrapper_target} "$@"\n',
                encoding="utf-8",
            )
            wrapper_path.chmod(0o755)

    def build_agent_shell_environment(self, session: WorkerSession) -> dict[str, str]:
        self._write_agent_shell_artifacts(session)
        return {
            "PATH": (
                f"{_AGENT_SHELL_BIN_DIR}:"
                f"{os.getenv('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')}"
            ),
            _RAGTIME_REDACTED_ENV_FILE_VAR: str(
                _AGENT_SHELL_ROOT / _AGENT_SHELL_ENV_METADATA_NAME
            ),
        }

    def build_agent_process_environment(self, session: WorkerSession) -> dict[str, str]:
        environment = dict(session.workspace_env)
        environment.update(self._object_storage_env_overrides.get(session.id, {}))
        environment.update(self.build_agent_shell_environment(session))
        return environment

    @staticmethod
    def _workspace_secret_redaction_items(
        session: WorkerSession,
    ) -> list[tuple[str, str, str]]:
        items: list[tuple[str, str, str]] = []
        for key, value in sorted(session.workspace_env.items()):
            if not key or not value:
                continue
            items.append((key, value, _RAGTIME_REDACTED_ENV_SENTINEL_SET))
        return items

    @staticmethod
    def _redact_secret_key_value(
        text: str,
        key: str,
        value: str,
        sentinel: str,
    ) -> str:
        escaped_key = re.escape(key)
        escaped_value = re.escape(value)
        patterns = (
            re.compile(rf"(?m)(\bexport\s+{escaped_key}=){escaped_value}(?=$|\s)"),
            re.compile(rf"(?m)(\b{escaped_key}=){escaped_value}(?=$|\s)"),
            re.compile(rf'("{escaped_key}"\s*:\s*")({escaped_value})(")'),
            re.compile(rf"('{escaped_key}'\s*:\s*')({escaped_value})(')"),
            re.compile(rf'(\b{escaped_key}\b\s*:\s*")({escaped_value})(")'),
            re.compile(rf"(\b{escaped_key}\b\s*:\s*')({escaped_value})(')"),
        )
        redacted = text
        for pattern in patterns:
            redacted = pattern.sub(
                lambda match: (
                    f"{match.group(1)}{sentinel}{match.group(3)}"
                    if (match.lastindex or 0) >= 3
                    else f"{match.group(1)}{sentinel}"
                ),
                redacted,
            )
        return redacted

    def redact_workspace_secret_output(
        self,
        session: WorkerSession,
        text: str,
    ) -> str:
        if not text:
            return text
        redacted_text = text
        for key, secret_value, sentinel in self._workspace_secret_redaction_items(
            session
        ):
            redacted_text = self._redact_secret_key_value(
                redacted_text,
                key,
                secret_value,
                sentinel,
            )
        return redacted_text

    def split_workspace_secret_output(
        self,
        session: WorkerSession,
        text: str,
        carry: str = "",
    ) -> tuple[str, str]:
        combined = f"{carry}{text}" if carry else text
        if not combined:
            return "", ""

        overlap = 0
        for _, secret_value, _ in self._workspace_secret_redaction_items(session):
            max_prefix_length = min(len(secret_value) - 1, len(combined))
            for prefix_length in range(max_prefix_length, 0, -1):
                if combined.endswith(secret_value[:prefix_length]):
                    overlap = max(overlap, prefix_length)
                    break

        if overlap > 0:
            output_text = combined[:-overlap]
            next_carry = combined[-overlap:]
        else:
            output_text = combined
            next_carry = ""

        return self.redact_workspace_secret_output(session, output_text), next_carry

    @staticmethod
    def _workspace_object_storage_config_path(workspace_root: Path) -> Path:
        return (
            workspace_root
            / _OBJECT_STORAGE_CONFIG_DIRNAME
            / _OBJECT_STORAGE_CONFIG_NAME
        )

    @staticmethod
    def _workspace_screenshot_dir(workspace_root: Path) -> Path:
        return workspace_root / "runtime-artifacts" / "screenshots"

    def _read_workspace_object_storage_config(
        self,
        workspace_root: Path,
    ) -> dict[str, Any] | None:
        config_path = self._workspace_object_storage_config_path(workspace_root)
        if not config_path.exists() or not config_path.is_file():
            return None
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _extract_object_storage_buckets(
        payload: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        buckets = payload.get("buckets")
        if not isinstance(buckets, list):
            return []
        return [bucket for bucket in buckets if isinstance(bucket, dict)]

    async def _terminate_object_storage_locked(self, session_id: str) -> None:
        process = self._object_storage_processes.pop(session_id, None)
        self._object_storage_env_overrides.pop(session_id, None)
        if process is None:
            return
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.kill()
            except Exception:
                return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except Exception:
                pass

    async def _start_object_storage_locked(
        self,
        session: WorkerSession,
    ) -> str | None:
        await self._terminate_object_storage_locked(session.id)

        config = self._read_workspace_object_storage_config(session.workspace_root)
        buckets = self._extract_object_storage_buckets(config)
        if not buckets:
            self._object_storage_env_overrides[session.id] = {}
            return None

        node_binary = shutil.which("node")
        if not node_binary:
            return "Workspace object storage requires Node.js in the runtime container."

        data_dir = session.workspace_root / _OBJECT_STORAGE_CONFIG_DIRNAME / "buckets"
        data_dir.mkdir(parents=True, exist_ok=True)
        bucket_names = [
            str(bucket.get("name") or "").strip()
            for bucket in buckets
            if str(bucket.get("name") or "").strip()
        ]
        if not bucket_names:
            self._object_storage_env_overrides[session.id] = {}
            return None

        port = self._pick_free_port()
        node_env = {**os.environ, "NODE_PATH": "/usr/local/lib/node_modules"}
        process = await asyncio.create_subprocess_exec(
            node_binary,
            str(_S3RVER_RUNNER_JS_PATH),
            "--directory",
            str(data_dir),
            "--port",
            str(port),
            "--buckets-json",
            json.dumps(bucket_names),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=node_env,
        )

        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            await self._terminate_object_storage_locked(session.id)
            return "Workspace object storage failed to start."

        try:
            ready_line = await asyncio.wait_for(
                stdout.readline(),
                timeout=_OBJECT_STORAGE_READY_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            try:
                process.kill()
            except Exception:
                pass
            stderr_tail = ""
            try:
                stderr_tail = (
                    (await asyncio.wait_for(stderr.read(), timeout=1))
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                stderr_tail = ""
            detail = "Workspace object storage timed out during startup."
            if stderr_tail:
                detail = f"{detail} {stderr_tail}"
            return f"{detail} ({exc})"

        if not ready_line:
            stderr_tail = ""
            try:
                stderr_tail = (
                    (await asyncio.wait_for(stderr.read(), timeout=1))
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                stderr_tail = ""
            detail = "Workspace object storage exited before becoming ready."
            if stderr_tail:
                detail = f"{detail} {stderr_tail}"
            return detail

        try:
            ready_payload = json.loads(ready_line.decode("utf-8", errors="replace"))
        except Exception:
            ready_payload = {}
        if ready_payload.get("type") != "ready":
            error = str(
                ready_payload.get("error") or "Invalid object storage handshake"
            )
            return f"Workspace object storage failed to start: {error}"

        endpoint = f"http://127.0.0.1:{port}"
        self._object_storage_processes[session.id] = process
        self._object_storage_env_overrides[session.id] = {
            _OBJECT_STORAGE_ENDPOINT_ENV_KEY: endpoint,
            _OBJECT_STORAGE_PORT_ENV_KEY: str(port),
        }
        return None

    @staticmethod
    def _mount_target_paths(mounts: list[dict[str, Any]]) -> set[str]:
        target_paths: set[str] = set()
        for mount in mounts:
            target = str(mount.get("target_path") or "").strip()
            if target:
                target_paths.add(target)
        return target_paths

    async def _materialize_workspace_mounts(self, session: WorkerSession) -> None:
        mounts = list(session.workspace_mounts or [])
        clear_targets = sorted(session.mount_targets_to_clear)
        if not mounts and not clear_targets:
            return
        async with self._mount_materialization_semaphore:
            await asyncio.to_thread(
                materialize_mounts,
                session.sandbox_spec,
                mounts,
                clear_targets=clear_targets,
            )

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

    def _read_runtime_bootstrap_config_sync(
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

    async def _read_runtime_bootstrap_config(
        self, workspace_root: Path
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self._read_runtime_bootstrap_config_sync,
            workspace_root,
        )

    def _runtime_bootstrap_config_digest_sync(self, workspace_root: Path) -> str | None:
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

    async def _runtime_bootstrap_config_digest(
        self, workspace_root: Path
    ) -> str | None:
        return await asyncio.to_thread(
            self._runtime_bootstrap_config_digest_sync,
            workspace_root,
        )

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

    def _bootstrap_expected_artifact_exists(
        self,
        *,
        session: WorkerSession,
        cwd_path: Path,
        cwd_value: str,
        artifact_relative_path: str,
    ) -> bool:
        normalized_artifact = (
            str(artifact_relative_path or "").strip().replace("\\", "/")
        )
        if not normalized_artifact:
            return False
        if (cwd_path / normalized_artifact).exists():
            return True

        rootfs_workspace = (
            session.sandbox_spec.rootfs_path / SANDBOX_WORKSPACE_MOUNT.lstrip("/")
        )
        rootfs_cwd = self._resolve_bootstrap_relative_path(rootfs_workspace, cwd_value)
        if rootfs_cwd is None:
            return False
        return (rootfs_cwd / normalized_artifact).exists()

    def _is_false_negative_npm_failure(
        self,
        *,
        session: WorkerSession,
        command_name: str,
        run: str,
        output: str,
        cwd_path: Path,
        cwd_value: str,
    ) -> bool:
        """Return True when npm exits non-zero after producing the expected artifact.

        Some runtime/container combinations print ``Exit handler never called``
        for successful npm operations. Treat that as a soft failure only when
        the command's expected artifact is now present in the workspace sandbox.
        """
        if "exit handler never called" not in (output or "").lower():
            return False

        cmd = (run or "").strip().lower()
        name = (command_name or "").strip().lower()
        expected_artifact: str | None = None

        if (
            name in {"npm_ci", "npm_install", "node_dependencies"}
            or cmd.startswith("npm ci")
            or cmd.startswith("npm install")
        ):
            expected_artifact = "node_modules"
        elif name == "node_tailwind_tooling" or (
            "npm install" in cmd and "tailwindcss" in cmd
        ):
            expected_artifact = "node_modules/.bin/tailwindcss"

        if not expected_artifact:
            return False

        return self._bootstrap_expected_artifact_exists(
            session=session,
            cwd_path=cwd_path,
            cwd_value=cwd_value,
            artifact_relative_path=expected_artifact,
        )

    @staticmethod
    def _extract_npm_debug_log_path(output: str) -> str | None:
        match = _NPM_DEBUG_LOG_PATH_RE.search(output or "")
        if not match:
            return None
        return str(match.group("path") or "").strip() or None

    def _read_sandbox_file_tail(
        self,
        *,
        sandbox_root: Path,
        absolute_path: str,
        max_chars: int = 4000,
    ) -> str:
        normalized = str(absolute_path or "").strip()
        if not normalized.startswith("/"):
            return ""
        try:
            host_path = (sandbox_root / normalized.lstrip("/")).resolve()
            resolved_root = sandbox_root.resolve()
        except Exception:
            return ""
        if host_path != resolved_root and resolved_root not in host_path.parents:
            return ""
        try:
            content = host_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return content[-max_chars:]

    def _extract_bootstrap_failure_detail(
        self,
        *,
        session: WorkerSession,
        command_name: str,
        run: str,
        output: str,
    ) -> str | None:
        cmd = (run or "").strip().lower()
        name = (command_name or "").strip().lower()
        is_npm_ci = name == "npm_ci" or cmd.startswith("npm ci")
        if not is_npm_ci:
            return None

        lowered_output = (output or "").lower()
        if "invalid comparator:" in lowered_output:
            marker_index = lowered_output.find("invalid comparator:")
            return output[marker_index:].splitlines()[0].strip()

        log_path = self._extract_npm_debug_log_path(output)
        if not log_path:
            return None

        log_tail = self._read_sandbox_file_tail(
            sandbox_root=session.sandbox_spec.rootfs_path,
            absolute_path=log_path,
        )
        if not log_tail:
            return None

        for line in reversed(log_tail.splitlines()):
            normalized_line = line.strip()
            lowered_line = normalized_line.lower()
            if not normalized_line:
                continue
            if "invalid comparator:" in lowered_line:
                marker_index = lowered_line.find("invalid comparator:")
                return normalized_line[marker_index:].strip()
            if "loadvirtual typeerror:" in lowered_line:
                marker_index = lowered_line.find("loadvirtual typeerror:")
                return normalized_line[marker_index:].strip()
        return None

    async def _run_workspace_bootstrap_if_needed(
        self,
        session: WorkerSession,
    ) -> str | None:
        workspace_root = session.workspace_files_path
        stamp_path = workspace_root / RUNTIME_BOOTSTRAP_STAMP_PATH
        config_digest = await self._runtime_bootstrap_config_digest(workspace_root)
        existing_digest = ""
        if stamp_path.exists() and stamp_path.is_file():
            try:
                existing_digest = await asyncio.to_thread(
                    stamp_path.read_text,
                    encoding="utf-8",
                )
                existing_digest = existing_digest.strip()
            except Exception:
                existing_digest = ""
            if config_digest and existing_digest == config_digest:
                return None
            if not config_digest and existing_digest:
                return None

        commands = await self._read_runtime_bootstrap_config(workspace_root)
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
            # Also check the rootfs workspace dir — the sandbox may already
            # have the artifact (e.g. node_modules) from a prior session's
            # copytree even though the canonical ``files/`` tree lacks it.
            if unless_exists:
                rootfs_ws = (
                    session.sandbox_spec.rootfs_path
                    / SANDBOX_WORKSPACE_MOUNT.lstrip("/")
                )
                rootfs_unless = self._resolve_bootstrap_relative_path(
                    rootfs_ws, unless_exists
                )
                if rootfs_unless is not None and rootfs_unless.exists():
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
                    ensure_ready=False,
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
                detail = self._extract_bootstrap_failure_detail(
                    session=session,
                    command_name=command_name,
                    run=run,
                    output=output,
                )

                if self._is_false_negative_npm_failure(
                    session=session,
                    command_name=command_name,
                    run=run,
                    output=output,
                    cwd_path=cwd_path,
                    cwd_value=cwd_value,
                ):
                    # npm occasionally exits non-zero with "Exit handler never called"
                    # even after bootstrap installs the requested artifact.
                    continue

                return (
                    f"Runtime bootstrap command '{command_name}' failed with code {returncode}: "
                    f"{(detail or output)[:300]}. {_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                )

        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_value = config_digest or self._utc_now().isoformat()
        await asyncio.to_thread(
            stamp_path.write_text,
            stamp_value,
            encoding="utf-8",
        )
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
                ensure_ready=False,
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

    def _playwright_broker_error_tail(self, slot: PlaywrightBrokerSlot) -> str:
        compact = " ".join(
            line.replace("\x00", "").strip()
            for line in slot.stderr_lines
            if line and line.strip()
        )
        compact = " ".join(compact.split())
        if len(compact) > _RUNTIME_DEVSERVER_LOG_TAIL_CHARS:
            return compact[-_RUNTIME_DEVSERVER_LOG_TAIL_CHARS:]
        return compact

    async def _drain_playwright_stderr(
        self,
        slot: PlaywrightBrokerSlot,
        stream: asyncio.StreamReader | None,
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    slot.stderr_lines.append(text)
        except Exception:
            pass

    async def _terminate_playwright_broker_slot(
        self,
        slot: PlaywrightBrokerSlot,
    ) -> None:
        process = slot.process
        stderr_task = slot.stderr_task
        slot.process = None
        slot.stderr_task = None

        if process is not None:
            stdin = process.stdin
            if stdin is not None:
                try:
                    stdin.close()
                except Exception:
                    pass
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except Exception:
                    process.kill()
                    try:
                        await process.wait()
                    except Exception:
                        pass

        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _terminate_playwright_brokers(self) -> None:
        for slot in self._playwright_slots:
            await self._terminate_playwright_broker_slot(slot)

    async def _ensure_playwright_broker_slot(
        self,
        slot: PlaywrightBrokerSlot,
    ) -> asyncio.subprocess.Process:
        process = slot.process
        if (
            process is not None
            and process.returncode is None
            and process.stdin is not None
            and process.stdout is not None
        ):
            return process

        if process is not None:
            await self._terminate_playwright_broker_slot(slot)

        node_binary = shutil.which("node")
        if not node_binary:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Runtime screenshot/content probe requires Node.js in runtime container "
                    "but it is not installed."
                ),
            )

        slot.stderr_lines.clear()
        node_env = {**os.environ, "NODE_PATH": "/usr/local/lib/node_modules"}
        process = await asyncio.create_subprocess_exec(
            node_binary,
            str(_PLAYWRIGHT_BROKER_JS_PATH),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=node_env,
        )
        slot.process = process
        slot.stderr_task = asyncio.create_task(
            self._drain_playwright_stderr(slot, process.stderr)
        )
        stdout = process.stdout
        if stdout is None:
            await self._terminate_playwright_broker_slot(slot)
            raise HTTPException(
                status_code=503,
                detail="Playwright broker stdout pipe was not created.",
            )

        try:
            ready_line = await asyncio.wait_for(
                stdout.readline(),
                timeout=_PLAYWRIGHT_BROKER_READY_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            await self._terminate_playwright_broker_slot(slot)
            stderr_tail = self._playwright_broker_error_tail(slot)
            detail = "Timed out starting Playwright broker in runtime container."
            if stderr_tail:
                detail = f"{detail} {stderr_tail}"
            raise HTTPException(status_code=503, detail=detail) from exc

        if not ready_line:
            await self._terminate_playwright_broker_slot(slot)
            stderr_tail = self._playwright_broker_error_tail(slot)
            detail = "Playwright broker exited before becoming ready."
            if stderr_tail:
                detail = f"{detail} {stderr_tail}"
            raise HTTPException(status_code=503, detail=detail)

        try:
            ready_payload = json.loads(ready_line.decode("utf-8", errors="replace"))
        except Exception:
            ready_payload = {}

        if ready_payload.get("type") != "ready":
            await self._terminate_playwright_broker_slot(slot)
            detail = str(
                ready_payload.get("error") or "Invalid Playwright broker handshake"
            )
            stderr_tail = self._playwright_broker_error_tail(slot)
            if stderr_tail:
                detail = f"{detail}. {stderr_tail}"
            raise HTTPException(status_code=503, detail=detail)

        return process

    async def _invoke_playwright_broker(
        self,
        request: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> dict[str, Any]:
        slot_index = await self._playwright_available_slots.get()
        slot = self._playwright_slots[slot_index]
        try:
            for attempt in range(2):
                try:
                    process = await self._ensure_playwright_broker_slot(slot)
                    if process.stdin is None or process.stdout is None:
                        raise RuntimeError("Playwright broker streams are unavailable")

                    self._playwright_request_counter += 1
                    request_id = f"pw-{slot.slot_id}-{self._playwright_request_counter}"
                    payload = {"id": request_id, **request}
                    process.stdin.write(
                        (json.dumps(payload, separators=(",", ":")) + "\n").encode(
                            "utf-8"
                        )
                    )
                    await process.stdin.drain()
                    response_line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=max(5.0, (int(timeout_ms) / 1000.0) + 5.0),
                    )

                    if not response_line:
                        raise RuntimeError(
                            "Playwright broker closed the response stream"
                        )

                    response = json.loads(
                        response_line.decode("utf-8", errors="replace")
                    )
                    if response.get("id") != request_id:
                        raise RuntimeError(
                            "Playwright broker returned a mismatched response"
                        )
                    if response.get("ok"):
                        result = response.get("result")
                        return result if isinstance(result, dict) else {}

                    code = str(response.get("code") or "")
                    detail = str(response.get("error") or "Playwright request failed")
                    stderr_tail = self._playwright_broker_error_tail(slot)
                    if stderr_tail:
                        detail = f"{detail}. {stderr_tail}"
                    status_code = (
                        503 if code in {"node_missing", "playwright_missing"} else 502
                    )
                    raise HTTPException(status_code=status_code, detail=detail)
                except HTTPException:
                    raise
                except Exception as exc:
                    await self._terminate_playwright_broker_slot(slot)
                    if attempt == 0:
                        continue
                    stderr_tail = self._playwright_broker_error_tail(slot)
                    detail = f"Playwright broker request failed: {exc}"
                    if stderr_tail:
                        detail = f"{detail}. {stderr_tail}"
                    raise HTTPException(status_code=502, detail=detail) from exc

            raise HTTPException(
                status_code=502, detail="Playwright broker request failed"
            )
        finally:
            try:
                self._playwright_available_slots.put_nowait(slot_index)
            except asyncio.QueueFull:
                pass

    @staticmethod
    def _probe_looks_like_startup_blank(probe: dict[str, Any]) -> bool:
        """Heuristic: detect transient "app still booting" blank captures.

        During Vite/devserver warm-up, probes can return HTTP 200 but with an
        almost-empty HTML shell and no title yet. Treat this as retryable
        startup noise rather than a definitive white-screen failure.
        """
        if not isinstance(probe, dict):
            return False
        try:
            status_code = int(probe.get("status_code") or 0)
        except Exception:
            status_code = 0
        try:
            html_length = int(probe.get("html_length") or 0)
        except Exception:
            html_length = 0
        title = str(probe.get("title") or "").strip()
        try:
            element_visible_count = int(probe.get("element_visible_count") or 0)
        except Exception:
            element_visible_count = 0
        console_errors = probe.get("console_errors")

        # Keep this intentionally conservative to avoid masking real regressions.
        return (
            status_code == 200
            and html_length <= 64
            and not title
            and element_visible_count == 0
            and not console_errors
        )

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

            for tool in _ENTRYPOINT_REQUIRED_TOOLS:
                if self._command_uses_tool(config_command, tool) and not shutil.which(
                    tool
                ):
                    return DevserverResolution(
                        error=self._missing_tool_error(tool),
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
        sleep_seconds = 0.1
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    response = await client.get(probe_url)
                    if response.status_code < 500:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(sleep_seconds)
                sleep_seconds = min(0.75, sleep_seconds * 1.5)
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
        await self._terminate_object_storage_locked(session.id)
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
            await self._terminate_object_storage_locked(session.id)
            session.state = "running"
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
                    self._set_operation_phase(session, "provisioning")
                    session.updated_at = self._utc_now()

                try:
                    await asyncio.to_thread(ensure_sandbox_ready, session.sandbox_spec)
                except Exception as exc:
                    await self._mark_operation_failed(
                        session_id,
                        operation_id,
                        f"Failed to prepare runtime sandbox: {exc}",
                    )
                    return

                try:
                    await self._materialize_workspace_mounts(session)
                except Exception as exc:
                    await self._mark_operation_failed(
                        session_id,
                        operation_id,
                        f"Failed to materialize workspace mounts: {exc}",
                    )
                    return

                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    session.mount_targets_to_clear = set()
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
                    object_storage_error = await self._start_object_storage_locked(
                        session
                    )
                    if object_storage_error:
                        session.state = "running"
                        session.devserver_running = False
                        session.last_error = object_storage_error
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return

                # --- Part 1: prepare for spawn (inside lock) ---
                # Resolve the command and set up the log file while holding the
                # lock. Do NOT call spawn_sandboxed here: it invokes
                # ensure_sandbox_ready() → provision_rootfs() →
                # shutil.copytree(), which can block for 10+ seconds on large
                # workspaces and starve every other coroutine waiting for
                # self._lock (including the 10 s manager timeout).
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
                        await self._terminate_object_storage_locked(session.id)
                        session.devserver_port = resolution.port
                        session.devserver_command = None
                        error = resolution.error or "Invalid runtime entrypoint"
                        session.state = "running"
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
                        session.state = "running"
                        session.devserver_running = False
                        session.last_error = (
                            f"Failed to initialize devserver log file: {exc}"
                        )
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return

                    self._devserver_log_paths[session.id] = log_path
                    self._devserver_log_handles[session.id] = log_handle
                    # Capture everything spawn_sandboxed needs so we can
                    # release the lock before the slow sandbox provisioning.
                    _sandbox_spec = session.sandbox_spec
                    _spawn_command = list(resolution.command)
                    _launch_cwd = self._resolve_launch_cwd(session)
                    _workspace_env = {
                        **session.workspace_env,
                        **self._object_storage_env_overrides.get(session.id, {}),
                    }

                # --- Part 2: spawn sandbox OUTSIDE the lock ---
                # ensure_sandbox_ready / shutil.copytree runs here without
                # holding self._lock.
                _process: asyncio.subprocess.Process | None = None
                _spawn_error: str | None = None
                try:
                    _process = await spawn_sandboxed(
                        _sandbox_spec,
                        _spawn_command,
                        cwd=_launch_cwd,
                        env=_workspace_env,
                        stdout=log_handle,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                        ensure_ready=False,
                    )
                except FileNotFoundError:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    _spawn_error = (
                        "Dev server command not found: "
                        f"{_spawn_command[0]}. Install the required runtime dependency or "
                        "set .ragtime/runtime-entrypoint.json command to an available executable. "
                        f"{_WORKSPACE_BOOTSTRAP_GUIDANCE}"
                    )
                except Exception as exc:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    _spawn_error = f"Failed to launch dev server: {exc}"

                # --- Part 3: commit spawn result (inside lock) ---
                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        # Session was invalidated while we were spawning.
                        if _process is not None:
                            try:
                                _process.kill()
                            except Exception:
                                pass
                        self._devserver_log_handles.pop(session_id, None)
                        return
                    if _spawn_error or _process is None:
                        await self._terminate_object_storage_locked(session.id)
                        self._devserver_log_handles.pop(session.id, None)
                        session.state = "running"
                        session.devserver_running = False
                        session.last_error = (
                            _spawn_error or "Failed to launch dev server"
                        )
                        self._set_operation_phase(session, "failed")
                        session.updated_at = self._utc_now()
                        return
                    self._devserver_processes[session.id] = _process
                    session.devserver_command = _spawn_command
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
                        await self._terminate_object_storage_locked(session.id)
                        session.state = "running"
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
                session.workspace_env = self._normalize_workspace_env(
                    request.workspace_env
                )
                session.workspace_env_visibility = (
                    self._normalize_workspace_env_visibility(
                        request.workspace_env_visibility,
                        session.workspace_env,
                    )
                )
                previous_targets = self._mount_target_paths(session.workspace_mounts)
                session.workspace_mounts = list(request.workspace_mounts or [])
                session.mount_targets_to_clear = (
                    previous_targets
                    | self._mount_target_paths(session.workspace_mounts)
                )
                self._schedule_startup_locked(session)
                session.updated_at = self._utc_now()
                return self._session_response(session)

            session_id = f"wkr-{request.workspace_id[:8]}-{os.urandom(4).hex()}"
            workspace_root, workspace_files, sandbox_spec = (
                self._resolve_workspace_root(request.workspace_id)
            )
            workspace_env = self._normalize_workspace_env(request.workspace_env)
            session = WorkerSession(
                id=session_id,
                workspace_id=request.workspace_id,
                provider_session_id=request.provider_session_id,
                workspace_root=workspace_root,
                workspace_files_path=workspace_files,
                sandbox_spec=sandbox_spec,
                pty_access_token=request.pty_access_token,
                workspace_env=workspace_env,
                workspace_env_visibility=self._normalize_workspace_env_visibility(
                    request.workspace_env_visibility,
                    workspace_env,
                ),
                workspace_mounts=list(request.workspace_mounts or []),
                mount_targets_to_clear=self._mount_target_paths(
                    list(request.workspace_mounts or [])
                ),
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
            await self._terminate_object_storage_locked(session.id)
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
        workspace_env_visibility: dict[str, bool] | None = None,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if workspace_env is not None:
                session.workspace_env = self._normalize_workspace_env(workspace_env)
            if workspace_env is not None or workspace_env_visibility is not None:
                session.workspace_env_visibility = (
                    self._normalize_workspace_env_visibility(
                        workspace_env_visibility,
                        session.workspace_env,
                    )
                )
            await self._terminate_object_storage_locked(session.id)
            previous_targets = self._mount_target_paths(session.workspace_mounts)
            if workspace_mounts is not None:
                session.workspace_mounts = list(workspace_mounts)
            session.mount_targets_to_clear = (
                previous_targets | self._mount_target_paths(session.workspace_mounts)
            )
            # Pick a fresh port to avoid TIME_WAIT "address already in use"
            session.devserver_port = self._pick_free_port()
            self._schedule_startup_locked(session)
            session.updated_at = self._utc_now()
            return self._session_response(session)

    async def refresh_mounts(
        self,
        worker_session_id: str,
        workspace_mounts: list[dict[str, Any]],
    ) -> WorkerSessionResponse:
        mount_specs = []
        target_paths: set[str] = set()
        for mount in workspace_mounts:
            target_path = str(mount.get("target_path") or "").strip()
            if not target_path:
                continue
            mount_specs.append(dict(mount))
            target_paths.add(target_path)
        if not mount_specs:
            raise HTTPException(
                status_code=400,
                detail="No workspace mounts were provided for refresh",
            )

        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(
                    status_code=409,
                    detail="Runtime session is not active",
                )
            workspace_id = session.workspace_id
            mounts_by_target: dict[str, dict[str, Any]] = {}
            for existing_mount in session.workspace_mounts:
                existing_target = str(existing_mount.get("target_path") or "").strip()
                if existing_target:
                    mounts_by_target[existing_target] = dict(existing_mount)
            for mount in mount_specs:
                mounts_by_target[str(mount.get("target_path") or "").strip()] = mount
            session.workspace_mounts = list(mounts_by_target.values())
            session.mount_targets_to_clear |= target_paths
            session.updated_at = self._utc_now()

        workspace_lock = self._workspace_startup_lock(workspace_id)
        async with workspace_lock:
            async with self._lock:
                session = self._sessions.get(worker_session_id)
                if not session:
                    raise HTTPException(
                        status_code=404, detail="Worker session not found"
                    )
            try:
                await self._materialize_workspace_mounts(session)
            except Exception as exc:
                error_message = f"Failed to refresh workspace mounts: {exc}"
                async with self._lock:
                    current = self._sessions.get(worker_session_id)
                    if current:
                        current.last_error = error_message
                        current.updated_at = self._utc_now()
                raise HTTPException(status_code=500, detail=error_message) from exc

            async with self._lock:
                current = self._sessions.get(worker_session_id)
                if not current:
                    raise HTTPException(
                        status_code=404, detail="Worker session not found"
                    )
                current.mount_targets_to_clear.difference_update(target_paths)
                current.last_error = None
                current.updated_at = self._utc_now()
                return self._session_response(current)

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
            agent_process_env = self.build_agent_process_environment(session)

        timeout_seconds = max(1, min(timeout_seconds, 120))
        timed_out = False
        truncated = False

        try:
            process = await spawn_sandboxed(
                sandbox_spec,
                ["sh", "-lc", command],
                cwd=sandbox_cwd,
                env=agent_process_env,
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

        stdout_text = self.redact_workspace_secret_output(
            session,
            stdout_bytes.decode("utf-8", errors="replace"),
        )
        stderr_text = self.redact_workspace_secret_output(
            session,
            stderr_bytes.decode("utf-8", errors="replace"),
        )

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

            output_dir = self._workspace_screenshot_dir(session.workspace_root)
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_candidate = f"{uuid.uuid4().hex}.png"

            output_path = output_dir / safe_candidate

        probe = await self._invoke_playwright_broker(
            {
                "type": "screenshot",
                "url": cache_busted_url,
                "output_path": str(output_path),
                "viewport_width": width,
                "viewport_height": height,
                "capture_full_page": bool(payload.full_page),
                "timeout_ms": int(payload.timeout_ms),
                "wait_for_selector": wait_selector,
                "capture_element": capture_element,
                "clip_padding_px": requested_clip_padding_px,
                "wait_after_load_ms": requested_wait_after_load_ms,
                "refresh_before_capture": bool(payload.refresh_before_capture),
                "max_pixels": MAX_USERSPACE_SCREENSHOT_PIXELS,
            },
            timeout_ms=int(payload.timeout_ms),
        )

        startup_retry_attempted = False
        startup_blank_detected = False
        if self._probe_looks_like_startup_blank(probe):
            startup_blank_detected = True
            startup_retry_attempted = True

            # One retry with a fresh cache-busted URL and a slightly longer
            # settle wait helps absorb Vite dependency pre-bundling churn.
            retry_wait_after_load_ms = min(
                max(requested_wait_after_load_ms + 2000, 1500),
                12000,
            )
            retry_url = (
                f"{upstream_url}{'&' if '?' in upstream_url else '?'}"
                f"_ragtime_screenshot_ts={int(time.time() * 1000)}"
            )
            retry_probe = await self._invoke_playwright_broker(
                {
                    "type": "screenshot",
                    "url": retry_url,
                    "output_path": str(output_path),
                    "viewport_width": width,
                    "viewport_height": height,
                    "capture_full_page": bool(payload.full_page),
                    "timeout_ms": int(payload.timeout_ms),
                    "wait_for_selector": wait_selector,
                    "capture_element": capture_element,
                    "clip_padding_px": requested_clip_padding_px,
                    "wait_after_load_ms": retry_wait_after_load_ms,
                    "refresh_before_capture": bool(payload.refresh_before_capture),
                    "max_pixels": MAX_USERSPACE_SCREENSHOT_PIXELS,
                },
                timeout_ms=int(payload.timeout_ms),
            )
            if isinstance(retry_probe, dict):
                probe = retry_probe

        if isinstance(probe, dict):
            probe["startup_blank_detected"] = startup_blank_detected
            probe["startup_retry_attempted"] = startup_retry_attempted

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

        probe = await self._invoke_playwright_broker(
            {
                "type": "content_probe",
                "url": upstream_url,
                "timeout_ms": int(payload.timeout_ms),
                "wait_after_load_ms": int(payload.wait_after_load_ms),
                "inject_mock_context": bool(
                    getattr(payload, "inject_mock_context", False)
                ),
            },
            timeout_ms=int(payload.timeout_ms),
        )

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

    def _build_external_browse_response(
        self,
        probe: dict[str, Any] | None,
        payload: RuntimeExternalBrowseRequest,
    ) -> RuntimeExternalBrowseResponse:
        probe_data = probe if isinstance(probe, dict) else {}
        raw_links = probe.get("links") if isinstance(probe, dict) else None
        link_models: list[RuntimeExternalBrowseLink] = []
        if isinstance(raw_links, list):
            for entry in raw_links:
                if not isinstance(entry, dict):
                    continue
                url_value = str(entry.get("url") or "").strip()
                if not url_value:
                    continue
                text_value = str(entry.get("text") or "").strip()
                link_models.append(
                    RuntimeExternalBrowseLink(url=url_value, text=text_value)
                )

        console_errors_raw = (
            probe.get("console_errors") if isinstance(probe, dict) else None
        )
        console_errors: list[str] = []
        if isinstance(console_errors_raw, list):
            console_errors = [str(item) for item in console_errors_raw[:5]]

        return RuntimeExternalBrowseResponse(
            ok=bool(probe_data.get("ok", False)),
            requested_url=str(probe_data.get("requested_url") or payload.url),
            url=str(probe_data.get("url") or payload.url),
            status_code=probe_data.get("status_code"),
            title=str(probe_data.get("title") or ""),
            text=str(probe_data.get("text") or ""),
            text_length=int(probe_data.get("text_length") or 0),
            truncated=bool(probe_data.get("truncated", False)),
            links=link_models,
            console_errors=console_errors,
        )

    async def external_browse(
        self,
        payload: RuntimeExternalBrowseRequest,
        worker_session_id: str | None = None,
    ) -> RuntimeExternalBrowseResponse:
        """Drive the Playwright broker for an arbitrary external URL.

        Used by the chat diagnostics path; not bound to a worker session or
        workspace devserver. Callers in the control plane validate the URL
        against the chat-diagnostics network policy before invoking this.
        """
        if worker_session_id:
            async with self._lock:
                session = self._sessions.get(worker_session_id)
                if not session:
                    raise HTTPException(
                        status_code=404, detail="Worker session not found"
                    )
                if session.state not in {"running", "starting"}:
                    raise HTTPException(
                        status_code=409, detail="Worker session not active"
                    )
                session.updated_at = self._utc_now()

        probe = await self._invoke_playwright_broker(
            {
                "type": "external_browse",
                "url": payload.url,
                "timeout_ms": int(payload.timeout_ms),
                "wait_after_load_ms": int(payload.wait_after_load_ms),
                "extract_links": bool(payload.extract_links),
                "max_text_chars": int(payload.max_text_chars),
                "max_links": int(payload.max_links),
                "user_agent": str(payload.user_agent or ""),
            },
            timeout_ms=int(payload.timeout_ms),
        )
        return self._build_external_browse_response(probe, payload)

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
                    "provisioning",
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
            for sid in list(self._object_storage_processes.keys()):
                await self._terminate_object_storage_locked(sid)
        await self._terminate_playwright_brokers()

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
