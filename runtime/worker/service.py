from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import hashlib
import html
import json
import logging
import os
import re
import shutil
import socket
import stat
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from runtime.manager.models import (
    RuntimeBridgeCredentialMetadata,
    RuntimeContentProbeRequest,
    RuntimeContentProbeResponse,
    RuntimeExecResponse,
    RuntimeExternalBrowseLink,
    RuntimeExternalBrowseRequest,
    RuntimeExternalBrowseResponse,
    RuntimeFileReadResponse,
    RuntimeMcpToolCallRequest,
    RuntimeMcpToolCallResponse,
    RuntimeMcpToolInfo,
    RuntimeMcpToolListResponse,
    RuntimePdfReadMatch,
    RuntimePdfReadRequest,
    RuntimePdfReadResponse,
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
    archive_workspace_mirror,
    cleanup_sandbox,
    detect_capabilities,
    ensure_sandbox_ready,
    get_sandbox_spec,
    materialize_mounts,
    recommended_startup_concurrency,
    reconcile_stopped_workspace_mirror,
    sandbox_diagnostics,
    spawn_sandboxed,
    terminate_process_group,
    workspace_mirror_required,
)

from ..core.secure_files import SecureFileError
from ..core.secure_files import delete_file as secure_delete_file
from ..core.secure_files import read_text as secure_read_text
from ..core.secure_files import write_text as secure_write_text
from ..core.shared import (
    RUNTIME_BOOTSTRAP_CONFIG_PATH,
    RUNTIME_BOOTSTRAP_STAMP_PATH,
    RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS,
    EntrypointStatus,
    RuntimeSessionState,
    normalize_file_path,
    parse_entrypoint_config,
)
from ..core.utils import get_positive_int_env, utc_now
from ..core.workspace_ops import (
    PLATFORM_MANAGED_GITIGNORE_PATTERNS,
    deduplicate_ancestor_paths,
    list_mount_source_tree_entries,
    list_workspace_tree_entries,
    resolve_workspace_mount_rooted_target,
    resolve_workspace_mount_source_path,
    sync_scope_relative_paths,
    workspace_mount_target_repo_relative_path,
    workspace_path_matches_mount_prefix,
)

logger = logging.getLogger(__name__)

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
_NPM_DEBUG_LOG_PATH_RE = re.compile(r"A complete log of this run can be found in:\s*(?P<path>/[^\s]+)")
_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
    "text/x-pdf",
}
_PDF_READ_USER_AGENT = "RagtimeBot/1.0"
_PDF_READ_RETRY_STATUS_CODES = {403, 406, 429}

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
# When the devserver log overflows the tail budget the bare tail often drops
# the most actionable line (e.g. ``Cannot find package 'foo'``). Patterns
# matched here are hoisted in front of the tail so that signal survives.
_DEVSERVER_ERROR_SUMMARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"Error \[ERR_MODULE_NOT_FOUND\]: "
        r"(?P<detail>Cannot find (?:package|module) '[^']+' imported from \S+)"
    ),
    re.compile(r"(?P<detail>ModuleNotFoundError: No module named '[^']+')"),
)
MAX_USERSPACE_SCREENSHOT_WIDTH = 1600
MAX_USERSPACE_SCREENSHOT_HEIGHT = 1200
MAX_USERSPACE_SCREENSHOT_PIXELS = 1_440_000
_SCREENSHOT_WAIT_AFTER_LOAD_FLOOR_MS = 900
_SCREENSHOT_WAIT_AFTER_LOAD_HMR_FLOOR_MS = 1800
_AGENT_SHELL_ROOT = Path("/tmp/.ragtime-agent-shell")
_AGENT_SHELL_BIN_DIR = _AGENT_SHELL_ROOT / "bin"
_AGENT_SHELL_ENV_METADATA_NAME = "workspace-env.json"
_AGENT_SHELL_ENV_VIEW_NAME = "redacted_env_view.py"
_AGENT_SHELL_INTERNAL_ENV_KEYS = ("RAGTIME_REDACTED_ENV_FILE",)
_RAGTIME_REDACTED_ENV_FILE_VAR = "RAGTIME_REDACTED_ENV_FILE"
_RAGTIME_REDACTED_ENV_SENTINEL_SET = "*****"
_RAGTIME_REDACTED_ENV_SENTINEL_MISSING = "__RAGTIME_SECRET_MISSING__"
_MAX_BRIDGE_REFRESH_REQUEST_HISTORY = 32

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PLAYWRIGHT_BROKER_JS_PATH = _TEMPLATES_DIR / "playwright_broker.js"
_PLAYWRIGHT_MCP_SERVER_PATH = Path(__file__).parent / "mcp_playwright_server.py"
_PLAYWRIGHT_MCP_TOOL_NAMES = frozenset(
    {
        "playwright_content_probe",
        "playwright_capture_screenshot",
        "playwright_external_browse",
        "playwright_debug_steps",
    }
)
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
class McpServerSpec:
    """Static definition of a runtime MCP server the worker can launch.

    Built-in servers are declared in ``_BUILTIN_MCP_SERVERS``. Adding a new
    internal server (Pylance, MyPy, TypeScript, ...) is just adding a spec;
    no per-server worker plumbing is required.
    """

    name: str
    server_id: str
    display_name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # When None, any tool name is accepted (e.g. dynamically discovered servers).
    tool_names: frozenset[str] | None = None
    # Tools callable without an active workspace session.
    sessionless_tools: frozenset[str] = frozenset()
    # Tools allowed to return ok=False without the worker raising (partial runs).
    partial_failure_tools: frozenset[str] = frozenset()
    # Tools reserved for UI/backend use and hidden from the conversational agent.
    agent_excluded_tools: frozenset[str] = frozenset()
    # Whether tool calls require a running devserver + preview-context injection.
    requires_devserver: bool = False
    inject_preview_context: bool = False
    pool_size: int = 0  # 0 -> use the worker default pool size


def _build_builtin_mcp_servers() -> dict[str, McpServerSpec]:
    playwright = McpServerSpec(
        name="playwright",
        server_id="runtime-playwright",
        display_name="Runtime Playwright",
        command=sys.executable,
        args=[str(_PLAYWRIGHT_MCP_SERVER_PATH)],
        env={"RAGTIME_PLAYWRIGHT_BROKER_JS_PATH": str(_PLAYWRIGHT_BROKER_JS_PATH)},
        tool_names=_PLAYWRIGHT_MCP_TOOL_NAMES,
        sessionless_tools=frozenset({"playwright_external_browse"}),
        partial_failure_tools=frozenset({"playwright_debug_steps"}),
        # UI/Backend specific tools are excluded from the conversational agent.
        # playwright_debug_steps is intentionally allowed so the dynamic MCP
        # binder can bind it natively!
        agent_excluded_tools=frozenset(
            {
                "playwright_content_probe",
                "playwright_capture_screenshot",
                "playwright_external_browse",
            }
        ),
        requires_devserver=True,
        inject_preview_context=True,
    )
    return {playwright.name: playwright}


_BUILTIN_MCP_SERVERS: dict[str, McpServerSpec] = _build_builtin_mcp_servers()


@dataclass
class _McpJob:
    """A single MCP operation queued for a slot's owner task.

    ``run`` receives the live MCP ``ClientSession`` and performs exactly one
    request (tool call or list-tools) so the SDK's anyio scopes stay on the
    owner task.
    """

    run: Callable[[Any], Awaitable[Any]]
    timeout_ms: int
    future: asyncio.Future[Any]


@dataclass
class McpServerSlot:
    """A warm, long-lived runtime MCP connection for one server.

    The MCP Python SDK's ``stdio_client``/``ClientSession`` build anyio task
    groups whose cancel scopes are bound to the task that enters them, so the
    enter/call/exit lifecycle must all happen on a single task. We therefore
    run one dedicated owner task per slot that owns the connection and pulls
    operations off ``queue``; request handlers submit jobs and await futures.
    This keeps server processes (and warm Chromium) alive across calls and
    tears down cleanly (no leaked subprocess) by cancelling the owner in-place.
    """

    slot_id: int
    spec: McpServerSpec
    queue: asyncio.Queue[_McpJob | None] | None = None
    owner_task: asyncio.Task[None] | None = None
    ready: asyncio.Event | None = None
    start_error: str | None = None


@dataclass
class _McpServerPool:
    """A fixed pool of warm connections for a single MCP server."""

    spec: McpServerSpec
    slots: list[McpServerSlot]
    available: asyncio.Queue[int]


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
    bridge_credential_mode: str = "env"
    bridge_session_id: str | None = None
    bridge_credential_revision: int = 0
    bridge_refresh_requests: dict[str, tuple[str, RuntimeBridgeCredentialMetadata]] = field(default_factory=dict)
    bridge_recent_tokens: list[str] = field(default_factory=list)
    bridge_token_file_initial_token: str | None = None


class WorkerService:
    def __init__(self) -> None:
        self._sessions: dict[str, WorkerSession] = {}
        self._provider_to_session: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._worker_name = os.getenv("RUNTIME_WORKER_NAME", "runtime-worker").strip()
        self._base_url = os.getenv("RUNTIME_WORKER_BASE_URL", "http://runtime:8090").strip().rstrip("/")
        self._root = Path(os.getenv("RUNTIME_WORKSPACE_ROOT", "/data/_userspace")).resolve()
        self._devserver_processes: dict[str, asyncio.subprocess.Process] = {}
        self._devserver_log_paths: dict[str, Path] = {}
        self._devserver_log_handles: dict[str, Any] = {}
        self._bootstrap_retry_flags: dict[str, bool] = {}
        self._devserver_start_timeout_seconds = int(os.getenv("RUNTIME_DEVSERVER_START_TIMEOUT_SECONDS", "90"))
        self._runtime_bootstrap_timeout_seconds = int(os.getenv("RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS", "180"))
        self._runtime_config_file = ".ragtime/runtime-entrypoint.json"
        self._startup_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_execs: dict[str, dict[int, Any]] = {}
        self._app_restart_requests: dict[tuple[str, str], WorkerSessionResponse] = {}
        self._workspace_startup_locks: dict[str, asyncio.Lock] = {}
        # Lock order: startup lock -> file lock -> mount semaphore. File APIs
        # use only the file lock and never await it while holding _lock.
        self._workspace_file_locks: dict[str, asyncio.Lock] = {}
        self._workspace_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._workspace_maintenance: dict[str, dict[str, tuple[bool, SandboxSpec]]] = {}
        self._background_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._startup_semaphore = asyncio.Semaphore(
            get_positive_int_env(
                "RUNTIME_STARTUP_CONCURRENCY",
                recommended_startup_concurrency(),
            )
        )
        self._mount_materialization_semaphore = asyncio.Semaphore(
            get_positive_int_env(
                "RUNTIME_MOUNT_MATERIALIZATION_CONCURRENCY",
                2,
            )
        )
        self._mcp_default_pool_size = get_positive_int_env(
            "RUNTIME_MCP_SERVER_POOL_SIZE",
            get_positive_int_env("RUNTIME_PLAYWRIGHT_BROKER_POOL_SIZE", 2),
        )
        self._mcp_pools: dict[str, _McpServerPool] = {}
        self._mcp_pools_lock = asyncio.Lock()
        self._mcp_tool_catalog_cache: dict[str, list[RuntimeMcpToolInfo]] = {}

    def _normalize_file_path(
        self,
        file_path: str,
        *,
        enforce_sqlite_managed: bool = False,
    ) -> str:
        if not isinstance(file_path, str) or "\x00" in file_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return normalize_file_path(
            file_path,
            enforce_sqlite_managed=enforce_sqlite_managed,
        )

    def _resolve_workspace_root(self, workspace_id: str) -> tuple[Path, Path, SandboxSpec]:
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
        workspace_tree_root = await self._active_workspace_tree_root(workspace_id, workspace_files_path)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(workspace_tree_root),
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
        tree_root = await self._active_workspace_tree_root(workspace_id, workspace_files_path)

        base_entries = await asyncio.to_thread(
            list_workspace_tree_entries,
            tree_root,
            include_dirs=include_dirs,
        )
        mount_prefixes = deduplicate_ancestor_paths(
            [repo_rel for spec in mount_specs if (repo_rel := workspace_mount_target_repo_relative_path(str(spec.get("target_path", "") or "")))]
        )
        if mount_prefixes and tree_root == workspace_files_path:
            base_entries = [entry for entry in base_entries if not any(workspace_path_matches_mount_prefix(entry.path, prefix) for prefix in mount_prefixes)]

        mount_entries = await asyncio.to_thread(
            list_mount_source_tree_entries,
            mount_specs,
            include_dirs=include_dirs,
        )
        entries_by_path = {entry.path: entry for entry in base_entries}
        for entry in mount_entries:
            entries_by_path.setdefault(entry.path, entry)

        return RuntimeWorkspaceFileListResponse(
            files=[self._workspace_file_info(entry) for entry in sorted(entries_by_path.values(), key=lambda item: item.path)]
        )

    async def _active_workspace_tree_root(self, workspace_id: str, workspace_files_path: Path) -> Path:
        async with self._lock:
            active_sessions = [
                session for session in self._sessions.values() if session.workspace_id == workspace_id and session.state in {"running", "starting"}
            ]
        if not active_sessions:
            return workspace_files_path
        session = max(active_sessions, key=lambda item: item.updated_at)
        return self._workspace_tree_root_for_session(session, workspace_files_path)

    def _workspace_tree_root_for_session(self, session: WorkerSession, fallback_workspace_files_path: Path | None = None) -> Path:
        workspace_files_path = fallback_workspace_files_path if fallback_workspace_files_path is not None else session.workspace_files_path
        if not workspace_mirror_required(session.sandbox_spec, detect_capabilities()):
            return workspace_files_path
        workspace_path = session.sandbox_spec.rootfs_path / session.sandbox_spec.sandbox_workspace.lstrip("/")
        return workspace_path if workspace_path.is_dir() else workspace_files_path

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
        workspace_tree_root = await self._active_workspace_tree_root(workspace_id, workspace_files_path)
        sync_scope_paths = await asyncio.to_thread(
            sync_scope_relative_paths,
            workspace_tree_root,
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
        current_commit_hash = commit_result[1].decode("utf-8", errors="replace").strip() if commit_result[0] == 0 else ""
        return RuntimeWorkspaceScmStatusResponse(
            has_sync_scope_files=bool(sync_scope_paths),
            has_uncommitted_changes=bool(status_result[1].decode("utf-8", errors="replace").strip()),
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
        return {str(key): str(value) for key, value in (raw_env or {}).items() if str(key).strip()}

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
        return WorkerService._agent_shell_host_root(spec) / _AGENT_SHELL_ENV_METADATA_NAME

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
                    "sentinel": (_RAGTIME_REDACTED_ENV_SENTINEL_SET if has_value else _RAGTIME_REDACTED_ENV_SENTINEL_MISSING),
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
                f'#!/bin/sh\nexec /usr/bin/python3 {wrapper_target} "$@"\n',
                encoding="utf-8",
            )
            wrapper_path.chmod(0o755)

    def build_agent_shell_environment(self, session: WorkerSession) -> dict[str, str]:
        self._write_agent_shell_artifacts(session)
        return {
            "PATH": (f"{_AGENT_SHELL_BIN_DIR}:{os.getenv('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')}"),
            _RAGTIME_REDACTED_ENV_FILE_VAR: str(_AGENT_SHELL_ROOT / _AGENT_SHELL_ENV_METADATA_NAME),
        }

    def build_agent_process_environment(self, session: WorkerSession) -> dict[str, str]:
        environment = dict(session.workspace_env)
        # Ensure ad-hoc agent/PTY execs target the same runtime-assigned
        # devserver port as the launched workspace process.
        if session.devserver_port:
            environment["PORT"] = str(session.devserver_port)
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
        for value in session.bridge_recent_tokens:
            if value:
                items.append(("RAGTIME_BRIDGE_TOKEN", value, _RAGTIME_REDACTED_ENV_SENTINEL_SET))
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
                lambda match: f"{match.group(1)}{sentinel}{match.group(3)}" if (match.lastindex or 0) >= 3 else f"{match.group(1)}{sentinel}",
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
        for key, secret_value, sentinel in self._workspace_secret_redaction_items(session):
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
    def _workspace_screenshot_dir(workspace_root: Path) -> Path:
        return workspace_root / "runtime-artifacts" / "screenshots"

    @staticmethod
    def _mount_target_paths(mounts: list[dict[str, Any]]) -> set[str]:
        target_paths: set[str] = set()
        for mount in mounts:
            target = str(mount.get("target_path") or "").strip()
            if target:
                target_paths.add(target)
        return target_paths

    _resolve_workspace_mount_file_path = staticmethod(resolve_workspace_mount_source_path)

    async def _materialize_workspace_mounts(self, session: WorkerSession) -> None:
        mounts = list(session.workspace_mounts or [])
        clear_targets = sorted(session.mount_targets_to_clear)
        if not mounts and not clear_targets:
            return
        async with self._mount_materialization_semaphore:
            cancel_event = threading.Event()
            materialization = asyncio.create_task(
                asyncio.to_thread(
                    materialize_mounts,
                    session.sandbox_spec,
                    mounts,
                    clear_targets=clear_targets,
                    cancel_event=cancel_event,
                    timeout_seconds=self._runtime_bootstrap_timeout_seconds,
                )
            )
            try:
                # Shield the task from caller cancellation: to_thread cannot
                # be cancelled, so the cancellation path below must retain the
                # filesystem fence until its synchronous work is finished.
                await asyncio.shield(materialization)
            except asyncio.CancelledError:
                cancel_event.set()
                while not materialization.done():
                    try:
                        await asyncio.shield(materialization)
                    except asyncio.CancelledError:
                        # Keep the filesystem fence until the helper has
                        # reaped its process group and the thread has exited.
                        continue
                    except Exception:
                        # The thread is done; preserve the caller's original
                        # cancellation after recording its terminal error.
                        break
                try:
                    materialization.result()
                except (asyncio.CancelledError, Exception):
                    logger.debug("Materialization stopped after cancellation", exc_info=True)
                raise

    @staticmethod
    def _decode_jwt_payload_metadata(token: str) -> dict[str, Any] | None:
        parts = str(token or "").split(".")
        if len(parts) != 3:
            return None
        payload_segment = parts[1]
        padding = "=" * (-len(payload_segment) % 4)
        try:
            decoded = base64.urlsafe_b64decode(f"{payload_segment}{padding}")
            payload = json.loads(decoded)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _coerce_datetime_claim(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            return None

    def _bridge_credential_metadata(
        self,
        session: WorkerSession,
    ) -> RuntimeBridgeCredentialMetadata | None:
        bridge_url = str(session.workspace_env.get("RAGTIME_BRIDGE_URL") or "").strip()
        token = str(session.workspace_env.get("RAGTIME_BRIDGE_TOKEN") or "").strip()
        if session.bridge_credential_mode == "worker_file":
            try:
                token = self._read_bridge_token_file(session) or ""
            except HTTPException:
                return None
        if not bridge_url or not token:
            return None
        payload = self._decode_jwt_payload_metadata(token)
        if payload is None:
            return None
        token_kind = str(payload.get("kind") or "").strip()
        workspace_id = str(payload.get("workspace_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        issued_at = self._coerce_datetime_claim(payload.get("iat"))
        expires_at = self._coerce_datetime_claim(payload.get("exp"))
        if not token_kind or not workspace_id or not session_id or issued_at is None or expires_at is None:
            return None
        return RuntimeBridgeCredentialMetadata(
            bridge_url=bridge_url,
            token_kind=token_kind,
            workspace_id=workspace_id,
            session_id=session_id,
            issued_at=issued_at,
            expires_at=expires_at,
            mode=session.bridge_credential_mode,
            revision=session.bridge_credential_revision,
        )

    @staticmethod
    def _bridge_token_path(session: WorkerSession) -> Path:
        return session.sandbox_spec.rootfs_path / "run" / ".ragtime-bridge" / "token"

    @staticmethod
    def _open_bridge_token_directory(session: WorkerSession) -> int:
        """Open the private token directory without following hostile links."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(session.sandbox_spec.rootfs_path, flags)
        except OSError as exc:
            raise HTTPException(status_code=409, detail="Unsafe bridge credential root") from exc
        try:
            try:
                os.mkdir("run", 0o755, dir_fd=root_fd)
            except FileExistsError:
                pass
            try:
                run_fd = os.open("run", flags, dir_fd=root_fd)
            except OSError as exc:
                raise HTTPException(status_code=409, detail="Unsafe bridge credential path") from exc
            try:
                try:
                    os.mkdir(".ragtime-bridge", 0o700, dir_fd=run_fd)
                except FileExistsError:
                    pass
                try:
                    credential_fd = os.open(".ragtime-bridge", flags, dir_fd=run_fd)
                except OSError as exc:
                    raise HTTPException(status_code=409, detail="Unsafe bridge credential path") from exc
            finally:
                os.close(run_fd)
        finally:
            os.close(root_fd)
        return credential_fd

    def _read_bridge_token_file(self, session: WorkerSession) -> str | None:
        directory_fd = self._open_bridge_token_directory(session)
        try:
            try:
                token_fd = os.open("token", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                return None
            try:
                return os.read(token_fd, 1024 * 1024).decode("utf-8").strip()
            finally:
                os.close(token_fd)
        finally:
            os.close(directory_fd)

    def _write_bridge_token_file(self, session: WorkerSession, token: str) -> None:
        directory_fd = self._open_bridge_token_directory(session)
        temporary = f".token-{os.urandom(8).hex()}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        try:
            os.write(fd, token.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(temporary, "token", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.chmod("token", 0o600, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(directory_fd)

    def _session_response(self, session: WorkerSession) -> WorkerSessionResponse:
        return WorkerSessionResponse(
            worker_session_id=session.id,
            workspace_id=session.workspace_id,
            state=session.state,
            preview_internal_url=f"{self._base_url}/worker/sessions/{session.id}/preview",
            launch_framework=session.launch_framework,
            launch_command=(" ".join(session.devserver_command) if session.devserver_command else None),
            launch_cwd=session.launch_cwd,
            launch_port=session.devserver_port,
            runtime_capabilities={**sandbox_diagnostics(), "bridge_credential_file": True, "sqlite_workspace_maintenance": True},
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            runtime_operation_id=session.runtime_operation_id,
            runtime_operation_phase=session.runtime_operation_phase,
            runtime_operation_started_at=session.runtime_operation_started_at,
            runtime_operation_updated_at=session.runtime_operation_updated_at,
            bridge_credential=self._bridge_credential_metadata(session),
            updated_at=session.updated_at,
        )

    def _workspace_startup_lock(self, workspace_id: str) -> asyncio.Lock:
        lock = self._workspace_startup_locks.get(workspace_id)
        if lock is None:
            lock = asyncio.Lock()
            self._workspace_startup_locks[workspace_id] = lock
        return lock

    def _workspace_file_lock(self, workspace_id: str) -> asyncio.Lock:
        lock = self._workspace_file_locks.get(workspace_id)
        if lock is None:
            lock = asyncio.Lock()
            self._workspace_file_locks[workspace_id] = lock
        return lock

    def _ensure_workspace_available_locked(self, workspace_id: str, *, require_full_release: bool = False, maintenance_lease_id: str | None = None) -> None:
        if self._has_durable_sqlite_maintenance_marker(workspace_id, maintenance_lease_id):
            raise HTTPException(status_code=423, detail="Workspace SQLite maintenance recovery is required")
        leases = self._workspace_maintenance.get(workspace_id, {})
        if leases and (require_full_release or any(maintenance for maintenance, _ in leases.values())):
            if maintenance_lease_id and leases.get(maintenance_lease_id, (False, None))[0]:
                return
            raise HTTPException(status_code=423, detail="Workspace SQLite maintenance is active")

    def _has_durable_sqlite_maintenance_marker(self, workspace_id: str, maintenance_lease_id: str | None = None) -> bool:
        """Fail closed on a protected sibling marker left by a crashed app."""
        workspace_id = self._validate_workspace_id(workspace_id)
        marker = self._root / "workspaces" / workspace_id / "sqlite_backups" / "sqlite-maintenance-intent.json"
        try:
            # The path is outside the sandbox/user files tree.  A symlink or
            # any unexpected entry is still an interrupted-maintenance signal.
            entry = os.lstat(marker)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        if not maintenance_lease_id or not stat.S_ISREG(entry.st_mode):
            return True
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        return payload.get("lease_id") != maintenance_lease_id

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> str:
        value = str(workspace_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
            raise HTTPException(status_code=400, detail="Invalid workspace ID")
        return value

    async def acquire_sqlite_workspace_access(self, workspace_id: str, lease_id: str, *, maintenance: bool) -> dict[str, str | bool]:
        """Fence a workspace and return the only safe SQLite source path."""
        workspace_id = self._validate_workspace_id(workspace_id)
        if not lease_id or len(lease_id) > 128:
            raise HTTPException(status_code=400, detail="Invalid SQLite maintenance lease")
        workspace_root, canonical, spec = self._resolve_workspace_root(workspace_id)
        del workspace_root
        async with self._lock:
            leases = self._workspace_maintenance.setdefault(workspace_id, {})
            existing = leases.get(lease_id)
            if existing and existing[0] != maintenance:
                raise HTTPException(status_code=409, detail="Workspace SQLite maintenance lease mode conflicts")
            if existing is None:
                if maintenance and leases:
                    raise HTTPException(status_code=409, detail="Workspace SQLite maintenance is already held")
                if not maintenance and any(active_maintenance for active_maintenance, _ in leases.values()):
                    raise HTTPException(status_code=409, detail="Workspace SQLite maintenance is already held")
                leases[lease_id] = (maintenance, spec)
            newly_registered = existing is None
            session_ids = [
                session.id for session in self._sessions.values() if session.workspace_id == workspace_id and session.state in {"starting", "running"}
            ]
        try:
            if maintenance:
                # Import lazily to avoid the worker service/API module cycle.
                from runtime.worker.api import evict_workspace_ptys

                if newly_registered:
                    await evict_workspace_ptys(workspace_id)
                    for session_id in session_ids:
                        try:
                            await self.stop_session(session_id, _maintenance_lease_id=lease_id)
                        except HTTPException as exc:
                            if exc.status_code != 404:
                                raise
                async with self._workspace_startup_lock(workspace_id):
                    async with self._workspace_file_lock(workspace_id):
                        caps = detect_capabilities()
                        if workspace_mirror_required(spec, caps):
                            await asyncio.to_thread(reconcile_stopped_workspace_mirror, spec)
                        authoritative = canonical
            else:
                # Never await either workspace lock while holding _lock.  Once
                # synchronized, choose from the current active session only.
                async with self._workspace_startup_lock(workspace_id):
                    async with self._workspace_file_lock(workspace_id):
                        async with self._lock:
                            active = [
                                session
                                for session in self._sessions.values()
                                if session.workspace_id == workspace_id and session.state in {"starting", "running"}
                            ]
                        if not active:
                            authoritative = canonical
                        else:
                            mirrors = {
                                session.sandbox_spec.rootfs_path / session.sandbox_spec.sandbox_workspace.lstrip("/")
                                for session in active
                                if workspace_mirror_required(session.sandbox_spec, detect_capabilities())
                            }
                            if not mirrors:
                                authoritative = canonical
                            elif len(mirrors) == 1 and next(iter(mirrors)).is_dir():
                                authoritative = next(iter(mirrors))
                            else:
                                raise HTTPException(status_code=503, detail="Authoritative runtime workspace is unavailable")
                caps = detect_capabilities()
                if not authoritative.is_dir():
                    raise HTTPException(status_code=503, detail="Authoritative runtime workspace is unavailable")
        except BaseException:
            # Only cleanup the lease if THIS call newly registered it.
            if newly_registered:
                async with self._lock:
                    leases = self._workspace_maintenance.get(workspace_id, {})
                    leases.pop(lease_id, None)
                    if not leases:
                        self._workspace_maintenance.pop(workspace_id, None)
            raise
        return {
            "workspace_id": workspace_id,
            "lease_id": lease_id,
            "authoritative_root": str(authoritative),
            "sandbox_mode": caps.mode,
            "maintenance": maintenance,
        }

    async def release_sqlite_workspace_access(self, workspace_id: str, lease_id: str) -> None:
        workspace_id = self._validate_workspace_id(workspace_id)
        async with self._lock:
            leases = self._workspace_maintenance.get(workspace_id, {})
            lease = leases.get(lease_id)
            if not lease:
                # A runtime restart loses the in-memory lease but must not make
                # verified Lane C recovery impossible.  The durable marker is
                # still the fail-closed authority until the control plane clears
                # it after receipt/candidate validation.
                if leases:
                    raise HTTPException(status_code=409, detail="Workspace SQLite maintenance lease owner mismatch")
                return
            maintenance, spec = lease
        if maintenance:
            async with self._workspace_startup_lock(workspace_id):
                async with self._workspace_file_lock(workspace_id):
                    caps = detect_capabilities()
                    if workspace_mirror_required(spec, caps):
                        await asyncio.to_thread(archive_workspace_mirror, spec)
        async with self._lock:
            leases = self._workspace_maintenance.get(workspace_id, {})
            if lease_id not in leases:
                raise HTTPException(status_code=409, detail="Workspace SQLite maintenance lease owner mismatch")
            leases.pop(lease_id)
            if not leases:
                self._workspace_maintenance.pop(workspace_id, None)

    async def assert_pty_available(self, worker_session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            self._ensure_workspace_available_locked(session.workspace_id)

    async def workspace_id_for_session(self, worker_session_id: str) -> str | None:
        """Expose only the session's workspace identity to the PTY drain path."""
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            return session.workspace_id if session else None

    @staticmethod
    async def _drain_file_io_task(task: asyncio.Task[Any]) -> Any:
        """Drain thread I/O before its workspace fence can be released."""
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _capture_file_target_locked(
        self,
        worker_session_id: str,
        rel_path: str,
        *,
        mutation: bool,
    ) -> tuple[WorkerSession, Path, str, str | None]:
        """Capture current root/policy only after the workspace file fence."""
        session = self._sessions.get(worker_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Worker session not found")
        mounted_target = resolve_workspace_mount_rooted_target(session.workspace_mounts, rel_path)
        if mutation and mounted_target and mounted_target.read_only:
            raise HTTPException(status_code=403, detail="Mounted workspace paths are read-only")
        workspace_tree_root = self._workspace_tree_root_for_session(session)
        return (
            session,
            mounted_target.root if mounted_target else workspace_tree_root,
            mounted_target.relative_path if mounted_target else rel_path,
            session.runtime_operation_id,
        )

    @staticmethod
    def _is_unsafe_file_error(exc: OSError) -> bool:
        return exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}

    async def _wait_for_workspace_cleanup(self, workspace_id: str) -> None:
        """Wait for a stop barrier that was registered before this startup."""
        while True:
            async with self._lock:
                cleanup_task = self._workspace_cleanup_tasks.get(workspace_id)
            if cleanup_task is None:
                return
            await asyncio.shield(cleanup_task)

    def _track_background_cleanup(self, task: asyncio.Task[None]) -> None:
        self._background_cleanup_tasks.add(task)
        task.add_done_callback(self._background_cleanup_tasks.discard)

    def _begin_operation(self, session: WorkerSession, phase: str) -> None:
        now = utc_now()
        session.runtime_operation_id = os.urandom(12).hex()
        session.runtime_operation_phase = phase
        session.runtime_operation_started_at = now
        session.runtime_operation_updated_at = now

    def _set_operation_phase(self, session: WorkerSession, phase: str) -> None:
        session.runtime_operation_phase = phase
        session.runtime_operation_updated_at = utc_now()

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

    @staticmethod
    def _load_runtime_bootstrap_config_dict_sync(workspace_root: Path) -> dict[str, Any] | None:
        config_path = workspace_root / RUNTIME_BOOTSTRAP_CONFIG_PATH
        if not config_path.exists() or not config_path.is_file():
            return None
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _normalize_bootstrap_watch_paths(raw_config: dict[str, Any] | None) -> list[str]:
        if not isinstance(raw_config, dict):
            return []
        watch_paths = raw_config.get("watch_paths")
        if not isinstance(watch_paths, list):
            return []
        normalized: list[str] = []
        for item in watch_paths:
            relative = str(item or "").strip().replace("\\", "/")
            if relative:
                normalized.append(relative)
        return normalized

    def _read_runtime_bootstrap_config_sync(self, workspace_root: Path) -> list[dict[str, str]]:
        raw = self._load_runtime_bootstrap_config_dict_sync(workspace_root)
        if raw is None:
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

    async def _read_runtime_bootstrap_config(self, workspace_root: Path) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self._read_runtime_bootstrap_config_sync,
            workspace_root,
        )

    def _sync_missing_bootstrap_watch_paths_to_sandbox_sync(self, session: WorkerSession) -> None:
        """Prune sandbox-mirror copies of bootstrap watch paths absent from canonical files.

        ``rootfs/workspace`` is an incremental ``copytree(dirs_exist_ok=True)``
        mirror, so files deleted from canonical ``files/`` persist there. When
        those files (e.g. ``package-lock.json``) gate bootstrap commands, the
        stale mirror copy steers ``npm ci`` toward a lock that no longer
        matches the canonical workspace. Pruning only the configured
        ``watch_paths`` keeps mirrored build artifacts (``node_modules`` etc.)
        intact.
        """
        rootfs_workspace = session.sandbox_spec.rootfs_path / SANDBOX_WORKSPACE_MOUNT.lstrip("/")
        if not rootfs_workspace.is_dir():
            return

        raw_config = self._load_runtime_bootstrap_config_dict_sync(session.workspace_files_path)
        for relative in self._normalize_bootstrap_watch_paths(raw_config):
            source = self._resolve_bootstrap_relative_path(session.workspace_files_path, relative)
            mirror = self._resolve_bootstrap_relative_path(rootfs_workspace, relative)
            if source is None or mirror is None or source.exists() or not mirror.exists():
                continue
            try:
                if mirror.is_symlink() or mirror.is_file():
                    mirror.unlink()
                elif mirror.is_dir():
                    shutil.rmtree(mirror)
            except Exception as exc:
                logger.warning(
                    "Failed to remove stale sandbox bootstrap path %s for workspace %s: %s",
                    relative,
                    session.workspace_id,
                    exc,
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

        watch_relatives = self._normalize_bootstrap_watch_paths(parsed)
        if not isinstance(parsed.get("watch_paths"), list):
            return hashlib.sha256(payload).hexdigest()

        digest = hashlib.sha256(payload)
        for relative in watch_relatives:
            resolved = self._resolve_bootstrap_relative_path(workspace_root, relative)
            if resolved is None:
                continue

            digest.update(relative.encode("utf-8", errors="ignore"))
            if not resolved.exists():
                digest.update(b"::missing")
                continue

            if resolved.is_file():
                digest.update(b"::file")
                self._update_digest_from_file(digest, resolved)
                continue

            if resolved.is_dir():
                digest.update(b"::dir")
                for child in sorted(path for path in resolved.rglob("*") if path.is_file()):
                    rel_child = str(child.relative_to(workspace_root)).replace("\\", "/")
                    digest.update(rel_child.encode("utf-8", errors="ignore"))
                    self._update_digest_from_file(digest, child)

        return digest.hexdigest()

    @staticmethod
    def _update_digest_from_file(digest: Any, path: Path) -> None:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)

    async def _runtime_bootstrap_config_digest(self, workspace_root: Path) -> str | None:
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
        normalized_artifact = str(artifact_relative_path or "").strip().replace("\\", "/")
        if not normalized_artifact:
            return False
        if (cwd_path / normalized_artifact).exists():
            return True

        rootfs_workspace = session.sandbox_spec.rootfs_path / SANDBOX_WORKSPACE_MOUNT.lstrip("/")
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

        if name in {"npm_ci", "npm_install", "node_dependencies"} or cmd.startswith("npm ci") or cmd.startswith("npm install"):
            expected_artifact = "node_modules"
        elif name == "node_tailwind_tooling" or ("npm install" in cmd and "tailwindcss" in cmd):
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

        # Bootstrap commands run inside the sandbox where ``rootfs/workspace``
        # is an additive mirror of canonical ``files/``. Prune stale mirror
        # copies of watched files (e.g. an orphaned ``package-lock.json``)
        # before any command observes them.
        await asyncio.to_thread(
            self._sync_missing_bootstrap_watch_paths_to_sandbox_sync,
            session,
        )

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
                return "Runtime bootstrap config has invalid cwd path. Use workspace-relative paths only."
            if when_exists and (when_path is None or not when_path.exists()):
                continue
            if unless_exists and unless_path is not None and unless_path.exists():
                continue
            # Also check the rootfs workspace dir — the sandbox may already
            # have the artifact (e.g. node_modules) from a prior session's
            # copytree even though the canonical ``files/`` tree lacks it.
            if unless_exists:
                rootfs_ws = session.sandbox_spec.rootfs_path / SANDBOX_WORKSPACE_MOUNT.lstrip("/")
                rootfs_unless = self._resolve_bootstrap_relative_path(rootfs_ws, unless_exists)
                if rootfs_unless is not None and rootfs_unless.exists():
                    continue

            try:
                # Resolve cwd relative to sandbox workspace mount
                sandbox_cwd = SANDBOX_WORKSPACE_MOUNT
                if cwd_value and cwd_value not in {".", "./"}:
                    sandbox_cwd = f"{SANDBOX_WORKSPACE_MOUNT}/{cwd_value}"
                # Embed an explicit ``cd`` so cwd is reliable even when the
                # sandbox launcher re-execs the workload under a fresh rootfs.
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
                return f"Runtime bootstrap command '{command_name}' timed out after {self._runtime_bootstrap_timeout_seconds}s."
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

                return f"Runtime bootstrap command '{command_name}' failed with code {returncode}: {(detail or output)[:300]}. {_WORKSPACE_BOOTSTRAP_GUIDANCE}"

        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_value = config_digest or utc_now().isoformat()
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
            return f"Auto-install of {framework} dependencies timed out after {self._runtime_bootstrap_timeout_seconds}s."
        except Exception as exc:
            return f"Auto-install of {framework} dependencies failed to launch: {exc}"

        returncode = process.returncode or 0
        if returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            output = stderr_text or stdout_text or "unknown error"
            return f"Auto-install of {framework} dependencies failed with code {returncode}: {output[:300]}"
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
        return f"Runtime entrypoint uses '{tool}' but it is not installed in this isolated runtime container. {_WORKSPACE_BOOTSTRAP_GUIDANCE}"

    def _resolve_devserver_log_path(self, session_id: str) -> Path:
        log_dir = Path(os.getenv("RUNTIME_DEVSERVER_LOG_DIR", _RUNTIME_DEVSERVER_LOG_DIR))
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{session_id}.log"

    @staticmethod
    def _normalize_log_text(text: str) -> str:
        """Strip null bytes (rejected by PG TEXT) and collapse whitespace."""
        return " ".join(text.replace("\x00", "").split())

    @staticmethod
    def _compact_devserver_log_for_error(content: str) -> str:
        compact = WorkerService._normalize_log_text(content)
        tail_chars = _RUNTIME_DEVSERVER_LOG_TAIL_CHARS
        if len(compact) <= tail_chars:
            return compact

        tail = compact[-tail_chars:]
        for pattern in _DEVSERVER_ERROR_SUMMARY_PATTERNS:
            match = pattern.search(compact)
            if not match:
                continue
            summary = match.group("detail").strip()
            if not summary or summary in tail:
                return tail
            separator = " ... "
            tail_budget = tail_chars - len(summary) - len(separator)
            if tail_budget <= 0:
                return summary[:tail_chars]
            return f"{summary}{separator}{tail[-tail_budget:]}"
        return tail

    def _read_devserver_log_tail(self, session_id: str) -> str:
        log_path = self._devserver_log_paths.get(session_id)
        if not log_path:
            return ""
        try:
            if not log_path.exists() or not log_path.is_file():
                return ""
            content = log_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return ""
        if not content:
            return ""
        return self._compact_devserver_log_for_error(content)

    async def _run_mcp_slot(self, slot: McpServerSlot) -> None:
        """Owner task: own one MCP connection and serve queued operations.

        Connect / initialize / run-op / teardown all run on this single task so
        the MCP SDK's anyio cancel scopes are entered and exited on the same
        task. Request handlers never touch the connection directly; they submit
        jobs to ``slot.queue`` and await the per-job future.
        """
        assert slot.queue is not None and slot.ready is not None
        queue = slot.queue
        ready = slot.ready
        spec = slot.spec

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # pragma: no cover - environment guard
            slot.start_error = f"Runtime MCP client dependencies are unavailable: {exc}"
            ready.set()
            self._fail_pending_mcp_jobs(queue, slot.start_error)
            return

        params = StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            env={**os.environ, **spec.env},
        )

        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as client:
                    await asyncio.wait_for(client.initialize(), timeout=15)
                    slot.start_error = None
                    ready.set()
                    while True:
                        job = await queue.get()
                        if job is None:
                            break
                        if job.future.cancelled():
                            continue
                        try:
                            op_result = await asyncio.wait_for(
                                job.run(client),
                                timeout=max(5.0, job.timeout_ms / 1000.0 + 10.0),
                            )
                        except Exception as exc:
                            if not job.future.done():
                                job.future.set_exception(exc)
                            # A failed op means the stdio transport is no longer
                            # trustworthy; end this owner so the next request
                            # rebuilds a fresh warm connection.
                            raise
                        if not job.future.done():
                            job.future.set_result(op_result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if slot.start_error is None:
                slot.start_error = str(exc)
        finally:
            if not ready.is_set():
                ready.set()
            slot.owner_task = None
            self._fail_pending_mcp_jobs(queue, slot.start_error or "Runtime MCP session ended")

    @staticmethod
    def _fail_pending_mcp_jobs(queue: "asyncio.Queue[_McpJob | None]", message: str) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job is None:
                continue
            if not job.future.done():
                job.future.set_exception(HTTPException(status_code=503, detail=message))

    async def _ensure_mcp_slot(self, slot: McpServerSlot) -> None:
        if slot.owner_task is not None and not slot.owner_task.done():
            if slot.ready is not None:
                try:
                    await asyncio.wait_for(slot.ready.wait(), timeout=30)
                except asyncio.TimeoutError as exc:
                    await self._terminate_mcp_slot(slot)
                    raise HTTPException(status_code=503, detail=f"Timed out waiting for runtime MCP server '{slot.spec.name}'") from exc
            if slot.start_error:
                detail = slot.start_error
                await self._terminate_mcp_slot(slot)
                raise HTTPException(status_code=503, detail=detail)
            return

        slot.queue = asyncio.Queue()
        slot.ready = asyncio.Event()
        slot.start_error = None
        slot.owner_task = asyncio.create_task(self._run_mcp_slot(slot))
        try:
            await asyncio.wait_for(slot.ready.wait(), timeout=30)
        except asyncio.TimeoutError as exc:
            await self._terminate_mcp_slot(slot)
            raise HTTPException(status_code=503, detail=f"Timed out starting runtime MCP server '{slot.spec.name}'") from exc
        if slot.start_error:
            detail = slot.start_error
            await self._terminate_mcp_slot(slot)
            raise HTTPException(status_code=503, detail=detail)

    async def _terminate_mcp_slot(self, slot: McpServerSlot) -> None:
        task = slot.owner_task
        slot.owner_task = None
        if slot.queue is not None:
            self._fail_pending_mcp_jobs(slot.queue, "Runtime MCP session terminated")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        slot.queue = None
        slot.ready = None

    async def _terminate_mcp_brokers(self) -> None:
        for pool in list(self._mcp_pools.values()):
            for slot in pool.slots:
                await self._terminate_mcp_slot(slot)

    @staticmethod
    def _resolve_mcp_spec(server_name: str) -> McpServerSpec:
        spec = _BUILTIN_MCP_SERVERS.get(server_name)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"Unknown runtime MCP server: {server_name}")
        return spec

    async def _acquire_mcp_pool(self, spec: McpServerSpec) -> _McpServerPool:
        pool = self._mcp_pools.get(spec.name)
        if pool is not None:
            return pool
        async with self._mcp_pools_lock:
            pool = self._mcp_pools.get(spec.name)
            if pool is None:
                size = max(1, spec.pool_size or self._mcp_default_pool_size)
                slots = [McpServerSlot(slot_id=i, spec=spec) for i in range(size)]
                available: asyncio.Queue[int] = asyncio.Queue(maxsize=size)
                for i in range(size):
                    available.put_nowait(i)
                pool = _McpServerPool(spec=spec, slots=slots, available=available)
                self._mcp_pools[spec.name] = pool
        return pool

    async def _run_on_mcp_pool(
        self,
        spec: McpServerSpec,
        run: Callable[[Any], Awaitable[Any]],
        *,
        timeout_ms: int,
    ) -> Any:
        """Acquire a warm slot for ``spec`` and run one MCP operation on it."""
        pool = await self._acquire_mcp_pool(spec)
        slot_index = await pool.available.get()
        slot = pool.slots[slot_index]
        result: Any = None
        try:
            for attempt in range(2):
                try:
                    await self._ensure_mcp_slot(slot)
                    queue = slot.queue
                    if queue is None:
                        raise HTTPException(status_code=503, detail=f"Runtime MCP server '{spec.name}' is unavailable")
                    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
                    await queue.put(_McpJob(run=run, timeout_ms=timeout_ms, future=future))
                    result = await asyncio.wait_for(future, timeout=max(10.0, timeout_ms / 1000.0 + 20.0))
                    break
                except asyncio.TimeoutError as exc:
                    await self._terminate_mcp_slot(slot)
                    if attempt == 0:
                        continue
                    raise HTTPException(status_code=504, detail="Runtime MCP tool call timed out") from exc
                except HTTPException:
                    # Raised on transient session loss (op failed mid-flight or
                    # startup error); retry once with a fresh warm connection.
                    if attempt == 0:
                        continue
                    raise
                except Exception as exc:
                    await self._terminate_mcp_slot(slot)
                    if attempt == 0:
                        continue
                    raise HTTPException(status_code=502, detail=f"Runtime MCP tool call failed: {exc}") from exc
        finally:
            try:
                pool.available.put_nowait(slot_index)
            except asyncio.QueueFull:
                pass
        return result

    async def _invoke_mcp_tool(
        self,
        session: WorkerSession | None,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_ms: int,
        screenshot_dir: Path | None = None,
    ) -> RuntimeMcpToolCallResponse:
        """Call a tool on a named runtime MCP server over stdio transport."""
        spec = self._resolve_mcp_spec(server_name)
        if session is None and tool_name not in spec.sessionless_tools:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        if spec.tool_names is not None and tool_name not in spec.tool_names:
            raise HTTPException(status_code=400, detail=f"Unknown runtime MCP tool: {server_name}/{tool_name}")

        request_payload = dict(arguments or {})
        if spec.inject_preview_context:
            if session:
                request_payload["__ragtime_preview_base_url"] = f"http://127.0.0.1:{session.devserver_port}"
            else:
                request_payload["__ragtime_preview_base_url"] = "http://127.0.0.1:0"
            if screenshot_dir:
                request_payload["__ragtime_screenshot_dir"] = str(screenshot_dir)

        result = await self._run_on_mcp_pool(
            spec,
            lambda client: client.call_tool(tool_name, request_payload),
            timeout_ms=timeout_ms,
        )

        request_payload.pop("__ragtime_preview_base_url", None)
        request_payload.pop("__ragtime_screenshot_dir", None)

        response_payload: dict[str, Any] = {}
        content_items = list(getattr(result, "content", []) or [])
        text_items: list[str] = []
        for item in content_items:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_items.append(text)
        if text_items:
            try:
                parsed = json.loads(text_items[0])
                if isinstance(parsed, dict):
                    response_payload = parsed
                else:
                    response_payload = {"text": text_items[0]}
            except Exception:
                response_payload = {"text": "\n".join(text_items)}
        if bool(getattr(result, "isError", False)) and "error" not in response_payload:
            response_payload["error"] = response_payload.get("text") or "MCP tool returned an error"
        if response_payload.get("ok") is False and tool_name not in spec.partial_failure_tools:
            raise HTTPException(status_code=502, detail=str(response_payload.get("error") or "Runtime MCP tool returned an error"))

        return RuntimeMcpToolCallResponse(
            ok=bool(response_payload.get("ok", True)),
            server_id=spec.server_id,
            server_name=spec.display_name,
            tool_name=tool_name,
            request=request_payload,
            response=response_payload,
        )

    async def _list_server_tools(self, spec: McpServerSpec) -> list[RuntimeMcpToolInfo]:
        cached = self._mcp_tool_catalog_cache.get(spec.name)
        if cached is not None:
            return cached
        list_result = await self._run_on_mcp_pool(
            spec,
            lambda client: client.list_tools(),
            timeout_ms=10000,
        )
        discovered: list[RuntimeMcpToolInfo] = []
        for tool in getattr(list_result, "tools", []) or []:
            name = str(getattr(tool, "name", "") or "")
            if not name:
                continue
            discovered.append(
                RuntimeMcpToolInfo(
                    server_name=spec.name,
                    server_id=spec.server_id,
                    name=name,
                    description=str(getattr(tool, "description", "") or ""),
                    input_schema=dict(getattr(tool, "inputSchema", {}) or {}),
                    agent_excluded=name in spec.agent_excluded_tools,
                )
            )
        self._mcp_tool_catalog_cache[spec.name] = discovered
        return discovered

    async def list_mcp_tools(self, worker_session_id: str) -> RuntimeMcpToolListResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
        tools: list[RuntimeMcpToolInfo] = []
        for spec in _BUILTIN_MCP_SERVERS.values():
            tools.extend(await self._list_server_tools(spec))
        return RuntimeMcpToolListResponse(tools=tools)

    async def list_global_mcp_tools(self) -> RuntimeMcpToolListResponse:
        """List built-in MCP tools that do not require an active workspace session."""
        tools: list[RuntimeMcpToolInfo] = []
        for spec in _BUILTIN_MCP_SERVERS.values():
            tools.extend(await self._list_server_tools(spec))
        return RuntimeMcpToolListResponse(tools=tools)

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
        return status_code == 200 and html_length <= 64 and not title and element_visible_count == 0 and not console_errors

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
            rewritten = f"{rewritten[: match.start()]}{replacement}{rewritten[match.end() :]}"
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
                if self._command_uses_tool(config_command, tool) and not shutil.which(tool):
                    return DevserverResolution(
                        error=self._missing_tool_error(tool),
                        framework=config_framework,
                        cwd=config_cwd,
                        port=effective_port,
                    )
            # Resolve the sandbox-internal cwd for the command. We embed an
            # explicit ``cd`` because the sandbox launcher re-execs the
            # workload under a fresh rootfs before the final command starts.
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
        deadline = asyncio.get_event_loop().time() + self._devserver_start_timeout_seconds
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
        process, log_handle = self._take_devserver_resources_locked(session_id)
        await self._terminate_devserver_resources(process, log_handle)

    def _take_devserver_resources_locked(
        self,
        session_id: str,
    ) -> tuple[asyncio.subprocess.Process | None, Any | None]:
        return (
            self._devserver_processes.pop(session_id, None),
            self._devserver_log_handles.pop(session_id, None),
        )

    async def _terminate_devserver_resources(
        self,
        process: asyncio.subprocess.Process | None,
        log_handle: Any | None,
    ) -> None:
        if process is None:
            if log_handle:
                try:
                    log_handle.close()
                except Exception:
                    pass
            return
        await self._terminate_devserver_process(process)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass

    async def _terminate_devserver_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout: float = 3,
    ) -> None:
        await terminate_process_group(process, timeout=timeout)

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
                session.last_error = f"Dev server exited with code {process.returncode}: {log_tail}"
            else:
                session.last_error = f"Dev server exited with code {process.returncode}"

            if not self._bootstrap_retry_flags.get(session.id, False) and self._should_retry_bootstrap_after_exit(
                process.returncode,
                log_tail,
            ):
                self._bootstrap_retry_flags[session.id] = True
                self._invalidate_bootstrap_stamp(session)
                session.last_error = f"{session.last_error} Retrying workspace bootstrap on next start."
            session.runtime_operation_phase = "failed"
            session.runtime_operation_updated_at = utc_now()
        session.updated_at = utc_now()

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
            session.state = "running"
            session.devserver_running = False
            session.last_error = error
            self._set_operation_phase(session, "failed")
            session.updated_at = utc_now()

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
        await self._wait_for_workspace_cleanup(workspace_id)
        async with workspace_lock:
            # A stop may have registered its barrier after the first check but
            # before this startup acquired the workspace lock.
            await self._wait_for_workspace_cleanup(workspace_id)
            async with self._lock:
                self._ensure_workspace_available_locked(workspace_id, require_full_release=True)
            async with self._startup_semaphore:
                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    self._set_operation_phase(session, "provisioning")
                    session.updated_at = utc_now()

                try:
                    # Provisioning and mount materialization can replace the
                    # active tree, so fence only this short phase. Bootstrap
                    # intentionally runs after releasing the file lock.
                    async with self._workspace_file_lock(workspace_id):
                        await asyncio.to_thread(ensure_sandbox_ready, session.sandbox_spec)
                        if session.bridge_credential_mode == "worker_file":
                            token = str(session.bridge_token_file_initial_token or "")
                            if not token:
                                raise HTTPException(status_code=400, detail="Missing worker file bridge credential")
                            self._write_bridge_token_file(session, token)
                            session.bridge_recent_tokens = [token]
                        await self._materialize_workspace_mounts(session)
                except Exception as exc:
                    await self._mark_operation_failed(
                        session_id,
                        operation_id,
                        f"Failed to prepare runtime sandbox or materialize workspace mounts: {exc}",
                    )
                    return

                async with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.runtime_operation_id != operation_id:
                        return
                    session.mount_targets_to_clear = set()
                    self._set_operation_phase(session, "bootstrapping")
                    session.updated_at = utc_now()

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
                    session.updated_at = utc_now()

                try:
                    deps_error = await self._ensure_entrypoint_dependencies(session)
                except asyncio.CancelledError:
                    raise
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
                    session.updated_at = utc_now()
                    port = session.devserver_port or self._pick_free_port()
                    resolution = self._resolve_devserver_command(session.workspace_files_path, port)
                    session.launch_framework = resolution.framework
                    session.launch_cwd = resolution.cwd
                    if not resolution.command:
                        session.devserver_port = resolution.port
                        session.devserver_command = None
                        error = resolution.error or "Invalid runtime entrypoint"
                        session.state = "running"
                        session.devserver_running = False
                        session.last_error = error
                        self._set_operation_phase(session, "failed")
                        session.updated_at = utc_now()
                        return

                    session.devserver_port = resolution.port or port
                    await self._terminate_devserver_locked(session.id)
                    log_path = self._resolve_devserver_log_path(session.id)
                    try:
                        log_handle = open(log_path, "wb", buffering=0)
                    except Exception as exc:
                        session.state = "running"
                        session.devserver_running = False
                        session.last_error = f"Failed to initialize devserver log file: {exc}"
                        self._set_operation_phase(session, "failed")
                        session.updated_at = utc_now()
                        return

                    self._devserver_log_paths[session.id] = log_path
                    self._devserver_log_handles[session.id] = log_handle
                    # Capture everything spawn_sandboxed needs so we can
                    # release the lock before process creation.
                    _sandbox_spec = session.sandbox_spec
                    _spawn_command = list(resolution.command)
                    _launch_cwd = self._resolve_launch_cwd(session)
                    _workspace_env = {
                        **session.workspace_env,
                    }

                # --- Part 2: spawn sandbox OUTSIDE the lock ---
                _process: asyncio.subprocess.Process | None = None
                _spawn_error: str | None = None
                try:
                    try:
                        _process = await spawn_sandboxed(
                            _sandbox_spec,
                            _spawn_command,
                            cwd=_launch_cwd,
                            env=_workspace_env,
                            stdout=log_handle,
                            stderr=asyncio.subprocess.STDOUT,
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
                                await self._terminate_devserver_process(_process)
                            tracked_log_handle = self._devserver_log_handles.get(session_id)
                            if tracked_log_handle is log_handle:
                                self._devserver_log_handles.pop(session_id, None)
                                try:
                                    log_handle.close()
                                except Exception:
                                    pass
                            return
                        if _spawn_error or _process is None:
                            self._devserver_log_handles.pop(session.id, None)
                            session.state = "running"
                            session.devserver_running = False
                            session.last_error = _spawn_error or "Failed to launch dev server"
                            self._set_operation_phase(session, "failed")
                            session.updated_at = utc_now()
                            return
                        self._devserver_processes[session.id] = _process
                        session.devserver_command = _spawn_command
                        self._set_operation_phase(session, "probing")
                        session.updated_at = utc_now()
                        target_port = session.devserver_port
                except asyncio.CancelledError:
                    if _process is not None:
                        await self._terminate_devserver_process(_process)
                    async with self._lock:
                        tracked_process = self._devserver_processes.get(session_id)
                        if tracked_process is _process:
                            self._devserver_processes.pop(session_id, None)
                        tracked_log_handle = self._devserver_log_handles.get(session_id)
                        if tracked_log_handle is log_handle:
                            self._devserver_log_handles.pop(session_id, None)
                        else:
                            tracked_log_handle = None
                    if tracked_log_handle:
                        try:
                            tracked_log_handle.close()
                        except Exception:
                            pass
                    raise

                ready = await self._wait_devserver_ready(target_port or 0)
                if not ready:
                    async with self._lock:
                        session = self._sessions.get(session_id)
                        if not session or session.runtime_operation_id != operation_id:
                            return
                        await self._sync_devserver_state_locked(session)
                        await self._terminate_devserver_locked(session.id)
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
                        session.updated_at = utc_now()
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
                    session.updated_at = utc_now()

    def _schedule_startup_locked(
        self,
        session: WorkerSession,
        *,
        replace_existing: bool = True,
    ) -> None:
        existing = self._startup_tasks.get(session.id)
        if existing and not existing.done():
            if not replace_existing:
                return
            existing.cancel()
        self._begin_operation(session, "queued")
        session.state = "starting"
        session.devserver_running = False
        session.devserver_port = None
        session.last_error = None
        session.updated_at = utc_now()
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
            self._ensure_workspace_available_locked(request.workspace_id, require_full_release=True)
            existing_session_id = self._provider_to_session.get(request.provider_session_id)
            if existing_session_id and existing_session_id in self._sessions:
                session = self._sessions[existing_session_id]
                if session.workspace_id != request.workspace_id:
                    raise HTTPException(
                        status_code=409,
                        detail="Worker session workspace does not match requested workspace",
                    )
                session.pty_access_token = request.pty_access_token
                session.workspace_env = self._normalize_workspace_env(request.workspace_env)
                session.bridge_credential_mode = request.bridge_credential_mode
                if request.bridge_credential_mode == "worker_file":
                    token = str(request.bridge_token_file_initial_token or "").strip()
                    if not token or "RAGTIME_BRIDGE_TOKEN" in session.workspace_env:
                        raise HTTPException(status_code=400, detail="worker_file mode requires separate bridge token")
                    session.workspace_env["RAGTIME_BRIDGE_TOKEN_FILE"] = "/run/.ragtime-bridge/token"
                    session.bridge_token_file_initial_token = token
                    session.bridge_recent_tokens = ([token] + session.bridge_recent_tokens)[:2]
                    session.bridge_session_id = str((self._decode_jwt_payload_metadata(token) or {}).get("session_id") or "") or None
                else:
                    session.bridge_token_file_initial_token = None
                session.workspace_env_visibility = self._normalize_workspace_env_visibility(
                    request.workspace_env_visibility,
                    session.workspace_env,
                )
                previous_targets = self._mount_target_paths(session.workspace_mounts)
                session.workspace_mounts = list(request.workspace_mounts or [])
                session.mount_targets_to_clear = previous_targets | self._mount_target_paths(session.workspace_mounts)
                self._schedule_startup_locked(session)
                session.updated_at = utc_now()
                return self._session_response(session)

            session_id = f"wkr-{request.workspace_id[:8]}-{os.urandom(4).hex()}"
            workspace_root, workspace_files, sandbox_spec = self._resolve_workspace_root(request.workspace_id)
            workspace_env = self._normalize_workspace_env(request.workspace_env)
            if request.bridge_credential_mode == "worker_file":
                token = str(request.bridge_token_file_initial_token or "").strip()
                if not token or "RAGTIME_BRIDGE_TOKEN" in workspace_env:
                    raise HTTPException(status_code=400, detail="worker_file mode requires separate bridge token")
                workspace_env["RAGTIME_BRIDGE_TOKEN_FILE"] = "/run/.ragtime-bridge/token"
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
                mount_targets_to_clear=self._mount_target_paths(list(request.workspace_mounts or [])),
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
                updated_at=utc_now(),
                bridge_credential_mode=request.bridge_credential_mode,
                bridge_session_id=str(
                    (
                        self._decode_jwt_payload_metadata(
                            token if request.bridge_credential_mode == "worker_file" else workspace_env.get("RAGTIME_BRIDGE_TOKEN", "")
                        )
                        or {}
                    ).get("session_id")
                    or ""
                )
                or None,
                bridge_token_file_initial_token=(token if request.bridge_credential_mode == "worker_file" else None),
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
            session.updated_at = utc_now()
            return self._session_response(session)

    async def stop_session(self, worker_session_id: str, *, _maintenance_lease_id: str | None = None) -> WorkerSessionResponse:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            workspace_id = session.workspace_id
            # Check admission before any mutation: shared lease blocks public stop.
            self._ensure_workspace_available_locked(workspace_id, require_full_release=True, maintenance_lease_id=_maintenance_lease_id)
            # Fence the operation before cancelling it. A startup pipeline can
            # be between its off-lock spawn and its guarded commit; clearing
            # the operation id makes that commit terminate its new process.
            session.runtime_operation_id = None
            session.runtime_operation_updated_at = utc_now()
            startup_task = self._startup_tasks.pop(session.id, None)
            if startup_task and not startup_task.done():
                startup_task.cancel()
            devserver_process, log_handle = self._take_devserver_resources_locked(session.id)
            active_execs = tuple(self._active_execs.pop(session.id, {}).values())
            for key in [key for key in self._app_restart_requests if key[0] == session.id]:
                del self._app_restart_requests[key]
            sandbox_spec = session.sandbox_spec
            cleanup_task = asyncio.create_task(
                self._finish_stop_cleanup(
                    worker_session_id=worker_session_id,
                    workspace_id=workspace_id,
                    startup_task=startup_task,
                    devserver_process=devserver_process,
                    log_handle=log_handle,
                    sandbox_spec=sandbox_spec,
                    active_execs=active_execs,
                )
            )
            # Publish the barrier before releasing the global lock so a newly
            # queued start observes it before it can provision this workspace.
            self._workspace_cleanup_tasks[workspace_id] = cleanup_task

        await asyncio.shield(cleanup_task)
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            return self._session_response(session)

    async def _finish_stop_cleanup(
        self,
        *,
        worker_session_id: str,
        workspace_id: str,
        startup_task: asyncio.Task[None] | None,
        devserver_process: asyncio.subprocess.Process | None,
        log_handle: Any | None,
        sandbox_spec: SandboxSpec,
        active_execs: tuple[Any, ...] = (),
    ) -> None:
        current_task = asyncio.current_task()
        try:
            if startup_task:
                try:
                    await startup_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Cancelled startup task failed during workspace cleanup")

            # Keep the workspace fence while cleanup runs, but never the global
            # state lock: unrelated status and preview requests remain responsive.
            workspace_lock = self._workspace_startup_lock(workspace_id)
            async with workspace_lock:
                async with self._workspace_file_lock(workspace_id):
                    for process in active_execs:
                        if hasattr(process, "communicate"):
                            with contextlib.suppress(Exception):
                                await terminate_process_group(process)
                    await self._terminate_devserver_resources(devserver_process, log_handle)
                    cleanup_thread = asyncio.create_task(asyncio.to_thread(cleanup_sandbox, sandbox_spec))
                    try:
                        await asyncio.shield(cleanup_thread)
                    except asyncio.CancelledError:
                        # Do not release either filesystem fence until the
                        # cleanup thread has stopped touching the sandbox root.
                        await asyncio.shield(cleanup_thread)
                        raise

                async with self._lock:
                    session = self._sessions.get(worker_session_id)
                    if session and session.runtime_operation_id is None:
                        session.state = "stopped"
                        session.devserver_running = False
                        session.last_error = None
                        self._set_operation_phase(session, "stopped")
                        session.updated_at = utc_now()
        finally:
            async with self._lock:
                if self._workspace_cleanup_tasks.get(workspace_id) is current_task:
                    self._workspace_cleanup_tasks.pop(workspace_id, None)

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
                if session.bridge_credential_mode == "worker_file":
                    # File-mode invariants survive env replacement: the raw token
                    # never enters the process env, and the app keeps the
                    # platform-fixed token-file path.
                    session.workspace_env.pop("RAGTIME_BRIDGE_TOKEN", None)
                    session.workspace_env["RAGTIME_BRIDGE_TOKEN_FILE"] = "/run/.ragtime-bridge/token"
            if workspace_env is not None or workspace_env_visibility is not None:
                session.workspace_env_visibility = self._normalize_workspace_env_visibility(
                    workspace_env_visibility,
                    session.workspace_env,
                )
            previous_targets = self._mount_target_paths(session.workspace_mounts)
            if workspace_mounts is not None:
                session.workspace_mounts = list(workspace_mounts)
            session.mount_targets_to_clear = previous_targets | self._mount_target_paths(session.workspace_mounts)
            # _schedule_startup_locked clears devserver_port so the pipeline picks fresh
            self._schedule_startup_locked(session)
            session.updated_at = utc_now()
            return self._session_response(session)

    async def restart_app(self, worker_session_id: str, request_id: str) -> WorkerSessionResponse:
        """Recycle only the devserver while retaining the worker session."""
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            key = (worker_session_id, request_id)
            existing = self._app_restart_requests.get(key)
            if existing is not None:
                return existing
            if self._active_execs.get(worker_session_id):
                raise HTTPException(status_code=409, detail="Runtime exec is active")
            startup = self._startup_tasks.get(worker_session_id)
            if startup and not startup.done():
                raise HTTPException(status_code=409, detail="Runtime startup is active")
            self._schedule_startup_locked(session, replace_existing=False)
            response = self._session_response(session)
            self._app_restart_requests[key] = response
            return response

    async def refresh_mounts(
        self,
        worker_session_id: str,
        workspace_mounts: list[dict[str, Any]],
        *,
        replace: bool = False,
    ) -> WorkerSessionResponse:
        mount_specs = []
        target_paths: set[str] = set()
        for mount in workspace_mounts:
            target_path = str(mount.get("target_path") or "").strip()
            if not target_path:
                continue
            mount_specs.append(dict(mount))
            target_paths.add(target_path)
        if not mount_specs and not replace:
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

        workspace_lock = self._workspace_startup_lock(workspace_id)
        async with workspace_lock:
            try:
                async with self._workspace_file_lock(workspace_id):
                    # Publish mount metadata under the same fence as its
                    # materialization, so queued file operations recapture a
                    # coherent root and read-only policy.
                    async with self._lock:
                        session = self._sessions.get(worker_session_id)
                        if not session:
                            raise HTTPException(status_code=404, detail="Worker session not found")
                        if replace:
                            previous_targets = self._mount_target_paths(session.workspace_mounts)
                            clear_target_paths = previous_targets | target_paths
                            session.workspace_mounts = mount_specs
                        else:
                            clear_target_paths = set(target_paths)
                            mounts_by_target: dict[str, dict[str, Any]] = {}
                            for existing_mount in session.workspace_mounts:
                                existing_target = str(existing_mount.get("target_path") or "").strip()
                                if existing_target:
                                    mounts_by_target[existing_target] = dict(existing_mount)
                            for mount in mount_specs:
                                mounts_by_target[str(mount.get("target_path") or "").strip()] = mount
                            session.workspace_mounts = list(mounts_by_target.values())
                        session.mount_targets_to_clear |= clear_target_paths
                        session.updated_at = utc_now()
                    await self._materialize_workspace_mounts(session)
            except Exception as exc:
                error_message = f"Failed to refresh workspace mounts: {exc}"
                async with self._lock:
                    current = self._sessions.get(worker_session_id)
                    if current:
                        current.last_error = error_message
                        current.updated_at = utc_now()
                raise HTTPException(status_code=500, detail=error_message) from exc

            async with self._lock:
                current = self._sessions.get(worker_session_id)
                if not current:
                    raise HTTPException(status_code=404, detail="Worker session not found")
                current.mount_targets_to_clear.difference_update(clear_target_paths)
                current.last_error = None
                current.updated_at = utc_now()
                return self._session_response(current)

    async def refresh_bridge_credential(
        self,
        worker_session_id: str,
        *,
        token: str,
        expected_session_id: str,
        expected_revision: int,
        request_id: str,
    ) -> RuntimeBridgeCredentialMetadata:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()
            previous = session.bridge_refresh_requests.get(request_id)
            if previous:
                if previous[0] != fingerprint:
                    raise HTTPException(status_code=409, detail="Credential request id payload conflict")
                return previous[1]
            if session.bridge_credential_mode != "worker_file":
                raise HTTPException(status_code=409, detail="Worker file credential mode is not active")
            if session.bridge_session_id != expected_session_id or session.bridge_credential_revision != expected_revision:
                raise HTTPException(status_code=409, detail="Bridge credential session or revision conflict")
            self._write_bridge_token_file(session, token)
            # Reprovision/recycle must retain the newest credential, not the
            # original startup delivery token.
            session.bridge_token_file_initial_token = token
            session.bridge_recent_tokens = ([token] + session.bridge_recent_tokens)[:2]
            session.bridge_credential_revision += 1
            metadata = self._bridge_credential_metadata(session)
            if metadata is None:
                raise HTTPException(status_code=400, detail="Invalid bridge credential")
            session.bridge_refresh_requests[request_id] = (fingerprint, metadata)
            if len(session.bridge_refresh_requests) > _MAX_BRIDGE_REFRESH_REQUEST_HISTORY:
                session.bridge_refresh_requests.pop(next(iter(session.bridge_refresh_requests)))
            return metadata

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
            workspace_id = session.workspace_id
            self._ensure_workspace_available_locked(workspace_id)

        file_lock = self._workspace_file_lock(workspace_id)
        async with file_lock:
            async with self._lock:
                session, root, root_relative_path, operation_id = self._capture_file_target_locked(worker_session_id, rel_path, mutation=False)
            io_task = asyncio.create_task(asyncio.to_thread(secure_read_text, root, root_relative_path))
            try:
                content = await self._drain_file_io_task(io_task)
            except asyncio.CancelledError:
                raise
        async with self._lock:
            current = self._sessions.get(worker_session_id)
            if current is not None and current is session and current.runtime_operation_id == operation_id:
                current.updated_at = utc_now()
            response_session = session
        if content is None:
            return self._runtime_file_response(response_session, rel_path, "", False)
        return self._runtime_file_response(response_session, rel_path, content, True)

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
            workspace_id = session.workspace_id
            self._ensure_workspace_available_locked(workspace_id)

        file_lock = self._workspace_file_lock(workspace_id)
        async with file_lock:
            async with self._lock:
                session, root, root_relative_path, operation_id = self._capture_file_target_locked(worker_session_id, rel_path, mutation=True)
            io_task = asyncio.create_task(asyncio.to_thread(secure_write_text, root, root_relative_path, content))
            try:
                await self._drain_file_io_task(io_task)
            except SecureFileError as exc:
                raise HTTPException(status_code=403, detail="Unsafe workspace file path") from exc
            except OSError as exc:
                if self._is_unsafe_file_error(exc):
                    raise HTTPException(status_code=403, detail="Unsafe workspace file path") from exc
                raise
        async with self._lock:
            current = self._sessions.get(worker_session_id)
            if current is not None and current is session and current.runtime_operation_id == operation_id:
                current.updated_at = utc_now()
            response_session = session
        return self._runtime_file_response(response_session, rel_path, content, True)

    async def delete_file(self, worker_session_id: str, file_path: str) -> dict[str, str | bool]:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            rel_path = self._normalize_file_path(
                file_path,
                enforce_sqlite_managed=True,
            )
            workspace_id = session.workspace_id
            self._ensure_workspace_available_locked(workspace_id)

        file_lock = self._workspace_file_lock(workspace_id)
        async with file_lock:
            async with self._lock:
                session, root, root_relative_path, operation_id = self._capture_file_target_locked(worker_session_id, rel_path, mutation=True)
            io_task = asyncio.create_task(asyncio.to_thread(secure_delete_file, root, root_relative_path))
            try:
                await self._drain_file_io_task(io_task)
            except SecureFileError as exc:
                raise HTTPException(status_code=403, detail="Unsafe workspace file path") from exc
            except OSError as exc:
                if self._is_unsafe_file_error(exc):
                    raise HTTPException(status_code=403, detail="Unsafe workspace file path") from exc
                raise
        async with self._lock:
            current = self._sessions.get(worker_session_id)
            if current is not None and current is session and current.runtime_operation_id == operation_id:
                current.updated_at = utc_now()
        return {"success": True, "path": rel_path}

    _EXEC_MAX_OUTPUT_BYTES = 60_000

    async def exec_command(
        self,
        worker_session_id: str,
        command: str,
        timeout_seconds: int = 120,
        cwd: str | None = None,
    ) -> RuntimeExecResponse:
        """Execute a shell command in the workspace sandbox."""
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")
            self._ensure_workspace_available_locked(session.workspace_id)

            # Resolve cwd as a sandbox-internal path
            if cwd:
                # Validate cwd doesn't escape workspace
                normalized_cwd = Path(cwd.replace("\\", "/"))
                if normalized_cwd.is_absolute() or any(part == ".." for part in normalized_cwd.parts):
                    raise HTTPException(
                        status_code=400,
                        detail="cwd must be within the workspace root",
                    )
                sandbox_cwd = f"{SANDBOX_WORKSPACE_MOUNT}/{normalized_cwd}"
            else:
                sandbox_cwd = SANDBOX_WORKSPACE_MOUNT

            sandbox_spec = session.sandbox_spec
            agent_process_env = self.build_agent_process_environment(session)
            reservation = object()
            self._active_execs.setdefault(worker_session_id, {})[id(reservation)] = reservation

        timeout_seconds = max(1, min(timeout_seconds, RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS))
        timed_out = False
        truncated = False

        process: asyncio.subprocess.Process | None = None
        try:
            process = await spawn_sandboxed(
                sandbox_spec,
                ["sh", "-lc", command],
                cwd=sandbox_cwd,
                env=agent_process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with self._lock:
                active = self._active_execs.setdefault(worker_session_id, {})
                active.pop(id(reservation), None)
                active[id(process)] = process
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                await terminate_process_group(process)  # type: ignore[arg-type]
            except Exception:
                pass
            stdout_bytes = b""
            stderr_bytes = f"Command timed out after {timeout_seconds}s".encode()
        except asyncio.CancelledError:
            if process is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(terminate_process_group(process))
            raise
        except Exception as exc:
            return RuntimeExecResponse(
                exit_code=-1,
                stdout="",
                stderr=f"Failed to execute command: {exc}",
                timed_out=False,
                truncated=False,
            )
        finally:
            async with self._lock:
                active = self._active_execs.get(worker_session_id, {})
                active.pop(id(reservation), None)
                if process is not None:
                    active.pop(id(process), None)

        exit_code = process.returncode if process is not None and process.returncode is not None else -1

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
                self._schedule_startup_locked(session, replace_existing=False)
            if not session.devserver_running or not session.devserver_port:
                raise HTTPException(
                    status_code=503,
                    detail=session.last_error or "Dev server is starting. Retry screenshot when runtime operation is ready.",
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
                    detail=("capture_element=true requires wait_for_selector to target a unique visible element"),
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
                normalized_preview_path = self._normalize_file_path(normalized_preview_path)
            output_dir = self._workspace_screenshot_dir(session.workspace_root)
            output_dir.mkdir(parents=True, exist_ok=True)

        mcp_result = await self._invoke_mcp_tool(
            session,
            "playwright",
            "playwright_capture_screenshot",
            {
                "path": normalized_preview_path,
                "width": width,
                "height": height,
                "full_page": bool(payload.full_page),
                "timeout_ms": int(payload.timeout_ms),
                "wait_for_selector": wait_selector,
                "capture_element": capture_element,
                "clip_padding_px": requested_clip_padding_px,
                "wait_after_load_ms": requested_wait_after_load_ms,
                "refresh_before_capture": bool(payload.refresh_before_capture),
            },
            timeout_ms=int(payload.timeout_ms),
            screenshot_dir=output_dir,
        )
        probe = mcp_result.response
        output_path = Path(str(probe.get("output_path") or ""))

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
            retry_mcp_result = await self._invoke_mcp_tool(
                session,
                "playwright",
                "playwright_capture_screenshot",
                {
                    "path": normalized_preview_path,
                    "width": width,
                    "height": height,
                    "full_page": bool(payload.full_page),
                    "timeout_ms": int(payload.timeout_ms),
                    "wait_for_selector": wait_selector,
                    "capture_element": capture_element,
                    "clip_padding_px": requested_clip_padding_px,
                    "wait_after_load_ms": retry_wait_after_load_ms,
                    "refresh_before_capture": bool(payload.refresh_before_capture),
                },
                timeout_ms=int(payload.timeout_ms),
                screenshot_dir=output_dir,
            )
            retry_probe = retry_mcp_result.response
            if isinstance(retry_probe, dict):
                probe = retry_probe
                mcp_result = retry_mcp_result
                output_path = Path(str(probe.get("output_path") or output_path))

        if isinstance(probe, dict):
            probe["startup_blank_detected"] = startup_blank_detected
            probe["startup_retry_attempted"] = startup_retry_attempted

        if not output_path.exists() or not output_path.is_file():
            raise HTTPException(
                status_code=502,
                detail=("Runtime screenshot capture reported success but no file was written"),
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
                "effective_wait_after_load_ms": int((probe.get("effective_wait_after_load_ms") or requested_wait_after_load_ms)),
                "refresh_before_capture": bool(payload.refresh_before_capture),
            },
            probe=probe if isinstance(probe, dict) else {},
            mcp=mcp_result.model_dump(mode="json"),
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
                self._schedule_startup_locked(session, replace_existing=False)
            if not session.devserver_running or not session.devserver_port:
                raise HTTPException(
                    status_code=503,
                    detail=session.last_error or "Dev server is starting. Retry when runtime is ready.",
                )

            normalized_preview_path = (payload.path or "").strip().lstrip("/")
            if normalized_preview_path:
                normalized_preview_path = self._normalize_file_path(normalized_preview_path)

        mcp_result = await self._invoke_mcp_tool(
            session,
            "playwright",
            "playwright_content_probe",
            {
                "path": normalized_preview_path,
                "timeout_ms": int(payload.timeout_ms),
                "wait_after_load_ms": int(payload.wait_after_load_ms),
                "inject_mock_context": bool(getattr(payload, "inject_mock_context", False)),
            },
            timeout_ms=int(payload.timeout_ms),
        )
        probe = mcp_result.response

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
            mcp=mcp_result.model_dump(mode="json"),
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
                link_models.append(RuntimeExternalBrowseLink(url=url_value, text=text_value))

        console_errors_raw = probe.get("console_errors") if isinstance(probe, dict) else None
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
        session: WorkerSession | None = None
        if worker_session_id:
            async with self._lock:
                session = self._sessions.get(worker_session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Worker session not found")
                if session.state not in {"running", "starting"}:
                    raise HTTPException(status_code=409, detail="Worker session not active")
                session.updated_at = utc_now()

        mcp_result = await self._invoke_mcp_tool(
            session,
            "playwright",
            "playwright_external_browse",
            {
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
        response = self._build_external_browse_response(mcp_result.response, payload)
        response.mcp = mcp_result.model_dump(mode="json")
        return response

    async def call_mcp_tool(
        self,
        worker_session_id: str,
        payload: RuntimeMcpToolCallRequest,
    ) -> RuntimeMcpToolCallResponse:
        spec = self._resolve_mcp_spec(payload.server_name)
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(status_code=409, detail="Worker session not active")
            screenshot_dir: Path | None = None
            if spec.requires_devserver:
                if not session.devserver_running or not session.devserver_port:
                    await self._sync_devserver_state_locked(session)
                if not session.devserver_running or not session.devserver_port:
                    raise HTTPException(
                        status_code=503,
                        detail=session.last_error or "Dev server is starting. Retry when runtime is ready.",
                    )
            session.updated_at = utc_now()
            if spec.inject_preview_context:
                screenshot_dir = self._workspace_screenshot_dir(session.workspace_root)
                screenshot_dir.mkdir(parents=True, exist_ok=True)

        return await self._invoke_mcp_tool(
            session,
            payload.server_name,
            payload.tool_name,
            payload.arguments,
            timeout_ms=int(payload.timeout_ms),
            screenshot_dir=screenshot_dir,
        )

    @staticmethod
    def _is_pdf_content_type(value: str) -> bool:
        content_type = (value or "").split(";", 1)[0].strip().lower()
        return content_type in _PDF_CONTENT_TYPES or content_type.endswith("+pdf")

    @staticmethod
    def _compact_pdf_text(value: str) -> str:
        lines = [line.rstrip() for line in str(value or "").splitlines()]
        compacted: list[str] = []
        blank_pending = False
        for line in lines:
            if line.strip():
                if blank_pending and compacted:
                    compacted.append("")
                compacted.append(line)
                blank_pending = False
            else:
                blank_pending = True
        return "\n".join(compacted).strip()

    @staticmethod
    def _compact_response_preview(value: str) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return " ".join(text.split())[:500]

    @classmethod
    async def _response_text_preview(
        cls,
        response: httpx.Response,
        *,
        max_bytes: int = 4096,
    ) -> str:
        content = bytearray()
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            remaining = max_bytes - len(content)
            if remaining <= 0:
                break
            content.extend(chunk[:remaining])
            if len(content) >= max_bytes:
                break
        if not content:
            return ""
        encoding = response.encoding or "utf-8"
        try:
            decoded = bytes(content).decode(encoding, errors="replace")
        except LookupError:
            decoded = bytes(content).decode("utf-8", errors="replace")
        return cls._compact_response_preview(decoded)

    @staticmethod
    def _pdf_http_error_message(status_code: int, body_preview: str) -> str:
        if status_code in {401, 403}:
            base = f"Upstream server denied PDF request with HTTP {status_code}"
        elif status_code == 404:
            base = "Upstream PDF URL was not found (HTTP 404)"
        else:
            base = f"Upstream PDF request failed with HTTP {status_code}"
        if body_preview:
            return f"{base}: {body_preview[:220]}"
        return base

    @staticmethod
    def _slice_pdf_text(text: str, *, start_char: int, max_chars: int) -> dict[str, Any]:
        text_length = len(text)
        bounded_start = max(0, min(int(start_char), text_length))
        end_char = min(text_length, bounded_start + max_chars)
        return {
            "text": text[bounded_start:end_char],
            "text_start_char": bounded_start,
            "text_end_char": end_char,
            "text_length": text_length,
            "truncated": end_char < text_length,
        }

    @staticmethod
    def _build_pdf_query_matches(
        text: str,
        *,
        query: str,
        max_matches: int,
        max_chars: int,
    ) -> list[RuntimePdfReadMatch]:
        cleaned_query = " ".join(str(query or "").split())
        if not cleaned_query:
            return []
        patterns = [re.compile(re.escape(cleaned_query), re.IGNORECASE)]
        terms = [term for term in re.split(r"\W+", cleaned_query) if len(term) >= 3]
        if len(terms) > 1:
            patterns.append(re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE))

        found: list[re.Match[str]] = []
        seen_starts: set[int] = set()
        for pattern in patterns:
            for match in pattern.finditer(text):
                if match.start() in seen_starts:
                    continue
                seen_starts.add(match.start())
                found.append(match)
                if len(found) >= max_matches:
                    break
            if found:
                break
        if not found:
            return []

        per_match_budget = max(200, max_chars // max(1, len(found)))
        context_chars = max(80, min(1200, per_match_budget // 2))
        matches: list[RuntimePdfReadMatch] = []
        for match in found[:max_matches]:
            snippet_start = max(0, match.start() - context_chars)
            snippet_end = min(len(text), match.end() + context_chars)
            matches.append(
                RuntimePdfReadMatch(
                    match_start_char=match.start(),
                    match_end_char=match.end(),
                    snippet_start_char=snippet_start,
                    snippet_end_char=snippet_end,
                    text=text[snippet_start:snippet_end],
                )
            )
        return matches

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        try:
            from ragtime.core.document_conversion import convert_document_bytes
        except ImportError as exc:
            raise RuntimeError("Runtime document conversion adapter is not available") from exc

        result = convert_document_bytes(content, ".pdf")
        failure = getattr(result, "failure", None)
        if failure is None:
            return str(getattr(result, "text", "") or "")

        failure_name = getattr(failure, "name", None) or str(failure)
        if failure_name in ("UNSUPPORTED", "NEEDS_OCR"):
            return ""

        detail = str(getattr(result, "detail", "") or "").strip()
        if detail:
            raise RuntimeError(detail)
        raise RuntimeError(f"Document conversion failed ({failure_name})")

    async def read_pdf(
        self,
        payload: RuntimePdfReadRequest,
        worker_session_id: str | None = None,
    ) -> RuntimePdfReadResponse:
        if worker_session_id:
            async with self._lock:
                session = self._sessions.get(worker_session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Worker session not found")
                if session.state not in {"running", "starting"}:
                    raise HTTPException(status_code=409, detail="Worker session not active")
                session.updated_at = utc_now()

        requested_user_agent = str(payload.user_agent or "").strip()
        user_agent_candidates: list[str | None] = []
        if requested_user_agent:
            user_agent_candidates.append(requested_user_agent)
        else:
            user_agent_candidates.append(None)
        for candidate in (None, _PDF_READ_USER_AGENT):
            if candidate not in user_agent_candidates:
                user_agent_candidates.append(candidate)

        timeout = httpx.Timeout(max(10.0, min(60.0, int(payload.max_bytes) / 512_000)))
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                for index, user_agent in enumerate(user_agent_candidates):
                    headers = {
                        "Accept": "application/pdf,application/octet-stream;q=0.8,*/*;q=0.2",
                    }
                    if user_agent:
                        headers["User-Agent"] = user_agent

                    async with client.stream("GET", payload.url, headers=headers) as response:
                        content_type = response.headers.get("content-type") or ""
                        content_length = response.headers.get("content-length") or ""
                        final_url = str(response.url)
                        if response.status_code >= 400:
                            body_preview = await self._response_text_preview(response)
                            error_response = RuntimePdfReadResponse(
                                status="error",
                                ok=False,
                                requested_url=payload.url,
                                url=final_url,
                                status_code=response.status_code,
                                content_type=content_type,
                                byte_limit=int(payload.max_bytes),
                                failure_mode="upstream_http_error",
                                body_preview=body_preview,
                                error=self._pdf_http_error_message(
                                    response.status_code,
                                    body_preview,
                                ),
                            )
                            if response.status_code in _PDF_READ_RETRY_STATUS_CODES and index < len(user_agent_candidates) - 1:
                                continue
                            return error_response

                        try:
                            declared_length = int(content_length)
                        except (TypeError, ValueError):
                            declared_length = 0
                        if declared_length > int(payload.max_bytes):
                            return RuntimePdfReadResponse(
                                status="too_large",
                                ok=False,
                                requested_url=payload.url,
                                url=final_url,
                                status_code=response.status_code,
                                content_type=content_type,
                                byte_limit=int(payload.max_bytes),
                                declared_bytes=declared_length,
                                failure_mode="pdf_too_large",
                                error="PDF exceeds web_read_pdf byte limit",
                            )

                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            content.extend(chunk)
                            if len(content) > int(payload.max_bytes):
                                return RuntimePdfReadResponse(
                                    status="too_large",
                                    ok=False,
                                    requested_url=payload.url,
                                    url=final_url,
                                    status_code=response.status_code,
                                    content_type=content_type,
                                    byte_limit=int(payload.max_bytes),
                                    downloaded_bytes=len(content),
                                    failure_mode="pdf_too_large",
                                    error="PDF exceeds web_read_pdf byte limit",
                                )
                        break
                else:
                    return RuntimePdfReadResponse(
                        status="error",
                        ok=False,
                        requested_url=payload.url,
                        url=payload.url,
                        byte_limit=int(payload.max_bytes),
                        failure_mode="request_error",
                        error="PDF request failed before any response was read",
                    )
        except Exception as exc:
            return RuntimePdfReadResponse(
                status="error",
                ok=False,
                requested_url=payload.url,
                url=payload.url,
                byte_limit=int(payload.max_bytes),
                failure_mode="request_error",
                error=f"PDF request failed ({exc.__class__.__name__}): {str(exc)[:200]}",
            )

        content_bytes = bytes(content)
        has_pdf_magic = content_bytes.lstrip()[:5] == b"%PDF-"
        if not has_pdf_magic and not self._is_pdf_content_type(content_type):
            return RuntimePdfReadResponse(
                status="not_pdf",
                ok=False,
                requested_url=payload.url,
                url=final_url,
                status_code=response.status_code,
                content_type=content_type,
                byte_count=len(content_bytes),
                byte_limit=int(payload.max_bytes),
                failure_mode="not_pdf",
                error="URL did not return PDF content",
            )

        try:
            text = self._compact_pdf_text(await asyncio.to_thread(self._extract_pdf_text, content_bytes))
        except Exception as exc:
            return RuntimePdfReadResponse(
                status="error",
                ok=False,
                requested_url=payload.url,
                url=final_url,
                status_code=response.status_code,
                content_type=content_type,
                byte_count=len(content_bytes),
                byte_limit=int(payload.max_bytes),
                failure_mode="pdf_extract_error",
                error=f"PDF extraction failed ({exc.__class__.__name__}): {str(exc)[:200]}",
            )

        if not text:
            return RuntimePdfReadResponse(
                status="empty",
                ok=False,
                requested_url=payload.url,
                url=final_url,
                status_code=response.status_code,
                content_type=content_type,
                byte_count=len(content_bytes),
                byte_limit=int(payload.max_bytes),
                failure_mode="empty_pdf_text",
                error="No text could be extracted from PDF",
            )

        query = " ".join(str(payload.query or "").split())
        if query:
            matches = self._build_pdf_query_matches(
                text,
                query=query,
                max_matches=int(payload.max_matches),
                max_chars=int(payload.max_chars),
            )
            joined = "\n\n---\n\n".join(match.text for match in matches)
            returned_text = joined[: int(payload.max_chars)]
            return RuntimePdfReadResponse(
                status="ok",
                ok=True,
                requested_url=payload.url,
                url=final_url,
                status_code=response.status_code,
                content_type=content_type,
                byte_count=len(content_bytes),
                byte_limit=int(payload.max_bytes),
                text=returned_text,
                text_length=len(text),
                truncated=len(joined) > int(payload.max_chars),
                query=query,
                max_chars=int(payload.max_chars),
                matches=matches,
                match_count=len(matches),
            )

        sliced = self._slice_pdf_text(
            text,
            start_char=int(payload.start_char),
            max_chars=int(payload.max_chars),
        )
        return RuntimePdfReadResponse(
            status="ok",
            ok=True,
            requested_url=payload.url,
            url=final_url,
            status_code=response.status_code,
            content_type=content_type,
            byte_count=len(content_bytes),
            byte_limit=int(payload.max_bytes),
            max_chars=int(payload.max_chars),
            **sliced,
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
                self._schedule_startup_locked(session, replace_existing=False)

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
            upstream_url = f"{upstream_base}/{normalized}" if normalized else f"{upstream_base}/"
            if query:
                upstream_url = f"{upstream_url}?{query}"
        return upstream_url

    async def verify_pty_token(self, worker_session_id: str, token: str) -> WorkerSession:
        async with self._lock:
            session = self._sessions.get(worker_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Worker session not found")
            if token != session.pty_access_token:
                raise HTTPException(status_code=403, detail="Invalid PTY token")
            return session

    async def shutdown(self) -> None:
        async with self._lock:
            startup_tasks = list(self._startup_tasks.values())
            self._startup_tasks.clear()
            for task in startup_tasks:
                if task and not task.done():
                    task.cancel()
            devserver_resources = [self._take_devserver_resources_locked(sid) for sid in list(self._devserver_processes)]
            cleanup_tasks = [
                *self._workspace_cleanup_tasks.values(),
                *self._background_cleanup_tasks,
            ]
        for task in startup_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for process, log_handle in devserver_resources:
            await self._terminate_devserver_resources(process, log_handle)
        if cleanup_tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in cleanup_tasks), return_exceptions=True)
        # Tear down warm Playwright MCP connections outside the lock so owner
        # task cancellation (and stdio subprocess teardown) cannot deadlock on it.
        await self._terminate_mcp_brokers()

    async def health(self) -> WorkerHealthResponse:
        async with self._lock:
            for session in self._sessions.values():
                await self._sync_devserver_state_locked(session)
            active_sessions = sum(1 for session in self._sessions.values() if session.state in {"running", "starting"})
            return WorkerHealthResponse(
                status="ok",
                service_mode="worker",
                active_sessions=active_sessions,
                metadata={
                    "worker_name": self._worker_name,
                    "runtime_capabilities": {"bridge_credential_file": True, "sqlite_workspace_maintenance": True},
                    "bridge_credential_file": True,
                    "sqlite_workspace_maintenance": True,
                    **sandbox_diagnostics(),
                },
            )


@lru_cache(maxsize=1)
def get_worker_service() -> WorkerService:
    return WorkerService()
