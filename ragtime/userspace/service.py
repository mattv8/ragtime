from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time as _time
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException
from jose import JWTError, jwt  # type: ignore[import-untyped]
from prisma import Json
from prisma import fields as prisma_fields

from ragtime.config import settings
from ragtime.core.app_settings import SettingsCache
from ragtime.core.auth import _get_ldap_connection, get_ldap_config
from ragtime.core.database import get_db
from ragtime.core.encryption import (CONNECTION_CONFIG_PASSWORD_FIELDS,
                                     decrypt_json_passwords, decrypt_secret,
                                     encrypt_json_passwords, encrypt_secret)
from ragtime.core.entrypoint_status import (EntrypointStatus,
                                            parse_entrypoint_config)
from ragtime.core.git import create_repository, parse_git_url
from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (DB_TYPE_POSTGRES,
                                    add_table_metadata_to_psql_output,
                                    enforce_max_results, format_query_result,
                                    validate_sql_query)
from ragtime.core.ssh import (USERSPACE_MOUNT_WATCH_INTERVAL_SECONDS,
                              USERSPACE_MOUNT_WATCH_JITTER_SECONDS, SSHTunnel,
                              build_ssh_tunnel_config,
                              check_remote_rsync_available,
                              execute_ssh_command, is_rsync_missing_error,
                              preview_ssh_directory_sync, rsync_ssh_directory,
                              ssh_config_from_dict,
                              ssh_tunnel_config_from_dict, sync_ssh_directory)
from ragtime.indexer.file_utils import build_authenticated_git_url
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.models import FilesystemConnectionConfig
from ragtime.indexer.repository import repository
from ragtime.rag.prompts import build_workspace_scm_setup_prompt
from ragtime.userspace.models import (
    ArtifactType, BrowseUserspaceMountSourceRequest,
    CreateUserspaceMountSourceRequest,
    CreateUserSpaceObjectStorageBucketRequest, CreateWorkspaceMountRequest,
    CreateWorkspaceRequest, DeleteUserspaceMountSourceResponse,
    DeleteUserSpaceObjectStorageBucketResponse, DeleteWorkspaceEnvVarResponse,
    DeleteWorkspaceMountResponse, ExecuteComponentRequest,
    ExecuteComponentResponse, MountableSource, MountSourceAffectedWorkspace,
    MountSourceAffectedWorkspacesResponse, PaginatedWorkspacesResponse,
    ShareAccessMode, SqliteImportResponse, SqlitePersistenceMode,
    SwitchSnapshotBranchRequest, UpdateSnapshotRequest,
    UpdateUserspaceMountSourceRequest,
    UpdateUserSpaceObjectStorageBucketRequest, UpdateWorkspaceMembersRequest,
    UpdateWorkspaceMountRequest, UpdateWorkspaceRequest,
    UpdateWorkspaceShareAccessRequest, UpsertWorkspaceEnvVarRequest,
    UpsertWorkspaceFileRequest, UserSpaceFileInfo, UserSpaceFileResponse,
    UserSpaceLiveDataCheck, UserSpaceLiveDataConnection, UserspaceMountBackend,
    UserspaceMountSource, UserspaceMountSourceType,
    UserSpaceObjectStorageBucket, UserSpaceObjectStorageConfig,
    UserSpaceSharedPreviewResponse, UserSpaceSnapshot, UserSpaceSnapshotBranch,
    UserSpaceSnapshotDiffFileSummary, UserSpaceSnapshotDiffSummaryResponse,
    UserSpaceSnapshotFileDiffResponse, UserSpaceSnapshotTimelineResponse,
    UserSpaceWorkspace, UserSpaceWorkspaceEnvVar,
    UserSpaceWorkspaceScmConnectionRequest,
    UserSpaceWorkspaceScmConnectionResponse,
    UserSpaceWorkspaceScmExportRequest, UserSpaceWorkspaceScmImportRequest,
    UserSpaceWorkspaceScmPreviewRequest, UserSpaceWorkspaceScmPreviewResponse,
    UserSpaceWorkspaceScmStatus, UserSpaceWorkspaceScmSyncResponse,
    UserSpaceWorkspaceShareLink, UserSpaceWorkspaceShareLinkStatus,
    WorkspaceMember, WorkspaceMount, WorkspaceMountBrowseRequest,
    WorkspaceMountBrowseResponse, WorkspaceMountDirectoryEntry,
    WorkspaceMountSyncMode, WorkspaceMountSyncPreviewRequest,
    WorkspaceMountSyncPreviewResponse, WorkspaceMountSyncRequest,
    WorkspaceMountSyncResponse, WorkspaceScmDirection,
    WorkspaceScmPreviewState, WorkspaceScmProvider,
    WorkspaceShareSlugAvailabilityResponse)
from ragtime.userspace.preview_host import \
    invalidate_preview_sessions_for_workspace
from ragtime.userspace.sqlite_import import (_MAX_IMPORT_SIZE_BYTES,
                                             SqlImportResult,
                                             detect_binary_pg_dump,
                                             detect_sql_dialect,
                                             import_sql_to_sqlite)

logger = get_logger(__name__)

_FILE_LIST_CACHE_TTL_SECONDS = 2
_ENTRYPOINT_STATUS_CACHE_TTL_SECONDS = 300  # 5-minute TTL for entrypoint status
_CHANGED_FILE_ACK_MAX_ROWS_PER_WORKSPACE_USER = (
    2000  # Threshold to bound growth of UserSpaceChangedFileAcknowledgement table
)
_SHARE_PASSWORD_ACCESS_TOKEN_KIND = "userspace_share_password_access"
_SHARE_PASSWORD_ACCESS_TTL_SECONDS = 60 * 30


class _ExecutionProofRecord:
    """Server-side proof of a successful execute-component call."""

    __slots__ = ("component_id", "row_count", "timestamp", "query_hash")

    def __init__(
        self,
        component_id: str,
        row_count: int,
        timestamp: float,
        query_hash: str,
    ) -> None:
        self.component_id = component_id
        self.row_count = row_count
        self.timestamp = timestamp
        self.query_hash = query_hash


class _ShareAuthorizationResult(TypedDict):
    workspace_id: str
    share_auth_token: str | None
    expires_at: datetime | None


class _GitCommandResult:
    """Async git command result payload."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _WorkspaceMountSyncPreviewRecord:
    """One-time destructive sync preview token state."""

    __slots__ = (
        "token",
        "workspace_id",
        "mount_id",
        "sync_mode",
        "state_fingerprint",
        "expires_at",
    )

    def __init__(
        self,
        *,
        token: str,
        workspace_id: str,
        mount_id: str,
        sync_mode: WorkspaceMountSyncMode,
        state_fingerprint: str,
        expires_at: datetime,
    ) -> None:
        self.token = token
        self.workspace_id = workspace_id
        self.mount_id = mount_id
        self.sync_mode = sync_mode
        self.state_fingerprint = state_fingerprint
        self.expires_at = expires_at


class _WorkspaceScmPreviewRecord:
    """One-time destructive SCM preview token state."""

    __slots__ = (
        "token",
        "workspace_id",
        "direction",
        "state_fingerprint",
        "expires_at",
    )

    def __init__(
        self,
        *,
        token: str,
        workspace_id: str,
        direction: WorkspaceScmDirection,
        state_fingerprint: str,
        expires_at: datetime,
    ) -> None:
        self.token = token
        self.workspace_id = workspace_id
        self.direction = direction
        self.state_fingerprint = state_fingerprint
        self.expires_at = expires_at


class _NonUtf8WorkspaceFileError(Exception):
    __slots__ = ("file_path",)

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        super().__init__(f"Non-UTF8 workspace file: {file_path}")


class _WorkspaceSnapshotFileDiff(TypedDict):
    path: str
    status: Literal["A", "D", "M", "R"]
    old_path: str | None
    additions: int
    deletions: int
    is_binary: bool
    is_untracked_in_current: bool


_EXECUTION_PROOF_MAX_AGE_SECONDS = 3600  # 1 hour
_SNAPSHOT_BRANCH_REF_PREFIX = "userspace/"
_GIT_EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf899d8e13f7beb2c"

_USPACE_EXEC_SUPPORTED_SQL_TOOLS = {"postgres", "mysql", "mssql", "influxdb"}
_USERSPACE_PREVIEW_ENTRY_PATH = "dashboard/main.ts"
_DEFAULT_SHARE_SLUG_PREFIX = "share"
_USERSPACE_PREVIEW_MAX_FILES = 200
_USERSPACE_PREVIEW_MAX_BYTES = 3_000_000
_SNAPSHOT_DIFF_MAX_FILE_BYTES = 1_048_576  # 1 MiB per side
_SQLITE_EXCLUDE_GLOBS = (
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.db3",
)
_WORKSPACE_DEFAULT_GITIGNORE_PATTERNS = (
    "node_modules/",
    "dist/",
    "__pycache__/",
)
_PLATFORM_MANAGED_GITIGNORE_PATTERNS = (
    ".ragtime/runtime-bootstrap.json",
    ".ragtime/.runtime-bootstrap.done",
)
_WORKSPACE_ENV_VAR_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORKSPACE_ENV_VAR_MAX_COUNT = 200
_WORKSPACE_SCM_PREVIEW_TTL_SECONDS = 300
_WORKSPACE_SCM_PREVIEW_SAMPLE_LIMIT = 100

_MODULE_SOURCE_EXTENSIONS = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".mts",
    ".cts",
)
_RUNTIME_BOOTSTRAP_CONFIG_PATH = ".ragtime/runtime-bootstrap.json"
_RUNTIME_BOOTSTRAP_TEMPLATE_VERSION = 7
_RUNTIME_BRIDGE_VERSION = 8
_RUNTIME_BRIDGE_VERSION_TAG = f"@ragtime/bridge v{_RUNTIME_BRIDGE_VERSION}"
_RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS = 310_000  # 300s + 10s buffer
_USERSPACE_TEMPLATES_DIR = Path(__file__).with_name("templates")
_WORKSPACE_OBJECT_STORAGE_DIRNAME = "s3"
_WORKSPACE_OBJECT_STORAGE_CONFIG_NAME = "config.json"
_WORKSPACE_OBJECT_STORAGE_REGION = "us-east-1"
_WORKSPACE_OBJECT_STORAGE_PUBLIC_PREFIX = "public"
_WORKSPACE_OBJECT_STORAGE_PRIVATE_PREFIX = "private"
_WORKSPACE_OBJECT_STORAGE_ENDPOINT_ENV_KEY = "RAGTIME_OBJECT_STORAGE_ENDPOINT"
_WORKSPACE_OBJECT_STORAGE_ACCESS_KEY_ENV_KEY = "RAGTIME_OBJECT_STORAGE_ACCESS_KEY_ID"
_WORKSPACE_OBJECT_STORAGE_SECRET_KEY_ENV_KEY = (
    "RAGTIME_OBJECT_STORAGE_SECRET_ACCESS_KEY"
)
_WORKSPACE_OBJECT_STORAGE_BUCKETS_ENV_KEY = "RAGTIME_OBJECT_STORAGE_BUCKETS_JSON"
_WORKSPACE_OBJECT_STORAGE_DEFAULT_BUCKET_ENV_KEY = (
    "RAGTIME_OBJECT_STORAGE_DEFAULT_BUCKET"
)
_WORKSPACE_OBJECT_STORAGE_FORCE_PATH_STYLE_ENV_KEY = (
    "RAGTIME_OBJECT_STORAGE_FORCE_PATH_STYLE"
)
_WORKSPACE_OBJECT_STORAGE_REGION_ENV_KEY = "RAGTIME_OBJECT_STORAGE_REGION"
_WORKSPACE_OBJECT_STORAGE_ENABLED_ENV_KEY = "RAGTIME_OBJECT_STORAGE_ENABLED"

# ---------------------------------------------------------------------------
# Node framework detection from command strings
# ---------------------------------------------------------------------------

_NODE_FRAMEWORK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnext\b"), "next"),
    (re.compile(r"\bnuxt\b"), "nuxt"),
    (re.compile(r"\bvite\b"), "vite"),
    (re.compile(r"\bexpress\b"), "express"),
]
_REPLIT_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_REPLIT_RUNTIME_KEYS = frozenset(
    {
        "run",
        "entrypoint",
        "localport",
        "externalport",
        "waitforport",
        "ignoreports",
    }
)
_REPLIT_NORMALIZATION_HEADER = [
    "# Ragtime import normalization:",
    "# .ragtime/runtime-entrypoint.json is the authoritative runtime config.",
    "# Ragtime's native preview proxy fronts the app directly, so Replit-only run/proxy/port directives are disabled below.",
]


def _guess_node_framework_from_command(command: str) -> str:
    """Return a framework hint from a shell command string, defaulting to ``node``."""
    lower = command.lower()
    for pattern, framework in _NODE_FRAMEWORK_PATTERNS:
        if pattern.search(lower):
            return framework
    return "node"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@lru_cache(maxsize=None)
def _load_userspace_template(template_name: str) -> str:
    return (_USERSPACE_TEMPLATES_DIR / template_name).read_text(encoding="utf-8")


def _render_userspace_template(
    template_name: str,
    replacements: dict[str, str] | None = None,
) -> str:
    content = _load_userspace_template(template_name)
    if not replacements:
        return content

    rendered = content
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _build_bridge_content(timeout_ms: int = _RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS) -> str:
    """Build the bridge.js content with a workspace-specific timeout."""
    timeout_seconds = max(timeout_ms // 1000, 60)
    return _render_userspace_template(
        "runtime_bridge.js",
        {
            "__RAGTIME_RUNTIME_BRIDGE_VERSION_TAG__": _RUNTIME_BRIDGE_VERSION_TAG,
            "__RAGTIME_RUNTIME_BRIDGE_TIMEOUT_MS__": str(timeout_ms),
            "__RAGTIME_RUNTIME_BRIDGE_TIMEOUT_LABEL__": f"{timeout_seconds}s",
        },
    )


def _compute_bridge_timeout_ms(selected_tool_ids: list[str] | None = None) -> int:
    """Derive the bridge postMessage timeout from workspace tool configs."""
    if not selected_tool_ids:
        return _RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS

    max_timeout = 300
    cached_configs = SettingsCache.get_instance()._tool_configs
    if cached_configs:
        id_set = set(selected_tool_ids)
        for cfg in cached_configs:
            if cfg.get("id") in id_set:
                timeout_seconds = cfg.get("timeout_max_seconds", 300) or 300
                if timeout_seconds > max_timeout:
                    max_timeout = timeout_seconds
    return (max(max_timeout, 300) + 10) * 1000


_SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
_SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})

_HIDDEN_DIRS = frozenset({".git", "node_modules", "__pycache__", ".ragtime", "dist"})
_AGENT_WRITABLE_RAGTIME_FILES = frozenset(
    {"runtime-entrypoint.json", "runtime-bootstrap.json"}
)
_AGENT_WRITABLE_RAGTIME_PREFIXES = ("db/migrations/", "scripts/")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError:
                return _utc_now()
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
    return _utc_now()


def _normalize_workspace_name_for_uniqueness(name: str) -> str:
    collapsed = re.sub(r"\s+", "_", (name or "").strip())
    return collapsed.lower()


def _is_workspace_name_conflict_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "owner_user_id_name_normalized" in message or (
        "unique" in message and "name_normalized" in message
    )


def _is_share_token_conflict_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unique" in message and (
        "sharetoken" in message or "share_token" in message or "share token" in message
    )


def _normalize_share_slug_for_uniqueness(slug: str) -> str:
    value = re.sub(r"\s+", "_", (slug or "").strip().lower())
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_-")[:80]


def _normalize_owner_username_for_share_path(username: str) -> str:
    value = (username or "").strip().lower()
    if value.startswith("local:"):
        value = value.split(":", 1)[1]
    return value


def _is_share_slug_conflict_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "owner_user_id_share_slug" in message or (
        "unique" in message and "share_slug" in message
    )


def _normalize_share_access_mode(value: str | None) -> ShareAccessMode:
    mode = (value or "token").strip().lower()
    allowed: set[str] = {
        "token",
        "password",
        "authenticated_users",
        "selected_users",
        "ldap_groups",
    }
    return cast(ShareAccessMode, mode if mode in allowed else "token")


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _requires_live_data_contract(
    relative_path: str,
    live_data_requested: bool,
    workspace_has_tools: bool = False,
) -> bool:
    normalized_path = (relative_path or "").strip().lower().replace("\\", "/")
    is_module_source = normalized_path.endswith(_MODULE_SOURCE_EXTENSIONS)
    if not is_module_source:
        return False

    # Auto-require live data contract only for the dashboard entry
    # module (dashboard/main.ts) when workspace has tools.  Helper
    # components under dashboard/ receive data as parameters and do
    # not need their own live data wiring.
    is_dashboard_entry = normalized_path == "dashboard/main.ts"
    if is_dashboard_entry and workspace_has_tools:
        return live_data_requested

    # All other module sources: only when explicitly requested.
    return live_data_requested


def _requires_entrypoint_wiring(
    relative_path: str,
    artifact_type: ArtifactType | None,
) -> bool:
    normalized_path = (relative_path or "").strip().lower().replace("\\", "/")
    if normalized_path == _USERSPACE_PREVIEW_ENTRY_PATH:
        return False

    is_module_source = normalized_path.endswith(_MODULE_SOURCE_EXTENSIONS)
    if not is_module_source:
        return False

    is_dashboard_module = artifact_type == "module_ts" or normalized_path.startswith(
        "dashboard/"
    )
    return is_dashboard_module


def _entrypoint_module_specifier_candidates(relative_path: str) -> list[str]:
    normalized_path = (relative_path or "").strip().replace("\\", "/")
    if not normalized_path.startswith("dashboard/"):
        return []

    module_rel = normalized_path[len("dashboard/") :]
    module_without_ext = module_rel
    for extension in _MODULE_SOURCE_EXTENSIONS:
        if module_without_ext.lower().endswith(extension):
            module_without_ext = module_without_ext[: -len(extension)]
            break

    candidates = [
        f"./{module_without_ext}",
        f"./{module_rel}",
    ]

    if "/" not in module_without_ext:
        candidates.extend(
            [
                f"./{module_without_ext}.ts",
                f"./{module_without_ext}.tsx",
                f"./{module_without_ext}.js",
                f"./{module_without_ext}.jsx",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def _entrypoint_references_module(main_content: str, candidates: list[str]) -> bool:
    if not main_content or not candidates:
        return False

    for candidate in candidates:
        if (
            f'"{candidate}"' in main_content
            or f"'{candidate}'" in main_content
            or f"`{candidate}`" in main_content
        ):
            return True
    return False


def _normalize_sqlite_persistence_mode(value: str | None) -> str:
    mode = (value or "exclude").strip().lower()
    return mode if mode in {"include", "exclude"} else "exclude"


def _is_sqlite_file_path(relative_path: str) -> bool:
    normalized = (relative_path or "").strip().replace("\\", "/").lstrip("/")
    if not normalized:
        return False
    return PurePosixPath(normalized).suffix.lower() in _SQLITE_FILE_EXTENSIONS


def _is_managed_sqlite_file_path(relative_path: str) -> bool:
    normalized = (relative_path or "").strip().replace("\\", "/").lstrip("/")
    return normalized.startswith(_SQLITE_MANAGED_DIR_PREFIX)


def _enforce_sqlite_file_path_policy(relative_path: str) -> None:
    if _is_sqlite_file_path(relative_path) and not _is_managed_sqlite_file_path(
        relative_path
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "SQLite persistence files must be managed under .ragtime/db/. "
                "Use paths like .ragtime/db/app.sqlite3."
            ),
        )


class UserSpaceService:
    _SSH_RSYNC_MISSING_RECHECK_SECONDS = 300.0
    _WORKSPACE_MOUNT_SYNC_PREVIEW_TTL_SECONDS = 120
    _WORKSPACE_MOUNT_SYNC_PREVIEW_SAMPLE_LIMIT = 200

    def __init__(self) -> None:
        self._base_dir = Path(settings.index_data_path) / "_userspace"
        self._workspaces_dir = self._base_dir / "workspaces"
        self._workspaces_dir.mkdir(parents=True, exist_ok=True)
        self._execution_proofs: dict[str, dict[str, _ExecutionProofRecord]] = {}
        # TTL-cached entrypoint status per workspace: {workspace_id: (EntrypointStatus, timestamp)}
        self._entrypoint_status_cache: dict[str, tuple[EntrypointStatus, float]] = {}
        # Short-lived file list cache: {workspace_id: (result, include_dirs, timestamp)}
        self._file_list_cache: dict[
            str, tuple[list[UserSpaceFileInfo], bool, float]
        ] = {}
        # Limit concurrent git status requests to avoid overloading process slots.
        self._git_status_semaphore = asyncio.Semaphore(8)
        # Limit concurrent mount sync jobs; each runs blocking SSH or rsync work
        # in a worker thread.
        self._workspace_mount_sync_semaphore = asyncio.Semaphore(
            self._positive_int_env("USERSPACE_MOUNT_SYNC_CONCURRENCY", 2)
        )
        self._workspace_mount_sync_tasks: dict[
            str, asyncio.Task[WorkspaceMountSyncResponse]
        ] = {}
        self._workspace_mount_sync_tasks_lock = asyncio.Lock()
        self._workspace_mount_operation_locks: dict[str, asyncio.Lock] = {}
        self._workspace_mount_operation_locks_lock = asyncio.Lock()
        self._workspace_mount_sync_previews: dict[
            str, _WorkspaceMountSyncPreviewRecord
        ] = {}
        self._workspace_mount_sync_previews_lock = asyncio.Lock()
        self._workspace_scm_previews: dict[str, _WorkspaceScmPreviewRecord] = {}
        self._workspace_scm_previews_lock = asyncio.Lock()
        self._workspace_scm_backfill_tasks: dict[str, asyncio.Task[None]] = {}
        self._workspace_scm_backfill_tasks_lock = asyncio.Lock()
        self._workspace_mount_watch_interval_seconds = (
            USERSPACE_MOUNT_WATCH_INTERVAL_SECONDS
        )
        self._workspace_mount_watch_jitter_seconds = (
            USERSPACE_MOUNT_WATCH_JITTER_SECONDS
        )
        self._workspace_mount_watch_task: asyncio.Task[Any] | None = None
        self._workspace_mount_watch_task_lock = asyncio.Lock()
        self._workspace_mount_watch_inflight: set[str] = set()
        self._workspace_mount_watch_next_due_monotonic: dict[str, float] = {}
        # Per-endpoint rsync capability state.
        # True  => rsync was observed working and can be attempted directly.
        # False => rsync was observed missing; float is the next monotonic time
        #          when it is worth re-checking remote availability.
        self._ssh_rsync_capability_cache: dict[str, tuple[bool, float]] = {}
        # Serialize snapshot mutations to keep timeline metadata consistent.
        self._snapshot_operation_semaphore = asyncio.Semaphore(1)
        # Startup drift reconciliation runs in a single background task.
        self._git_drift_startup_task: asyncio.Task[Any] | None = None

    @staticmethod
    def _positive_int_env(name: str, default_value: int) -> int:
        raw_value = os.getenv(name, str(default_value)).strip()
        try:
            parsed = int(raw_value)
        except Exception:
            return default_value
        return parsed if parsed > 0 else default_value

    @staticmethod
    def _ssh_rsync_capability_cache_key(ssh_config: Any) -> str:
        return (
            f"{getattr(ssh_config, 'user', '')}@{getattr(ssh_config, 'host', '')}:"
            f"{getattr(ssh_config, 'port', 22)}"
        )

    @staticmethod
    def _ssh_rsync_fallback_notice() -> str:
        return (
            "Remote server does not have rsync installed. Falling back to "
            "Ragtime's built-in SSH sync for this mount. Sync will still work, "
            "but large trees may be slower until rsync is installed remotely."
        )

    @staticmethod
    def _normalize_workspace_mount_sync_mode(
        value: str | None,
        *,
        legacy_sync_deletes: bool = False,
    ) -> WorkspaceMountSyncMode:
        if value == "source_authoritative":
            return "source_authoritative"
        if value == "target_authoritative":
            return "target_authoritative"
        if value == "merge":
            return "merge"
        return "source_authoritative" if legacy_sync_deletes else "merge"

    @staticmethod
    def _is_destructive_workspace_mount_sync_mode(
        sync_mode: WorkspaceMountSyncMode,
    ) -> bool:
        return sync_mode != "merge"

    @classmethod
    def _validate_workspace_mount_sync_configuration(
        cls,
        *,
        source_type: str,
        sync_mode: WorkspaceMountSyncMode,
        auto_sync_enabled: bool,
    ) -> None:
        if source_type != "ssh" and sync_mode != "merge":
            raise HTTPException(
                status_code=400,
                detail="Non-SSH mounts only support merge sync mode",
            )
        # Auto-sync is allowed for all modes; destructive syncs use
        # the configured mode directly and the user is expected to
        # preview/dry-run before enabling auto-sync.

    async def _get_workspace_mount_operation_lock(
        self,
        mount_id: str,
    ) -> asyncio.Lock:
        async with self._workspace_mount_operation_locks_lock:
            existing = self._workspace_mount_operation_locks.get(mount_id)
            if existing is None:
                existing = asyncio.Lock()
                self._workspace_mount_operation_locks[mount_id] = existing
            return existing

    async def _store_workspace_mount_sync_preview(
        self,
        record: _WorkspaceMountSyncPreviewRecord,
    ) -> None:
        async with self._workspace_mount_sync_previews_lock:
            self._workspace_mount_sync_previews[record.mount_id] = record

    async def _pop_workspace_mount_sync_preview(
        self,
        mount_id: str,
    ) -> _WorkspaceMountSyncPreviewRecord | None:
        async with self._workspace_mount_sync_previews_lock:
            return self._workspace_mount_sync_previews.pop(mount_id, None)

    async def _invalidate_workspace_mount_sync_preview(self, mount_id: str) -> None:
        async with self._workspace_mount_sync_previews_lock:
            self._workspace_mount_sync_previews.pop(mount_id, None)

    @staticmethod
    def _workspace_scm_preview_key(
        workspace_id: str,
        direction: WorkspaceScmDirection,
    ) -> str:
        return f"{workspace_id}:{direction}"

    async def _store_workspace_scm_preview(
        self,
        record: _WorkspaceScmPreviewRecord,
    ) -> None:
        async with self._workspace_scm_previews_lock:
            key = self._workspace_scm_preview_key(record.workspace_id, record.direction)
            self._workspace_scm_previews[key] = record

    async def _pop_workspace_scm_preview(
        self,
        workspace_id: str,
        direction: WorkspaceScmDirection,
    ) -> _WorkspaceScmPreviewRecord | None:
        async with self._workspace_scm_previews_lock:
            key = self._workspace_scm_preview_key(workspace_id, direction)
            return self._workspace_scm_previews.pop(key, None)

    async def _consume_workspace_scm_preview(
        self,
        *,
        workspace_id: str,
        direction: WorkspaceScmDirection,
        preview_token: str | None,
        state_fingerprint: str,
    ) -> None:
        preview_record = await self._pop_workspace_scm_preview(workspace_id, direction)
        if (
            preview_record is None
            or preview_record.token != (preview_token or "")
            or preview_record.workspace_id != workspace_id
            or preview_record.direction != direction
            or preview_record.state_fingerprint != state_fingerprint
            or preview_record.expires_at <= _utc_now()
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "SCM overwrite preview is missing, expired, or stale. "
                    "Run preview again before forcing sync."
                ),
            )

    def _prune_workspace_scm_backfill_task(
        self,
        workspace_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._workspace_scm_backfill_tasks.get(workspace_id) is task:
            self._workspace_scm_backfill_tasks.pop(workspace_id, None)

    def _finalize_workspace_scm_backfill_task(
        self,
        workspace_id: str,
        task: asyncio.Task[None],
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug(
                "Workspace SCM backfill task cancelled for workspace %s",
                workspace_id,
            )
        except Exception as exc:
            logger.warning(
                "Workspace SCM backfill task failed for workspace %s: %s",
                workspace_id,
                exc,
                exc_info=True,
            )
        finally:
            self._prune_workspace_scm_backfill_task(workspace_id, task)

    def _attach_workspace_scm_backfill_task_cleanup(
        self,
        workspace_id: str,
        task: asyncio.Task[None],
    ) -> None:
        task.add_done_callback(
            partial(self._finalize_workspace_scm_backfill_task, workspace_id)
        )

    @staticmethod
    def _has_destructive_auto_sync_approval(
        mount: Any,
        sync_mode: WorkspaceMountSyncMode,
    ) -> bool:
        approved_mode = getattr(mount, "destructiveAutoSyncConfirmedMode", None)
        approved_at = getattr(mount, "destructiveAutoSyncConfirmedAt", None)
        return bool(approved_at) and str(approved_mode or "") == sync_mode

    async def _consume_workspace_mount_sync_preview(
        self,
        *,
        mount_id: str,
        workspace_id: str,
        ssh_config: Any,
        remote_path: str,
        cache_dir: Path,
        sync_mode: WorkspaceMountSyncMode,
        preview_token: str | None,
    ) -> None:
        preview_record = await self._pop_workspace_mount_sync_preview(mount_id)
        if (
            preview_record is None
            or preview_record.token != (preview_token or "")
            or preview_record.workspace_id != workspace_id
            or preview_record.mount_id != mount_id
            or preview_record.sync_mode != sync_mode
            or preview_record.expires_at <= _utc_now()
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Destructive sync preview is missing, expired, or stale. "
                    "Run preview again before syncing."
                ),
            )

        current_preview = await asyncio.to_thread(
            preview_ssh_directory_sync,
            ssh_config,
            remote_path,
            str(cache_dir),
            sync_mode=sync_mode,
            sample_limit=self._WORKSPACE_MOUNT_SYNC_PREVIEW_SAMPLE_LIMIT,
        )
        if not current_preview.success:
            raise HTTPException(
                status_code=400,
                detail=(
                    "; ".join(current_preview.errors[:5])
                    or "Failed to preview destructive sync"
                ),
            )
        if current_preview.state_fingerprint != preview_record.state_fingerprint:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Sync preview is stale because the source or target changed. "
                    "Run preview again before syncing."
                ),
            )

    def _prune_workspace_mount_sync_task(
        self,
        mount_id: str,
        task: asyncio.Task[WorkspaceMountSyncResponse],
    ) -> None:
        if self._workspace_mount_sync_tasks.get(mount_id) is task:
            self._workspace_mount_sync_tasks.pop(mount_id, None)

    def _attach_workspace_mount_sync_task_cleanup(
        self,
        mount_id: str,
        task: asyncio.Task[WorkspaceMountSyncResponse],
    ) -> None:
        task.add_done_callback(partial(self._prune_workspace_mount_sync_task, mount_id))

    def _set_remote_rsync_availability(
        self,
        ssh_config: Any,
        *,
        available: bool,
    ) -> None:
        next_recheck_monotonic = 0.0
        if not available:
            next_recheck_monotonic = (
                _time.monotonic() + self._SSH_RSYNC_MISSING_RECHECK_SECONDS
            )
        self._ssh_rsync_capability_cache[
            self._ssh_rsync_capability_cache_key(ssh_config)
        ] = (available, next_recheck_monotonic)

    async def _probe_remote_rsync_availability(
        self,
        ssh_config: Any,
    ) -> bool | None:
        available, _message = await asyncio.to_thread(
            check_remote_rsync_available,
            ssh_config,
        )
        if available is not None:
            self._set_remote_rsync_availability(
                ssh_config,
                available=available,
            )
        return available

    def _remember_remote_rsync_available(
        self,
        ssh_config: Any,
    ) -> None:
        self._set_remote_rsync_availability(ssh_config, available=True)

    def _remember_remote_rsync_missing(
        self,
        ssh_config: Any,
    ) -> None:
        self._set_remote_rsync_availability(ssh_config, available=False)

    async def _resolve_ssh_sync_backend(
        self,
        ssh_config: Any,
        *,
        probe_if_unknown: bool = False,
        force_recheck_missing: bool = False,
    ) -> tuple[str, str | None]:
        cache_key = self._ssh_rsync_capability_cache_key(ssh_config)
        cached = self._ssh_rsync_capability_cache.get(cache_key)
        now_monotonic = _time.monotonic()

        if cached is not None:
            cached_available, next_recheck_monotonic = cached
            if cached_available:
                return "rsync", None
            if now_monotonic < next_recheck_monotonic and not force_recheck_missing:
                return "paramiko", self._ssh_rsync_fallback_notice()

            rechecked = await self._probe_remote_rsync_availability(ssh_config)
            if rechecked is False:
                return "paramiko", self._ssh_rsync_fallback_notice()
            if rechecked is True:
                return "rsync", None

        if probe_if_unknown:
            probed_available = await self._probe_remote_rsync_availability(ssh_config)
            if probed_available is False:
                return "paramiko", self._ssh_rsync_fallback_notice()
        return "rsync", None

    def record_execution_proof(
        self,
        workspace_id: str,
        component_id: str,
        row_count: int,
        query: str,
    ) -> None:
        """Record a server-side proof of successful tool/component execution."""
        proof = _ExecutionProofRecord(
            component_id=component_id,
            row_count=row_count,
            timestamp=_utc_now().timestamp(),
            query_hash=hashlib.sha256(query.strip().encode()).hexdigest(),
        )
        self._execution_proofs.setdefault(workspace_id, {})[component_id] = proof

    def verify_execution_proofs(
        self,
        workspace_id: str,
        component_ids: set[str],
    ) -> list[str]:
        """Return component_ids that lack valid (non-expired) server-side proofs."""
        now = _utc_now().timestamp()
        workspace_proofs = self._execution_proofs.get(workspace_id, {})
        missing: list[str] = []
        for cid in sorted(component_ids):
            proof = workspace_proofs.get(cid)
            if not proof or (now - proof.timestamp) > _EXECUTION_PROOF_MAX_AGE_SECONDS:
                missing.append(cid)
        return missing

    @property
    def root_path(self) -> Path:
        return self._base_dir

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self._workspaces_dir / workspace_id

    def _workspace_meta_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "workspace.json"

    def _workspace_files_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "files"

    @staticmethod
    def _read_workspace_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _detect_imported_replit_features(
        self, workspace_id: str
    ) -> tuple[list[str], bool]:
        """Return deterministic Replit import markers and legacy storage status."""

        files_dir = self._workspace_files_dir(workspace_id)
        features: list[str] = []
        has_legacy_object_storage = False

        replit_config_path = files_dir / ".replit"
        if replit_config_path.is_file():
            features.append(".replit config")

        replit_nix_path = files_dir / "replit.nix"
        if replit_nix_path.is_file():
            features.append("replit.nix")

        replit_integrations_dir = files_dir / "server" / "replit_integrations"
        if replit_integrations_dir.is_dir():
            features.append("server/replit_integrations")

        package_json_path = files_dir / "package.json"
        package_json_text = self._read_workspace_text_file(package_json_path)
        if '"@google-cloud/storage"' in package_json_text:
            features.append("@google-cloud/storage dependency")
            has_legacy_object_storage = True

        object_storage_path = (
            files_dir
            / "server"
            / "replit_integrations"
            / "object_storage"
            / "objectStorage.ts"
        )
        object_storage_text = self._read_workspace_text_file(object_storage_path)
        if object_storage_text:
            features.append("Replit object-storage adapter")
            legacy_markers = (
                "REPLIT_SIDECAR_ENDPOINT",
                "signed-object-url",
                "@google-cloud/storage",
                "127.0.0.1:1106",
            )
            if any(marker in object_storage_text for marker in legacy_markers):
                has_legacy_object_storage = True
                features.append("legacy Replit object-storage flow")

        deduped_features = list(dict.fromkeys(features))
        return deduped_features, has_legacy_object_storage

    def _workspace_rootfs_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "rootfs"

    def _workspace_object_storage_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / _WORKSPACE_OBJECT_STORAGE_DIRNAME

    def _workspace_object_storage_config_path(self, workspace_id: str) -> Path:
        return (
            self._workspace_object_storage_dir(workspace_id)
            / _WORKSPACE_OBJECT_STORAGE_CONFIG_NAME
        )

    def _workspace_object_storage_buckets_dir(self, workspace_id: str) -> Path:
        return self._workspace_object_storage_dir(workspace_id) / "buckets"

    @staticmethod
    def _normalize_object_storage_bucket_name(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not normalized:
            raise HTTPException(status_code=400, detail="Bucket name is required")
        if len(normalized) < 3 or len(normalized) > 63:
            raise HTTPException(
                status_code=400,
                detail="Bucket name must be between 3 and 63 characters",
            )
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", normalized):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Bucket name must use lowercase letters, numbers, and hyphens, "
                    "and must start and end with a letter or number"
                ),
            )
        return normalized

    @staticmethod
    def _normalize_object_storage_bucket_description(
        description: str | None,
    ) -> str | None:
        if description is None:
            return None
        normalized = description.strip()
        return normalized or None

    @staticmethod
    def _default_object_storage_bucket_name(workspace_id: str) -> str:
        suffix = re.sub(r"[^a-z0-9]", "", workspace_id.lower())[:12] or "default"
        return f"workspace-{suffix}"

    @staticmethod
    def _build_object_storage_bucket_record(
        *,
        name: str,
        description: str | None,
        now: datetime,
        public_prefix: str = _WORKSPACE_OBJECT_STORAGE_PUBLIC_PREFIX,
        private_prefix: str = _WORKSPACE_OBJECT_STORAGE_PRIVATE_PREFIX,
    ) -> dict[str, Any]:
        timestamp = now.isoformat()
        return {
            "name": name,
            "description": description,
            "public_prefix": public_prefix,
            "private_prefix": private_prefix,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def _default_object_storage_config(self, workspace_id: str) -> dict[str, Any]:
        now = _utc_now()
        default_bucket = self._default_object_storage_bucket_name(workspace_id)
        secret = secrets.token_urlsafe(32)
        return {
            "version": 1,
            "managed_by": "ragtime",
            "region": _WORKSPACE_OBJECT_STORAGE_REGION,
            "access_key_id": f"ragtime-{workspace_id[:8]}",
            "secret_access_key_encrypted": encrypt_secret(secret),
            "default_bucket_name": default_bucket,
            "buckets": [
                self._build_object_storage_bucket_record(
                    name=default_bucket,
                    description="Default workspace bucket",
                    now=now,
                )
            ],
        }

    def _write_object_storage_config(
        self,
        workspace_id: str,
        payload: dict[str, Any],
    ) -> None:
        storage_dir = self._workspace_object_storage_dir(workspace_id)
        config_path = self._workspace_object_storage_config_path(workspace_id)
        buckets_dir = self._workspace_object_storage_buckets_dir(workspace_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        buckets_dir.mkdir(parents=True, exist_ok=True)
        for bucket in payload.get("buckets") or []:
            bucket_name = str((bucket or {}).get("name") or "").strip()
            if bucket_name:
                (buckets_dir / bucket_name).mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _ensure_object_storage_config(self, workspace_id: str) -> dict[str, Any]:
        config_path = self._workspace_object_storage_config_path(workspace_id)
        if config_path.exists() and config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Workspace object storage config is invalid",
                ) from exc
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Workspace object storage config is invalid",
                )
            self._write_object_storage_config(workspace_id, payload)
            return payload

        payload = self._default_object_storage_config(workspace_id)
        self._write_object_storage_config(workspace_id, payload)
        return payload

    @staticmethod
    def _parse_object_storage_datetime(raw_value: Any) -> datetime:
        if isinstance(raw_value, datetime):
            return raw_value
        text = str(raw_value or "").strip()
        if text:
            try:
                return datetime.fromisoformat(text)
            except Exception:
                pass
        return _utc_now()

    @staticmethod
    def _bucket_public_object_root(bucket: dict[str, Any]) -> str:
        prefix = str(
            bucket.get("public_prefix") or _WORKSPACE_OBJECT_STORAGE_PUBLIC_PREFIX
        )
        return f"/{bucket['name']}/{prefix.strip('/')}"

    @staticmethod
    def _bucket_private_object_root(bucket: dict[str, Any]) -> str:
        prefix = str(
            bucket.get("private_prefix") or _WORKSPACE_OBJECT_STORAGE_PRIVATE_PREFIX
        )
        return f"/{bucket['name']}/{prefix.strip('/')}"

    def _object_storage_bucket_model(
        self,
        bucket: dict[str, Any],
        *,
        default_bucket_name: str | None,
    ) -> UserSpaceObjectStorageBucket:
        bucket_name = str(bucket.get("name") or "").strip()
        return UserSpaceObjectStorageBucket(
            name=bucket_name,
            description=self._normalize_object_storage_bucket_description(
                cast(str | None, bucket.get("description"))
            ),
            public_prefix=str(
                bucket.get("public_prefix") or _WORKSPACE_OBJECT_STORAGE_PUBLIC_PREFIX
            ),
            private_prefix=str(
                bucket.get("private_prefix") or _WORKSPACE_OBJECT_STORAGE_PRIVATE_PREFIX
            ),
            is_default=bucket_name == (default_bucket_name or ""),
            created_at=self._parse_object_storage_datetime(bucket.get("created_at")),
            updated_at=self._parse_object_storage_datetime(bucket.get("updated_at")),
        )

    def _object_storage_config_model(
        self,
        workspace_id: str,
        payload: dict[str, Any],
    ) -> UserSpaceObjectStorageConfig:
        raw_buckets = payload.get("buckets")
        buckets_raw: list[dict[str, Any]] = (
            [bucket for bucket in raw_buckets if isinstance(bucket, dict)]
            if isinstance(raw_buckets, list)
            else []
        )
        default_bucket_name = (
            str(payload.get("default_bucket_name") or "").strip() or None
        )
        buckets = [
            self._object_storage_bucket_model(
                bucket,
                default_bucket_name=default_bucket_name,
            )
            for bucket in buckets_raw
            if str(bucket.get("name") or "").strip()
        ]
        public_paths = [
            self._bucket_public_object_root(bucket)
            for bucket in buckets_raw
            if str(bucket.get("name") or "").strip()
        ]
        private_path = None
        for bucket in buckets_raw:
            if str(bucket.get("name") or "").strip() == (default_bucket_name or ""):
                private_path = self._bucket_private_object_root(bucket)
                break
        return UserSpaceObjectStorageConfig(
            workspace_id=workspace_id,
            region=str(payload.get("region") or _WORKSPACE_OBJECT_STORAGE_REGION),
            endpoint_env_key=_WORKSPACE_OBJECT_STORAGE_ENDPOINT_ENV_KEY,
            access_key_env_key=_WORKSPACE_OBJECT_STORAGE_ACCESS_KEY_ENV_KEY,
            secret_key_env_key=_WORKSPACE_OBJECT_STORAGE_SECRET_KEY_ENV_KEY,
            default_bucket_name=default_bucket_name,
            public_object_search_paths=public_paths,
            private_object_dir=private_path,
            buckets=buckets,
        )

    def _build_object_storage_runtime_env(self, workspace_id: str) -> dict[str, str]:
        payload = self._ensure_object_storage_config(workspace_id)
        default_bucket_name = str(payload.get("default_bucket_name") or "").strip()
        access_key_id = str(payload.get("access_key_id") or "").strip()
        encrypted_secret = str(payload.get("secret_access_key_encrypted") or "").strip()
        secret_access_key = decrypt_secret(encrypted_secret) if encrypted_secret else ""
        config_model = self._object_storage_config_model(workspace_id, payload)

        env: dict[str, str] = {
            _WORKSPACE_OBJECT_STORAGE_ENABLED_ENV_KEY: "true",
            _WORKSPACE_OBJECT_STORAGE_REGION_ENV_KEY: config_model.region,
            _WORKSPACE_OBJECT_STORAGE_FORCE_PATH_STYLE_ENV_KEY: "true",
            _WORKSPACE_OBJECT_STORAGE_BUCKETS_ENV_KEY: json.dumps(
                [bucket.model_dump(mode="json") for bucket in config_model.buckets],
                separators=(",", ":"),
            ),
        }
        if access_key_id:
            env[_WORKSPACE_OBJECT_STORAGE_ACCESS_KEY_ENV_KEY] = access_key_id
        if secret_access_key:
            env[_WORKSPACE_OBJECT_STORAGE_SECRET_KEY_ENV_KEY] = secret_access_key
        if default_bucket_name:
            env[_WORKSPACE_OBJECT_STORAGE_DEFAULT_BUCKET_ENV_KEY] = default_bucket_name
        if config_model.public_object_search_paths:
            env["PUBLIC_OBJECT_SEARCH_PATHS"] = ",".join(
                config_model.public_object_search_paths
            )
        if config_model.private_object_dir:
            env["PRIVATE_OBJECT_DIR"] = config_model.private_object_dir
        return env

    async def get_workspace_object_storage_config(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceObjectStorageConfig:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        payload = self._ensure_object_storage_config(workspace_id)
        return self._object_storage_config_model(workspace_id, payload)

    async def get_workspace_object_storage_summary(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceObjectStorageConfig:
        """Return non-secret workspace object-storage metadata for any member.

        This is safe for prompt context because it only includes bucket names,
        prefixes, and public env-key names. Credentials are never returned.
        """

        await self._enforce_workspace_access(workspace_id, user_id)
        payload = self._ensure_object_storage_config(workspace_id)
        return self._object_storage_config_model(workspace_id, payload)

    async def create_workspace_object_storage_bucket(
        self,
        workspace_id: str,
        user_id: str,
        request: CreateUserSpaceObjectStorageBucketRequest,
    ) -> UserSpaceObjectStorageConfig:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        payload = self._ensure_object_storage_config(workspace_id)
        raw_buckets = payload.get("buckets")
        buckets: list[dict[str, Any]] = (
            [bucket for bucket in raw_buckets if isinstance(bucket, dict)]
            if isinstance(raw_buckets, list)
            else []
        )
        bucket_name = self._normalize_object_storage_bucket_name(request.name)
        if any(
            str((bucket or {}).get("name") or "") == bucket_name for bucket in buckets
        ):
            raise HTTPException(status_code=409, detail="Bucket already exists")

        now = _utc_now()
        buckets.append(
            self._build_object_storage_bucket_record(
                name=bucket_name,
                description=self._normalize_object_storage_bucket_description(
                    request.description
                ),
                now=now,
            )
        )
        payload["buckets"] = buckets
        if (
            request.make_default
            or not str(payload.get("default_bucket_name") or "").strip()
        ):
            payload["default_bucket_name"] = bucket_name

        self._write_object_storage_config(workspace_id, payload)
        db = await get_db()
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": _utc_now()},
        )
        return self._object_storage_config_model(workspace_id, payload)

    async def update_workspace_object_storage_bucket(
        self,
        workspace_id: str,
        user_id: str,
        bucket_name: str,
        request: UpdateUserSpaceObjectStorageBucketRequest,
    ) -> UserSpaceObjectStorageConfig:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        normalized_name = self._normalize_object_storage_bucket_name(bucket_name)
        payload = self._ensure_object_storage_config(workspace_id)
        raw_buckets = payload.get("buckets")
        buckets: list[dict[str, Any]] = (
            [bucket for bucket in raw_buckets if isinstance(bucket, dict)]
            if isinstance(raw_buckets, list)
            else []
        )
        target: dict[str, Any] | None = None
        for bucket in buckets:
            if str(bucket.get("name") or "") == normalized_name:
                target = bucket
                break
        if target is None:
            raise HTTPException(status_code=404, detail="Bucket not found")

        if request.description is not None:
            target["description"] = self._normalize_object_storage_bucket_description(
                request.description
            )
        target["updated_at"] = _utc_now().isoformat()
        if request.make_default:
            payload["default_bucket_name"] = normalized_name

        self._write_object_storage_config(workspace_id, payload)
        db = await get_db()
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": _utc_now()},
        )
        return self._object_storage_config_model(workspace_id, payload)

    async def delete_workspace_object_storage_bucket(
        self,
        workspace_id: str,
        user_id: str,
        bucket_name: str,
    ) -> DeleteUserSpaceObjectStorageBucketResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        normalized_name = self._normalize_object_storage_bucket_name(bucket_name)
        payload = self._ensure_object_storage_config(workspace_id)
        raw_buckets = payload.get("buckets")
        buckets: list[dict[str, Any]] = (
            [bucket for bucket in raw_buckets if isinstance(bucket, dict)]
            if isinstance(raw_buckets, list)
            else []
        )
        remaining = [
            bucket
            for bucket in buckets
            if str(bucket.get("name") or "") != normalized_name
        ]
        if len(remaining) == len(buckets):
            raise HTTPException(status_code=404, detail="Bucket not found")
        if not remaining:
            raise HTTPException(
                status_code=400,
                detail="At least one workspace bucket must remain configured",
            )

        payload["buckets"] = remaining
        if str(payload.get("default_bucket_name") or "") == normalized_name:
            payload["default_bucket_name"] = str(remaining[0].get("name") or "")

        bucket_dir = (
            self._workspace_object_storage_buckets_dir(workspace_id) / normalized_name
        )
        try:
            if bucket_dir.exists() and bucket_dir.is_dir():
                shutil.rmtree(bucket_dir)
        except Exception as exc:
            logger.warning(
                "Failed to remove workspace object storage bucket data for %s/%s: %s",
                workspace_id,
                normalized_name,
                exc,
            )

        self._write_object_storage_config(workspace_id, payload)
        db = await get_db()
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": _utc_now()},
        )
        return DeleteUserSpaceObjectStorageBucketResponse(
            success=True,
            bucket_name=normalized_name,
            workspace_id=workspace_id,
        )

    @staticmethod
    def _normalize_workspace_env_var_key(raw_key: str) -> str:
        key = (raw_key or "").strip()
        if not key:
            raise HTTPException(
                status_code=400, detail="Environment variable key is required"
            )
        if len(key) > 128:
            raise HTTPException(
                status_code=400,
                detail="Environment variable key must be 128 characters or fewer",
            )
        if not _WORKSPACE_ENV_VAR_KEY_PATTERN.fullmatch(key):
            raise HTTPException(
                status_code=400,
                detail=("Environment variable key must match [A-Za-z_][A-Za-z0-9_]*"),
            )
        return key

    @staticmethod
    def _workspace_env_var_model(db: Any) -> Any:
        return getattr(db, "workspaceenvironmentvariable")

    @staticmethod
    def _runtime_audit_model(db: Any) -> Any:
        return getattr(db, "userspaceruntimeauditevent")

    @staticmethod
    def _workspace_env_var_from_record(record: Any) -> UserSpaceWorkspaceEnvVar:
        description = getattr(record, "description", None)
        return UserSpaceWorkspaceEnvVar(
            key=str(getattr(record, "key", "") or ""),
            has_value=bool(str(getattr(record, "value", "") or "")),
            description=str(description) if description is not None else None,
            created_at=getattr(record, "createdAt"),
            updated_at=getattr(record, "updatedAt"),
        )

    async def _audit_workspace_env_var_event(
        self,
        workspace_id: str,
        user_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        db = await get_db()
        model = self._runtime_audit_model(db)
        payload_json = prisma_fields.Json(json.loads(json.dumps(payload, default=str)))
        now = _utc_now()
        shapes: list[dict[str, Any]] = [
            {
                "workspaceId": workspace_id,
                "userId": user_id,
                "eventType": event_type,
                "eventPayload": payload_json,
                "createdAt": now,
            },
            {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "event_type": event_type,
                "event_payload": payload_json,
                "created_at": now,
            },
        ]
        for data in shapes:
            try:
                await model.create(data=data)
                return
            except Exception:
                continue
        logger.debug(
            "Failed to persist env-var audit event for workspace %s", workspace_id
        )

    @staticmethod
    def _sanitize_workspace_env_map(raw_items: list[Any]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for item in raw_items:
            key = str(getattr(item, "key", "") or "").strip()
            encrypted_value = str(getattr(item, "value", "") or "")
            if not key or not encrypted_value:
                continue
            resolved[key] = decrypt_secret(encrypted_value)
        return resolved

    @staticmethod
    def _default_runtime_bootstrap_config() -> dict[str, Any]:
        return {
            "version": 1,
            "managed_by": "ragtime",
            "template_version": _RUNTIME_BOOTSTRAP_TEMPLATE_VERSION,
            "auto_update": True,
            "watch_paths": [
                "package.json",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "bun.lock",
                "bun.lockb",
                "requirements.txt",
                "pyproject.toml",
                "uv.lock",
                "poetry.lock",
                "Pipfile",
                "Pipfile.lock",
                ".ragtime/scripts/sqlite_migrate.py",
                ".ragtime/db/migrations",
            ],
            "commands": [
                {
                    "name": "node_dependencies",
                    "when_exists": "package.json",
                    "run": "if [ -f pnpm-lock.yaml ]; then pnpm install --frozen-lockfile; elif [ -f yarn.lock ]; then yarn install --frozen-lockfile; elif [ -f bun.lock ] || [ -f bun.lockb ]; then bun install --frozen-lockfile; elif [ -f package-lock.json ]; then npm ci; elif [ ! -d node_modules ]; then npm install; fi",
                },
                {
                    "name": "node_tailwind_tooling",
                    "when_exists": "package.json",
                    "unless_exists": "node_modules/.bin/tailwindcss",
                    "run": "if [ -f pnpm-lock.yaml ]; then pnpm add -D tailwindcss @tailwindcss/cli; elif [ -f yarn.lock ]; then yarn add -D tailwindcss @tailwindcss/cli; elif [ -f bun.lock ] || [ -f bun.lockb ]; then bun add -d tailwindcss @tailwindcss/cli; else npm install -D tailwindcss @tailwindcss/cli; fi",
                },
                {
                    "name": "python_dependencies",
                    "run": "if [ -f uv.lock ]; then uv sync --frozen; elif [ -f poetry.lock ] || ( [ -f pyproject.toml ] && grep -q '\\[tool\\.poetry\\]' pyproject.toml ); then poetry install --no-interaction --no-root; elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then pipenv install; elif [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; elif [ -f pyproject.toml ]; then uv sync; fi",
                },
                {
                    "name": "sqlite_migrations",
                    "when_exists": ".ragtime/scripts/sqlite_migrate.py",
                    "run": "python3 .ragtime/scripts/sqlite_migrate.py --db .ragtime/db/app.sqlite3 --migrations .ragtime/db/migrations",
                },
            ],
        }

    @staticmethod
    def _merge_workspace_mount_sync_notices(*notices: str | None) -> str | None:
        merged: list[str] = []
        for notice in notices:
            normalized = str(notice or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        if not merged:
            return None
        return " ".join(merged)

    @staticmethod
    def _is_legacy_default_bootstrap(payload: dict[str, Any]) -> bool:
        if (
            payload.get("managed_by") is not None
            or payload.get("template_version") is not None
        ):
            return False
        if int(payload.get("version") or 0) != 1:
            return False
        commands = payload.get("commands")
        if not isinstance(commands, list):
            return False
        command_names = {
            str(item.get("name") or "").strip()
            for item in commands
            if isinstance(item, dict)
        }
        return command_names == {"npm_ci", "npm_install", "pip_requirements"}

    def _sync_runtime_bootstrap_config(self, workspace_id: str) -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        config_path = files_dir / _RUNTIME_BOOTSTRAP_CONFIG_PATH
        default_payload = self._default_runtime_bootstrap_config()

        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(default_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return

        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(existing, dict):
            return

        managed_by = str(existing.get("managed_by") or "").strip()
        auto_update = bool(existing.get("auto_update", managed_by == "ragtime"))
        template_version = int(existing.get("template_version") or 0)
        is_legacy_default = self._is_legacy_default_bootstrap(existing)
        should_update = (
            managed_by == "ragtime"
            and auto_update
            and template_version < _RUNTIME_BOOTSTRAP_TEMPLATE_VERSION
        ) or is_legacy_default
        if not should_update:
            return

        config_path.write_text(
            json.dumps(default_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _seed_runtime_bootstrap_config(self, workspace_id: str) -> None:
        self._sync_runtime_bootstrap_config(workspace_id)
        self._seed_sqlite_migrate_script(workspace_id)

    def _seed_sqlite_migrate_script(self, workspace_id: str) -> None:
        """Ensure the workspace has a default SQLite migration runner script."""
        files_dir = self._workspace_files_dir(workspace_id)
        script_path = files_dir / ".ragtime" / "scripts" / "sqlite_migrate.py"
        if script_path.exists():
            return
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            _render_userspace_template("sqlite_migrate.py"),
            encoding="utf-8",
        )

    async def build_runtime_bridge_content(
        self, workspace_id: str | None = None
    ) -> str:
        if not workspace_id:
            return _build_bridge_content()

        db = await get_db()
        workspace = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"toolSelections": True},
        )
        if not workspace:
            return _build_bridge_content()

        selected_tool_ids = self._selected_tool_ids_from_workspace_record(workspace)
        return _build_bridge_content(_compute_bridge_timeout_ms(selected_tool_ids))

    @staticmethod
    def _default_runtime_entrypoint_config() -> dict[str, str]:
        """Default runtime entrypoint for new workspaces (static HTML fallback).

        Uses ``$PORT`` so the runtime can substitute the actual port at launch
        time via the ``PORT`` environment variable.
        """
        return {
            "command": "python3 -m http.server $PORT --bind 0.0.0.0 --directory .",
            "cwd": ".",
            "framework": "static",
        }

    def _seed_runtime_entrypoint_config(self, workspace_id: str) -> None:
        """Create ``.ragtime/runtime-entrypoint.json`` for a new workspace.

        Only writes the file when it does not already exist so that
        user/agent-managed entrypoints are never overwritten.
        """
        files_dir = self._workspace_files_dir(workspace_id)
        config_path = files_dir / ".ragtime" / "runtime-entrypoint.json"
        if config_path.exists():
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self._default_runtime_entrypoint_config(), indent=2) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Import-time entrypoint inference
    # ------------------------------------------------------------------

    def _infer_entrypoint_from_imported_files(
        self,
        workspace_id: str,
    ) -> dict[str, str] | None:
        """Attempt to infer a runtime entrypoint from imported workspace files.

        Detection is purely file/config-based (no repo-name heuristics).
        Returns a ``{command, cwd, framework}`` dict on success, or ``None``
        if no confident inference can be made.

        Checked signals (in priority order):
        1. ``.replit`` ``run`` field — explicit run command from Replit config.
        2. ``package.json`` ``scripts.dev`` or ``scripts.start`` — Node/npm
           convention for dev servers.
        """
        files_dir = self._workspace_files_dir(workspace_id)

        # --- Signal 1: .replit run field ---
        replit_path = files_dir / ".replit"
        if replit_path.is_file():
            inferred = self._parse_replit_run_command(replit_path)
            if inferred:
                return inferred

        # --- Signal 2: package.json scripts ---
        pkg_path = files_dir / "package.json"
        if pkg_path.is_file():
            inferred = self._parse_package_json_entrypoint(pkg_path)
            if inferred:
                return inferred

        return None

    @staticmethod
    def _parse_replit_run_command(replit_path: Path) -> dict[str, str] | None:
        """Extract a run command from a ``.replit`` file.

        The ``.replit`` format is TOML-like.  We look for a top-level
        ``run = "..."`` assignment.  Returns ``None`` when no usable
        command is found.
        """
        try:
            text = replit_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("run") and "=" in stripped:
                _, _, value = stripped.partition("=")
                value = value.strip().strip('"').strip("'").strip()
                if not value:
                    continue
                framework = _guess_node_framework_from_command(value)
                return {
                    "command": value,
                    "cwd": ".",
                    "framework": framework,
                }
        return None

    @staticmethod
    def _parse_package_json_entrypoint(pkg_path: Path) -> dict[str, str] | None:
        """Infer a dev-server entrypoint from ``package.json`` scripts.

        Checks ``scripts.dev`` first (common for Vite/Next/Nuxt),
        then ``scripts.start``.  Returns ``None`` when nothing useful
        is found.
        """
        try:
            data = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        scripts = data.get("scripts")
        if not isinstance(scripts, dict):
            return None

        for key in ("dev", "start"):
            cmd = scripts.get(key, "").strip()
            if not cmd:
                continue
            npm_cmd = f"npm run {key}"
            framework = _guess_node_framework_from_command(cmd)
            return {
                "command": npm_cmd,
                "cwd": ".",
                "framework": framework,
            }
        return None

    def _normalize_imported_replit_runtime_artifacts(
        self,
        workspace_id: str,
    ) -> list[str]:
        files_dir = self._workspace_files_dir(workspace_id)
        replit_path = files_dir / ".replit"
        if not replit_path.is_file():
            return []

        original_text = self._read_workspace_text_file(replit_path)
        if not original_text:
            return []

        current_section = ""
        normalized_lines: list[str] = []
        changed = False
        actions: list[str] = []

        for line in original_text.splitlines():
            stripped = line.strip()
            section_match = _REPLIT_SECTION_RE.match(stripped)
            if section_match:
                current_section = section_match.group(1).strip().lower()

            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip().lower()
                if key in _REPLIT_RUNTIME_KEYS:
                    normalized_lines.append(f"# Ragtime normalized: {line}")
                    actions.append("commented Replit runtime directives in .replit")
                    changed = True
                    continue
                if current_section == "env" and key == "port":
                    normalized_lines.append(f"# Ragtime normalized: {line}")
                    actions.append("removed Replit PORT override from .replit")
                    changed = True
                    continue

            normalized_lines.append(line)

        if not changed:
            return []

        normalized_text = "\n".join(normalized_lines)
        if not normalized_text.startswith(_REPLIT_NORMALIZATION_HEADER[0]):
            normalized_text = (
                "\n".join(_REPLIT_NORMALIZATION_HEADER) + "\n\n" + normalized_text
            )
        if original_text.endswith("\n"):
            normalized_text += "\n"
        replit_path.write_text(normalized_text, encoding="utf-8")
        return _dedupe_preserve_order(actions)

    def _seed_entrypoint_from_import(self, workspace_id: str) -> dict[str, str] | None:
        """Infer and write an entrypoint for an imported workspace.

        Only writes when the entrypoint is currently missing or is the
        default static server placeholder (never overwrites a real
        user/agent-configured entrypoint).

        Returns the inferred config dict that was written, or ``None`` if
        no inference was made or the entrypoint was already valid.
        """
        status = self.get_workspace_entrypoint_status(workspace_id)
        if status.state == "valid" and not self.is_default_static_entrypoint(
            workspace_id, status
        ):
            return None

        inferred = self._infer_entrypoint_from_imported_files(workspace_id)
        if not inferred:
            return None

        files_dir = self._workspace_files_dir(workspace_id)
        config_path = files_dir / ".ragtime" / "runtime-entrypoint.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(inferred, indent=2) + "\n",
            encoding="utf-8",
        )
        self.invalidate_entrypoint_cache(workspace_id)
        logger.info(
            "Inferred runtime entrypoint for workspace %s: framework=%s, command=%s",
            workspace_id,
            inferred.get("framework"),
            inferred.get("command"),
        )
        return inferred

    def get_workspace_entrypoint_status(self, workspace_id: str) -> EntrypointStatus:
        """Return the canonical entrypoint status for *workspace_id*.

        Uses the shared :func:`runtime.shared.parse_entrypoint_config`
        parser so that the ragtime app and the runtime worker always agree
        on what constitutes a valid/missing/invalid entrypoint.

        Results are cached per workspace with a short TTL to avoid
        redundant filesystem reads within the same request flow.
        """
        cached = self._entrypoint_status_cache.get(workspace_id)
        now = _utc_now().timestamp()
        if cached is not None:
            status, ts = cached
            if (now - ts) < _ENTRYPOINT_STATUS_CACHE_TTL_SECONDS:
                return status
        status = parse_entrypoint_config(self._workspace_files_dir(workspace_id))
        self._entrypoint_status_cache[workspace_id] = (status, now)
        return status

    def is_default_static_entrypoint(
        self,
        workspace_id: str,
        status: EntrypointStatus | None = None,
    ) -> bool:
        """Return True when the entrypoint is the seeded default static server.

        The default seed (``python3 -m http.server ...``, framework ``static``)
        is semantically valid JSON but not a real user/agent choice.  Prompt
        nudges should treat this the same as a missing entrypoint so the agent
        is encouraged to choose a proper framework.

        Accepts an optional pre-fetched *status* to avoid redundant reads
        when the caller already has the value.
        """
        if status is None:
            status = self.get_workspace_entrypoint_status(workspace_id)
        if status.state != "valid":
            return False
        default = self._default_runtime_entrypoint_config()
        return status.command == default.get("command", "") and (
            status.framework or ""
        ) == default.get("framework", "")

    def invalidate_entrypoint_cache(self, workspace_id: str) -> None:
        """Drop cached entrypoint status for *workspace_id*.

        Call after any write that may change ``.ragtime/runtime-entrypoint.json``.
        """
        self._entrypoint_status_cache.pop(workspace_id, None)

    def _workspace_git_dir(self, workspace_id: str) -> Path:
        return self._workspace_files_dir(workspace_id) / ".git"

    def _ensure_workspace_gitignore(self, files_dir: Path) -> None:
        gitignore = files_dir / ".gitignore"
        existing_lines: list[str] = []
        if gitignore.exists():
            try:
                existing_lines = gitignore.read_text(encoding="utf-8").splitlines()
            except Exception:
                existing_lines = []

        existing_set = {line.strip() for line in existing_lines if line.strip()}
        required = (
            *_WORKSPACE_DEFAULT_GITIGNORE_PATTERNS,
            *_PLATFORM_MANAGED_GITIGNORE_PATTERNS,
        )
        missing = [pattern for pattern in required if pattern not in existing_set]
        if not missing and gitignore.exists():
            return

        merged = list(existing_lines)
        if merged and merged[-1].strip():
            merged.append("")
        merged.extend(missing)
        content = "\n".join(merged).strip("\n") + "\n"
        gitignore.write_text(content, encoding="utf-8")

    async def _run_git_raw(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        files_dir = self._workspace_files_dir(workspace_id)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(files_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            returncode = process.returncode if process.returncode is not None else 1
            if check and returncode != 0:
                stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                raise HTTPException(
                    status_code=500,
                    detail=f"Git snapshot operation failed: {stderr or 'unknown error'}",
                )
            return returncode, stdout_bytes, stderr_bytes
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="Git binary not available for User Space snapshots",
            ) from exc

    async def _run_git(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> _GitCommandResult:
        returncode, stdout_bytes, stderr_bytes = await self._run_git_raw(
            workspace_id,
            args,
            check=check,
            env=env,
        )
        return _GitCommandResult(
            returncode=returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def _run_git_bytes(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        return await self._run_git_raw(workspace_id, args, check=check, env=env)

    async def _run_git_in_dir_raw(
        self,
        directory: Path,
        args: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(directory),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            returncode = process.returncode if process.returncode is not None else 1
            if check and returncode != 0:
                stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                raise HTTPException(
                    status_code=500,
                    detail=f"Git workspace SCM operation failed: {stderr or 'unknown error'}",
                )
            return returncode, stdout_bytes, stderr_bytes
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="Git binary not available for workspace SCM operations",
            ) from exc

    async def _run_git_in_dir(
        self,
        directory: Path,
        args: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> _GitCommandResult:
        returncode, stdout_bytes, stderr_bytes = await self._run_git_in_dir_raw(
            directory,
            args,
            check=check,
            env=env,
        )
        return _GitCommandResult(
            returncode=returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def _ensure_workspace_git_repo(self, workspace_id: str) -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        git_dir = self._workspace_git_dir(workspace_id)
        if git_dir.exists() and git_dir.is_dir():
            self._ensure_workspace_gitignore(files_dir)
            return

        await self._run_git(workspace_id, ["init"])
        await self._run_git(
            workspace_id,
            ["config", "user.name", "Ragtime User Space"],
        )
        await self._run_git(
            workspace_id,
            ["config", "user.email", "userspace@ragtime.local"],
        )

        self._ensure_workspace_gitignore(files_dir)

    async def _reconcile_workspace_git_drift(
        self,
        workspace_id: str,
        reason: str,
    ) -> None:
        """Ensure the workspace Git policy includes platform-managed ignore rules."""
        try:
            await self._ensure_workspace_git_repo(workspace_id)
            logger.debug(
                "Completed userspace git drift reconciliation for %s (%s)",
                workspace_id,
                reason,
            )
        except Exception as exc:
            logger.debug(
                "Skipped userspace git drift reconciliation for %s (%s): %s",
                workspace_id,
                reason,
                exc,
            )

    def schedule_startup_git_drift_reconciliation(self) -> None:
        """Start best-effort Git policy reconciliation for existing workspaces."""
        if (
            self._git_drift_startup_task is not None
            and not self._git_drift_startup_task.done()
        ):
            return
        self._git_drift_startup_task = asyncio.create_task(
            self._startup_git_drift_reconciliation()
        )

    def schedule_workspace_mount_watch(self) -> None:
        """Start optional background loop that auto-syncs watch-enabled SSH mounts."""
        if (
            self._workspace_mount_watch_task is not None
            and not self._workspace_mount_watch_task.done()
        ):
            return
        self._workspace_mount_watch_task = asyncio.create_task(
            self._workspace_mount_watch_loop(),
            name="userspace-mount-watch-loop",
        )

    async def _workspace_mount_watch_loop(self) -> None:
        poll_seconds = max(
            1.0, min(5.0, self._workspace_mount_watch_interval_seconds / 3.0)
        )
        while True:
            await asyncio.sleep(poll_seconds)
            try:
                await self._workspace_mount_watch_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(
                    "Workspace mount watch tick failed: %s", exc, exc_info=True
                )

    def _workspace_mount_watch_stagger_seconds(self, mount_id: str) -> float:
        if self._workspace_mount_watch_jitter_seconds <= 0:
            return 0.0
        # Deterministic per-mount jitter avoids bursty sync waves.
        jitter_fraction = (abs(hash(mount_id)) % 1000) / 1000.0
        return jitter_fraction * self._workspace_mount_watch_jitter_seconds

    async def _workspace_mount_watch_tick(self) -> None:
        db = await get_db()
        mounts = await db.workspacemount.find_many(
            where={"autoSyncEnabled": True, "enabled": True},
            include={"mountSource": True},
        )

        active_mount_ids: set[str] = set()
        now_monotonic = _time.monotonic()

        for mount in mounts:
            mount_id = str(getattr(mount, "id", "") or "")
            workspace_id = str(getattr(mount, "workspaceId", "") or "")
            if not mount_id or not workspace_id:
                continue
            active_mount_ids.add(mount_id)

            mount_source = getattr(mount, "mountSource", None)
            if not mount_source:
                continue
            if str(getattr(mount_source, "sourceType", "") or "") != "ssh":
                continue
            if not bool(getattr(mount_source, "enabled", False)):
                continue
            if mount_id in self._workspace_mount_watch_inflight:
                continue

            due_at = self._workspace_mount_watch_next_due_monotonic.get(mount_id, 0.0)
            if now_monotonic < due_at:
                continue

            # Per-source interval from DB, falling back to the global default.
            source_interval = float(
                getattr(mount_source, "syncIntervalSeconds", None)
                or self._workspace_mount_watch_interval_seconds
            )

            self._workspace_mount_watch_inflight.add(mount_id)
            # Compute the next due time when the run completes so each mount
            # gets a full cooldown interval after finishing a sync.
            asyncio.create_task(
                self._run_watched_workspace_mount_sync(
                    workspace_id,
                    mount_id,
                    source_interval,
                ),
                name=f"userspace-mount-watch-sync:{workspace_id}:{mount_id}",
            )

        stale_mount_ids = (
            set(self._workspace_mount_watch_next_due_monotonic) - active_mount_ids
        )
        for mount_id in stale_mount_ids:
            self._workspace_mount_watch_next_due_monotonic.pop(mount_id, None)
            self._workspace_mount_watch_inflight.discard(mount_id)

    async def _run_watched_workspace_mount_sync(
        self,
        workspace_id: str,
        mount_id: str,
        source_interval_seconds: float,
    ) -> None:
        t0 = _time.monotonic()
        try:
            db = await get_db()
            mount = await db.workspacemount.find_first(
                where={"id": mount_id, "workspaceId": workspace_id},
                include={"mountSource": True},
            )
            if not mount or not bool(getattr(mount, "autoSyncEnabled", False)):
                return

            mount_source = getattr(mount, "mountSource", None)
            if (
                not mount_source
                or str(getattr(mount_source, "sourceType", "") or "") != "ssh"
            ):
                return

            sync_mode = self._normalize_workspace_mount_sync_mode(
                getattr(mount, "syncMode", None),
                legacy_sync_deletes=bool(getattr(mount, "syncDeletes", False)),
            )
            if self._is_destructive_workspace_mount_sync_mode(
                sync_mode
            ) and not self._has_destructive_auto_sync_approval(mount, sync_mode):
                await self._finalize_workspace_mount_sync(
                    db,
                    mount_id=mount_id,
                    workspace_id=workspace_id,
                    sync_mode=sync_mode,
                    sync_status="error",
                    files_synced=0,
                    sync_backend=str(getattr(mount, "syncBackend", "") or "") or None,
                    sync_notice=str(getattr(mount, "syncNotice", "") or "") or None,
                    last_sync_error=(
                        "Destructive auto-sync requires confirmation. Disable Auto and enable it again to review the dry run."
                    ),
                )
                try:
                    from ragtime.userspace.runtime_service import \
                        userspace_runtime_service

                    await userspace_runtime_service.bump_workspace_generation(
                        workspace_id,
                        "mount_sync",
                    )
                except Exception:
                    logger.debug(
                        "Failed to bump workspace generation after auto-sync status update for %s/%s",
                        workspace_id,
                        mount_id,
                        exc_info=True,
                    )
                return
            logger.debug(
                "Auto-syncing mount %s/%s in %s mode",
                workspace_id,
                mount_id,
                sync_mode,
            )
            await self._sync_workspace_mount_record(
                db,
                mount,
                allow_destructive_auto_sync_approval=True,
            )
            try:
                from ragtime.userspace.runtime_service import \
                    userspace_runtime_service

                await userspace_runtime_service.bump_workspace_generation(
                    workspace_id,
                    "mount_sync",
                )
            except Exception:
                logger.debug(
                    "Failed to bump workspace generation after auto-sync for %s/%s",
                    workspace_id,
                    mount_id,
                    exc_info=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Auto-sync watch run failed for mount %s/%s: %s",
                workspace_id,
                mount_id,
                exc,
                exc_info=True,
            )
        finally:
            elapsed = _time.monotonic() - t0
            if elapsed > source_interval_seconds:
                logger.warning(
                    "Mount %s sync took %.1fs, exceeding configured interval of %.0fs; "
                    "next sync deferred until cooldown elapses",
                    mount_id,
                    elapsed,
                    source_interval_seconds,
                )
            # Schedule from completion so the interval is measured between the
            # end of one sync and the start of the next.
            self._workspace_mount_watch_next_due_monotonic[mount_id] = (
                _time.monotonic()
                + source_interval_seconds
                + self._workspace_mount_watch_stagger_seconds(mount_id)
            )
            self._workspace_mount_watch_inflight.discard(mount_id)

    async def _startup_git_drift_reconciliation(self) -> None:
        """Reconcile existing workspaces sequentially in one background task."""
        try:
            db = await get_db()
            rows = await db.query_raw(
                """
                SELECT id
                FROM workspaces
                ORDER BY updated_at DESC
                """
            )
            for row in rows:
                workspace_id = str(row.get("id") or "").strip()
                if not workspace_id:
                    continue
                try:
                    await self._reconcile_workspace_git_drift(
                        workspace_id, reason="startup"
                    )
                except Exception as exc:
                    logger.debug(
                        "Startup drift reconcile failed for %s: %s",
                        workspace_id,
                        exc,
                    )
                # Yield between workspaces so startup reconciliation stays low impact.
                await asyncio.sleep(0)
        except Exception as exc:
            logger.debug("Skipped startup userspace git drift reconciliation: %s", exc)

    async def shutdown_git_drift_reconciliation(self) -> None:
        """Cancel the startup drift reconciliation task if it is still running."""
        if self._git_drift_startup_task and not self._git_drift_startup_task.done():
            self._git_drift_startup_task.cancel()
            try:
                await self._git_drift_startup_task
            except (asyncio.CancelledError, Exception):
                pass
        self._git_drift_startup_task = None

    async def shutdown_workspace_mount_watch(self) -> None:
        """Cancel the workspace mount watch task and clear watch state."""
        if (
            self._workspace_mount_watch_task
            and not self._workspace_mount_watch_task.done()
        ):
            self._workspace_mount_watch_task.cancel()
            try:
                await self._workspace_mount_watch_task
            except (asyncio.CancelledError, Exception):
                pass
        self._workspace_mount_watch_task = None
        self._workspace_mount_watch_inflight.clear()
        self._workspace_mount_watch_next_due_monotonic.clear()

    def _resolve_workspace_file_path(
        self, workspace_id: str, relative_path: str
    ) -> Path:
        if not relative_path or relative_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")

        _enforce_sqlite_file_path_policy(relative_path)

        files_dir = self._workspace_files_dir(workspace_id)
        target = (files_dir / relative_path).resolve()
        if files_dir.resolve() not in target.parents and target != files_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid file path")
        return target

    def resolve_workspace_file_path(
        self, workspace_id: str, relative_path: str
    ) -> Path:
        return self._resolve_workspace_file_path(workspace_id, relative_path)

    async def _resolve_workspace_tree_file_path(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> Path:
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if not normalized_path or normalized_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")

        best_prefix = ""
        best_source_dir: Path | None = None
        mount_specs = await self.resolve_workspace_mounts_for_runtime(workspace_id)
        for spec in mount_specs:
            repo_relative_prefix = self._workspace_mount_target_repo_relative_path(
                str(spec.get("target_path", "") or "")
            )
            if (
                not repo_relative_prefix
                or not self._workspace_path_matches_mount_prefix(
                    normalized_path,
                    repo_relative_prefix,
                )
            ):
                continue

            source_local_path = str(spec.get("source_local_path", "") or "").strip()
            if not source_local_path:
                continue

            if len(repo_relative_prefix) > len(best_prefix):
                best_prefix = repo_relative_prefix
                best_source_dir = Path(source_local_path)

        if best_source_dir is None:
            return self._resolve_workspace_file_path(workspace_id, normalized_path)

        suffix = normalized_path[len(best_prefix) :].lstrip("/")
        target = best_source_dir if not suffix else best_source_dir / suffix
        resolved_source_dir = best_source_dir.resolve()
        resolved_target = target.resolve()
        if (
            resolved_target != resolved_source_dir
            and resolved_source_dir not in resolved_target.parents
        ):
            raise HTTPException(status_code=400, detail="Invalid file path")
        return resolved_target

    @staticmethod
    def _normalize_workspace_relative_path(relative_path: str) -> str:
        normalized = (relative_path or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.lstrip("/")

    def _parse_git_status_changed_file_paths(self, git_status_output: str) -> list[str]:
        if not git_status_output:
            return []

        tokens = git_status_output.split("\x00")
        changed_file_paths: set[str] = set()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if not token:
                i += 1
                continue

            if len(token) < 3:
                i += 1
                continue

            status = token[:2]
            path = token[3:]
            candidate_path = path

            # In porcelain -z mode, renames/copies encode old path in the first token
            # and new path in the following token.
            if status and status[0] in {"R", "C"} and (i + 1) < len(tokens):
                next_token = tokens[i + 1]
                if next_token:
                    candidate_path = next_token
                i += 1

            normalized = self._normalize_workspace_relative_path(candidate_path)
            if not normalized or self._is_reserved_internal_path(normalized):
                i += 1
                continue

            changed_file_paths.add(normalized)
            i += 1

        return sorted(changed_file_paths)

    def _is_reserved_internal_path(self, relative_path: str) -> bool:
        normalized = relative_path.strip("/")
        if normalized.endswith(".artifact.json"):
            return True
        parts = Path(normalized).parts
        # Allow agent access to specific .ragtime/ files (e.g. runtime-entrypoint.json)
        if (
            len(parts) == 2
            and parts[0] == ".ragtime"
            and parts[1] in _AGENT_WRITABLE_RAGTIME_FILES
        ):
            return False
        # Allow agent access to writable .ragtime/ subtrees (e.g. db/migrations/)
        if len(parts) >= 2 and parts[0] == ".ragtime":
            sub_path = "/".join(parts[1:])
            for prefix in _AGENT_WRITABLE_RAGTIME_PREFIXES:
                # Match files under the prefix or the prefix directory itself
                if sub_path.startswith(prefix) or (sub_path + "/").startswith(prefix):
                    return False
        return bool(_HIDDEN_DIRS.intersection(parts))

    def is_reserved_internal_path(self, relative_path: str) -> bool:
        return self._is_reserved_internal_path(relative_path)

    async def _get_snapshot_record(
        self,
        workspace_id: str,
        snapshot_id: str,
    ) -> dict[str, Any]:
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT s.id, s.workspace_id, s.branch_id, s.git_commit_hash, s.message,
                 s.remote_commit_hash, s.file_count, s.parent_snapshot_id,
                 s.created_at,
                   b.name AS branch_name
            FROM userspace_snapshots s
            JOIN userspace_snapshot_branches b ON b.id = s.branch_id
            WHERE s.workspace_id = {self._sql_quote(workspace_id)}
              AND s.id = {self._sql_quote(snapshot_id)}
            LIMIT 1
            """
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return rows[0]

    @staticmethod
    def _decode_optional_text_content(content: bytes) -> str | None:
        if not content:
            return ""
        if b"\x00" in content:
            return None
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _count_text_lines(content: str) -> int:
        if not content:
            return 0
        return content.count("\n") + (0 if content.endswith("\n") else 1)

    async def _read_git_text_content(
        self,
        workspace_id: str,
        git_object: str,
    ) -> tuple[str, bool]:
        returncode, stdout_bytes, _ = await self._run_git_bytes(
            workspace_id,
            ["show", git_object],
            check=False,
        )
        if returncode != 0:
            return "", False

        decoded = self._decode_optional_text_content(stdout_bytes)
        if decoded is None:
            return "", True
        return decoded, False

    async def _read_workspace_text_content(self, file_path: Path) -> tuple[str, bool]:
        file_bytes = await asyncio.to_thread(file_path.read_bytes)
        decoded = self._decode_optional_text_content(file_bytes)
        if decoded is None:
            return "", True
        return decoded, False

    def _parse_name_status_line(
        self,
        raw_line: str,
    ) -> tuple[str, str | None, str | None]:
        parts = raw_line.split("\t")
        if len(parts) < 2:
            raise ValueError("Invalid git name-status line")
        status_token = parts[0].strip().upper()
        status = status_token[:1]
        if status not in {"A", "D", "M", "R"}:
            status = "M"
        if status == "R" and len(parts) >= 3:
            return status, parts[2], parts[1]
        return status, parts[1], None

    async def _apply_numstat_batch(
        self,
        workspace_id: str,
        git_refs: list[str],
        summary_by_path: dict[str, _WorkspaceSnapshotFileDiff],
    ) -> None:
        """Run a single git diff --numstat and apply additions/deletions/is_binary to all summaries."""
        numstat_result = await self._run_git(
            workspace_id,
            ["diff", "--find-renames", "--numstat", *git_refs, "--"],
            check=False,
        )
        if numstat_result.returncode != 0:
            return
        # Build a reverse lookup: old_path -> summary for renames
        old_path_lookup: dict[str, _WorkspaceSnapshotFileDiff] = {}
        for summary in summary_by_path.values():
            if summary["status"] == "R" and summary["old_path"]:
                old_path_lookup[summary["old_path"]] = summary
        for raw_line in numstat_result.stdout.splitlines():
            parts = raw_line.split("\t")
            if len(parts) < 3:
                continue
            add_token, delete_token = parts[0].strip(), parts[1].strip()
            numstat_path = parts[2].strip()
            # For renames, numstat shows the new path with --find-renames
            # but may use {old => new} notation — try the raw path, then old_path lookup
            matched: _WorkspaceSnapshotFileDiff | None = summary_by_path.get(
                numstat_path
            ) or old_path_lookup.get(numstat_path)
            if matched is None:
                # Try normalizing in case of path format differences
                normalized = self._normalize_workspace_relative_path(numstat_path)
                if normalized:
                    matched = summary_by_path.get(normalized) or old_path_lookup.get(
                        normalized
                    )
            if matched is None:
                continue
            if add_token == "-" or delete_token == "-":
                matched["is_binary"] = True
                matched["additions"] = 0
                matched["deletions"] = 0
            else:
                try:
                    matched["additions"] = int(add_token)
                    matched["deletions"] = int(delete_token)
                except ValueError:
                    pass

    async def _resolve_snapshot_parent_ref(
        self,
        workspace_id: str,
        snapshot_commit_hash: str,
    ) -> str:
        """Return the parent commit hash, or the empty-tree hash for root commits."""
        result = await self._run_git(
            workspace_id,
            ["rev-parse", "--verify", f"{snapshot_commit_hash}^"],
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        empty_tree = await self._run_git(
            workspace_id,
            ["hash-object", "-t", "tree", "/dev/null"],
            check=False,
        )
        if empty_tree.returncode == 0 and empty_tree.stdout.strip():
            return empty_tree.stdout.strip()
        return _GIT_EMPTY_TREE_HASH

    async def _build_snapshot_own_diff_summary_map(
        self,
        workspace_id: str,
        snapshot_commit_hash: str,
        parent_ref: str,
    ) -> dict[str, _WorkspaceSnapshotFileDiff]:
        """Diff parent (or empty tree) -> snapshot to show what the snapshot itself contains."""
        summary_by_path: dict[str, _WorkspaceSnapshotFileDiff] = {}
        diff_result = await self._run_git(
            workspace_id,
            [
                "diff",
                "--find-renames",
                "--name-status",
                parent_ref,
                snapshot_commit_hash,
            ],
            check=False,
        )
        if diff_result.returncode != 0:
            return summary_by_path

        for line in diff_result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed_status, candidate_path, old_path = self._parse_name_status_line(
                    stripped
                )
            except ValueError:
                continue

            normalized_path = self._normalize_workspace_relative_path(
                candidate_path or ""
            )
            normalized_old_path = (
                self._normalize_workspace_relative_path(old_path or "") or None
            )
            if not normalized_path or self._is_reserved_internal_path(normalized_path):
                continue
            if normalized_old_path and self._is_reserved_internal_path(
                normalized_old_path
            ):
                normalized_old_path = None
            summary_by_path[normalized_path] = cast(
                _WorkspaceSnapshotFileDiff,
                {
                    "path": normalized_path,
                    "status": cast(Literal["A", "D", "M", "R"], parsed_status),
                    "old_path": normalized_old_path,
                    "additions": 0,
                    "deletions": 0,
                    "is_binary": False,
                    "is_untracked_in_current": False,
                },
            )

        await self._apply_numstat_batch(
            workspace_id, [parent_ref, snapshot_commit_hash], summary_by_path
        )

        return summary_by_path

    async def _build_snapshot_diff_summary_map(
        self,
        workspace_id: str,
        snapshot_commit_hash: str,
    ) -> dict[str, _WorkspaceSnapshotFileDiff]:
        summary_by_path: dict[str, _WorkspaceSnapshotFileDiff] = {}
        diff_result = await self._run_git(
            workspace_id,
            ["diff", "--find-renames", "--name-status", snapshot_commit_hash, "--"],
            check=False,
        )
        if diff_result.returncode != 0:
            stderr = (diff_result.stderr or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compute snapshot diff summary: {stderr or 'git diff failed'}",
            )

        for line in diff_result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed_status, candidate_path, old_path = self._parse_name_status_line(
                    stripped
                )
            except ValueError:
                continue

            normalized_path = self._normalize_workspace_relative_path(
                candidate_path or ""
            )
            normalized_old_path = (
                self._normalize_workspace_relative_path(old_path or "") or None
            )
            if not normalized_path or self._is_reserved_internal_path(normalized_path):
                continue
            if normalized_old_path and self._is_reserved_internal_path(
                normalized_old_path
            ):
                normalized_old_path = None
            summary_by_path[normalized_path] = cast(
                _WorkspaceSnapshotFileDiff,
                {
                    "path": normalized_path,
                    "status": cast(Literal["A", "D", "M", "R"], parsed_status),
                    "old_path": normalized_old_path,
                    "additions": 0,
                    "deletions": 0,
                    "is_binary": False,
                    "is_untracked_in_current": False,
                },
            )

        await self._apply_numstat_batch(
            workspace_id, [snapshot_commit_hash], summary_by_path
        )

        async with self._git_status_semaphore:
            status_result = await self._run_git(
                workspace_id,
                ["status", "--porcelain=1", "-z", "--untracked-files=all"],
                check=False,
            )
        if status_result.returncode != 0:
            return summary_by_path

        tokens = status_result.stdout.split("\x00")
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token or len(token) < 3:
                index += 1
                continue
            status_token = token[:2]
            candidate_path = token[3:]
            previous_path: str | None = None
            if (
                status_token
                and status_token[0] in {"R", "C"}
                and (index + 1) < len(tokens)
            ):
                next_token = tokens[index + 1]
                if next_token:
                    previous_path = candidate_path
                    candidate_path = next_token
                index += 1

            normalized_path = self._normalize_workspace_relative_path(candidate_path)
            normalized_previous_path = (
                self._normalize_workspace_relative_path(previous_path or "") or None
            )
            if not normalized_path or self._is_reserved_internal_path(normalized_path):
                index += 1
                continue

            entry = summary_by_path.get(normalized_path)
            if status_token == "??":
                file_path = self._resolve_workspace_file_path(
                    workspace_id, normalized_path
                )
                if file_path.exists() and file_path.is_file():
                    raw_content = await asyncio.to_thread(file_path.read_bytes)
                    decoded = self._decode_optional_text_content(raw_content)
                    summary_by_path[normalized_path] = cast(
                        _WorkspaceSnapshotFileDiff,
                        {
                            "path": normalized_path,
                            "status": "A",
                            "old_path": None,
                            "additions": (
                                0
                                if decoded is None
                                else self._count_text_lines(decoded)
                            ),
                            "deletions": 0,
                            "is_binary": decoded is None,
                            "is_untracked_in_current": True,
                        },
                    )
                index += 1
                continue

            if entry is None:
                derived_status = "M"
                if "A" in status_token:
                    derived_status = "A"
                elif "D" in status_token:
                    derived_status = "D"
                elif "R" in status_token:
                    derived_status = "R"
                summary_by_path[normalized_path] = cast(
                    _WorkspaceSnapshotFileDiff,
                    {
                        "path": normalized_path,
                        "status": cast(Literal["A", "D", "M", "R"], derived_status),
                        "old_path": normalized_previous_path,
                        "additions": 0,
                        "deletions": 0,
                        "is_binary": False,
                        "is_untracked_in_current": False,
                    },
                )
            index += 1

        return summary_by_path

    async def _prepare_snapshot_diff_context(
        self,
        workspace_id: str,
        snapshot_id: str,
        user_id: str,
    ) -> tuple[str, dict[str, _WorkspaceSnapshotFileDiff], bool]:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        snapshot_row = await self._get_snapshot_record(workspace_id, snapshot_id)
        snapshot_commit_hash = str(snapshot_row.get("git_commit_hash") or "")
        summary_map = await self._build_snapshot_diff_summary_map(
            workspace_id,
            snapshot_commit_hash,
        )

        is_snapshot_own_diff = False
        if not summary_map and snapshot_commit_hash:
            parent_ref = await self._resolve_snapshot_parent_ref(
                workspace_id, snapshot_commit_hash
            )
            own_diff_map = await self._build_snapshot_own_diff_summary_map(
                workspace_id, snapshot_commit_hash, parent_ref
            )
            if own_diff_map:
                summary_map = own_diff_map
                is_snapshot_own_diff = True

        return snapshot_commit_hash, summary_map, is_snapshot_own_diff

    async def get_snapshot_diff_summary(
        self,
        workspace_id: str,
        snapshot_id: str,
        user_id: str,
    ) -> UserSpaceSnapshotDiffSummaryResponse:
        snapshot_commit_hash, summary_map, is_snapshot_own_diff = (
            await self._prepare_snapshot_diff_context(
                workspace_id,
                snapshot_id,
                user_id,
            )
        )
        files = [
            UserSpaceSnapshotDiffFileSummary(
                path=item["path"],
                status=item["status"],
                old_path=item["old_path"],
                additions=item["additions"],
                deletions=item["deletions"],
                is_binary=item["is_binary"],
            )
            for item in sorted(summary_map.values(), key=lambda value: value["path"])
        ]
        return UserSpaceSnapshotDiffSummaryResponse(
            workspace_id=workspace_id,
            snapshot_id=snapshot_id,
            snapshot_commit_hash=snapshot_commit_hash or None,
            files=files,
            is_snapshot_own_diff=is_snapshot_own_diff,
        )

    async def _lookup_single_file_diff_entry(
        self,
        workspace_id: str,
        snapshot_commit_hash: str,
        normalized_path: str,
    ) -> tuple[_WorkspaceSnapshotFileDiff | None, bool]:
        """Lightweight lookup for a single file's diff metadata without rebuilding the full summary map.

        Returns (file_entry, is_snapshot_own_diff).
        """
        # Try working-tree vs snapshot first (the common case)
        name_status = await self._run_git(
            workspace_id,
            [
                "diff",
                "--find-renames",
                "--name-status",
                snapshot_commit_hash,
                "--",
                normalized_path,
            ],
            check=False,
        )
        entry: _WorkspaceSnapshotFileDiff | None = None
        is_snapshot_own_diff = False

        if name_status.returncode == 0:
            for line in name_status.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed_status, candidate_path, old_path = (
                        self._parse_name_status_line(stripped)
                    )
                except ValueError:
                    continue
                np = self._normalize_workspace_relative_path(candidate_path or "")
                if np == normalized_path:
                    nop = (
                        self._normalize_workspace_relative_path(old_path or "") or None
                    )
                    entry = cast(
                        _WorkspaceSnapshotFileDiff,
                        {
                            "path": normalized_path,
                            "status": cast(Literal["A", "D", "M", "R"], parsed_status),
                            "old_path": nop,
                            "additions": 0,
                            "deletions": 0,
                            "is_binary": False,
                            "is_untracked_in_current": False,
                        },
                    )
                    break

        # Also check renames where normalized_path is the old name
        if entry is None and name_status.returncode == 0:
            rename_check = await self._run_git(
                workspace_id,
                [
                    "diff",
                    "--find-renames",
                    "--name-status",
                    snapshot_commit_hash,
                    "--",
                ],
                check=False,
            )
            if rename_check.returncode == 0:
                for line in rename_check.stdout.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed_status, candidate_path, old_path = (
                            self._parse_name_status_line(stripped)
                        )
                    except ValueError:
                        continue
                    np = self._normalize_workspace_relative_path(candidate_path or "")
                    if np == normalized_path:
                        nop = (
                            self._normalize_workspace_relative_path(old_path or "")
                            or None
                        )
                        entry = cast(
                            _WorkspaceSnapshotFileDiff,
                            {
                                "path": normalized_path,
                                "status": cast(
                                    Literal["A", "D", "M", "R"], parsed_status
                                ),
                                "old_path": nop,
                                "additions": 0,
                                "deletions": 0,
                                "is_binary": False,
                                "is_untracked_in_current": False,
                            },
                        )
                        break

        # Check git status for untracked/staged-only files invisible to git diff
        if entry is None:
            async with self._git_status_semaphore:
                status_result = await self._run_git(
                    workspace_id,
                    [
                        "status",
                        "--porcelain=1",
                        "-z",
                        "--untracked-files=all",
                        "--",
                        normalized_path,
                    ],
                    check=False,
                )
            if status_result.returncode == 0 and status_result.stdout:
                tokens = status_result.stdout.split("\x00")
                for token in tokens:
                    if not token or len(token) < 3:
                        continue
                    status_token = token[:2]
                    candidate_path = token[3:]
                    np = self._normalize_workspace_relative_path(candidate_path)
                    if np != normalized_path:
                        continue
                    if status_token == "??":
                        file_path = self._resolve_workspace_file_path(
                            workspace_id, normalized_path
                        )
                        is_binary = False
                        additions = 0
                        if file_path.exists() and file_path.is_file():
                            raw_content = await asyncio.to_thread(file_path.read_bytes)
                            decoded = self._decode_optional_text_content(raw_content)
                            is_binary = decoded is None
                            additions = (
                                0
                                if decoded is None
                                else self._count_text_lines(decoded)
                            )
                        entry = cast(
                            _WorkspaceSnapshotFileDiff,
                            {
                                "path": normalized_path,
                                "status": "A",
                                "old_path": None,
                                "additions": additions,
                                "deletions": 0,
                                "is_binary": is_binary,
                                "is_untracked_in_current": True,
                            },
                        )
                    else:
                        derived_status: Literal["A", "D", "M", "R"] = "M"
                        if "A" in status_token:
                            derived_status = "A"
                        elif "D" in status_token:
                            derived_status = "D"
                        elif "R" in status_token:
                            derived_status = "R"
                        entry = cast(
                            _WorkspaceSnapshotFileDiff,
                            {
                                "path": normalized_path,
                                "status": derived_status,
                                "old_path": None,
                                "additions": 0,
                                "deletions": 0,
                                "is_binary": False,
                                "is_untracked_in_current": False,
                            },
                        )
                    break

        # Fallback: try snapshot's own diff (parent -> snapshot)
        if entry is None and snapshot_commit_hash:
            parent_ref = await self._resolve_snapshot_parent_ref(
                workspace_id, snapshot_commit_hash
            )
            own_status = await self._run_git(
                workspace_id,
                [
                    "diff",
                    "--find-renames",
                    "--name-status",
                    parent_ref,
                    snapshot_commit_hash,
                    "--",
                ],
                check=False,
            )
            if own_status.returncode == 0:
                for line in own_status.stdout.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed_status, candidate_path, old_path = (
                            self._parse_name_status_line(stripped)
                        )
                    except ValueError:
                        continue
                    np = self._normalize_workspace_relative_path(candidate_path or "")
                    if np == normalized_path:
                        nop = (
                            self._normalize_workspace_relative_path(old_path or "")
                            or None
                        )
                        entry = cast(
                            _WorkspaceSnapshotFileDiff,
                            {
                                "path": normalized_path,
                                "status": cast(
                                    Literal["A", "D", "M", "R"], parsed_status
                                ),
                                "old_path": nop,
                                "additions": 0,
                                "deletions": 0,
                                "is_binary": False,
                                "is_untracked_in_current": False,
                            },
                        )
                        is_snapshot_own_diff = True
                        break

        # Apply numstat for the single file
        if entry is not None:
            single_map = {normalized_path: entry}
            if is_snapshot_own_diff:
                parent_ref = await self._resolve_snapshot_parent_ref(
                    workspace_id, snapshot_commit_hash
                )
                await self._apply_numstat_batch(
                    workspace_id, [parent_ref, snapshot_commit_hash], single_map
                )
            else:
                await self._apply_numstat_batch(
                    workspace_id, [snapshot_commit_hash], single_map
                )

        return entry, is_snapshot_own_diff

    async def get_snapshot_file_diff(
        self,
        workspace_id: str,
        snapshot_id: str,
        relative_path: str,
        user_id: str,
    ) -> UserSpaceSnapshotFileDiffResponse:
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if not normalized_path or self._is_reserved_internal_path(normalized_path):
            raise HTTPException(status_code=400, detail="Invalid file path")
        normalized_path = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_path,
        )

        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        snapshot_row = await self._get_snapshot_record(workspace_id, snapshot_id)
        snapshot_commit_hash = str(snapshot_row.get("git_commit_hash") or "")

        file_summary, is_snapshot_own_diff = await self._lookup_single_file_diff_entry(
            workspace_id, snapshot_commit_hash, normalized_path
        )
        if file_summary is None:
            raise HTTPException(status_code=404, detail="Snapshot diff file not found")

        before_path = file_summary["old_path"] or normalized_path
        after_path = normalized_path
        before_content = ""
        after_content = ""
        is_binary = bool(file_summary["is_binary"])
        is_deleted_in_current = file_summary["status"] == "D"
        is_untracked_in_current = bool(file_summary["is_untracked_in_current"])

        if is_snapshot_own_diff:
            parent_ref = await self._resolve_snapshot_parent_ref(
                workspace_id, snapshot_commit_hash
            )
            if file_summary["status"] != "A":
                before_content, before_is_binary = await self._read_git_text_content(
                    workspace_id,
                    f"{parent_ref}:{before_path}",
                )
                is_binary = is_binary or before_is_binary
            after_content, after_is_binary = await self._read_git_text_content(
                workspace_id,
                f"{snapshot_commit_hash}:{after_path}",
            )
            is_binary = is_binary or after_is_binary
        else:
            if file_summary["status"] != "A":
                before_content, before_is_binary = await self._read_git_text_content(
                    workspace_id,
                    f"{snapshot_commit_hash}:{before_path}",
                )
                is_binary = is_binary or before_is_binary

            current_file_path = self._resolve_workspace_file_path(
                workspace_id, after_path
            )
            if (
                not is_deleted_in_current
                and current_file_path.exists()
                and current_file_path.is_file()
            ):
                after_content, after_is_binary = (
                    await self._read_workspace_text_content(current_file_path)
                )
                is_binary = is_binary or after_is_binary

        is_truncated = False
        if not is_binary:
            content_size = len(before_content.encode("utf-8", errors="replace")) + len(
                after_content.encode("utf-8", errors="replace")
            )
            if content_size > _SNAPSHOT_DIFF_MAX_FILE_BYTES:
                is_truncated = True
                before_content = ""
                after_content = ""

        message: str | None = None
        if is_binary:
            message = (
                "Binary or non-UTF-8 content cannot be rendered in the diff viewer."
            )
            before_content = ""
            after_content = ""
        elif is_truncated:
            message = "File content is too large to display in the diff viewer."
        elif file_summary["status"] == "D":
            message = (
                "File exists in the snapshot but is deleted in the current workspace."
            )
        elif is_untracked_in_current:
            message = "File is untracked in the current workspace."

        return UserSpaceSnapshotFileDiffResponse(
            workspace_id=workspace_id,
            snapshot_id=snapshot_id,
            path=normalized_path,
            status=file_summary["status"],
            old_path=file_summary["old_path"],
            before_path=before_path,
            after_path=after_path,
            before_content=before_content,
            after_content=after_content,
            additions=file_summary["additions"],
            deletions=file_summary["deletions"],
            is_binary=is_binary,
            is_deleted_in_current=is_deleted_in_current,
            is_untracked_in_current=is_untracked_in_current,
            is_snapshot_own_diff=is_snapshot_own_diff,
            is_truncated=is_truncated,
            message=message,
        )

    @staticmethod
    def _generate_share_token() -> str:
        return secrets.token_urlsafe(24)

    def _build_workspace_share_url(
        self,
        owner_username: str,
        share_slug: str,
        base_url: str | None = None,
    ) -> str:
        normalized_base = (base_url or "").strip().rstrip("/")
        owner_segment = quote(owner_username, safe="")
        slug_segment = quote(share_slug, safe="")
        share_path = f"/{owner_segment}/{slug_segment}"
        return f"{normalized_base}{share_path}" if normalized_base else share_path

    def _build_workspace_anonymous_share_url(
        self,
        share_token: str,
        base_url: str | None = None,
    ) -> str:
        normalized_base = (base_url or "").strip().rstrip("/")
        share_path = f"/shared/{quote(share_token, safe='')}"
        return f"{normalized_base}{share_path}" if normalized_base else share_path

    def build_workspace_anonymous_share_url(
        self,
        share_token: str,
        base_url: str | None = None,
    ) -> str:
        return self._build_workspace_anonymous_share_url(
            share_token,
            base_url=base_url,
        )

    def is_direct_share_subdomain_allowed(self, workspace_record: Any) -> bool:
        share_token = str(getattr(workspace_record, "shareToken", "") or "").strip()
        mode, password_encrypted, _, _ = self._extract_share_access_state(
            workspace_record
        )
        return bool(share_token) and mode == "token" and not bool(password_encrypted)

    async def is_public_direct_share_host_enabled(self, workspace_id: str) -> bool:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            return False
        return self.is_direct_share_subdomain_allowed(workspace_record)

    async def has_active_share_link(self, workspace_id: str) -> bool:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            return False
        return bool(str(getattr(workspace_record, "shareToken", "") or "").strip())

    async def get_share_access_mode(self, workspace_id: str) -> str | None:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            return None
        mode, _, _, _ = self._extract_share_access_state(workspace_record)
        return mode

    async def get_share_token(self, workspace_id: str) -> str | None:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            return None
        return str(getattr(workspace_record, "shareToken", "") or "").strip() or None

    def _share_prompt_metadata_from_record(
        self, workspace_record: Any
    ) -> tuple[str | None, str | None]:
        workspace_name = (
            str(getattr(workspace_record, "name", "") or "").strip() or None
        )
        owner_obj = getattr(workspace_record, "owner", None)
        owner_display_name = (
            str(getattr(owner_obj, "displayName", "") or "").strip() or None
        )
        return workspace_name, owner_display_name

    async def _resolve_share_owner_ids(self, owner_username: str) -> list[str]:
        normalized_owner = _normalize_owner_username_for_share_path(owner_username)
        if not normalized_owner:
            return []

        db = await get_db()
        candidate_usernames = [normalized_owner]
        if not normalized_owner.startswith("local:"):
            candidate_usernames.append(f"local:{normalized_owner}")

        users = await db.user.find_many(
            where={"username": {"in": candidate_usernames, "mode": "insensitive"}},
            take=5,
        )
        return [
            str(getattr(user, "id", ""))
            for user in users
            if _normalize_owner_username_for_share_path(
                str(getattr(user, "username", "") or "")
            )
            == normalized_owner
        ]

    async def get_share_prompt_metadata_by_token(
        self,
        share_token: str,
    ) -> tuple[str | None, str | None]:
        token = (share_token or "").strip()
        if not token:
            return None, None

        db = await get_db()
        workspace = await db.workspace.find_first(
            where={"shareToken": token},
            include={"owner": True},
        )
        if not workspace:
            return None, None
        return self._share_prompt_metadata_from_record(workspace)

    async def get_share_prompt_metadata_by_slug(
        self,
        owner_username: str,
        share_slug: str,
    ) -> tuple[str | None, str | None]:
        normalized_slug = _normalize_share_slug_for_uniqueness(share_slug)
        if not normalized_slug:
            return None, None

        owner_ids = await self._resolve_share_owner_ids(owner_username)
        if not owner_ids:
            return None, None

        db = await get_db()
        workspace = await db.workspace.find_first(
            where={
                "ownerUserId": {"in": owner_ids},
                "shareSlug": normalized_slug,
                "shareToken": {"not": None},
            },
            include={"owner": True},
        )
        if not workspace:
            return None, None
        return self._share_prompt_metadata_from_record(workspace)

    async def _resolve_workspace_id_from_share_token(self, share_token: str) -> str:
        token = (share_token or "").strip()
        if not token:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        db = await get_db()
        workspace = await db.workspace.find_first(where={"shareToken": token})
        if not workspace:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        return str(workspace.id)

    async def _resolve_workspace_id_from_share_slug(
        self,
        owner_username: str,
        share_slug: str,
    ) -> str:
        normalized_slug = _normalize_share_slug_for_uniqueness(share_slug)
        if not normalized_slug:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        owner_ids = await self._resolve_share_owner_ids(owner_username)
        if not owner_ids:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        db = await get_db()
        workspace = await db.workspace.find_first(
            where={
                "ownerUserId": {"in": owner_ids},
                "shareSlug": normalized_slug,
                "shareToken": {"not": None},
            },
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        return str(workspace.id)

    async def _get_public_owner_username(self, owner_user_id: str) -> str:
        db = await get_db()
        owner = await db.user.find_unique(where={"id": owner_user_id})
        if not owner:
            raise HTTPException(status_code=404, detail="Workspace owner not found")

        normalized = _normalize_owner_username_for_share_path(
            str(getattr(owner, "username", "") or "")
        )
        if not normalized:
            raise HTTPException(
                status_code=400, detail="Workspace owner username invalid"
            )
        return normalized

    async def _allocate_next_share_slug(
        self,
        owner_user_id: str,
        preferred_name: str,
    ) -> str:
        del owner_user_id, preferred_name
        return f"{_DEFAULT_SHARE_SLUG_PREFIX}_{secrets.token_hex(4)}"

    def _extract_share_access_state(
        self,
        workspace_record: Any,
    ) -> tuple[ShareAccessMode, str | None, list[str], list[str]]:
        mode = _normalize_share_access_mode(
            str(getattr(workspace_record, "shareAccessMode", "token") or "token")
        )
        password_encrypted = str(getattr(workspace_record, "sharePassword", "") or "")
        selected_user_ids = _normalize_string_list(
            getattr(workspace_record, "shareSelectedUserIds", [])
        )
        selected_ldap_groups = _normalize_string_list(
            getattr(workspace_record, "shareSelectedLdapGroups", [])
        )
        return (
            mode,
            (password_encrypted or None),
            selected_user_ids,
            selected_ldap_groups,
        )

    async def _is_user_in_ldap_group(self, user: Any, group_dn: str) -> bool:
        if not getattr(user, "ldapDn", None):
            return False

        ldap_config = await get_ldap_config()
        if not ldap_config.serverUrl:
            return False

        return await asyncio.to_thread(
            self._is_user_in_ldap_group_sync,
            str(getattr(user, "ldapDn", "") or ""),
            group_dn,
            str(ldap_config.serverUrl),
            str(ldap_config.bindDn),
            str(ldap_config.bindPassword),
            bool(ldap_config.allowSelfSigned),
        )

    @staticmethod
    def _is_user_in_ldap_group_sync(
        user_ldap_dn: str,
        group_dn: str,
        server_url: str,
        bind_dn: str,
        bind_password_encrypted: str,
        allow_self_signed: bool,
    ) -> bool:
        bind_password = decrypt_secret(bind_password_encrypted)
        conn = _get_ldap_connection(
            server_url,
            bind_dn,
            bind_password,
            allow_self_signed,
        )
        if not conn:
            return False

        try:
            conn.search(
                search_base=user_ldap_dn,
                search_filter="(objectClass=*)",
                search_scope="BASE",
                attributes=["memberOf", "primaryGroupID"],
            )
            if not conn.entries:
                return False

            entry = conn.entries[0]
            member_of: list[str] = []
            if hasattr(entry, "memberOf") and entry.memberOf:
                member_of = [str(value).lower() for value in entry.memberOf]
            if group_dn.lower() in member_of:
                return True

            primary_group_id = None
            if hasattr(entry, "primaryGroupID") and entry.primaryGroupID:
                primary_group_id = int(str(entry.primaryGroupID))

            if primary_group_id:
                conn.search(
                    search_base=group_dn,
                    search_filter="(objectClass=*)",
                    search_scope="BASE",
                    attributes=["primaryGroupToken"],
                )
                if conn.entries:
                    group_entry = conn.entries[0]
                    if (
                        hasattr(group_entry, "primaryGroupToken")
                        and group_entry.primaryGroupToken
                    ):
                        group_rid = int(str(group_entry.primaryGroupToken))
                        return group_rid == primary_group_id

            return False
        except Exception:
            return False
        finally:
            if conn.bound:
                conn.unbind()

    async def _enforce_share_access(
        self,
        workspace_record: Any,
        current_user: Any | None,
        provided_password: str | None,
    ) -> None:
        mode, password_encrypted, selected_user_ids, selected_ldap_groups = (
            self._extract_share_access_state(workspace_record)
        )

        if mode == "token":
            return

        if mode == "password":
            if not password_encrypted:
                raise HTTPException(
                    status_code=403, detail="Share password not configured"
                )
            provided = (provided_password or "").strip()
            if not provided:
                raise HTTPException(status_code=401, detail="Password required")
            if not hmac.compare_digest(
                decrypt_secret(password_encrypted) or "", provided
            ):
                raise HTTPException(status_code=401, detail="Invalid password")
            return

        # Owner bypass: workspace owner skips group/user-list checks.
        # Password mode is handled above and always requires the password.
        if current_user is not None:
            owner_id = str(getattr(workspace_record, "ownerUserId", "") or "").strip()
            if (
                owner_id
                and owner_id == str(getattr(current_user, "id", "") or "").strip()
            ):
                return

        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        if mode == "authenticated_users":
            return

        if mode == "selected_users":
            if str(getattr(current_user, "id", "")) not in selected_user_ids:
                raise HTTPException(
                    status_code=403, detail="User not allowed for this share"
                )
            return

        if mode == "ldap_groups":
            if (
                getattr(current_user, "authProvider", None) == "local"
                and getattr(current_user, "role", None) == "admin"
            ):
                return
            for group_dn in selected_ldap_groups:
                if await self._is_user_in_ldap_group(current_user, group_dn):
                    return
            raise HTTPException(
                status_code=403, detail="User not in allowed LDAP groups"
            )

    @staticmethod
    def _share_password_access_proof(workspace_record: Any) -> str | None:
        password_encrypted = str(getattr(workspace_record, "sharePassword", "") or "")
        share_token = str(getattr(workspace_record, "shareToken", "") or "")
        workspace_id = str(getattr(workspace_record, "id", "") or "")
        if not password_encrypted or not share_token or not workspace_id:
            return None
        return hashlib.sha256(
            f"{workspace_id}:{share_token}:{password_encrypted}".encode("utf-8")
        ).hexdigest()

    def _build_share_password_access_token(
        self,
        workspace_record: Any,
    ) -> tuple[str, datetime]:
        workspace_id = str(getattr(workspace_record, "id", "") or "").strip()
        proof = self._share_password_access_proof(workspace_record)
        if not workspace_id or not proof:
            raise HTTPException(status_code=403, detail="Share password not configured")

        now = _utc_now()
        expires_at = now + timedelta(seconds=_SHARE_PASSWORD_ACCESS_TTL_SECONDS)
        payload = {
            "kind": _SHARE_PASSWORD_ACCESS_TOKEN_KIND,
            "workspace_id": workspace_id,
            "share_access_mode": "password",
            "proof": proof,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(uuid4()),
        }
        token = jwt.encode(
            payload, settings.encryption_key, algorithm=settings.jwt_algorithm
        )
        return token, expires_at

    def _verify_share_password_access_token(
        self,
        token: str,
        workspace_record: Any,
    ) -> datetime:
        try:
            claims = cast(
                dict[str, Any],
                jwt.decode(
                    token,
                    settings.encryption_key,
                    algorithms=[settings.jwt_algorithm],
                ),
            )
        except JWTError as exc:
            raise HTTPException(status_code=401, detail="Password required") from exc

        workspace_id = str(getattr(workspace_record, "id", "") or "").strip()
        expected_proof = self._share_password_access_proof(workspace_record)
        if (
            str(claims.get("kind") or "").strip() != _SHARE_PASSWORD_ACCESS_TOKEN_KIND
            or str(claims.get("workspace_id") or "").strip() != workspace_id
            or str(claims.get("share_access_mode") or "").strip() != "password"
            or not expected_proof
            or not hmac.compare_digest(str(claims.get("proof") or ""), expected_proof)
        ):
            raise HTTPException(status_code=401, detail="Password required")

        exp = claims.get("exp")
        if not isinstance(exp, (int, float)):
            raise HTTPException(status_code=401, detail="Password required")
        return datetime.fromtimestamp(int(exp), tz=timezone.utc)

    async def _authorize_shared_workspace_record(
        self,
        workspace_record: Any,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> _ShareAuthorizationResult:
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        mode, _, _, _ = self._extract_share_access_state(workspace_record)
        resolved_token: str | None = None
        expires_at: datetime | None = None

        if mode == "password":
            candidate_token = str(share_auth_token or "").strip()
            if candidate_token:
                try:
                    expires_at = self._verify_share_password_access_token(
                        candidate_token,
                        workspace_record,
                    )
                    resolved_token = candidate_token
                except HTTPException:
                    if not str(password or "").strip():
                        raise

            if resolved_token is None:
                await self._enforce_share_access(
                    workspace_record,
                    current_user,
                    password,
                )
                resolved_token, expires_at = self._build_share_password_access_token(
                    workspace_record
                )
        else:
            await self._enforce_share_access(
                workspace_record,
                current_user,
                password,
            )

        return {
            "workspace_id": str(getattr(workspace_record, "id", "") or ""),
            "share_auth_token": resolved_token,
            "expires_at": expires_at,
        }

    async def _allocate_next_default_workspace_name(self, user_id: str) -> str:
        db = await get_db()
        owner_workspaces = await db.workspace.find_many(
            where={"ownerUserId": user_id},
            take=10000,
            order={"createdAt": "asc"},
        )
        used: set[str] = {
            str(getattr(workspace, "nameNormalized", "") or "")
            for workspace in owner_workspaces
            if str(getattr(workspace, "nameNormalized", "") or "")
        }

        next_index = 1
        while True:
            candidate = f"Workspace {next_index}"
            normalized = _normalize_workspace_name_for_uniqueness(candidate)
            if normalized not in used:
                return candidate
            next_index += 1

    @staticmethod
    def _read_artifact_sidecar(
        file_path: Path,
    ) -> tuple[
        ArtifactType | None,
        list[UserSpaceLiveDataConnection] | None,
        list[UserSpaceLiveDataCheck] | None,
    ]:
        artifact_type: ArtifactType | None = None
        live_data_connections: list[UserSpaceLiveDataConnection] | None = None
        live_data_checks: list[UserSpaceLiveDataCheck] | None = None

        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if not sidecar.exists():
            return artifact_type, live_data_connections, live_data_checks

        try:
            sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_value = sidecar_data.get("artifact_type")
            if sidecar_value == "module_ts":
                artifact_type = cast(ArtifactType, sidecar_value)

            raw_connections = sidecar_data.get("live_data_connections")
            if isinstance(raw_connections, list):
                parsed_connections: list[UserSpaceLiveDataConnection] = []
                for item in raw_connections:
                    if not isinstance(item, dict):
                        continue
                    try:
                        parsed_connections.append(
                            UserSpaceLiveDataConnection.model_validate(item)
                        )
                    except Exception:
                        continue
                live_data_connections = parsed_connections or None

            raw_checks = sidecar_data.get("live_data_checks")
            if isinstance(raw_checks, list):
                parsed_checks: list[UserSpaceLiveDataCheck] = []
                for item in raw_checks:
                    if not isinstance(item, dict):
                        continue
                    try:
                        parsed_checks.append(
                            UserSpaceLiveDataCheck.model_validate(item)
                        )
                    except Exception:
                        continue
                live_data_checks = parsed_checks or None
        except Exception:
            artifact_type = None
            live_data_connections = None
            live_data_checks = None

        return artifact_type, live_data_connections, live_data_checks

    async def _touch_workspace(
        self, workspace_id: str, ts: datetime | None = None
    ) -> None:
        db = await get_db()
        try:
            await db.workspace.update(
                where={"id": workspace_id},
                data={"updatedAt": ts or _utc_now()},
            )
        except Exception:
            logger.debug("Failed to update workspace timestamp for %s", workspace_id)

    async def touch_workspace(
        self, workspace_id: str, ts: datetime | None = None
    ) -> None:
        await self._touch_workspace(workspace_id, ts=ts)

    def _build_dashboard_entrypoint_content(self, relative_path: str) -> str:
        normalized = (relative_path or "").replace("\\", "/")
        module_rel = (
            normalized[len("dashboard/") :]
            if normalized.startswith("dashboard/")
            else normalized
        )
        module_without_ext = module_rel
        for extension in _MODULE_SOURCE_EXTENSIONS:
            if module_without_ext.lower().endswith(extension):
                module_without_ext = module_without_ext[: -len(extension)]
                break
        specifier = f"./{module_without_ext}".replace("//", "/")
        return _render_userspace_template(
            "dashboard_entrypoint.js",
            {"__RAGTIME_DASHBOARD_MODULE_SPECIFIER__": specifier},
        )

    def _append_dashboard_entrypoint_reference(
        self,
        existing_content: str,
        relative_path: str,
    ) -> str:
        candidates = _entrypoint_module_specifier_candidates(relative_path)
        if _entrypoint_references_module(existing_content, candidates):
            return existing_content
        if candidates:
            specifier = candidates[0]
            trimmed = existing_content.rstrip()
            suffix = f"\nimport '{specifier}';\n"
            return f"{trimmed}{suffix}" if trimmed else suffix
        return existing_content

    @staticmethod
    def _workspace_mount_target_repo_relative_path(target_path: str) -> str | None:
        raw = (target_path or "").strip()
        if not raw or "\x00" in raw:
            return None
        normalized_target = posixpath.normpath(raw)
        if not normalized_target.startswith("/workspace/"):
            return None
        relative = normalized_target[len("/workspace/") :].strip("/")
        if not relative or relative == ".":
            return None
        # Reject any traversal segments that survived normalization.
        parts = relative.split("/")
        if any(part in ("..", ".", "") for part in parts):
            return None
        return relative

    @staticmethod
    def _workspace_path_matches_mount_prefix(path: str, prefix: str) -> bool:
        normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
        normalized_prefix = (prefix or "").strip().replace("\\", "/").lstrip("/")
        if not normalized_path or not normalized_prefix:
            return False
        return normalized_path == normalized_prefix or normalized_path.startswith(
            normalized_prefix + "/"
        )

    @staticmethod
    def _deduplicate_ancestor_paths(paths: list[str]) -> list[str]:
        """Remove paths that are children of other paths in the list."""
        if len(paths) <= 1:
            return list(paths)
        sorted_paths = sorted(paths)
        result: list[str] = []
        for path in sorted_paths:
            if result and (path == result[-1] or path.startswith(result[-1] + "/")):
                continue
            result.append(path)
        return result

    async def _list_workspace_mount_target_repo_paths(
        self,
        workspace_id: str,
    ) -> list[str]:
        db = await get_db()
        rows = await db.workspacemount.find_many(
            where={"workspaceId": workspace_id},
            order={"createdAt": "asc"},
        )
        paths: list[str] = []
        seen: set[str] = set()
        for row in rows:
            relative = self._workspace_mount_target_repo_relative_path(
                str(getattr(row, "targetPath", "") or "")
            )
            if not relative or relative in seen:
                continue
            seen.add(relative)
            paths.append(relative)
        return self._deduplicate_ancestor_paths(paths)

    async def _list_disabled_workspace_mount_target_repo_paths(
        self,
        workspace_id: str,
    ) -> list[str]:
        db = await get_db()
        rows = await db.workspacemount.find_many(
            where={"workspaceId": workspace_id, "enabled": False},
            order={"createdAt": "asc"},
        )
        paths: list[str] = []
        seen: set[str] = set()
        for row in rows:
            relative = self._workspace_mount_target_repo_relative_path(
                str(getattr(row, "targetPath", "") or "")
            )
            if not relative or relative in seen:
                continue
            seen.add(relative)
            paths.append(relative)
        return self._deduplicate_ancestor_paths(paths)

    async def ensure_workspace_path_not_in_disabled_mount(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> str:
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if not normalized_path:
            raise HTTPException(status_code=400, detail="Invalid file path")

        disabled_prefixes = await self._list_disabled_workspace_mount_target_repo_paths(
            workspace_id
        )
        for prefix in disabled_prefixes:
            if self._workspace_path_matches_mount_prefix(normalized_path, prefix):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Path is inside an unmounted workspace mount and is not accessible"
                    ),
                )
        return normalized_path

    async def _stage_workspace_snapshot_files(self, workspace_id: str) -> None:
        """Stage all workspace files then purge paths that must not be snapshotted.

        Excluded from the index in a single pass after ``git add -A``:
        - Platform-managed files (bridge, bootstrap config/marker)
        - Mount materialized directories (from DB workspace_mounts)
        - SQLite databases (when workspace sqlite persistence mode is 'exclude')
        """
        # Collect all paths/patterns to purge from the index.
        rm_cached_paths: list[str] = []
        reset_patterns: list[str] = []

        # Platform-managed files — always excluded.
        rm_cached_paths.extend(_PLATFORM_MANAGED_GITIGNORE_PATTERNS)

        # Mount targets — authoritative exclusion from DB records.
        mount_paths = await self._list_workspace_mount_target_repo_paths(workspace_id)
        if mount_paths:
            rm_cached_paths.extend(mount_paths)
            logger.debug(
                "Excluding mount target paths from snapshot for %s: %s",
                workspace_id,
                mount_paths,
            )

        # SQLite exclusion policy.
        db = await get_db()
        workspace = await db.workspace.find_unique(where={"id": workspace_id})
        sqlite_mode = _normalize_sqlite_persistence_mode(
            str(getattr(workspace, "sqlitePersistenceMode", "exclude") or "exclude")
            if workspace
            else "exclude"
        )
        if sqlite_mode == "exclude":
            reset_patterns.extend(_SQLITE_EXCLUDE_GLOBS)

        # Stage everything first.
        await self._run_git(workspace_id, ["add", "-A"])

        # Purge excluded paths from the index.
        if rm_cached_paths:
            await self._run_git(
                workspace_id,
                [
                    "rm",
                    "-r",
                    "--cached",
                    "--ignore-unmatch",
                    "--",
                    *rm_cached_paths,
                ],
                check=False,
            )
        for pattern in reset_patterns:
            await self._run_git(
                workspace_id,
                ["reset", "--", pattern],
                check=False,
            )

    def _workspace_from_record(self, record: Any) -> UserSpaceWorkspace:
        member_rows = list(getattr(record, "members", []) or [])
        members: list[WorkspaceMember] = []
        owner_present = False

        for member in member_rows:
            member_user_id = getattr(member, "userId", "")
            role_value = getattr(member, "role", "viewer")
            role = (
                role_value
                if isinstance(role_value, str)
                else str(getattr(role_value, "value", role_value))
            )
            if member_user_id == getattr(record, "ownerUserId", "") and role == "owner":
                owner_present = True
            members.append(
                WorkspaceMember(
                    user_id=member_user_id,
                    role=cast(Any, role),
                )
            )

        if not owner_present and getattr(record, "ownerUserId", None):
            members.insert(
                0,
                WorkspaceMember(user_id=record.ownerUserId, role="owner"),
            )

        tool_rows = list(getattr(record, "toolSelections", []) or [])
        selected_tool_ids = [
            getattr(tool_row, "toolConfigId", "")
            for tool_row in tool_rows
            if getattr(tool_row, "toolConfigId", None)
        ]

        group_rows = list(getattr(record, "toolGroupSelections", []) or [])
        selected_tool_group_ids = [
            getattr(row, "toolGroupId", "")
            for row in group_rows
            if getattr(row, "toolGroupId", None)
        ]

        owner_obj = getattr(record, "owner", None)
        owner_username: str | None = None
        owner_display_name: str | None = None
        if owner_obj is not None:
            owner_username = getattr(owner_obj, "username", None)
            owner_display_name = getattr(owner_obj, "displayName", None)

        scm_provider = self._normalize_workspace_scm_provider(
            getattr(record, "scmProvider", None),
            getattr(record, "scmGitUrl", None),
        )
        scm = UserSpaceWorkspaceScmStatus(
            connected=bool(getattr(record, "scmGitUrl", None)),
            git_url=getattr(record, "scmGitUrl", None),
            git_branch=getattr(record, "scmGitBranch", None),
            provider=scm_provider,
            repo_visibility=getattr(record, "scmRepoVisibility", None),
            has_stored_token=bool(getattr(record, "scmToken", None)),
            connected_at=getattr(record, "scmConnectedAt", None),
            last_sync_at=getattr(record, "scmLastSyncAt", None),
            last_sync_direction=getattr(record, "scmLastSyncDirection", None),
            last_sync_status=getattr(record, "scmLastSyncStatus", None),
            last_sync_message=getattr(record, "scmLastSyncMessage", None),
            last_remote_commit_hash=getattr(record, "scmLastRemoteCommitHash", None),
            last_synced_snapshot_id=getattr(record, "scmLastSyncedSnapshotId", None),
        )

        return UserSpaceWorkspace(
            id=record.id,
            name=record.name,
            description=record.description,
            sqlite_persistence_mode=cast(
                SqlitePersistenceMode,
                _normalize_sqlite_persistence_mode(
                    str(
                        getattr(record, "sqlitePersistenceMode", "exclude") or "exclude"
                    )
                ),
            ),
            owner_user_id=record.ownerUserId,
            owner_username=owner_username,
            owner_display_name=owner_display_name,
            selected_tool_ids=selected_tool_ids,
            selected_tool_group_ids=selected_tool_group_ids,
            conversation_ids=[],
            members=members,
            scm=scm,
            created_at=record.createdAt,
            updated_at=record.updatedAt,
        )

    @staticmethod
    def _normalize_workspace_scm_provider(
        provider: str | None,
        git_url: str | None,
    ) -> WorkspaceScmProvider | None:
        normalized = str(provider or "").strip().lower()
        if normalized in {"github", "gitlab", "generic"}:
            return cast(WorkspaceScmProvider, normalized)
        parsed = parse_git_url(str(git_url or "")) if git_url else None
        if parsed:
            return cast(WorkspaceScmProvider, parsed.provider.value)
        return None

    @staticmethod
    def _normalize_workspace_scm_branch(branch: str | None) -> str:
        value = str(branch or "").strip()
        return value or "main"

    @staticmethod
    def _sync_scope_relative_paths(root: Path) -> dict[str, Path]:
        results: dict[str, Path] = {}
        if not root.exists():
            return results
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith(".git/") or relative == ".git":
                continue
            if relative in _PLATFORM_MANAGED_GITIGNORE_PATTERNS:
                continue
            results[relative] = path
        return results

    @staticmethod
    def _compute_file_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _collect_scm_diff_sample(
        self,
        left_root: Path,
        right_root: Path,
    ) -> list[str]:
        left = self._sync_scope_relative_paths(left_root)
        right = self._sync_scope_relative_paths(right_root)
        changed: list[str] = []
        for relative_path in sorted(set(left) | set(right)):
            left_path = left.get(relative_path)
            right_path = right.get(relative_path)
            if left_path is None or right_path is None:
                changed.append(relative_path)
            elif self._compute_file_sha256(left_path) != self._compute_file_sha256(
                right_path
            ):
                changed.append(relative_path)
            if len(changed) >= _WORKSPACE_SCM_PREVIEW_SAMPLE_LIMIT:
                break
        return changed

    async def _workspace_has_sync_scope_files(self, workspace_id: str) -> bool:
        files_dir = self._workspace_files_dir(workspace_id)
        return bool(await asyncio.to_thread(self._sync_scope_relative_paths, files_dir))

    async def _workspace_has_uncommitted_changes(self, workspace_id: str) -> bool:
        result = await self._run_git(
            workspace_id,
            ["status", "--porcelain", "--untracked-files=all"],
            check=False,
        )
        return bool(result.stdout.strip())

    async def _workspace_current_commit_hash(self, workspace_id: str) -> str | None:
        result = await self._run_git(
            workspace_id,
            ["rev-parse", "HEAD"],
            check=False,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    @staticmethod
    def _git_network_env() -> dict[str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    async def _clone_remote_branch_to_tempdir(
        self,
        git_url: str,
        git_branch: str,
        git_token: str | None,
        *,
        full_history: bool = False,
        checkout: bool = True,
    ) -> tuple[Path | None, str | None, str | None]:
        temp_dir = Path(tempfile.mkdtemp(prefix="ragtime-workspace-scm-"))
        auth_url = build_authenticated_git_url(git_url, git_token)
        env = self._git_network_env()
        try:
            clone_args = [
                "clone",
                "--single-branch",
                "--branch",
                git_branch,
            ]
            if not checkout:
                clone_args.append("--no-checkout")
            if not full_history:
                clone_args.extend(["--depth", "1"])
            clone_args.extend([auth_url, str(temp_dir)])
            result = await self._run_git_in_dir(
                temp_dir.parent,
                clone_args,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return (
                    None,
                    None,
                    result.stderr.strip() or result.stdout.strip() or "Clone failed",
                )
            head = await self._run_git_in_dir(
                temp_dir, ["rev-parse", "HEAD"], check=False
            )
            return temp_dir, head.stdout.strip() or None, None
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, str(exc)

    async def _schedule_workspace_scm_remote_commit_backfill(
        self,
        workspace_id: str,
        *,
        git_branch: str,
        remote_dir: Path,
        remote_head_hash: str,
        branch_id: str,
        imported_snapshot_id: str,
    ) -> None:
        async with self._workspace_scm_backfill_tasks_lock:
            existing_task = self._workspace_scm_backfill_tasks.get(workspace_id)
            if existing_task is not None and not existing_task.done():
                return

            backfill_task = asyncio.create_task(
                self._backfill_workspace_scm_remote_commits(
                    workspace_id,
                    git_branch=git_branch,
                    remote_dir=remote_dir,
                    remote_head_hash=remote_head_hash,
                    branch_id=branch_id,
                    imported_snapshot_id=imported_snapshot_id,
                ),
                name=f"userspace-scm-backfill:{workspace_id}",
            )
            self._attach_workspace_scm_backfill_task_cleanup(
                workspace_id,
                backfill_task,
            )
            self._workspace_scm_backfill_tasks[workspace_id] = backfill_task

    async def _backfill_workspace_scm_remote_commits(
        self,
        workspace_id: str,
        *,
        git_branch: str,
        remote_dir: Path,
        remote_head_hash: str,
        branch_id: str,
        imported_snapshot_id: str,
    ) -> None:
        """Import the remote branch's full commit history as snapshot records.

        Each remote commit becomes a restorable snapshot in the timeline.
        The remote's git objects are fetched into the local workspace repo
        so that ``git checkout <hash>`` works for any historical commit.

        Takes ownership of *remote_dir* and deletes it when finished.
        """
        try:
            await self._backfill_workspace_scm_remote_commits_inner(
                workspace_id,
                git_branch=git_branch,
                remote_dir=remote_dir,
                remote_head_hash=remote_head_hash,
                branch_id=branch_id,
                imported_snapshot_id=imported_snapshot_id,
            )
        finally:
            shutil.rmtree(remote_dir, ignore_errors=True)

    async def _backfill_workspace_scm_remote_commits_inner(
        self,
        workspace_id: str,
        *,
        git_branch: str,
        remote_dir: Path,
        remote_head_hash: str,
        branch_id: str,
        imported_snapshot_id: str,
    ) -> None:
        """Import the remote branch's full commit history as snapshot records.

        Each remote commit becomes a restorable snapshot in the timeline.
        The remote's git objects are fetched into the local workspace repo
        so that ``git checkout <hash>`` works for any historical commit.
        """
        # ------------------------------------------------------------------
        # 1. Read first-parent history (oldest → newest) from the clone.
        # ------------------------------------------------------------------
        log_result = await self._run_git_in_dir(
            remote_dir,
            [
                "log",
                "--first-parent",
                "--reverse",
                "--format=%H%x00%ct%x00%s",
                "HEAD",
            ],
            check=False,
        )
        if log_result.returncode != 0 or not log_result.stdout.strip():
            logger.debug(
                "No commit history to import for workspace %s from %s",
                workspace_id,
                git_branch,
            )
            return

        all_commits: list[dict[str, Any]] = []
        for line in log_result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\x00", 2)
            if len(parts) < 3:
                continue
            all_commits.append(
                {
                    "hash": parts[0],
                    "timestamp": int(parts[1]),
                    "message": parts[2][:200],
                }
            )

        # HEAD is already represented by the imported snapshot — skip it.
        history_commits = [c for c in all_commits if c["hash"] != remote_head_hash]
        if not history_commits:
            logger.debug(
                "Remote has only one commit; no additional history for workspace %s",
                workspace_id,
            )
            return

        # ------------------------------------------------------------------
        # 2. Compute file counts per historical commit.
        # ------------------------------------------------------------------
        for commit in history_commits:
            tree_result = await self._run_git_in_dir(
                remote_dir,
                ["ls-tree", "-r", "--name-only", commit["hash"]],
                check=False,
            )
            if tree_result.returncode == 0:
                commit["file_count"] = sum(
                    1
                    for f in tree_result.stdout.splitlines()
                    if f.strip() and not self._is_reserved_internal_path(f)
                )
            else:
                commit["file_count"] = 0

        # ------------------------------------------------------------------
        # 3. Fetch remote objects into the local workspace git repo so that
        #    every historical commit is locally restorable.
        # ------------------------------------------------------------------
        env = self._git_network_env()
        await self._run_git(
            workspace_id,
            ["remote", "add", "_scm_import", str(remote_dir)],
            check=False,
        )
        try:
            await self._run_git(
                workspace_id,
                ["fetch", "--no-tags", "_scm_import", git_branch],
                env=env,
            )
        finally:
            await self._run_git(
                workspace_id,
                ["remote", "remove", "_scm_import"],
                check=False,
            )

        # Persistent ref prevents GC from pruning the fetched objects.
        await self._run_git(
            workspace_id,
            ["update-ref", f"refs/scm-history/{git_branch}", remote_head_hash],
        )

        # ------------------------------------------------------------------
        # 4. Determine which commits already have snapshot records.
        # ------------------------------------------------------------------
        db = await get_db()
        all_hashes = [c["hash"] for c in history_commits]
        in_clause = ", ".join(self._sql_quote(h) for h in all_hashes)
        existing_rows = await db.query_raw(
            f"""
            SELECT git_commit_hash, id
            FROM userspace_snapshots
            WHERE workspace_id = {self._sql_quote(workspace_id)}
              AND branch_id = {self._sql_quote(branch_id)}
              AND git_commit_hash IN ({in_clause})
            """
        )
        existing_by_hash: dict[str, str] = {
            str(r["git_commit_hash"]): str(r["id"]) for r in existing_rows
        }

        # ------------------------------------------------------------------
        # 5. Create snapshot records (oldest → newest) with parent chain.
        # ------------------------------------------------------------------
        created_count = 0
        prev_snapshot_id: str | None = None
        for commit in history_commits:
            if commit["hash"] in existing_by_hash:
                prev_snapshot_id = existing_by_hash[commit["hash"]]
                continue

            snapshot_id = str(uuid4())
            created_at = datetime.fromtimestamp(commit["timestamp"], tz=timezone.utc)
            await db.execute_raw(
                f"""
                INSERT INTO userspace_snapshots
                    (id, workspace_id, branch_id, git_commit_hash,
                     remote_commit_hash, message, file_count,
                     parent_snapshot_id, created_by_user_id,
                     created_at, updated_at)
                VALUES (
                    {self._sql_quote(snapshot_id)},
                    {self._sql_quote(workspace_id)},
                    {self._sql_quote(branch_id)},
                    {self._sql_quote(commit["hash"])},
                    {self._sql_quote(commit["hash"])},
                    {self._sql_quote(commit["message"])},
                    {commit["file_count"]},
                    {self._sql_quote(prev_snapshot_id)},
                    NULL,
                    {self._sql_quote(created_at.isoformat())},
                    {self._sql_quote(created_at.isoformat())}
                )
                """
            )
            existing_by_hash[commit["hash"]] = snapshot_id
            prev_snapshot_id = snapshot_id
            created_count += 1

        # ------------------------------------------------------------------
        # 6. Reparent the imported snapshot so the timeline is continuous.
        # ------------------------------------------------------------------
        if prev_snapshot_id:
            await db.execute_raw(
                f"""
                UPDATE userspace_snapshots
                SET parent_snapshot_id = {self._sql_quote(prev_snapshot_id)},
                    updated_at = NOW()
                WHERE id = {self._sql_quote(imported_snapshot_id)}
                """
            )

        # Also mark any previously-pushed local snapshots whose hashes
        # happen to match remote commits (original backfill behaviour).
        updated_count = 0
        chunk_size = 500
        full_hash_list = [c["hash"] for c in all_commits]
        for start in range(0, len(full_hash_list), chunk_size):
            chunk = full_hash_list[start : start + chunk_size]
            chunk_clause = ", ".join(self._sql_quote(h) for h in chunk)
            await db.execute_raw(
                f"""
                UPDATE userspace_snapshots
                SET remote_commit_hash = git_commit_hash,
                    updated_at = NOW()
                WHERE workspace_id = {self._sql_quote(workspace_id)}
                  AND remote_commit_hash IS NULL
                  AND git_commit_hash IN ({chunk_clause})
                """
            )

        logger.info(
            "Imported %s historical snapshots for workspace %s from %s (%s remote commits)",
            created_count,
            workspace_id,
            git_branch,
            len(all_commits),
        )

    async def _stage_temp_repo_from_workspace(
        self,
        workspace_id: str,
    ) -> Path:
        files_dir = self._workspace_files_dir(workspace_id)
        temp_dir = Path(tempfile.mkdtemp(prefix="ragtime-workspace-export-"))
        staged_files = await asyncio.to_thread(
            self._sync_scope_relative_paths, files_dir
        )
        for relative_path, source_path in staged_files.items():
            target_path = temp_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        await self._run_git_in_dir(temp_dir, ["init"])
        await self._run_git_in_dir(
            temp_dir, ["config", "user.name", "Ragtime Workspace Sync"]
        )
        await self._run_git_in_dir(
            temp_dir, ["config", "user.email", "userspace-sync@ragtime.local"]
        )
        await self._run_git_in_dir(temp_dir, ["add", "."])
        return temp_dir

    async def _update_workspace_scm_sync_metadata(
        self,
        workspace_id: str,
        *,
        git_url: str | None = None,
        git_branch: str | None = None,
        provider: WorkspaceScmProvider | None = None,
        repo_visibility: str | None = None,
        git_token: str | None = None,
        remote_commit_hash: str | None = None,
        snapshot_id: str | None = None,
        status: str,
        message: str,
        direction: WorkspaceScmDirection = "export",
    ) -> UserSpaceWorkspaceScmStatus:
        db = await get_db()
        current_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not current_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        update_data: dict[str, Any] = {
            "scmLastSyncAt": _utc_now(),
            "scmLastSyncDirection": direction,
            "scmLastSyncStatus": status,
            "scmLastSyncMessage": message,
            "scmConnectedAt": getattr(current_record, "scmConnectedAt", None)
            or _utc_now(),
        }
        if git_url is not None:
            update_data["scmGitUrl"] = git_url
        if git_branch is not None:
            update_data["scmGitBranch"] = git_branch
        if provider is not None:
            update_data["scmProvider"] = provider
        if repo_visibility is not None:
            update_data["scmRepoVisibility"] = repo_visibility
        if git_token:
            update_data["scmToken"] = encrypt_secret(git_token.strip())
        if remote_commit_hash is not None:
            update_data["scmLastRemoteCommitHash"] = remote_commit_hash
        if snapshot_id is not None:
            update_data["scmLastSyncedSnapshotId"] = snapshot_id

        updated = await db.workspace.update(
            where={"id": workspace_id},
            data=update_data,
        )
        return self._workspace_from_record(updated).scm or UserSpaceWorkspaceScmStatus()

    async def _push_workspace_snapshot_commit(
        self,
        workspace_id: str,
        *,
        snapshot: UserSpaceSnapshot,
        git_url: str,
        git_branch: str,
        git_token: str | None,
        remote_commit_hash: str | None = None,
        allow_force: bool = False,
    ) -> str:
        commit_hash = (snapshot.git_commit_hash or "").strip()
        if not commit_hash:
            raise HTTPException(
                status_code=409,
                detail="Snapshot commit is missing and cannot be pushed",
            )

        push_args = [
            "push",
            build_authenticated_git_url(git_url, git_token),
            f"{commit_hash}:refs/heads/{git_branch}",
        ]
        if allow_force and remote_commit_hash:
            push_args.insert(
                1,
                f"--force-with-lease=refs/heads/{git_branch}:{remote_commit_hash}",
            )
        elif allow_force:
            push_args.insert(1, "--force")

        await self._run_git(
            workspace_id,
            push_args,
            env=self._git_network_env(),
        )
        return commit_hash

    async def _maybe_auto_push_snapshot_to_scm(
        self,
        workspace_id: str,
        snapshot: UserSpaceSnapshot,
    ) -> None:
        """Auto-push snapshot to remote if SCM is configured.

        When a remote is configured, snapshots are treated as 'commit + push' operations.
        Local snapshot is always created, but sync status is tracked and reported.
        If push fails for a configured remote, error is logged prominently.
        """
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            return

        git_url = str(getattr(workspace_record, "scmGitUrl", "") or "").strip()
        if not git_url:
            return

        git_branch = self._normalize_workspace_scm_branch(
            getattr(workspace_record, "scmGitBranch", None)
        )
        stored_token_encrypted = getattr(workspace_record, "scmToken", None)
        git_token = (
            decrypt_secret(stored_token_encrypted) if stored_token_encrypted else None
        )
        parsed = parse_git_url(git_url, git_token)
        if not parsed:
            await self._update_workspace_scm_sync_metadata(
                workspace_id,
                git_url=git_url,
                git_branch=git_branch,
                status="error",
                message="Automatic snapshot push skipped because the SCM URL is invalid.",
            )
            logger.error(
                "SCM URL is invalid for workspace %s; snapshot push skipped: %s",
                workspace_id,
                git_url,
            )
            return

        provider = cast(WorkspaceScmProvider, parsed.provider.value)
        repo_visibility = getattr(workspace_record, "scmRepoVisibility", None)
        last_remote_commit_hash = getattr(
            workspace_record, "scmLastRemoteCommitHash", None
        )

        try:
            remote_commit_hash = await self._push_workspace_snapshot_commit(
                workspace_id,
                snapshot=snapshot,
                git_url=git_url,
                git_branch=git_branch,
                git_token=git_token,
                remote_commit_hash=(
                    str(last_remote_commit_hash) if last_remote_commit_hash else None
                ),
                allow_force=False,
            )
            db = await get_db()
            await db.execute_raw(
                f"""
                UPDATE userspace_snapshots
                SET remote_commit_hash = {self._sql_quote(remote_commit_hash)},
                    updated_at = NOW()
                WHERE id = {self._sql_quote(snapshot.id)}
                """
            )
            await self._update_workspace_scm_sync_metadata(
                workspace_id,
                git_url=git_url,
                git_branch=git_branch,
                provider=provider,
                repo_visibility=(
                    str(repo_visibility) if repo_visibility is not None else None
                ),
                remote_commit_hash=remote_commit_hash,
                snapshot_id=snapshot.id,
                status="success",
                message=f"Snapshot pushed to {git_branch}",
                direction="export",
            )
            logger.info(
                "Snapshot %s auto-pushed to remote for workspace %s",
                snapshot.id,
                workspace_id,
            )
        except HTTPException as exc:
            detail = (
                exc.detail if isinstance(exc.detail, str) else "Snapshot push failed"
            )
            await self._update_workspace_scm_sync_metadata(
                workspace_id,
                git_url=git_url,
                git_branch=git_branch,
                provider=provider,
                repo_visibility=(
                    str(repo_visibility) if repo_visibility is not None else None
                ),
                status="error",
                message=f"Failed to push snapshot: {detail}",
                direction="export",
            )
            logger.error(
                "Automatic snapshot push FAILED for workspace %s (remote %s): %s. "
                "Snapshot exists locally but is not synced to remote.",
                workspace_id,
                git_url,
                detail,
            )

    async def _replace_workspace_sync_scope_from_dir(
        self,
        workspace_id: str,
        source_root: Path,
    ) -> None:
        target_root = self._workspace_files_dir(workspace_id)

        def _sync() -> None:
            target_files = self._sync_scope_relative_paths(target_root)
            source_files = self._sync_scope_relative_paths(source_root)

            for relative_path, target_path in target_files.items():
                if relative_path not in source_files and target_path.exists():
                    target_path.unlink()

            for relative_path, source_path in source_files.items():
                destination_path = target_root / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                if destination_path.exists() and self._compute_file_sha256(
                    destination_path
                ) == self._compute_file_sha256(source_path):
                    continue
                shutil.copy2(source_path, destination_path)

            for directory in sorted(
                [path for path in target_root.rglob("*") if path.is_dir()],
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                if directory == target_root:
                    continue
                relative = directory.relative_to(target_root).as_posix()
                if relative == ".git" or relative.startswith(".git/"):
                    continue
                if relative == ".ragtime":
                    continue
                try:
                    next(directory.iterdir())
                except StopIteration:
                    directory.rmdir()

        await asyncio.to_thread(_sync)
        self._seed_runtime_bootstrap_config(workspace_id)

    def _workspace_scm_setup_prompt(
        self,
        workspace_id: str,
        git_url: str,
        git_branch: str,
        inferred_entrypoint: dict[str, str] | None = None,
        normalization_actions: list[str] | None = None,
    ) -> str:
        detected_replit_features, has_legacy_replit_object_storage = (
            self._detect_imported_replit_features(workspace_id)
        )
        prompt_builder = cast(Any, build_workspace_scm_setup_prompt)
        return prompt_builder(  # type: ignore[call-arg]
            git_url=git_url,
            git_branch=git_branch,
            inferred_entrypoint=inferred_entrypoint,
            detected_replit_features=detected_replit_features,
            has_legacy_replit_object_storage=has_legacy_replit_object_storage,
            normalization_actions=normalization_actions,
        )

    async def _resolve_workspace_scm_target(
        self,
        workspace_record: Any,
        request: UserSpaceWorkspaceScmPreviewRequest,
    ) -> tuple[str, str, str | None, WorkspaceScmProvider, str | None]:
        git_url = str(
            request.git_url or getattr(workspace_record, "scmGitUrl", "") or ""
        ).strip()
        if not git_url:
            raise HTTPException(
                status_code=400, detail="Git repository URL is required"
            )

        git_branch = self._normalize_workspace_scm_branch(
            request.git_branch or getattr(workspace_record, "scmGitBranch", None)
        )
        provided_token = (request.git_token or "").strip() or None
        stored_token_encrypted = getattr(workspace_record, "scmToken", None)
        stored_token = (
            decrypt_secret(stored_token_encrypted) if stored_token_encrypted else None
        )
        git_token = provided_token or stored_token
        parsed = parse_git_url(git_url, git_token)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid Git repository URL")

        provider = cast(WorkspaceScmProvider, parsed.provider.value)
        repo_visibility = str(
            request.create_repo_private is False
            and "public"
            or getattr(workspace_record, "scmRepoVisibility", None)
            or "private"
        )
        return git_url, git_branch, git_token, provider, repo_visibility

    async def _build_workspace_scm_preview(
        self,
        workspace_id: str,
        user_id: str,
        direction: WorkspaceScmDirection,
        request: UserSpaceWorkspaceScmPreviewRequest,
        *,
        store_preview: bool,
    ) -> tuple[UserSpaceWorkspaceScmPreviewResponse, str | None]:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        git_url, git_branch, git_token, provider, repo_visibility = (
            await self._resolve_workspace_scm_target(workspace_record, request)
        )

        current_snapshot_id = getattr(workspace_record, "currentSnapshotId", None)
        last_synced_snapshot_id = getattr(
            workspace_record, "scmLastSyncedSnapshotId", None
        )
        last_remote_commit_hash = getattr(
            workspace_record, "scmLastRemoteCommitHash", None
        )
        local_commit_hash = await self._workspace_current_commit_hash(workspace_id)
        local_has_uncommitted_changes = await self._workspace_has_uncommitted_changes(
            workspace_id
        )
        local_has_files = await self._workspace_has_sync_scope_files(workspace_id)
        local_changed = (
            local_has_uncommitted_changes
            or bool(
                current_snapshot_id and current_snapshot_id != last_synced_snapshot_id
            )
            or (not current_snapshot_id and local_has_files)
        )

        remote_dir, remote_commit_hash, clone_error = (
            await self._clone_remote_branch_to_tempdir(
                git_url,
                git_branch,
                git_token,
            )
        )

        remote_exists = remote_commit_hash is not None
        remote_changed = bool(
            remote_commit_hash and remote_commit_hash != last_remote_commit_hash
        )

        state: WorkspaceScmPreviewState
        summary: str
        will_overwrite_local = False
        will_overwrite_remote = False
        can_proceed_without_force = False
        changed_files_sample: list[str] = []

        if remote_dir is not None:
            changed_files_sample = await asyncio.to_thread(
                self._collect_scm_diff_sample,
                self._workspace_files_dir(workspace_id),
                remote_dir,
            )

        if not remote_exists:
            if direction == "import":
                state = "missing_branch"
                summary = (
                    f"Branch '{git_branch}' was not found in the remote repository."
                )
            else:
                state = "safe"
                can_proceed_without_force = True
                if request.create_repo_if_missing:
                    summary = (
                        "Remote repository or branch is missing. Ragtime can create "
                        "the repository if needed and push this workspace as the first commit."
                    )
                else:
                    summary = f"Branch '{git_branch}' does not exist remotely yet. Export can create it safely."
        elif direction == "import":
            if not local_changed and remote_changed:
                state = "safe"
                can_proceed_without_force = True
                summary = "Remote changes are available and the workspace has no unsynced local state."
            elif not local_changed and not remote_changed:
                state = "up_to_date"
                summary = "Workspace and remote repository are already in sync."
            else:
                state = "destructive"
                will_overwrite_local = True
                summary = "Import would overwrite local workspace state. Preview the changed files and confirm overwrite to continue."
        else:
            if local_changed and not remote_changed:
                state = "safe"
                can_proceed_without_force = True
                summary = "Workspace changes are ready to export and the remote has not moved since the last sync."
            elif not local_changed and not remote_changed:
                state = "up_to_date"
                summary = "Workspace and remote repository are already in sync."
            else:
                state = "destructive"
                will_overwrite_remote = True
                summary = "Export would overwrite newer or unknown remote state. Preview and confirm overwrite to continue."

        state_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "workspace_id": workspace_id,
                    "direction": direction,
                    "git_url": git_url,
                    "git_branch": git_branch,
                    "current_snapshot_id": current_snapshot_id,
                    "last_synced_snapshot_id": last_synced_snapshot_id,
                    "last_remote_commit_hash": last_remote_commit_hash,
                    "remote_commit_hash": remote_commit_hash,
                    "local_changed": local_changed,
                    "remote_changed": remote_changed,
                    "will_overwrite_local": will_overwrite_local,
                    "will_overwrite_remote": will_overwrite_remote,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        preview_token: str | None = None
        preview_expires_at: datetime | None = None
        if state == "destructive" and store_preview:
            preview_token = secrets.token_urlsafe(24)
            preview_expires_at = _utc_now() + timedelta(
                seconds=_WORKSPACE_SCM_PREVIEW_TTL_SECONDS
            )
            await self._store_workspace_scm_preview(
                _WorkspaceScmPreviewRecord(
                    token=preview_token,
                    workspace_id=workspace_id,
                    direction=direction,
                    state_fingerprint=state_fingerprint,
                    expires_at=preview_expires_at,
                )
            )

        if remote_dir is not None:
            shutil.rmtree(remote_dir, ignore_errors=True)

        if clone_error and not remote_exists and direction == "import":
            summary = clone_error

        return (
            UserSpaceWorkspaceScmPreviewResponse(
                workspace_id=workspace_id,
                direction=direction,
                state=state,
                summary=summary,
                git_url=git_url,
                git_branch=git_branch,
                provider=provider,
                repo_visibility=repo_visibility,
                local_changed=local_changed,
                remote_changed=remote_changed,
                local_has_uncommitted_changes=local_has_uncommitted_changes,
                will_overwrite_local=will_overwrite_local,
                will_overwrite_remote=will_overwrite_remote,
                can_proceed_without_force=can_proceed_without_force,
                local_commit_hash=local_commit_hash,
                remote_commit_hash=remote_commit_hash,
                current_snapshot_id=current_snapshot_id,
                changed_files_sample=changed_files_sample,
                preview_token=preview_token,
                preview_expires_at=preview_expires_at,
            ),
            state_fingerprint,
        )

    async def get_workspace_scm_connection(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceWorkspaceScmConnectionResponse:
        await self._enforce_workspace_access(workspace_id, user_id)
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return UserSpaceWorkspaceScmConnectionResponse(
            workspace_id=workspace_id,
            scm=self._workspace_from_record(workspace_record).scm
            or UserSpaceWorkspaceScmStatus(),
        )

    async def update_workspace_scm_connection(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmConnectionRequest,
    ) -> UserSpaceWorkspaceScmConnectionResponse:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        parsed = parse_git_url(request.git_url, request.git_token)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid Git repository URL")

        update_data: dict[str, Any] = {
            "scmGitUrl": request.git_url.strip(),
            "scmGitBranch": self._normalize_workspace_scm_branch(request.git_branch),
            "scmProvider": parsed.provider.value,
            "scmRepoVisibility": (
                request.repo_visibility
                or getattr(workspace_record, "scmRepoVisibility", None)
                or "private"
            ),
            "scmConnectedAt": getattr(workspace_record, "scmConnectedAt", None)
            or _utc_now(),
            "updatedAt": _utc_now(),
        }
        if request.git_token:
            update_data["scmToken"] = encrypt_secret(request.git_token.strip())

        updated = await db.workspace.update(
            where={"id": workspace_id}, data=update_data
        )
        return UserSpaceWorkspaceScmConnectionResponse(
            workspace_id=workspace_id,
            scm=self._workspace_from_record(updated).scm
            or UserSpaceWorkspaceScmStatus(),
        )

    async def preview_workspace_scm_import(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmPreviewRequest,
    ) -> UserSpaceWorkspaceScmPreviewResponse:
        preview, _ = await self._build_workspace_scm_preview(
            workspace_id,
            user_id,
            "import",
            request,
            store_preview=True,
        )
        return preview

    async def preview_workspace_scm_export(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmPreviewRequest,
    ) -> UserSpaceWorkspaceScmPreviewResponse:
        preview, _ = await self._build_workspace_scm_preview(
            workspace_id,
            user_id,
            "export",
            request,
            store_preview=True,
        )
        return preview

    async def preview_workspace_scm_sync(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmPreviewRequest,
    ) -> UserSpaceWorkspaceScmPreviewResponse:
        """Auto-detect sync direction and return preview.

        Logic:
        - Only remote changed → import (pull)
        - Only local changed → export (push)
        - Both changed → import (destructive, overwrites local)
        - Neither changed → up-to-date (checked via import preview)
        """
        preview, _ = await self._build_workspace_scm_preview(
            workspace_id,
            user_id,
            "import",
            request,
            store_preview=True,
        )
        if preview.local_changed and not preview.remote_changed:
            preview, _ = await self._build_workspace_scm_preview(
                workspace_id,
                user_id,
                "export",
                request,
                store_preview=True,
            )
        return preview

    async def import_workspace_from_scm(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmImportRequest,
    ) -> UserSpaceWorkspaceScmSyncResponse:
        preview, state_fingerprint = await self._build_workspace_scm_preview(
            workspace_id,
            user_id,
            "import",
            request,
            store_preview=False,
        )
        if preview.state == "missing_branch":
            raise HTTPException(status_code=404, detail=preview.summary)
        if preview.state == "destructive":
            await self._consume_workspace_scm_preview(
                workspace_id=workspace_id,
                direction="import",
                preview_token=request.overwrite_preview_token,
                state_fingerprint=state_fingerprint or "",
            )

        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")
        _, _, git_token, _, _ = await self._resolve_workspace_scm_target(
            workspace_record,
            request,
        )

        remote_dir, remote_commit_hash, clone_error = (
            await self._clone_remote_branch_to_tempdir(
                preview.git_url,
                preview.git_branch,
                git_token,
                full_history=True,
            )
        )
        if remote_dir is None or not remote_commit_hash:
            raise HTTPException(
                status_code=400,
                detail=clone_error or "Failed to clone remote repository branch",
            )

        if preview.will_overwrite_local and (
            preview.local_changed or preview.local_has_uncommitted_changes
        ):
            await self.create_snapshot(
                workspace_id,
                user_id,
                f"SCM backup before import from {preview.git_url} ({preview.git_branch})",
                auto_sync_to_scm=False,
            )

        await self._replace_workspace_sync_scope_from_dir(workspace_id, remote_dir)

        # Attempt to infer a runtime entrypoint from the imported files
        # when no valid user-configured entrypoint exists yet.
        inferred_entrypoint = self._seed_entrypoint_from_import(workspace_id)
        normalization_actions = _dedupe_preserve_order(
            self._normalize_imported_replit_runtime_artifacts(workspace_id)
        )

        imported_snapshot = await self.create_snapshot(
            workspace_id,
            user_id,
            f"Import from {preview.git_url} ({preview.git_branch})",
            auto_sync_to_scm=False,
        )

        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE userspace_snapshots
            SET remote_commit_hash = {self._sql_quote(remote_commit_hash)},
                updated_at = NOW()
            WHERE id = {self._sql_quote(imported_snapshot.id)}
            """
        )
        imported_snapshot.remote_commit_hash = remote_commit_hash

        scm = await self._update_workspace_scm_sync_metadata(
            workspace_id,
            git_url=preview.git_url,
            git_branch=preview.git_branch,
            provider=preview.provider,
            repo_visibility=preview.repo_visibility,
            git_token=request.git_token,
            remote_commit_hash=remote_commit_hash,
            snapshot_id=imported_snapshot.id,
            status="success",
            message=f"Imported from {preview.git_branch}",
            direction="import",
        )

        # Import full remote commit history as snapshot records inline
        # so the timeline is complete before returning to the caller.
        await self._backfill_workspace_scm_remote_commits(
            workspace_id,
            git_branch=preview.git_branch,
            remote_dir=remote_dir,
            remote_head_hash=remote_commit_hash,
            branch_id=imported_snapshot.branch_id,
            imported_snapshot_id=imported_snapshot.id,
        )
        return UserSpaceWorkspaceScmSyncResponse(
            workspace_id=workspace_id,
            direction="import",
            state="success",
            summary="Workspace imported successfully.",
            scm=scm,
            snapshot=imported_snapshot,
            remote_commit_hash=remote_commit_hash,
            suggested_setup_prompt=self._workspace_scm_setup_prompt(
                workspace_id,
                preview.git_url,
                preview.git_branch,
                inferred_entrypoint=inferred_entrypoint,
                normalization_actions=normalization_actions,
            ),
        )

    async def export_workspace_to_scm(
        self,
        workspace_id: str,
        user_id: str,
        request: UserSpaceWorkspaceScmExportRequest,
    ) -> UserSpaceWorkspaceScmSyncResponse:
        preview, state_fingerprint = await self._build_workspace_scm_preview(
            workspace_id,
            user_id,
            "export",
            request,
            store_preview=False,
        )
        if preview.state == "destructive":
            await self._consume_workspace_scm_preview(
                workspace_id=workspace_id,
                direction="export",
                preview_token=request.overwrite_preview_token,
                state_fingerprint=state_fingerprint or "",
            )

        export_request = request
        git_url = preview.git_url
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")
        _, _, git_token, _, _ = await self._resolve_workspace_scm_target(
            workspace_record,
            request,
        )

        if not preview.remote_commit_hash and request.create_repo_if_missing:
            create_result = await create_repository(
                git_url,
                git_token or "",
                private=request.create_repo_private,
                description=request.create_repo_description,
            )
            if not create_result.success or not create_result.git_url:
                raise HTTPException(status_code=400, detail=create_result.message)
            git_url = create_result.git_url
            default_branch = create_result.default_branch or preview.git_branch
            if default_branch:
                preview.git_branch = default_branch
        elif not preview.remote_commit_hash and not request.create_repo_if_missing:
            preview.remote_commit_hash = None

        exported_snapshot = cast(UserSpaceSnapshot | None, None)
        if preview.local_has_uncommitted_changes or not preview.current_snapshot_id:
            exported_snapshot = await self.create_snapshot(
                workspace_id,
                user_id,
                f"Export to {git_url} ({preview.git_branch})",
                auto_sync_to_scm=False,
            )
        else:
            timeline = await self.get_snapshot_timeline(workspace_id, user_id)
            exported_snapshot = next(
                (
                    snapshot
                    for snapshot in timeline.snapshots
                    if snapshot.id == preview.current_snapshot_id
                ),
                None,
            )
            if exported_snapshot is None:
                exported_snapshot = await self.create_snapshot(
                    workspace_id,
                    user_id,
                    f"Export to {git_url} ({preview.git_branch})",
                    auto_sync_to_scm=False,
                )

        try:
            remote_commit_hash = await self._push_workspace_snapshot_commit(
                workspace_id,
                snapshot=exported_snapshot,
                git_url=git_url,
                git_branch=preview.git_branch,
                git_token=git_token,
                remote_commit_hash=preview.remote_commit_hash,
                allow_force=preview.state == "destructive",
            )
            db = await get_db()
            await db.execute_raw(
                f"""
                UPDATE userspace_snapshots
                SET remote_commit_hash = {self._sql_quote(remote_commit_hash)},
                    updated_at = NOW()
                WHERE id = {self._sql_quote(exported_snapshot.id)}
                """
            )
            exported_snapshot.remote_commit_hash = remote_commit_hash
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Export failed"
            await self._update_workspace_scm_sync_metadata(
                workspace_id,
                git_url=git_url,
                git_branch=preview.git_branch,
                provider=preview.provider,
                repo_visibility=(
                    request.create_repo_private
                    and "private"
                    or preview.repo_visibility
                    or "public"
                ),
                git_token=export_request.git_token,
                status="error",
                message=detail,
                direction="export",
            )
            raise

        scm = await self._update_workspace_scm_sync_metadata(
            workspace_id,
            git_url=git_url,
            git_branch=preview.git_branch,
            provider=preview.provider,
            repo_visibility=(
                request.create_repo_private
                and "private"
                or preview.repo_visibility
                or "public"
            ),
            git_token=export_request.git_token,
            remote_commit_hash=remote_commit_hash,
            snapshot_id=exported_snapshot.id,
            status="success",
            message=f"Exported to {preview.git_branch}",
            direction="export",
        )

        return UserSpaceWorkspaceScmSyncResponse(
            workspace_id=workspace_id,
            direction="export",
            state="success",
            summary="Workspace exported successfully.",
            scm=scm,
            snapshot=exported_snapshot,
            remote_commit_hash=remote_commit_hash,
            suggested_setup_prompt=None,
        )

    @staticmethod
    def _selected_tool_ids_from_workspace_record(workspace_record: Any) -> list[str]:
        tool_rows = list(getattr(workspace_record, "toolSelections", []) or [])
        return [
            getattr(row, "toolConfigId", "")
            for row in tool_rows
            if getattr(row, "toolConfigId", None)
        ]

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        db = await get_db()

        workspace = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

        if is_admin:
            self._sync_runtime_bootstrap_config(workspace_id)
            return self._workspace_from_record(workspace)

        user_role: str | None = None
        if workspace.ownerUserId == user_id:
            user_role = "owner"
        else:
            for member in list(getattr(workspace, "members", []) or []):
                if getattr(member, "userId", None) != user_id:
                    continue
                role_value = getattr(member, "role", "viewer")
                user_role = (
                    role_value
                    if isinstance(role_value, str)
                    else str(getattr(role_value, "value", role_value))
                )
                break

        if user_role is None:
            # Check if user is an admin — admins get owner-level access to all workspaces
            user_record = await db.user.find_unique(where={"id": user_id})
            if user_record and getattr(user_record, "role", None) == "admin":
                self._sync_runtime_bootstrap_config(workspace_id)
                return self._workspace_from_record(workspace)
            raise HTTPException(status_code=404, detail="Workspace not found")

        if required_role == "editor" and user_role not in {"owner", "editor"}:
            raise HTTPException(status_code=403, detail="Editor access required")
        if required_role == "owner" and user_role != "owner":
            raise HTTPException(status_code=403, detail="Owner access required")

        self._sync_runtime_bootstrap_config(workspace_id)
        return self._workspace_from_record(workspace)

    async def _get_workspace_record(self, workspace_id: str) -> Any:
        db = await get_db()
        return await db.workspace.find_unique(
            where={"id": workspace_id},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
        )

    async def list_workspaces(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 50,
        is_admin: bool = False,
    ) -> PaginatedWorkspacesResponse:
        db = await get_db()

        where_clause: dict[str, Any] = {}
        if not is_admin:
            where_clause = {
                "OR": [
                    {"ownerUserId": user_id},
                    {"members": {"some": {"userId": user_id}}},
                ]
            }

        rows = await db.workspace.find_many(
            where=where_clause,
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
            order={"updatedAt": "desc"},
            skip=offset,
            take=limit,
        )
        total = await db.workspace.count(where=where_clause)

        return PaginatedWorkspacesResponse(
            items=[self._workspace_from_record(row) for row in rows],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def create_workspace(
        self, request: CreateWorkspaceRequest, user_id: str
    ) -> UserSpaceWorkspace:
        db = await get_db()

        now = _utc_now()
        workspace_id = str(uuid4())
        requested_name = (request.name or "").strip()

        if requested_name:
            final_name = requested_name
        else:
            final_name = await self._allocate_next_default_workspace_name(user_id)

        name_normalized = _normalize_workspace_name_for_uniqueness(final_name)
        if not name_normalized:
            raise HTTPException(status_code=400, detail="Workspace name is required")

        try:
            await db.workspace.create(
                data={
                    "id": workspace_id,
                    "name": final_name,
                    "nameNormalized": name_normalized,
                    "description": request.description,
                    "sqlitePersistenceMode": _normalize_sqlite_persistence_mode(
                        request.sqlite_persistence_mode
                    ),
                    "ownerUserId": user_id,
                    "createdAt": now,
                    "updatedAt": now,
                },
                include={
                    "members": True,
                    "toolSelections": True,
                    "toolGroupSelections": True,
                },
            )
        except Exception as exc:
            if _is_workspace_name_conflict_error(exc):
                raise HTTPException(
                    status_code=409,
                    detail="A workspace with that name already exists for this owner",
                ) from exc
            raise

        await db.workspacemember.create(
            data={
                "workspaceId": workspace_id,
                "userId": user_id,
                "role": "owner",
            }
        )

        requested_tool_ids = request.selected_tool_ids
        if requested_tool_ids is None:
            enabled_tools = await db.toolconfig.find_many(where={"enabled": True})
            requested_tool_ids = [str(tool.id) for tool in enabled_tools if tool.id]

        for tool_id in requested_tool_ids:
            tool = await db.toolconfig.find_unique(where={"id": tool_id})
            if not tool or not tool.enabled:
                continue
            await db.workspacetoolselection.create(
                data={
                    "workspaceId": workspace_id,
                    "toolConfigId": tool_id,
                }
            )

        if request.selected_tool_group_ids:
            for group_id in request.selected_tool_group_ids:
                grp = await db.toolgroup.find_unique(where={"id": group_id})
                if not grp:
                    continue
                await db.workspacetoolgroupselection.create(
                    data={
                        "workspaceId": workspace_id,
                        "toolGroupId": group_id,
                    }
                )

        self._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        self._ensure_object_storage_config(workspace_id)
        self._seed_runtime_bootstrap_config(workspace_id)
        self._seed_runtime_entrypoint_config(workspace_id)
        await self._ensure_workspace_git_repo(workspace_id)

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
        )
        if not refreshed:
            raise HTTPException(status_code=500, detail="Failed to create workspace")

        return self._workspace_from_record(refreshed)

    async def get_workspace(
        self, workspace_id: str, user_id: str
    ) -> UserSpaceWorkspace:
        return await self._enforce_workspace_access(workspace_id, user_id)

    async def get_workspace_share_link_status(
        self,
        workspace_id: str,
        user_id: str,
        base_url: str | None = None,
    ) -> UserSpaceWorkspaceShareLinkStatus:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        share_token = str(getattr(workspace_record, "shareToken", "") or "")
        share_slug = str(getattr(workspace_record, "shareSlug", "") or "")
        access_mode, password_encrypted, selected_user_ids, selected_ldap_groups = (
            self._extract_share_access_state(workspace_record)
        )
        owner_username = await self._get_public_owner_username(
            str(getattr(workspace_record, "ownerUserId", "") or "")
        )
        created_at = getattr(workspace_record, "shareTokenCreatedAt", None)
        if not share_token or not share_slug:
            return UserSpaceWorkspaceShareLinkStatus(
                workspace_id=workspace_id,
                has_share_link=False,
                owner_username=owner_username,
                share_slug=share_slug or None,
                share_token=None,
                share_url=None,
                created_at=None,
                share_access_mode=access_mode,
                selected_user_ids=selected_user_ids,
                selected_ldap_groups=selected_ldap_groups,
                has_password=bool(password_encrypted),
            )

        return UserSpaceWorkspaceShareLinkStatus(
            workspace_id=workspace_id,
            has_share_link=True,
            owner_username=owner_username,
            share_slug=share_slug,
            share_token=share_token,
            share_url=self._build_workspace_share_url(
                owner_username,
                share_slug,
                base_url=base_url,
            ),
            created_at=created_at,
            share_access_mode=access_mode,
            selected_user_ids=selected_user_ids,
            selected_ldap_groups=selected_ldap_groups,
            has_password=bool(password_encrypted),
        )

    async def revoke_workspace_share_link(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceWorkspaceShareLinkStatus:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        owner_username = await self._get_public_owner_username(
            str(getattr(workspace_record, "ownerUserId", "") or "")
        )
        share_slug = str(getattr(workspace_record, "shareSlug", "") or "")

        try:
            await db.workspace.update(
                where={"id": workspace_id},
                data={
                    "shareToken": None,
                    "shareTokenCreatedAt": None,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

        return UserSpaceWorkspaceShareLinkStatus(
            workspace_id=workspace_id,
            has_share_link=False,
            owner_username=owner_username,
            share_slug=share_slug or None,
            share_token=None,
            share_url=None,
            created_at=None,
            share_access_mode=_normalize_share_access_mode(
                str(getattr(workspace_record, "shareAccessMode", "token") or "token")
            ),
            selected_user_ids=_normalize_string_list(
                getattr(workspace_record, "shareSelectedUserIds", [])
            ),
            selected_ldap_groups=_normalize_string_list(
                getattr(workspace_record, "shareSelectedLdapGroups", [])
            ),
            has_password=bool(
                str(getattr(workspace_record, "sharePassword", "") or "")
            ),
        )

    async def create_workspace_share_link(
        self,
        workspace_id: str,
        user_id: str,
        base_url: str | None = None,
        rotate_token: bool = False,
    ) -> UserSpaceWorkspaceShareLink:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        existing_token = str(getattr(workspace_record, "shareToken", "") or "")
        owner_user_id = str(getattr(workspace_record, "ownerUserId", "") or "")
        existing_slug = str(getattr(workspace_record, "shareSlug", "") or "")
        share_slug = _normalize_share_slug_for_uniqueness(existing_slug)
        if not share_slug:
            share_slug = await self._allocate_next_share_slug(
                owner_user_id,
                str(getattr(workspace_record, "name", "") or ""),
            )

        if existing_token and not rotate_token:
            share_token = existing_token
            if share_slug != existing_slug:
                try:
                    await db.workspace.update(
                        where={"id": workspace_id},
                        data={"shareSlug": share_slug},
                    )
                except Exception as exc:
                    if _is_share_slug_conflict_error(exc):
                        raise HTTPException(
                            status_code=409,
                            detail="A share slug with that value already exists for this owner",
                        ) from exc
                    raise
        else:
            share_token = ""
            max_attempts = 20
            for _ in range(max_attempts):
                candidate = self._generate_share_token()
                try:
                    await db.workspace.update(
                        where={"id": workspace_id},
                        data={
                            "shareToken": candidate,
                            "shareTokenCreatedAt": _utc_now(),
                            "shareSlug": share_slug,
                        },
                    )
                    share_token = candidate
                    break
                except Exception as exc:
                    if _is_share_token_conflict_error(
                        exc
                    ) or _is_share_slug_conflict_error(exc):
                        continue
                    raise
            if not share_token:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate workspace share link",
                )
        owner_username = await self._get_public_owner_username(owner_user_id)
        share_url = self._build_workspace_share_url(
            owner_username,
            share_slug,
            base_url=base_url,
        )

        return UserSpaceWorkspaceShareLink(
            workspace_id=workspace_id,
            share_token=share_token,
            owner_username=owner_username,
            share_slug=share_slug,
            share_url=share_url,
        )

    async def update_workspace_share_slug(
        self,
        workspace_id: str,
        slug: str,
        user_id: str,
        base_url: str | None = None,
    ) -> UserSpaceWorkspaceShareLinkStatus:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        normalized_slug = _normalize_share_slug_for_uniqueness(slug)
        if not normalized_slug:
            raise HTTPException(status_code=400, detail="Share slug is required")

        db = await get_db()
        try:
            await db.workspace.update(
                where={"id": workspace_id},
                data={
                    "shareSlug": normalized_slug,
                    "updatedAt": _utc_now(),
                },
            )
        except Exception as exc:
            if _is_share_slug_conflict_error(exc):
                raise HTTPException(
                    status_code=409,
                    detail="A share slug with that value already exists for this owner",
                ) from exc
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

        return await self.get_workspace_share_link_status(
            workspace_id,
            user_id,
            base_url=base_url,
        )

    async def check_workspace_share_slug_availability(
        self,
        workspace_id: str,
        slug: str,
        user_id: str,
    ) -> WorkspaceShareSlugAvailabilityResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        normalized_slug = _normalize_share_slug_for_uniqueness(slug)
        if not normalized_slug:
            raise HTTPException(status_code=400, detail="Share slug is required")

        db = await get_db()
        workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        existing = await db.workspace.find_first(
            where={
                "ownerUserId": str(getattr(workspace_record, "ownerUserId", "") or ""),
                "shareSlug": normalized_slug,
                "NOT": {"id": workspace_id},
            }
        )
        return WorkspaceShareSlugAvailabilityResponse(
            slug=normalized_slug,
            available=existing is None,
        )

    async def update_workspace_share_access(
        self,
        workspace_id: str,
        request: UpdateWorkspaceShareAccessRequest,
        user_id: str,
        base_url: str | None = None,
    ) -> UserSpaceWorkspaceShareLinkStatus:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        mode = _normalize_share_access_mode(request.share_access_mode)
        selected_user_ids = _normalize_string_list(request.selected_user_ids)
        selected_ldap_groups = _normalize_string_list(request.selected_ldap_groups)

        update_data: dict[str, Any] = {
            "shareAccessMode": mode,
            "shareSelectedUserIds": Json(selected_user_ids),
            "shareSelectedLdapGroups": Json(selected_ldap_groups),
            "updatedAt": _utc_now(),
        }

        if mode == "password":
            password = (request.password or "").strip()
            if not password:
                raise HTTPException(status_code=400, detail="Password is required")
            update_data["sharePassword"] = encrypt_secret(password)
        else:
            update_data["sharePassword"] = None

        if mode == "selected_users" and not selected_user_ids:
            raise HTTPException(
                status_code=400,
                detail="At least one user is required for selected-users mode",
            )
        if mode == "ldap_groups" and not selected_ldap_groups:
            raise HTTPException(
                status_code=400,
                detail="At least one LDAP group is required for ldap-groups mode",
            )

        db = await get_db()
        await db.workspace.update(where={"id": workspace_id}, data=update_data)

        # Invalidate stale preview sessions so visitors must re-authenticate
        # through the proper share URL with the new access mode.
        await invalidate_preview_sessions_for_workspace(workspace_id)
        from ragtime.userspace.runtime_service import userspace_runtime_service

        await userspace_runtime_service.invalidate_preview_session_cache(workspace_id)

        return await self.get_workspace_share_link_status(
            workspace_id,
            user_id,
            base_url=base_url,
        )

    async def get_shared_preview(
        self,
        share_token: str,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        return await self._build_shared_preview_response(
            workspace_id,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def _get_authorized_shared_workspace_record(
        self,
        workspace_id: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> Any:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        await self._authorize_shared_workspace_record(
            workspace_record,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )
        return workspace_record

    async def authorize_shared_workspace_access(
        self,
        share_token: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> _ShareAuthorizationResult:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        return await self._authorize_shared_workspace_record(
            workspace_record,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def resolve_shared_workspace_id(
        self,
        share_token: str,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> str:
        authorization = await self.authorize_shared_workspace_access(
            share_token,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )
        return authorization["workspace_id"]

    async def get_shared_preview_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_id = await self._resolve_workspace_id_from_share_slug(
            owner_username,
            share_slug,
        )
        return await self._build_shared_preview_response(
            workspace_id,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def authorize_shared_workspace_access_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> _ShareAuthorizationResult:
        workspace_id = await self._resolve_workspace_id_from_share_slug(
            owner_username,
            share_slug,
        )
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        return await self._authorize_shared_workspace_record(
            workspace_record,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def resolve_shared_workspace_id_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> str:
        authorization = await self.authorize_shared_workspace_access_by_slug(
            owner_username,
            share_slug,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )
        return authorization["workspace_id"]

    async def _build_shared_preview_response(
        self,
        workspace_id: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_record = await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

        files_dir = self._workspace_files_dir(workspace_id)
        if not files_dir.exists() or not files_dir.is_dir():
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        workspace_files = await asyncio.to_thread(
            self._collect_preview_workspace_files,
            files_dir,
            _USERSPACE_PREVIEW_ENTRY_PATH,
        )

        entry_file_path = files_dir / _USERSPACE_PREVIEW_ENTRY_PATH
        _, live_data_connections, _ = await asyncio.to_thread(
            self._read_artifact_sidecar,
            entry_file_path,
        )

        return UserSpaceSharedPreviewResponse(
            workspace_id=workspace_id,
            workspace_name=str(getattr(workspace_record, "name", "User Space")),
            entry_path=_USERSPACE_PREVIEW_ENTRY_PATH,
            workspace_files=workspace_files,
            live_data_connections=live_data_connections,
        )

    async def delete_workspace(
        self, workspace_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="owner", is_admin=is_admin
        )
        db = await get_db()
        try:
            await db.workspace.delete(where={"id": workspace_id})
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

        workspace_dir = self._workspace_dir(workspace_id)
        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_dir)

    async def enforce_workspace_role(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        return await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role=required_role,
            is_admin=is_admin,
        )

    async def update_workspace(
        self,
        workspace_id: str,
        request: UpdateWorkspaceRequest,
        user_id: str,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        current_ws = await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor", is_admin=is_admin
        )
        db = await get_db()

        update_data: dict[str, Any] = {"updatedAt": _utc_now()}

        if request.owner_user_id is not None:
            if not is_admin:
                raise HTTPException(
                    status_code=403,
                    detail="Only admins can transfer workspace ownership",
                )
            new_owner = await db.user.find_unique(where={"id": request.owner_user_id})
            if not new_owner:
                raise HTTPException(status_code=404, detail="Target user not found")
            update_data["ownerUserId"] = request.owner_user_id
            # Ensure new owner is a member with owner role
            existing_member = await db.workspacemember.find_first(
                where={"workspaceId": workspace_id, "userId": request.owner_user_id}
            )
            if existing_member:
                await db.workspacemember.update(
                    where={"id": existing_member.id},
                    data={"role": "owner"},
                )
            else:
                await db.workspacemember.create(
                    data={
                        "workspaceId": workspace_id,
                        "userId": request.owner_user_id,
                        "role": "owner",
                    }
                )
            # Downgrade previous owner to editor in members
            old_owner_id = current_ws.owner_user_id
            if old_owner_id != request.owner_user_id:
                old_owner_member = await db.workspacemember.find_first(
                    where={"workspaceId": workspace_id, "userId": old_owner_id}
                )
                if old_owner_member:
                    await db.workspacemember.update(
                        where={"id": old_owner_member.id},
                        data={"role": "editor"},
                    )

        if request.name is not None:
            normalized_name = _normalize_workspace_name_for_uniqueness(request.name)
            if not normalized_name:
                raise HTTPException(
                    status_code=400, detail="Workspace name is required"
                )
            update_data["name"] = request.name.strip()
            update_data["nameNormalized"] = normalized_name
        if request.description is not None:
            update_data["description"] = request.description
        if request.sqlite_persistence_mode is not None:
            update_data["sqlitePersistenceMode"] = _normalize_sqlite_persistence_mode(
                request.sqlite_persistence_mode
            )

        try:
            await db.workspace.update(where={"id": workspace_id}, data=update_data)
        except Exception as exc:
            if _is_workspace_name_conflict_error(exc):
                raise HTTPException(
                    status_code=409,
                    detail="A workspace with that name already exists for this owner",
                ) from exc
            raise

        if request.selected_tool_ids is not None:
            await db.workspacetoolselection.delete_many(
                where={"workspaceId": workspace_id}
            )
            for tool_id in request.selected_tool_ids:
                tool = await db.toolconfig.find_unique(where={"id": tool_id})
                if not tool or not tool.enabled:
                    continue
                await db.workspacetoolselection.create(
                    data={
                        "workspaceId": workspace_id,
                        "toolConfigId": tool_id,
                    }
                )

        if request.selected_tool_group_ids is not None:
            await db.workspacetoolgroupselection.delete_many(
                where={"workspaceId": workspace_id}
            )
            for group_id in request.selected_tool_group_ids:
                grp = await db.toolgroup.find_unique(where={"id": group_id})
                if not grp:
                    continue
                await db.workspacetoolgroupselection.create(
                    data={
                        "workspaceId": workspace_id,
                        "toolGroupId": group_id,
                    }
                )

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
        )
        if not refreshed:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return self._workspace_from_record(refreshed)

    async def update_workspace_members(
        self,
        workspace_id: str,
        request: UpdateWorkspaceMembersRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        workspace = await self._enforce_workspace_access(
            workspace_id, user_id, required_role="owner"
        )
        db = await get_db()

        normalized_members: dict[str, WorkspaceMember] = {
            workspace.owner_user_id: WorkspaceMember(
                user_id=workspace.owner_user_id, role="owner"
            )
        }
        for member in request.members:
            if member.user_id == workspace.owner_user_id:
                continue
            normalized_role = "editor" if member.role == "owner" else member.role
            normalized_members[member.user_id] = WorkspaceMember(
                user_id=member.user_id,
                role=normalized_role,
            )

        await db.workspacemember.delete_many(where={"workspaceId": workspace_id})
        for member in normalized_members.values():
            user = await db.user.find_unique(where={"id": member.user_id})
            if not user:
                continue
            await db.workspacemember.create(
                data={
                    "workspaceId": workspace_id,
                    "userId": member.user_id,
                    "role": member.role,
                }
            )

        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": _utc_now()},
        )

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
                "owner": True,
            },
        )
        if not refreshed:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return self._workspace_from_record(refreshed)

    async def list_workspace_env_vars(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[UserSpaceWorkspaceEnvVar]:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        db = await get_db()
        rows = await self._workspace_env_var_model(db).find_many(
            where={"workspaceId": workspace_id},
            order={"key": "asc"},
        )
        return [self._workspace_env_var_from_record(row) for row in rows]

    async def list_workspace_env_var_summaries(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[UserSpaceWorkspaceEnvVar]:
        """List env-var metadata (keys + has_value) for any workspace member.

        Values are never returned by this method. It is safe to use for
        prompt/reminder context and agent guidance.
        """

        await self._enforce_workspace_access(workspace_id, user_id)
        db = await get_db()
        rows = await self._workspace_env_var_model(db).find_many(
            where={"workspaceId": workspace_id},
            order={"key": "asc"},
        )
        return [self._workspace_env_var_from_record(row) for row in rows]

    async def upsert_workspace_env_var(
        self,
        workspace_id: str,
        user_id: str,
        request: UpsertWorkspaceEnvVarRequest,
    ) -> UserSpaceWorkspaceEnvVar:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        db = await get_db()
        model = self._workspace_env_var_model(db)

        key = self._normalize_workspace_env_var_key(request.key)
        target_key = self._normalize_workspace_env_var_key(request.new_key or key)
        description = request.description
        now = _utc_now()

        existing = await model.find_first(
            where={"workspaceId": workspace_id, "key": key}
        )

        if existing is None:
            count = await model.count(where={"workspaceId": workspace_id})
            if count >= _WORKSPACE_ENV_VAR_MAX_COUNT:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Workspace environment variable limit reached ({_WORKSPACE_ENV_VAR_MAX_COUNT})"
                    ),
                )

            conflict = await model.find_first(
                where={"workspaceId": workspace_id, "key": target_key}
            )
            if conflict is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Environment variable '{target_key}' already exists",
                )

            created = await model.create(
                data={
                    "id": str(uuid4()),
                    "workspaceId": workspace_id,
                    "key": target_key,
                    # Empty string means "placeholder key" (no secret set yet).
                    "value": (
                        encrypt_secret(request.value)
                        if request.value is not None
                        else ""
                    ),
                    "description": description,
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            await db.workspace.update(
                where={"id": workspace_id},
                data={"updatedAt": now},
            )
            await self._audit_workspace_env_var_event(
                workspace_id,
                user_id,
                "env_var_created",
                payload={"key": target_key},
            )
            return self._workspace_env_var_from_record(created)

        if target_key != key:
            conflict = await model.find_first(
                where={
                    "workspaceId": workspace_id,
                    "key": target_key,
                    "NOT": {"id": str(getattr(existing, "id", "") or "")},
                }
            )
            if conflict is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Environment variable '{target_key}' already exists",
                )

        encrypted_value = getattr(existing, "value", "")
        if request.value is not None:
            encrypted_value = encrypt_secret(request.value)

        next_description = (
            str(getattr(existing, "description", "") or "")
            if description is None
            else description
        )

        updated = await model.update(
            where={"id": str(getattr(existing, "id", "") or "")},
            data={
                "key": target_key,
                "value": encrypted_value,
                "description": next_description,
                "updatedAt": now,
            },
        )
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": now},
        )

        event_type = "env_var_updated"
        payload: dict[str, Any] = {"key": key}
        if target_key != key:
            event_type = "env_var_renamed"
            payload["new_key"] = target_key
        if request.value is not None:
            payload["value_replaced"] = True
        await self._audit_workspace_env_var_event(
            workspace_id,
            user_id,
            event_type,
            payload=payload,
        )

        return self._workspace_env_var_from_record(updated)

    async def delete_workspace_env_var(
        self,
        workspace_id: str,
        user_id: str,
        key: str,
    ) -> DeleteWorkspaceEnvVarResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="owner",
        )
        normalized_key = self._normalize_workspace_env_var_key(key)
        db = await get_db()
        model = self._workspace_env_var_model(db)
        existing = await model.find_first(
            where={"workspaceId": workspace_id, "key": normalized_key}
        )
        if existing is None:
            raise HTTPException(
                status_code=404, detail="Environment variable not found"
            )

        now = _utc_now()
        await model.delete(where={"id": str(getattr(existing, "id", "") or "")})
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": now},
        )
        await self._audit_workspace_env_var_event(
            workspace_id,
            user_id,
            "env_var_deleted",
            payload={"key": normalized_key},
        )
        return DeleteWorkspaceEnvVarResponse(success=True, key=normalized_key)

    async def get_workspace_runtime_environment(
        self,
        workspace_id: str,
    ) -> dict[str, str]:
        db = await get_db()
        rows = await self._workspace_env_var_model(db).find_many(
            where={"workspaceId": workspace_id},
            order={"key": "asc"},
        )
        env_map = self._sanitize_workspace_env_map(list(rows))
        env_map.update(self._build_object_storage_runtime_env(workspace_id))
        return env_map

    async def get_workspace_runtime_environment_visibility(
        self,
        workspace_id: str,
    ) -> dict[str, bool]:
        db = await get_db()
        rows = await self._workspace_env_var_model(db).find_many(
            where={"workspaceId": workspace_id},
            order={"key": "asc"},
        )
        visibility: dict[str, bool] = {}
        for row in rows:
            key = str(getattr(row, "key", "") or "").strip()
            if not key:
                continue
            visibility[key] = bool(str(getattr(row, "value", "") or ""))
        return visibility

    # ------------------------------------------------------------------
    # Workspace Mounts
    # ------------------------------------------------------------------

    _WORKSPACE_MOUNT_MAX_COUNT = 50
    _RESERVED_MOUNT_TARGETS = {
        "/workspace",
        "/proc",
        "/dev",
        "/sys",
        "/run",
        "/tmp",
        "/bin",
        "/sbin",
        "/usr",
        "/lib",
        "/lib64",
        "/etc",
        "/var",
    }

    @staticmethod
    def _normalize_mount_source_name(name: str) -> str:
        normalized = str(name or "").strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Mount source name is required")
        return normalized

    @staticmethod
    def _normalize_mount_source_description(description: str | None) -> str | None:
        if description is None:
            return None
        normalized = description.strip()
        return normalized or None

    @staticmethod
    def _load_connection_config(raw_config: Any) -> dict[str, Any]:
        if isinstance(raw_config, str):
            raw_config = json.loads(raw_config)
        if not isinstance(raw_config, dict):
            return {}
        return dict(raw_config)

    @classmethod
    def _load_mount_source_connection_config(cls, record: Any) -> dict[str, Any]:
        raw_config = cls._load_connection_config(
            getattr(record, "connectionConfig", None)
        )
        return decrypt_json_passwords(raw_config, CONNECTION_CONFIG_PASSWORD_FIELDS)

    @classmethod
    def _load_tool_connection_config(cls, tool_record: Any) -> dict[str, Any]:
        """Load and decrypt connection config from a related ToolConfig record."""
        raw_config = cls._load_connection_config(
            getattr(tool_record, "connectionConfig", None)
        )
        return decrypt_json_passwords(raw_config, CONNECTION_CONFIG_PASSWORD_FIELDS)

    @classmethod
    def _load_mount_source_approved_paths(cls, record: Any) -> list[str]:
        raw_paths = getattr(record, "approvedPaths", None)
        if isinstance(raw_paths, str):
            raw_paths = json.loads(raw_paths)
        if not isinstance(raw_paths, list):
            raw_paths = []
        return cls._normalize_mount_source_paths([str(path) for path in raw_paths])

    @classmethod
    def _mount_backend_from_source(
        cls,
        source_type: str,
        connection_config: dict[str, Any],
    ) -> UserspaceMountBackend:
        if source_type == "ssh":
            return "ssh"
        mount_type = str(connection_config.get("mount_type") or "docker_volume").strip()
        if mount_type not in {"docker_volume", "smb", "nfs", "local"}:
            raise HTTPException(
                status_code=400, detail=f"Unsupported mount backend '{mount_type}'"
            )
        return cast(UserspaceMountBackend, mount_type)

    @classmethod
    def _normalize_mount_source_payload(
        cls,
        *,
        source_type: UserspaceMountSourceType,
        connection_config: dict[str, Any],
        approved_paths: list[str] | None,
    ) -> tuple[dict[str, Any], list[str], UserspaceMountBackend]:
        if not isinstance(connection_config, dict):
            raise HTTPException(
                status_code=400,
                detail="Mount source connection config must be an object",
            )

        normalized_config = dict(connection_config)
        if source_type == "ssh":
            host = str(normalized_config.get("host") or "").strip()
            user = str(normalized_config.get("user") or "").strip()
            if not host:
                raise HTTPException(
                    status_code=400, detail="SSH mount sources require a host"
                )
            if not user:
                raise HTTPException(
                    status_code=400, detail="SSH mount sources require a user"
                )
            normalized_config["host"] = host
            normalized_config["user"] = user
            port = normalized_config.get("port")
            if port not in (None, ""):
                try:
                    port_str = str(port)
                    normalized_config["port"] = int(port_str)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=400, detail="SSH port must be an integer"
                    ) from exc
            backend: UserspaceMountBackend = "ssh"
        else:
            backend = cls._mount_backend_from_source(source_type, normalized_config)
            normalized_config["mount_type"] = backend
            base_path = str(normalized_config.get("base_path") or "").strip()
            if backend in {"docker_volume", "local"}:
                if not base_path:
                    raise HTTPException(
                        status_code=400,
                        detail="Filesystem mount sources require a base_path for local and docker_volume backends",
                    )
                normalized_config["base_path"] = base_path
            elif backend == "smb":
                if not str(normalized_config.get("smb_host") or "").strip():
                    raise HTTPException(
                        status_code=400, detail="SMB mount sources require smb_host"
                    )
                if not str(normalized_config.get("smb_share") or "").strip():
                    raise HTTPException(
                        status_code=400, detail="SMB mount sources require smb_share"
                    )
                normalized_config["base_path"] = base_path or "/"
            elif backend == "nfs":
                if not str(normalized_config.get("nfs_host") or "").strip():
                    raise HTTPException(
                        status_code=400, detail="NFS mount sources require nfs_host"
                    )
                if not str(normalized_config.get("nfs_export") or "").strip():
                    raise HTTPException(
                        status_code=400, detail="NFS mount sources require nfs_export"
                    )
                normalized_config["base_path"] = base_path or "/"

        normalized_paths = cls._normalize_mount_source_paths(approved_paths)
        if not normalized_paths:
            normalized_paths = ["."]
        return normalized_config, normalized_paths, backend

    @classmethod
    def _userspace_mount_source_from_record(
        cls, record: Any, usage_count: int = 0
    ) -> UserspaceMountSource:
        connection_config = cls._load_mount_source_connection_config(record)
        source_type = cast(
            UserspaceMountSourceType,
            str(getattr(record, "sourceType", "") or "filesystem"),
        )
        tool_config_id = getattr(record, "toolConfigId", None)
        tool_record = getattr(record, "toolConfig", None) if tool_config_id else None
        tool_name = str(getattr(tool_record, "name", "")) if tool_record else None
        # When backed by a tool, prefer the tool's connection config so credential
        # changes propagate automatically.
        if tool_record is not None:
            tool_connection = cls._load_tool_connection_config(tool_record)
            if tool_connection:
                connection_config = tool_connection
        return UserspaceMountSource(
            id=str(getattr(record, "id", "") or ""),
            name=str(getattr(record, "name", "") or ""),
            description=cls._normalize_mount_source_description(
                getattr(record, "description", None)
            ),
            enabled=bool(getattr(record, "enabled", True)),
            source_type=source_type,
            mount_backend=cls._mount_backend_from_source(
                source_type, connection_config
            ),
            tool_config_id=str(tool_config_id) if tool_config_id else None,
            tool_name=tool_name or None,
            connection_config=connection_config,
            approved_paths=cls._load_mount_source_approved_paths(record),
            sync_interval_seconds=getattr(record, "syncIntervalSeconds", 30) or 30,
            usage_count=usage_count,
            created_at=getattr(record, "createdAt"),
            updated_at=getattr(record, "updatedAt"),
        )

    @staticmethod
    def _workspace_mount_from_record(
        record: Any,
        mount_source: UserspaceMountSource | None = None,
    ) -> WorkspaceMount:
        source_type = mount_source.source_type if mount_source else None
        sync_backend = getattr(record, "syncBackend", None)
        sync_mode = UserSpaceService._normalize_workspace_mount_sync_mode(
            getattr(record, "syncMode", None),
            legacy_sync_deletes=bool(getattr(record, "syncDeletes", False)),
        )
        if source_type == "ssh" and not sync_backend:
            sync_backend = "rsync"
        return WorkspaceMount(
            id=str(getattr(record, "id", "") or ""),
            workspace_id=str(getattr(record, "workspaceId", "") or ""),
            mount_source_id=str(getattr(record, "mountSourceId", "") or ""),
            source_path=str(getattr(record, "sourcePath", "") or ""),
            target_path=str(getattr(record, "targetPath", "") or ""),
            description=str(getattr(record, "description", "") or "").strip() or None,
            enabled=bool(getattr(record, "enabled", True)),
            sync_mode=sync_mode,
            sync_status=str(getattr(record, "syncStatus", "pending") or "pending"),  # type: ignore[arg-type]
            sync_backend=str(sync_backend) if sync_backend else None,
            sync_notice=str(getattr(record, "syncNotice", "") or "").strip() or None,
            last_sync_at=getattr(record, "lastSyncAt", None),
            last_sync_error=getattr(record, "lastSyncError", None),
            auto_sync_enabled=bool(getattr(record, "autoSyncEnabled", False)),
            source_name=mount_source.name if mount_source else None,
            source_type=source_type,
            mount_backend=mount_source.mount_backend if mount_source else None,
            created_at=getattr(record, "createdAt"),
            updated_at=getattr(record, "updatedAt"),
        )

    async def _check_mount_source_available(
        self,
        row: Any,
        mount_source: UserspaceMountSource | None,
    ) -> bool:
        """Check whether the backing source directory is reachable.

        For filesystem mounts this verifies the resolved local path exists,
        which will be False when the backing Docker volume has been removed.
        SSH mounts are considered available if the sync cache exists and the
        mount source is enabled; actual remote reachability is checked at sync
        time, not here.
        """
        if mount_source is None:
            return False
        if not mount_source.enabled:
            return False

        source_path = str(getattr(row, "sourcePath", "") or "")

        if mount_source.source_type == "filesystem":
            try:
                resolved = await self._resolve_filesystem_mount_source_local_path(
                    mount_source_id=mount_source.id,
                    connection_config=mount_source.connection_config,
                    source_path=source_path,
                )
                return Path(resolved).is_dir()
            except Exception:
                return False

        # SSH mounts — source is "available" when the mount source record is
        # enabled (remote reachability is validated at sync time).
        return True

    async def _sync_workspace_mount_record(
        self,
        db: Any,
        mount: Any,
        *,
        force_backend_recheck: bool = False,
        preview_token: str | None = None,
        allow_destructive_auto_sync_approval: bool = False,
    ) -> WorkspaceMountSyncResponse:
        mount_id = str(getattr(mount, "id", "") or "")
        if not mount_id:
            raise HTTPException(status_code=400, detail="Mount not found")
        sync_mode = self._normalize_workspace_mount_sync_mode(
            getattr(mount, "syncMode", None),
            legacy_sync_deletes=bool(getattr(mount, "syncDeletes", False)),
        )
        if preview_token is not None or self._is_destructive_workspace_mount_sync_mode(
            sync_mode
        ):
            return await self._sync_workspace_mount_record_once(
                db,
                mount,
                force_backend_recheck=force_backend_recheck,
                preview_token=preview_token,
                allow_destructive_auto_sync_approval=allow_destructive_auto_sync_approval,
            )

        async with self._workspace_mount_sync_tasks_lock:
            existing_task = self._workspace_mount_sync_tasks.get(mount_id)
            if existing_task is None or existing_task.done():
                existing_task = asyncio.create_task(
                    self._sync_workspace_mount_record_once(
                        db,
                        mount,
                        force_backend_recheck=force_backend_recheck,
                        preview_token=preview_token,
                        allow_destructive_auto_sync_approval=allow_destructive_auto_sync_approval,
                    ),
                    name=f"userspace-mount-sync:{mount_id}",
                )
                self._attach_workspace_mount_sync_task_cleanup(
                    mount_id,
                    existing_task,
                )
                self._workspace_mount_sync_tasks[mount_id] = existing_task

        try:
            return await asyncio.shield(existing_task)
        finally:
            if existing_task.done():
                self._prune_workspace_mount_sync_task(mount_id, existing_task)

    async def _sync_workspace_mount_record_once(
        self,
        db: Any,
        mount: Any,
        *,
        force_backend_recheck: bool = False,
        preview_token: str | None = None,
        allow_destructive_auto_sync_approval: bool = False,
    ) -> WorkspaceMountSyncResponse:
        context = await self._build_workspace_mount_sync_context(
            mount,
            force_backend_recheck=force_backend_recheck,
        )
        mount_id = context["mount_id"]
        workspace_id = context["workspace_id"]
        ssh_config = context["ssh_config"]
        remote_path = context["remote_path"]
        target_path = context["target_path"]
        cache_dir = context["cache_dir"]
        sync_mode = context["sync_mode"]
        preferred_backend = context["preferred_backend"]
        preferred_notice = context["preferred_notice"]
        mount_lock = await self._get_workspace_mount_operation_lock(mount_id)

        try:
            async with mount_lock:
                async with self._workspace_mount_sync_semaphore:
                    await asyncio.to_thread(
                        self._stage_runtime_mount_into_sync_cache,
                        workspace_id,
                        target_path,
                        cache_dir,
                    )
                    if self._is_destructive_workspace_mount_sync_mode(sync_mode):
                        if preview_token is not None:
                            await self._consume_workspace_mount_sync_preview(
                                mount_id=mount_id,
                                workspace_id=workspace_id,
                                ssh_config=ssh_config,
                                remote_path=remote_path,
                                cache_dir=cache_dir,
                                sync_mode=sync_mode,
                                preview_token=preview_token,
                            )
                        elif not (
                            allow_destructive_auto_sync_approval
                            and self._has_destructive_auto_sync_approval(
                                mount, sync_mode
                            )
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    "Destructive sync preview is missing, expired, or stale. "
                                    "Run preview again before syncing."
                                ),
                            )

                    if preferred_backend == "paramiko":
                        result = await asyncio.to_thread(
                            sync_ssh_directory,
                            ssh_config,
                            remote_path,
                            str(cache_dir),
                            sync_mode=sync_mode,
                        )
                        result.notice = preferred_notice
                    else:
                        result = await asyncio.to_thread(
                            rsync_ssh_directory,
                            ssh_config,
                            remote_path,
                            str(cache_dir),
                            sync_mode=sync_mode,
                        )
                        if result.success:
                            self._remember_remote_rsync_available(ssh_config)
                        elif is_rsync_missing_error("\n".join(result.errors)):
                            self._remember_remote_rsync_missing(ssh_config)
                            result = await asyncio.to_thread(
                                sync_ssh_directory,
                                ssh_config,
                                remote_path,
                                str(cache_dir),
                                sync_mode=sync_mode,
                            )
                            result.notice = self._ssh_rsync_fallback_notice()
            sync_status = "synced" if result.success else "error"
            sync_backend = result.backend_used or preferred_backend
            sync_notice = result.notice
            if result.success:
                refresh_notice = (
                    await self._maybe_refresh_active_runtime_mount_after_sync(
                        workspace_id,
                        mount_id,
                    )
                )
                sync_notice = self._merge_workspace_mount_sync_notices(
                    sync_notice,
                    refresh_notice,
                )
            last_error = "; ".join(result.errors[:5]) if result.errors else None
            return await self._finalize_workspace_mount_sync(
                db,
                mount_id=mount_id,
                workspace_id=workspace_id,
                sync_mode=sync_mode,
                sync_status=sync_status,
                files_synced=result.files_synced,
                sync_backend=sync_backend,
                sync_notice=sync_notice,
                last_sync_error=last_error,
            )
        except Exception as exc:
            logger.error("Mount sync failed for %s/%s: %s", workspace_id, mount_id, exc)
            return await self._finalize_workspace_mount_sync(
                db,
                mount_id=mount_id,
                workspace_id=workspace_id,
                sync_mode=sync_mode,
                sync_status="error",
                files_synced=0,
                sync_backend=preferred_backend,
                sync_notice=preferred_notice,
                last_sync_error=str(exc)[:500],
            )

    async def _maybe_refresh_active_runtime_mount_after_sync(
        self,
        workspace_id: str,
        mount_id: str,
    ) -> str | None:
        try:
            from ragtime.userspace.runtime_service import \
                userspace_runtime_service

            return await userspace_runtime_service.refresh_workspace_mount_after_sync(
                workspace_id,
                mount_id,
            )
        except Exception as exc:
            logger.warning(
                "Mount sync completed but automatic runtime refresh failed for %s/%s: %s",
                workspace_id,
                mount_id,
                exc,
            )
            detail = str(exc).strip() or "unknown error"
            return (
                "Sync completed, but the active runtime was not refreshed automatically: "
                f"{detail[:240]}"
            )

    async def _build_workspace_mount_sync_context(
        self,
        mount: Any,
        *,
        force_backend_recheck: bool = False,
        sync_mode_override: WorkspaceMountSyncMode | None = None,
    ) -> dict[str, Any]:
        mount_source_record = getattr(mount, "mountSource", None)
        source_type = str(getattr(mount_source_record, "sourceType", "") or "")
        mount_id = str(getattr(mount, "id", "") or "")
        workspace_id = str(getattr(mount, "workspaceId", "") or "")
        if source_type != "ssh":
            raise HTTPException(
                status_code=400,
                detail="Sync is only supported for SSH-backed mounts",
            )

        connection_config = self._load_mount_source_connection_config(
            mount_source_record
        )

        ssh_config = ssh_config_from_dict(connection_config)
        remote_path = self._resolve_ssh_mount_remote_path(
            connection_config,
            str(getattr(mount, "sourcePath", "") or ""),
        )
        target_path = str(getattr(mount, "targetPath", "") or "")
        cache_dir = self._base_dir / "mount_cache" / workspace_id / mount_id
        sync_mode = sync_mode_override or self._normalize_workspace_mount_sync_mode(
            getattr(mount, "syncMode", None),
            legacy_sync_deletes=bool(getattr(mount, "syncDeletes", False)),
        )
        preferred_backend, preferred_notice = await self._resolve_ssh_sync_backend(
            ssh_config,
            force_recheck_missing=force_backend_recheck,
        )

        return {
            "mount_id": mount_id,
            "workspace_id": workspace_id,
            "ssh_config": ssh_config,
            "remote_path": remote_path,
            "target_path": target_path,
            "cache_dir": cache_dir,
            "sync_mode": sync_mode,
            "preferred_backend": preferred_backend,
            "preferred_notice": preferred_notice,
        }

    async def _preview_workspace_mount_record(
        self,
        db: Any,
        mount: Any,
        *,
        force_backend_recheck: bool = False,
        sync_mode_override: WorkspaceMountSyncMode | None = None,
    ) -> WorkspaceMountSyncPreviewResponse:
        context = await self._build_workspace_mount_sync_context(
            mount,
            force_backend_recheck=force_backend_recheck,
            sync_mode_override=sync_mode_override,
        )
        mount_id = context["mount_id"]
        workspace_id = context["workspace_id"]
        ssh_config = context["ssh_config"]
        remote_path = context["remote_path"]
        target_path = context["target_path"]
        cache_dir = context["cache_dir"]
        sync_mode = context["sync_mode"]
        preferred_backend = context["preferred_backend"]
        preferred_notice = context["preferred_notice"]
        mount_lock = await self._get_workspace_mount_operation_lock(mount_id)

        try:
            async with mount_lock:
                async with self._workspace_mount_sync_semaphore:
                    await asyncio.to_thread(
                        self._stage_runtime_mount_into_sync_cache,
                        workspace_id,
                        target_path,
                        cache_dir,
                    )
                    preview = await asyncio.to_thread(
                        preview_ssh_directory_sync,
                        ssh_config,
                        remote_path,
                        str(cache_dir),
                        sync_mode=sync_mode,
                        sample_limit=self._WORKSPACE_MOUNT_SYNC_PREVIEW_SAMPLE_LIMIT,
                    )

            if not preview.success:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "; ".join(preview.errors[:5])
                        or "Failed to preview workspace mount sync"
                    ),
                )
        except Exception as exc:
            error_detail = self._format_workspace_mount_sync_error_detail(exc)[:500]
            logger.warning(
                "Mount sync preview failed for %s/%s: %s",
                workspace_id,
                mount_id,
                error_detail,
            )
            await self._write_workspace_mount_sync_state(
                db,
                mount_id=mount_id,
                workspace_id=workspace_id,
                sync_status="error",
                sync_backend=preferred_backend,
                sync_notice=preferred_notice,
                last_sync_error=error_detail,
                update_last_sync_at=False,
            )
            raise

        preview_token = secrets.token_urlsafe(24)
        preview_expires_at = _utc_now() + timedelta(
            seconds=self._WORKSPACE_MOUNT_SYNC_PREVIEW_TTL_SECONDS
        )
        await self._store_workspace_mount_sync_preview(
            _WorkspaceMountSyncPreviewRecord(
                token=preview_token,
                workspace_id=workspace_id,
                mount_id=mount_id,
                sync_mode=sync_mode,
                state_fingerprint=preview.state_fingerprint,
                expires_at=preview_expires_at,
            )
        )

        return WorkspaceMountSyncPreviewResponse(
            mount_id=mount_id,
            sync_mode=sync_mode,
            sync_backend=preferred_backend,
            sync_notice=preferred_notice,
            requires_confirmation=self._is_destructive_workspace_mount_sync_mode(
                sync_mode
            ),
            preview_token=preview_token,
            preview_expires_at=preview_expires_at,
            delete_from_source_count=preview.delete_from_source_count,
            delete_from_target_count=preview.delete_from_target_count,
            delete_from_source_paths=preview.delete_from_source_paths,
            delete_from_target_paths=preview.delete_from_target_paths,
            sample_limit=preview.sample_limit,
            last_sync_error=None,
        )

    def _stage_runtime_mount_into_sync_cache(
        self,
        workspace_id: str,
        target_path: str,
        cache_dir: Path,
    ) -> bool:
        runtime_dir = self._resolve_workspace_mount_runtime_target_dir(
            workspace_id,
            target_path,
        )
        if runtime_dir is None or not runtime_dir.is_dir():
            return False

        cache_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                f"{runtime_dir}/",
                f"{cache_dir}/",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "Failed to stage runtime mount content %s into sync cache %s: %s",
                runtime_dir,
                cache_dir,
                (
                    proc.stderr or proc.stdout or f"rsync exited {proc.returncode}"
                ).strip()[:300],
            )
            return False

        return True

    def _stage_runtime_mounts_into_sync_cache_sync(
        self,
        workspace_id: str,
        stage_specs: list[tuple[str, Path]],
    ) -> bool:
        staged_any = False
        for target_path, cache_dir in stage_specs:
            if self._stage_runtime_mount_into_sync_cache(
                workspace_id,
                target_path,
                cache_dir,
            ):
                staged_any = True
        return staged_any

    def _format_workspace_mount_sync_error_detail(self, error: Exception) -> str:
        if isinstance(error, HTTPException):
            detail = error.detail
            if isinstance(detail, list):
                joined = "; ".join(
                    str(item).strip() for item in detail if str(item).strip()
                )
                if joined:
                    return joined
            if detail is not None:
                text = str(detail).strip()
                if text:
                    return text

        text = str(error).strip()
        if text:
            return text
        return "Workspace mount sync failed"

    async def _write_workspace_mount_sync_state(
        self,
        db: Any,
        *,
        mount_id: str,
        workspace_id: str,
        sync_status: str,
        sync_backend: str | None,
        sync_notice: str | None,
        last_sync_error: str | None,
        update_last_sync_at: bool,
    ) -> None:
        now = _utc_now()
        data: dict[str, Any] = {
            "syncStatus": sync_status,
            "syncBackend": sync_backend,
            "syncNotice": sync_notice,
            "lastSyncError": last_sync_error,
            "updatedAt": now,
        }
        if update_last_sync_at:
            data["lastSyncAt"] = now

        await db.workspacemount.update(
            where={"id": mount_id},
            data=data,
        )
        self.invalidate_file_list_cache(workspace_id)

    async def _finalize_workspace_mount_sync(
        self,
        db: Any,
        *,
        mount_id: str,
        workspace_id: str,
        sync_mode: WorkspaceMountSyncMode,
        sync_status: str,
        files_synced: int,
        sync_backend: str | None,
        sync_notice: str | None,
        last_sync_error: str | None,
    ) -> WorkspaceMountSyncResponse:
        await self._write_workspace_mount_sync_state(
            db,
            mount_id=mount_id,
            workspace_id=workspace_id,
            sync_status=sync_status,
            sync_backend=sync_backend,
            sync_notice=sync_notice,
            last_sync_error=last_sync_error,
            update_last_sync_at=True,
        )
        return WorkspaceMountSyncResponse(
            mount_id=mount_id,
            sync_mode=sync_mode,
            sync_status=cast(Any, sync_status),
            files_synced=files_synced,
            sync_backend=sync_backend,
            sync_notice=sync_notice,
            last_sync_error=last_sync_error,
        )

    def _validate_mount_target_path(self, target_path: str) -> str:
        target = target_path.strip()
        if not target.startswith("/"):
            target = f"/{target}"
        target = posixpath.normpath(target)
        if target in self._RESERVED_MOUNT_TARGETS:
            raise HTTPException(
                status_code=400,
                detail=f"Target path '{target}' is reserved and cannot be used as a mount point",
            )
        for reserved in self._RESERVED_MOUNT_TARGETS:
            if reserved == "/workspace":
                # Mounts are allowed within the workspace tree, but not over the
                # workspace root itself.
                continue
            if target.startswith(reserved + "/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Target path '{target}' conflicts with reserved path '{reserved}'",
                )
        return target

    @staticmethod
    def _is_same_or_descendant_mount_path(path: str, candidate_ancestor: str) -> bool:
        normalized_path = posixpath.normpath(path)
        normalized_ancestor = posixpath.normpath(candidate_ancestor)
        return normalized_path == normalized_ancestor or normalized_path.startswith(
            normalized_ancestor + "/"
        )

    @staticmethod
    def _normalize_mount_description(description: str | None) -> str | None:
        if description is None:
            return None
        normalized = description.strip()
        return normalized or None

    @staticmethod
    def _normalize_mount_source_path(source_path: str) -> str:
        normalized = posixpath.normpath((source_path or "").strip().replace("\\", "/"))
        if normalized in {"", "."}:
            return "."
        if (
            normalized.startswith("/")
            or normalized == ".."
            or normalized.startswith("../")
        ):
            raise HTTPException(
                status_code=400,
                detail="Approved mount paths must be relative to the mount source root",
            )
        return normalized

    @classmethod
    def _normalize_mount_source_paths(cls, source_paths: list[str] | None) -> list[str]:
        normalized_paths: list[str] = []
        seen: set[str] = set()
        for raw_path in source_paths or []:
            normalized = cls._normalize_mount_source_path(raw_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            normalized_paths.append(normalized)
        return normalized_paths

    @staticmethod
    def _normalize_mount_browser_path(path: str) -> str:
        normalized_parts: list[str] = []
        for part in (path or "/").replace("\\", "/").split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                if normalized_parts:
                    normalized_parts.pop()
                continue
            normalized_parts.append(part)
        return "/" + "/".join(normalized_parts)

    @classmethod
    def _browser_path_to_mount_source_path(cls, path: str) -> str:
        normalized = cls._normalize_mount_browser_path(path)
        if normalized == "/":
            return "."
        return cls._normalize_mount_source_path(normalized.lstrip("/"))

    @classmethod
    def _is_mount_source_within_root(
        cls, source_path: str, root_source_path: str
    ) -> bool:
        normalized_source = cls._normalize_mount_source_path(source_path)
        normalized_root = cls._normalize_mount_source_path(root_source_path)
        if normalized_root == ".":
            return True
        return normalized_source == normalized_root or normalized_source.startswith(
            normalized_root + "/"
        )

    @classmethod
    def _ensure_mount_source_within_approved_paths(
        cls,
        source_path: str,
        approved_paths: list[str],
    ) -> None:
        normalized_source = cls._normalize_mount_source_path(source_path)
        if any(
            cls._is_mount_source_within_root(normalized_source, approved_path)
            for approved_path in approved_paths
        ):
            return
        raise HTTPException(
            status_code=400,
            detail="Source path is not within the mount source's approved paths",
        )

    @classmethod
    def _resolve_filesystem_mount_source_path(
        cls,
        connection_config: dict[str, Any],
        source_path: str,
    ) -> str:
        normalized = cls._normalize_mount_source_path(source_path)
        base_path = str(connection_config.get("base_path") or "").strip()
        if not base_path:
            raise HTTPException(
                status_code=400,
                detail="Filesystem mount source is unavailable because base_path is not configured",
            )
        if normalized == ".":
            return str(Path(base_path))
        return str(Path(base_path) / normalized)

    @classmethod
    def _resolve_ssh_mount_remote_path(
        cls,
        connection_config: dict[str, Any],
        source_path: str,
    ) -> str:
        normalized = cls._normalize_mount_source_path(source_path)
        working_directory = str(
            connection_config.get("working_directory") or ""
        ).strip()
        if not working_directory:
            return normalized
        if normalized == ".":
            return posixpath.normpath(working_directory)
        return posixpath.normpath(posixpath.join(working_directory, normalized))

    @staticmethod
    def _build_filesystem_mount_config(
        connection_config: dict[str, Any],
        mount_source_id: str,
    ) -> FilesystemConnectionConfig:
        fs_config_data = dict(connection_config)
        fs_config_data.setdefault("index_name", f"userspace-mount-{mount_source_id}")
        return FilesystemConnectionConfig(**fs_config_data)

    async def _resolve_filesystem_mount_source_local_path(
        self,
        *,
        mount_source_id: str,
        connection_config: dict[str, Any],
        source_path: str,
    ) -> str:
        fs_config = self._build_filesystem_mount_config(
            connection_config, mount_source_id
        )
        normalized = self._normalize_mount_source_path(source_path)
        async with filesystem_indexer.filesystem_access(
            fs_config, mount_source_id
        ) as base_path:
            base = Path(base_path)
            resolved = base if normalized == "." else base / normalized
            return str(resolved)

    async def _create_workspace_mount_source_directory(
        self,
        *,
        mount_source_id: str,
        source_type: UserspaceMountSourceType,
        mount_backend: UserspaceMountBackend,
        connection_config: dict[str, Any],
        source_path: str,
    ) -> None:
        if source_type == "filesystem":
            resolved_path = Path(
                await self._resolve_filesystem_mount_source_local_path(
                    mount_source_id=mount_source_id,
                    connection_config=connection_config,
                    source_path=source_path,
                )
            )

            def _mkdir() -> None:
                resolved_path.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(_mkdir)
            return

        if source_type == "ssh":
            ssh_config = ssh_config_from_dict(connection_config)
            remote_path = self._resolve_ssh_mount_remote_path(
                connection_config,
                source_path,
            )
            command = f"mkdir -p -- {shlex.quote(remote_path)}"
            result = await asyncio.to_thread(execute_ssh_command, ssh_config, command)
            if not result.success:
                error_msg = (
                    result.stderr
                    or result.stdout
                    or "Failed to create source directory"
                )
                raise HTTPException(status_code=400, detail=error_msg)
            return

        raise HTTPException(
            status_code=400,
            detail=f"Creating source directories is not supported for mount backend '{mount_backend}'",
        )

    async def _create_workspace_mount_target_directory(
        self,
        workspace_id: str,
        target_path: str,
    ) -> None:
        if not target_path.startswith("/workspace/"):
            raise HTTPException(
                status_code=400,
                detail="Mount target directories can only be created under /workspace",
            )

        relative_path = self._normalize_workspace_relative_path(
            target_path[len("/workspace/") :]
        )
        if not relative_path:
            raise HTTPException(
                status_code=400, detail="Invalid mount target directory"
            )

        self._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        target_dir_path = self._resolve_workspace_file_path(workspace_id, relative_path)

        def _mkdir() -> None:
            target_dir_path.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(_mkdir)

    async def _get_mount_source_record(
        self,
        db: Any,
        mount_source_id: str,
        *,
        require_enabled: bool = False,
    ) -> Any:
        record = await db.userspacemountsource.find_unique(
            where={"id": mount_source_id},
            include={"toolConfig": True},
        )
        if not record:
            raise HTTPException(status_code=404, detail="Mount source not found")
        if require_enabled and not bool(getattr(record, "enabled", False)):
            raise HTTPException(status_code=400, detail="Mount source is disabled")
        return record

    async def list_userspace_mount_sources(self) -> list[UserspaceMountSource]:
        db = await get_db()
        rows = await db.userspacemountsource.find_many(
            order={"name": "asc"},
            include={"toolConfig": True},
        )
        # Count workspace mounts per source in a single query
        count_rows: list[dict[str, Any]] = await db.query_raw(
            "SELECT mount_source_id, COUNT(*)::int AS cnt FROM workspace_mounts GROUP BY mount_source_id"
        )
        counts: dict[str, int] = {
            str(r.get("mount_source_id", "")): int(r.get("cnt", 0)) for r in count_rows
        }
        return [
            self._userspace_mount_source_from_record(
                row, usage_count=counts.get(str(getattr(row, "id", "")), 0)
            )
            for row in rows
        ]

    async def create_userspace_mount_source(
        self,
        request: CreateUserspaceMountSourceRequest,
    ) -> UserspaceMountSource:
        db = await get_db()
        tool_config_id: str | None = None

        # Resolve source_type and connection_config from backing tool when provided
        if request.tool_config_id:
            tool_record = await db.toolconfig.find_unique(
                where={"id": request.tool_config_id}
            )
            if not tool_record:
                raise HTTPException(status_code=400, detail="Backing tool not found")
            tool_type = str(getattr(tool_record, "toolType", ""))
            if tool_type not in ("ssh_shell", "filesystem_indexer"):
                raise HTTPException(
                    status_code=400,
                    detail="Only SSH and filesystem tools can back mount sources",
                )
            source_type: UserspaceMountSourceType = (
                "ssh" if tool_type == "ssh_shell" else "filesystem"
            )
            tool_connection = self._load_tool_connection_config(tool_record)
            tool_config_id = request.tool_config_id
        else:
            if not request.source_type:
                raise HTTPException(
                    status_code=400,
                    detail="source_type is required when tool_config_id is not provided",
                )
            source_type = request.source_type
            tool_connection = request.connection_config

        connection_config, approved_paths, _mount_backend = (
            self._normalize_mount_source_payload(
                source_type=source_type,
                connection_config=tool_connection,
                approved_paths=request.approved_paths,
            )
        )
        now = _utc_now()
        data: dict[str, Any] = {
            "id": str(uuid4()),
            "name": self._normalize_mount_source_name(request.name),
            "description": self._normalize_mount_source_description(
                request.description
            ),
            "enabled": bool(request.enabled),
            "sourceType": source_type,
            "connectionConfig": Json(
                encrypt_json_passwords(
                    connection_config,
                    CONNECTION_CONFIG_PASSWORD_FIELDS,
                )
            ),
            "approvedPaths": Json(approved_paths),
            "syncIntervalSeconds": (
                request.sync_interval_seconds
                if request.sync_interval_seconds is not None
                else 30
            ),
            "createdAt": now,
            "updatedAt": now,
        }
        if tool_config_id:
            data["toolConfigId"] = tool_config_id
        created = await db.userspacemountsource.create(
            data=data,
            include={"toolConfig": True},
        )
        return self._userspace_mount_source_from_record(created)

    async def get_mount_source_affected_workspaces(
        self,
        mount_source_id: str,
    ) -> MountSourceAffectedWorkspacesResponse:
        db = await get_db()
        record = await self._get_mount_source_record(db, mount_source_id)
        source = self._userspace_mount_source_from_record(record)
        rows: list[dict[str, Any]] = await db.query_raw(
            "SELECT w.id AS workspace_id, w.name AS workspace_name,"
            " w.owner_user_id, COUNT(wm.id)::int AS mount_count"
            " FROM workspace_mounts wm"
            " JOIN workspaces w ON w.id = wm.workspace_id"
            " WHERE wm.mount_source_id = $1"
            " GROUP BY w.id, w.name, w.owner_user_id"
            " ORDER BY w.name",
            mount_source_id,
        )
        workspaces = [
            MountSourceAffectedWorkspace(
                workspace_id=str(r.get("workspace_id", "")),
                workspace_name=str(r.get("workspace_name", "")),
                owner_user_id=str(r.get("owner_user_id", "")),
                mount_count=int(r.get("mount_count", 0)),
            )
            for r in rows
        ]
        return MountSourceAffectedWorkspacesResponse(
            mount_source_id=source.id,
            mount_source_name=source.name,
            source_type=source.source_type,
            total_mounts=sum(w.mount_count for w in workspaces),
            workspaces=workspaces,
        )

    async def update_userspace_mount_source(
        self,
        mount_source_id: str,
        request: UpdateUserspaceMountSourceRequest,
    ) -> UserspaceMountSource:
        db = await get_db()
        existing_record = await self._get_mount_source_record(db, mount_source_id)
        existing = self._userspace_mount_source_from_record(existing_record)
        fields_set = request.model_fields_set

        next_name = existing.name
        if "name" in fields_set:
            next_name = self._normalize_mount_source_name(request.name or "")

        next_description = existing.description
        if "description" in fields_set:
            next_description = self._normalize_mount_source_description(
                request.description
            )

        next_enabled = (
            existing.enabled if "enabled" not in fields_set else bool(request.enabled)
        )
        next_connection_config = (
            existing.connection_config
            if "connection_config" not in fields_set
            else dict(request.connection_config or {})
        )
        next_approved_paths = (
            existing.approved_paths
            if "approved_paths" not in fields_set
            else list(request.approved_paths or [])
        )
        next_sync_interval = (
            existing.sync_interval_seconds
            if "sync_interval_seconds" not in fields_set
            else request.sync_interval_seconds
        )

        normalized_connection_config, normalized_approved_paths, _mount_backend = (
            self._normalize_mount_source_payload(
                source_type=existing.source_type,
                connection_config=next_connection_config,
                approved_paths=next_approved_paths,
            )
        )
        update_data: dict[str, Any] = {
            "name": next_name,
            "description": next_description,
            "enabled": next_enabled,
            "connectionConfig": Json(
                encrypt_json_passwords(
                    normalized_connection_config,
                    CONNECTION_CONFIG_PASSWORD_FIELDS,
                )
            ),
            "approvedPaths": Json(normalized_approved_paths),
            "updatedAt": _utc_now(),
        }
        if next_sync_interval is not None:
            update_data["syncIntervalSeconds"] = max(
                1, min(2592000, next_sync_interval)
            )
        updated = await db.userspacemountsource.update(
            where={"id": mount_source_id},
            data=update_data,
            include={"toolConfig": True},
        )

        # Cascade: when disabling a mount source, stop auto-sync on all
        # workspace mounts using this source so the watch loop no longer
        # schedules new syncs.
        if not next_enabled and existing.enabled:
            await db.workspacemount.update_many(
                where={"mountSourceId": mount_source_id, "autoSyncEnabled": True},
                data={"autoSyncEnabled": False, "updatedAt": _utc_now()},
            )
            # Evict any in-flight watch entries so the background loop
            # immediately stops tracking these mounts.
            affected_ids: list[dict[str, Any]] = await db.query_raw(
                "SELECT id FROM workspace_mounts WHERE mount_source_id = $1",
                mount_source_id,
            )
            for row in affected_ids:
                mid = str(row.get("id", ""))
                self._workspace_mount_watch_next_due_monotonic.pop(mid, None)
                self._workspace_mount_watch_inflight.discard(mid)

        return self._userspace_mount_source_from_record(updated)

    async def delete_userspace_mount_source(
        self,
        mount_source_id: str,
    ) -> DeleteUserspaceMountSourceResponse:
        db = await get_db()
        await self._get_mount_source_record(db, mount_source_id)
        mount_count = await db.workspacemount.count(
            where={"mountSourceId": mount_source_id}
        )
        if mount_count > 0:
            raise HTTPException(
                status_code=400,
                detail="Mount source is still attached to one or more workspaces",
            )
        await db.userspacemountsource.delete(where={"id": mount_source_id})
        return DeleteUserspaceMountSourceResponse(
            success=True,
            mount_source_id=mount_source_id,
        )

    async def browse_userspace_mount_source(
        self,
        mount_source_id: str,
        request: BrowseUserspaceMountSourceRequest,
    ) -> WorkspaceMountBrowseResponse:
        db = await get_db()
        mount_source_record = await self._get_mount_source_record(db, mount_source_id)
        mount_source = self._userspace_mount_source_from_record(mount_source_record)
        browser_path = self._normalize_mount_browser_path(request.path)
        source_path = self._browser_path_to_mount_source_path(browser_path)

        if mount_source.source_type == "ssh":
            return await self._browse_ssh_workspace_mount_source(
                connection_config=mount_source.connection_config,
                browser_path=browser_path,
                source_path=source_path,
            )

        return await self._browse_filesystem_workspace_mount_source(
            mount_source_id=mount_source.id,
            connection_config=mount_source.connection_config,
            browser_path=browser_path,
            source_path=source_path,
        )

    async def browse_tool_config(
        self,
        tool_config_id: str,
        request: BrowseUserspaceMountSourceRequest,
    ) -> WorkspaceMountBrowseResponse:
        """Browse a tool config's filesystem directly (before a mount source exists)."""
        db = await get_db()
        tool_record = await db.toolconfig.find_unique(where={"id": tool_config_id})
        if not tool_record:
            raise HTTPException(status_code=404, detail="Tool config not found")

        tool_type = str(getattr(tool_record, "toolType", ""))
        if tool_type not in ("ssh_shell", "filesystem_indexer"):
            raise HTTPException(
                status_code=400,
                detail="Only SSH and filesystem tools can be browsed",
            )

        connection_config = self._load_tool_connection_config(tool_record)
        source_type: UserspaceMountSourceType = (
            "ssh" if tool_type == "ssh_shell" else "filesystem"
        )

        browser_path = self._normalize_mount_browser_path(request.path)
        source_path = self._browser_path_to_mount_source_path(browser_path)

        if source_type == "ssh":
            return await self._browse_ssh_workspace_mount_source(
                connection_config=connection_config,
                browser_path=browser_path,
                source_path=source_path,
            )

        return await self._browse_filesystem_workspace_mount_source(
            mount_source_id=tool_config_id,
            connection_config=connection_config,
            browser_path=browser_path,
            source_path=source_path,
        )

    async def list_workspace_mounts(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[WorkspaceMount]:
        await self._enforce_workspace_access(workspace_id, user_id)
        db = await get_db()
        rows = await db.workspacemount.find_many(
            where={"workspaceId": workspace_id},
            order={"createdAt": "asc"},
            include={"mountSource": True},
        )
        results: list[WorkspaceMount] = []
        for row in rows:
            mount_source_record = getattr(row, "mountSource", None)
            mount_source = (
                self._userspace_mount_source_from_record(mount_source_record)
                if mount_source_record is not None
                else None
            )
            mount = self._workspace_mount_from_record(row, mount_source)
            mount.source_available = await self._check_mount_source_available(
                row, mount_source
            )
            results.append(mount)
        return results

    async def list_mountable_sources(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[MountableSource]:
        await self._enforce_workspace_access(workspace_id, user_id)
        db = await get_db()
        rows = await db.userspacemountsource.find_many(
            where={"enabled": True},
            order={"name": "asc"},
        )
        sources: list[MountableSource] = []
        for row in rows:
            mount_source = self._userspace_mount_source_from_record(row)
            for path in mount_source.approved_paths:
                sources.append(
                    MountableSource(
                        mount_source_id=mount_source.id,
                        source_name=mount_source.name,
                        source_type=mount_source.source_type,
                        mount_backend=mount_source.mount_backend,
                        source_path=path,
                    )
                )
        return sources

    async def create_workspace_mount(
        self,
        workspace_id: str,
        user_id: str,
        request: CreateWorkspaceMountRequest,
    ) -> WorkspaceMount:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        mount_source_record = await self._get_mount_source_record(
            db,
            request.mount_source_id,
            require_enabled=True,
        )
        mount_source = self._userspace_mount_source_from_record(mount_source_record)
        normalized_source_path = self._normalize_mount_source_path(request.source_path)
        self._ensure_mount_source_within_approved_paths(
            normalized_source_path,
            mount_source.approved_paths,
        )

        target_path = self._validate_mount_target_path(request.target_path)
        source_directory_to_create: str | None = None
        if request.source_directory_to_create:
            source_directory_to_create = self._normalize_mount_source_path(
                request.source_directory_to_create
            )
            self._ensure_mount_source_within_approved_paths(
                source_directory_to_create,
                mount_source.approved_paths,
            )
            if not self._is_mount_source_within_root(
                normalized_source_path,
                source_directory_to_create,
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Source directory to create must be the selected mount path or its parent",
                )

        target_directory_to_create: str | None = None
        if request.target_directory_to_create:
            target_directory_to_create = self._validate_mount_target_path(
                request.target_directory_to_create
            )
            if not target_directory_to_create.startswith("/workspace/"):
                raise HTTPException(
                    status_code=400,
                    detail="Target directory to create must be within /workspace",
                )
            if not self._is_same_or_descendant_mount_path(
                target_path,
                target_directory_to_create,
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Target directory to create must be the selected mount path or its parent",
                )

        count = await db.workspacemount.count(where={"workspaceId": workspace_id})
        if count >= self._WORKSPACE_MOUNT_MAX_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Workspace mount limit reached ({self._WORKSPACE_MOUNT_MAX_COUNT})",
            )

        if source_directory_to_create:
            await self._create_workspace_mount_source_directory(
                mount_source_id=mount_source.id,
                source_type=mount_source.source_type,
                mount_backend=mount_source.mount_backend,
                connection_config=mount_source.connection_config,
                source_path=source_directory_to_create,
            )
        if target_directory_to_create:
            await self._create_workspace_mount_target_directory(
                workspace_id,
                target_directory_to_create,
            )

        self._validate_workspace_mount_sync_configuration(
            source_type=mount_source.source_type,
            sync_mode=request.sync_mode,
            auto_sync_enabled=bool(request.auto_sync_enabled),
        )

        now = _utc_now()
        initial_sync_status = (
            "synced" if mount_source.source_type == "filesystem" else "pending"
        )
        initial_sync_backend: str | None = None
        initial_sync_notice: str | None = None
        if mount_source.source_type == "ssh":
            ssh_config = ssh_config_from_dict(mount_source.connection_config)
            initial_sync_backend, initial_sync_notice = (
                await self._resolve_ssh_sync_backend(
                    ssh_config,
                    probe_if_unknown=True,
                )
            )
        created = await db.workspacemount.create(
            data={
                "id": str(uuid4()),
                "workspaceId": workspace_id,
                "mountSourceId": mount_source.id,
                "sourcePath": normalized_source_path,
                "targetPath": target_path,
                "autoSyncEnabled": bool(request.auto_sync_enabled),
                "syncMode": request.sync_mode,
                "description": self._normalize_mount_description(request.description),
                "syncStatus": initial_sync_status,
                "syncBackend": initial_sync_backend,
                "syncNotice": initial_sync_notice,
                "createdAt": now,
                "updatedAt": now,
            }
        )
        await db.workspace.update(where={"id": workspace_id}, data={"updatedAt": now})
        return self._workspace_mount_from_record(created, mount_source)

    async def update_workspace_mount(
        self,
        workspace_id: str,
        user_id: str,
        mount_id: str,
        request: UpdateWorkspaceMountRequest,
    ) -> WorkspaceMount:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        existing = await db.workspacemount.find_first(
            where={"id": mount_id, "workspaceId": workspace_id},
            include={"mountSource": True},
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Mount not found")

        mount_source_record = getattr(existing, "mountSource", None)
        source_type = (
            str(getattr(mount_source_record, "sourceType", "") or "")
            if mount_source_record
            else ""
        )
        current_sync_mode = self._normalize_workspace_mount_sync_mode(
            getattr(existing, "syncMode", None),
            legacy_sync_deletes=bool(getattr(existing, "syncDeletes", False)),
        )
        next_sync_mode = request.sync_mode or current_sync_mode
        next_auto_sync_enabled = (
            bool(request.auto_sync_enabled)
            if request.auto_sync_enabled is not None
            else bool(getattr(existing, "autoSyncEnabled", False))
        )
        if request.enabled is not None and not request.enabled:
            next_auto_sync_enabled = False

        self._validate_workspace_mount_sync_configuration(
            source_type=source_type,
            sync_mode=next_sync_mode,
            auto_sync_enabled=next_auto_sync_enabled,
        )

        existing_has_destructive_auto_sync_approval = (
            self._has_destructive_auto_sync_approval(
                existing,
                next_sync_mode,
            )
        )
        needs_destructive_auto_sync_confirmation = (
            next_auto_sync_enabled
            and self._is_destructive_workspace_mount_sync_mode(next_sync_mode)
            and (
                not existing_has_destructive_auto_sync_approval
                or request.auto_sync_enabled is True
                or request.sync_mode is not None
                or request.target_path is not None
            )
        )

        update_data: dict[str, Any] = {"updatedAt": _utc_now()}
        if request.target_path is not None:
            update_data["targetPath"] = self._validate_mount_target_path(
                request.target_path
            )
        if request.description is not None:
            update_data["description"] = self._normalize_mount_description(
                request.description
            )
        if request.auto_sync_enabled is not None:
            update_data["autoSyncEnabled"] = next_auto_sync_enabled
        if request.sync_mode is not None:
            update_data["syncMode"] = request.sync_mode
        if request.enabled is not None:
            update_data["enabled"] = bool(request.enabled)
            if not request.enabled:
                update_data["autoSyncEnabled"] = False

        clear_destructive_auto_sync_confirmation = (
            not next_auto_sync_enabled
            or not self._is_destructive_workspace_mount_sync_mode(next_sync_mode)
            or request.target_path is not None
            or request.sync_mode is not None
            or (request.enabled is not None and not request.enabled)
        )

        if needs_destructive_auto_sync_confirmation:
            preview_token = request.destructive_auto_sync_preview_token
            if not preview_token:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Destructive auto-sync requires a fresh preview confirmation. "
                        "Run a dry run before enabling Auto."
                    ),
                )
            context = await self._build_workspace_mount_sync_context(
                existing,
                force_backend_recheck=True,
                sync_mode_override=next_sync_mode,
            )
            mount_lock = await self._get_workspace_mount_operation_lock(mount_id)
            async with mount_lock:
                async with self._workspace_mount_sync_semaphore:
                    await asyncio.to_thread(
                        self._stage_runtime_mount_into_sync_cache,
                        workspace_id,
                        context["target_path"],
                        context["cache_dir"],
                    )
                    await self._consume_workspace_mount_sync_preview(
                        mount_id=context["mount_id"],
                        workspace_id=context["workspace_id"],
                        ssh_config=context["ssh_config"],
                        remote_path=context["remote_path"],
                        cache_dir=context["cache_dir"],
                        sync_mode=context["sync_mode"],
                        preview_token=preview_token,
                    )
            update_data["destructiveAutoSyncConfirmedAt"] = update_data["updatedAt"]
            update_data["destructiveAutoSyncConfirmedMode"] = next_sync_mode
            clear_destructive_auto_sync_confirmation = False

        if clear_destructive_auto_sync_confirmation:
            update_data["destructiveAutoSyncConfirmedAt"] = None
            update_data["destructiveAutoSyncConfirmedMode"] = None

        await self._invalidate_workspace_mount_sync_preview(mount_id)

        updated = await db.workspacemount.update(
            where={"id": mount_id}, data=update_data
        )
        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": update_data["updatedAt"]},
        )

        target_path = str(getattr(existing, "targetPath", "") or "")

        # Filesystem side-effects when enabled state changes.
        if request.enabled is not None and target_path:
            if not request.enabled:
                if source_type == "ssh":
                    # SSH mounts are sync-mounts: retain files on disable so
                    # content is preserved for re-enable.  The 409 guard
                    # already prevents access while the mount is disabled.
                    logger.info(
                        "Disabled SSH mount %s; retaining synced files",
                        mount_id,
                    )
                else:
                    # Non-SSH (filesystem) mounts: clear materialized rootfs
                    # content.  Filesystem mounts re-materialize on next launch.
                    runtime_dir = self._resolve_workspace_mount_runtime_target_dir(
                        workspace_id,
                        target_path,
                    )
                    if runtime_dir is not None and runtime_dir.exists():
                        try:
                            shutil.rmtree(runtime_dir)
                            logger.info(
                                "Cleared rootfs mount target %s for disabled mount %s",
                                runtime_dir,
                                mount_id,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to clear rootfs mount target %s: %s",
                                runtime_dir,
                                exc,
                            )

                self.invalidate_file_list_cache(workspace_id)

            else:
                # Re-enabling: for SSH mounts, trigger a fresh sync so the
                # content reappears. Filesystem mounts will re-materialize on
                # next runtime launch automatically.
                if source_type == "ssh" and next_sync_mode == "merge":
                    try:
                        await self._sync_workspace_mount_record(db, existing)
                    except Exception as exc:
                        logger.warning(
                            "Failed to re-sync SSH mount %s on re-enable: %s",
                            mount_id,
                            exc,
                        )
                self.invalidate_file_list_cache(workspace_id)

        refreshed = await db.workspacemount.find_first(
            where={"id": mount_id, "workspaceId": workspace_id},
            include={"mountSource": True},
        )
        mount_source_record = (
            getattr(refreshed, "mountSource", None)
            if refreshed
            else getattr(existing, "mountSource", None)
        )
        mount_source = (
            self._userspace_mount_source_from_record(mount_source_record)
            if mount_source_record is not None
            else None
        )
        return self._workspace_mount_from_record(refreshed or updated, mount_source)

    async def delete_workspace_mount(
        self,
        workspace_id: str,
        user_id: str,
        mount_id: str,
    ) -> DeleteWorkspaceMountResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        existing = await db.workspacemount.find_first(
            where={"id": mount_id, "workspaceId": workspace_id},
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Mount not found")

        now = _utc_now()
        target_path = str(getattr(existing, "targetPath", "") or "")
        await db.workspacemount.delete(where={"id": mount_id})
        await db.workspace.update(where={"id": workspace_id}, data={"updatedAt": now})

        # Clean up materialized content and sync cache.
        if target_path:
            runtime_dir = self._resolve_workspace_mount_runtime_target_dir(
                workspace_id,
                target_path,
            )
            if runtime_dir is not None and runtime_dir.exists():
                try:
                    shutil.rmtree(runtime_dir)
                except Exception as exc:
                    logger.warning(
                        "Failed to clear rootfs on mount delete %s: %s",
                        mount_id,
                        exc,
                    )
        ssh_cache_dir = self._base_dir / "mount_cache" / workspace_id / mount_id
        if ssh_cache_dir.exists():
            try:
                shutil.rmtree(ssh_cache_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to clear SSH mount cache on delete %s: %s",
                    mount_id,
                    exc,
                )
        await self._invalidate_workspace_mount_sync_preview(mount_id)
        self.invalidate_file_list_cache(workspace_id)

        return DeleteWorkspaceMountResponse(success=True, mount_id=mount_id)

    async def preview_workspace_mount_sync(
        self,
        workspace_id: str,
        user_id: str,
        mount_id: str,
        request: WorkspaceMountSyncPreviewRequest | None = None,
    ) -> WorkspaceMountSyncPreviewResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        mount = await db.workspacemount.find_first(
            where={"id": mount_id, "workspaceId": workspace_id},
            include={"mountSource": True},
        )
        if not mount:
            raise HTTPException(status_code=404, detail="Mount not found")
        return await self._preview_workspace_mount_record(
            db,
            mount,
            force_backend_recheck=True,
            sync_mode_override=request.sync_mode if request else None,
        )

    async def sync_workspace_mount(
        self,
        workspace_id: str,
        user_id: str,
        mount_id: str,
        request: WorkspaceMountSyncRequest,
    ) -> WorkspaceMountSyncResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        mount = await db.workspacemount.find_first(
            where={"id": mount_id, "workspaceId": workspace_id},
            include={"mountSource": True},
        )
        if not mount:
            raise HTTPException(status_code=404, detail="Mount not found")
        # Manual sync requests should re-check previously-missing rsync endpoints
        # immediately so UI state can recover without waiting for cooldown expiry.
        return await self._sync_workspace_mount_record(
            db,
            mount,
            force_backend_recheck=True,
            preview_token=request.preview_token,
        )

    async def browse_workspace_mount_source(
        self,
        workspace_id: str,
        user_id: str,
        request: WorkspaceMountBrowseRequest,
    ) -> WorkspaceMountBrowseResponse:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        db = await get_db()
        mount_source_record = await self._get_mount_source_record(
            db,
            request.mount_source_id,
            require_enabled=True,
        )
        mount_source = self._userspace_mount_source_from_record(mount_source_record)

        root_source_path = self._normalize_mount_source_path(request.root_source_path)
        if root_source_path not in mount_source.approved_paths:
            raise HTTPException(
                status_code=400,
                detail="Browse root is not one of the mount source's approved paths",
            )

        browser_path = self._normalize_mount_browser_path(request.path)
        source_path = self._browser_path_to_mount_source_path(browser_path)
        if not self._is_mount_source_within_root(source_path, root_source_path):
            raise HTTPException(
                status_code=400,
                detail="Browse path must stay within the approved mount root",
            )

        if mount_source.source_type == "ssh":
            return await self._browse_ssh_workspace_mount_source(
                connection_config=mount_source.connection_config,
                browser_path=browser_path,
                source_path=source_path,
            )

        return await self._browse_filesystem_workspace_mount_source(
            mount_source_id=mount_source.id,
            connection_config=mount_source.connection_config,
            browser_path=browser_path,
            source_path=source_path,
        )

    async def _browse_ssh_workspace_mount_source(
        self,
        *,
        connection_config: dict[str, Any],
        browser_path: str,
        source_path: str,
    ) -> WorkspaceMountBrowseResponse:
        ssh_config = ssh_config_from_dict(connection_config)
        remote_path = self._resolve_ssh_mount_remote_path(
            connection_config, source_path
        )
        command = f"ls -p1a {shlex.quote(remote_path)}"

        try:
            result = await asyncio.to_thread(execute_ssh_command, ssh_config, command)
        except Exception as exc:
            return WorkspaceMountBrowseResponse(
                path=browser_path, entries=[], error=str(exc)
            )

        if not result.success:
            error_msg = result.stderr or result.stdout or "Failed to list directory"
            return WorkspaceMountBrowseResponse(
                path=browser_path,
                entries=[],
                error=error_msg,
            )

        entries: list[WorkspaceMountDirectoryEntry] = []
        for line in result.stdout.splitlines():
            name = line.strip()
            if name in {".", "./", "..", "../"} or not name:
                continue
            if not name.endswith("/"):
                continue
            clean_name = name.rstrip("/")
            child_browser_path = (
                f"/{clean_name}"
                if browser_path == "/"
                else f"{browser_path.rstrip('/')}/{clean_name}"
            )
            entries.append(
                WorkspaceMountDirectoryEntry(
                    name=clean_name,
                    path=child_browser_path,
                    is_dir=True,
                    size=None,
                )
            )

        entries.sort(key=lambda entry: entry.name.lower())
        return WorkspaceMountBrowseResponse(path=browser_path, entries=entries)

    async def _browse_filesystem_workspace_mount_source(
        self,
        *,
        mount_source_id: str,
        connection_config: dict[str, Any],
        browser_path: str,
        source_path: str,
    ) -> WorkspaceMountBrowseResponse:
        try:
            resolved_path = Path(
                await self._resolve_filesystem_mount_source_local_path(
                    mount_source_id=mount_source_id,
                    connection_config=connection_config,
                    source_path=source_path,
                )
            )
        except Exception as exc:
            return WorkspaceMountBrowseResponse(
                path=browser_path, entries=[], error=str(exc)
            )

        def _browse() -> WorkspaceMountBrowseResponse:
            if not resolved_path.exists():
                return WorkspaceMountBrowseResponse(
                    path=browser_path,
                    entries=[],
                    error=f"Path does not exist: {browser_path}",
                )
            if not resolved_path.is_dir():
                return WorkspaceMountBrowseResponse(
                    path=browser_path,
                    entries=[],
                    error=f"Not a directory: {browser_path}",
                )

            try:
                entries: list[WorkspaceMountDirectoryEntry] = []
                for entry in sorted(
                    resolved_path.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                ):
                    try:
                        if not entry.is_dir():
                            continue
                    except (PermissionError, OSError):
                        continue
                    child_browser_path = (
                        f"/{entry.name}"
                        if browser_path == "/"
                        else f"{browser_path.rstrip('/')}/{entry.name}"
                    )
                    entries.append(
                        WorkspaceMountDirectoryEntry(
                            name=entry.name,
                            path=child_browser_path,
                            is_dir=True,
                            size=None,
                        )
                    )
                return WorkspaceMountBrowseResponse(path=browser_path, entries=entries)
            except PermissionError:
                return WorkspaceMountBrowseResponse(
                    path=browser_path,
                    entries=[],
                    error=f"Permission denied: {browser_path}",
                )
            except Exception as exc:
                return WorkspaceMountBrowseResponse(
                    path=browser_path,
                    entries=[],
                    error=str(exc),
                )

        return await asyncio.to_thread(_browse)

    async def resolve_workspace_mounts_for_runtime(
        self,
        workspace_id: str,
        mount_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve persisted mounts into runtime-ready specs.

        Returns a list of dicts with keys: source_local_path, target_path,
        source_type, mount_backend, read_only. For SSH mounts the
        source_local_path points to the sync cache. For filesystem mounts it
        points to the resolved local path for the configured backend.
        """
        db = await get_db()
        rows = await db.workspacemount.find_many(
            where={"workspaceId": workspace_id},
            include={"mountSource": True},
        )
        if not rows:
            return []

        specs: list[dict[str, Any]] = []
        mount_id_filter = (
            {str(mount_id).strip() for mount_id in mount_ids if str(mount_id).strip()}
            if mount_ids is not None
            else None
        )
        for row in rows:
            mount_id = str(getattr(row, "id", "") or "")
            if mount_id_filter is not None and mount_id not in mount_id_filter:
                continue
            mount_source_record = getattr(row, "mountSource", None)
            if not mount_source_record:
                continue
            mount_source = self._userspace_mount_source_from_record(mount_source_record)
            if not mount_source.enabled:
                continue
            if not bool(getattr(row, "enabled", True)):
                continue
            source_path = str(getattr(row, "sourcePath", "") or "")
            target_path = str(getattr(row, "targetPath", "") or "")

            if mount_source.source_type == "ssh":
                local_path = str(
                    self._base_dir / "mount_cache" / workspace_id / mount_id
                )
                specs.append(
                    {
                        "source_local_path": local_path,
                        "target_path": target_path,
                        "source_type": mount_source.source_type,
                        "mount_backend": mount_source.mount_backend,
                        "read_only": True,
                    }
                )
            elif mount_source.source_type == "filesystem":
                try:
                    resolved = await self._resolve_filesystem_mount_source_local_path(
                        mount_source_id=mount_source.id,
                        connection_config=mount_source.connection_config,
                        source_path=source_path,
                    )
                except Exception:
                    logger.warning(
                        "Skipping invalid filesystem userspace mount %s for mount source %s",
                        mount_id,
                        mount_source.id,
                    )
                    continue
                specs.append(
                    {
                        "source_local_path": resolved,
                        "target_path": target_path,
                        "source_type": mount_source.source_type,
                        "mount_backend": mount_source.mount_backend,
                        "read_only": True,
                    }
                )

        return specs

    async def stage_runtime_mounts_into_sync_cache(
        self,
        workspace_id: str,
        mount_ids: list[str] | None = None,
    ) -> bool:
        db = await get_db()
        rows = await db.workspacemount.find_many(
            where={"workspaceId": workspace_id},
            include={"mountSource": True},
        )
        if not rows:
            return False

        mount_id_filter = (
            {str(mount_id).strip() for mount_id in mount_ids if str(mount_id).strip()}
            if mount_ids is not None
            else None
        )
        stage_specs: list[tuple[str, Path]] = []

        for row in rows:
            mount_id = str(getattr(row, "id", "") or "")
            if mount_id_filter is not None and mount_id not in mount_id_filter:
                continue
            if not bool(getattr(row, "enabled", True)):
                continue

            mount_source_record = getattr(row, "mountSource", None)
            if not mount_source_record:
                continue
            if str(getattr(mount_source_record, "sourceType", "") or "") != "ssh":
                continue

            target_path = str(getattr(row, "targetPath", "") or "").strip()
            if not target_path:
                continue

            stage_specs.append(
                (
                    target_path,
                    self._base_dir / "mount_cache" / workspace_id / mount_id,
                )
            )

        if not stage_specs:
            return False

        staged_any = await asyncio.to_thread(
            self._stage_runtime_mounts_into_sync_cache_sync,
            workspace_id,
            stage_specs,
        )
        if staged_any:
            self.invalidate_file_list_cache(workspace_id)
        return staged_any

    def invalidate_file_list_cache(self, workspace_id: str) -> None:
        self._file_list_cache.pop(workspace_id, None)

    def _resolve_workspace_mount_runtime_target_dir(
        self,
        workspace_id: str,
        target_path: str,
    ) -> Path | None:
        normalized = (target_path or "").strip().replace("\\", "/").lstrip("/")
        if not normalized:
            return None
        rootfs_dir = self._workspace_rootfs_dir(workspace_id)
        candidate = rootfs_dir / normalized
        try:
            candidate.relative_to(rootfs_dir)
        except ValueError:
            return None
        return candidate

    def _runtime_mount_content_signature_sync(
        self,
        workspace_id: str,
        target_paths: list[str],
    ) -> tuple[Any, ...]:
        signatures: list[tuple[Any, ...]] = []
        for target_path in sorted(
            {path.strip() for path in target_paths if path.strip()}
        ):
            target_dir = self._resolve_workspace_mount_runtime_target_dir(
                workspace_id,
                target_path,
            )
            if target_dir is None or not target_dir.exists():
                signatures.append((target_path, False, 0, 0, 0, ""))
                continue

            digest = hashlib.sha1()
            entry_count = 0
            total_size = 0
            max_mtime_ns = 0

            def _add_entry(path: Path, relative: str) -> None:
                nonlocal entry_count, total_size, max_mtime_ns
                try:
                    stat = path.stat()
                except OSError:
                    return
                is_file = path.is_file()
                is_dir = path.is_dir()
                if not is_file and not is_dir:
                    return
                entry_count += 1
                size_bytes = stat.st_size if is_file else 0
                total_size += size_bytes
                max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
                entry_kind = "d" if is_dir else "f"
                digest.update(relative.encode("utf-8", errors="ignore"))
                digest.update(b"\0")
                digest.update(entry_kind.encode("ascii"))
                digest.update(b"\0")
                digest.update(str(size_bytes).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
                digest.update(b"\0")

            _add_entry(target_dir, ".")
            if target_dir.is_dir():
                for root, dirnames, filenames in os.walk(target_dir):
                    dirnames.sort()
                    filenames.sort()
                    root_path = Path(root)
                    for dirname in dirnames:
                        child = root_path / dirname
                        relative = str(child.relative_to(target_dir))
                        _add_entry(child, relative)
                    for filename in filenames:
                        child = root_path / filename
                        relative = str(child.relative_to(target_dir))
                        _add_entry(child, relative)

            signatures.append(
                (
                    target_path,
                    True,
                    entry_count,
                    total_size,
                    max_mtime_ns,
                    digest.hexdigest(),
                )
            )

        return tuple(signatures)

    async def get_runtime_mount_content_signature(
        self,
        workspace_id: str,
        target_paths: list[str],
    ) -> tuple[Any, ...]:
        return await asyncio.to_thread(
            self._runtime_mount_content_signature_sync,
            workspace_id,
            target_paths,
        )

    async def list_workspace_changed_file_paths(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[str]:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        async with self._git_status_semaphore:
            status_result = await self._run_git(
                workspace_id,
                ["status", "--porcelain=1", "-z", "--untracked-files=all"],
                check=False,
            )
        if status_result.returncode != 0:
            stderr = (status_result.stderr or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compute workspace changedFile paths: {stderr or 'git status failed'}",
            )

        return self._parse_git_status_changed_file_paths(status_result.stdout)

    async def list_workspace_changed_file_acknowledgements(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[str]:
        await self._enforce_workspace_access(workspace_id, user_id)
        db = await get_db()
        rows = await db.userspacechangedfileacknowledgement.find_many(
            where={
                "workspaceId": workspace_id,
                "userId": user_id,
            }
        )

        paths = {
            self._normalize_workspace_relative_path(str(getattr(row, "path", "") or ""))
            for row in rows
        }
        return sorted(path for path in paths if path)

    async def _garbage_collect_workspace_changed_file_acknowledgements(
        self,
        workspace_id: str,
        user_id: str,
    ) -> None:
        db = await get_db()
        total_rows = await db.userspacechangedfileacknowledgement.count(
            where={
                "workspaceId": workspace_id,
                "userId": user_id,
            }
        )
        if total_rows <= _CHANGED_FILE_ACK_MAX_ROWS_PER_WORKSPACE_USER:
            return

        trim_to_rows = int(_CHANGED_FILE_ACK_MAX_ROWS_PER_WORKSPACE_USER * 0.75)
        stale_rows_count = total_rows - trim_to_rows
        if stale_rows_count <= 0:
            return

        stale_rows = await db.userspacechangedfileacknowledgement.find_many(
            where={
                "workspaceId": workspace_id,
                "userId": user_id,
            },
            order={"updatedAt": "desc"},
            skip=trim_to_rows,
            take=stale_rows_count,
        )
        stale_ids = [
            str(getattr(row, "id", "") or "")
            for row in stale_rows
            if getattr(row, "id", None)
        ]
        if not stale_ids:
            return

        await db.userspacechangedfileacknowledgement.delete_many(
            where={"id": {"in": stale_ids}}
        )

    async def acknowledge_workspace_changed_file_path(
        self,
        workspace_id: str,
        user_id: str,
        relative_path: str,
    ) -> list[str]:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        normalized = self._normalize_workspace_relative_path(relative_path)
        if not normalized or self._is_reserved_internal_path(normalized):
            raise HTTPException(status_code=400, detail="Invalid file path")

        db = await get_db()
        await db.userspacechangedfileacknowledgement.delete_many(
            where={
                "workspaceId": workspace_id,
                "userId": user_id,
                "path": normalized,
            }
        )
        await db.userspacechangedfileacknowledgement.create(
            data={
                "workspaceId": workspace_id,
                "userId": user_id,
                "path": normalized,
            }
        )
        # Keep newest acknowledgement rows and trim older ones in-band on writes.
        await self._garbage_collect_workspace_changed_file_acknowledgements(
            workspace_id,
            user_id,
        )
        return await self.list_workspace_changed_file_acknowledgements(
            workspace_id, user_id
        )

    async def clear_workspace_changed_file_acknowledgements(
        self,
        workspace_id: str,
        user_id: str,
        relative_path: str | None = None,
    ) -> list[str]:
        await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )

        where: dict[str, Any] = {
            "workspaceId": workspace_id,
            "userId": user_id,
        }
        if relative_path is not None:
            normalized = self._normalize_workspace_relative_path(relative_path)
            if not normalized:
                raise HTTPException(status_code=400, detail="Invalid file path")
            where["path"] = normalized

        db = await get_db()
        await db.userspacechangedfileacknowledgement.delete_many(where=where)
        return await self.list_workspace_changed_file_acknowledgements(
            workspace_id, user_id
        )

    async def clear_workspace_changed_file_acknowledgements_for_all_users(
        self,
        workspace_id: str,
    ) -> None:
        db = await get_db()
        await db.userspacechangedfileacknowledgement.delete_many(
            where={"workspaceId": workspace_id}
        )

    async def clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
        self,
        workspace_id: str,
        relative_paths: list[str],
    ) -> None:
        normalized_paths_set: set[str] = set()
        for path in relative_paths:
            normalized = self._normalize_workspace_relative_path(path)
            if normalized:
                normalized_paths_set.add(normalized)
        normalized_paths = sorted(normalized_paths_set)
        if not normalized_paths:
            return

        db = await get_db()
        await db.userspacechangedfileacknowledgement.delete_many(
            where={
                "workspaceId": workspace_id,
                "path": {"in": normalized_paths},
            }
        )

    async def list_workspace_files(
        self,
        workspace_id: str,
        user_id: str,
        include_dirs: bool = False,
    ) -> list[UserSpaceFileInfo]:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)
        files_dir = self._workspace_files_dir(workspace_id)
        if not files_dir.exists():
            return []

        cached = self._file_list_cache.get(workspace_id)
        if cached is not None:
            cached_result, cached_include_dirs, cached_ts = cached
            if (
                cached_include_dirs == include_dirs
                and (_time.monotonic() - cached_ts) < _FILE_LIST_CACHE_TTL_SECONDS
            ):
                return cached_result

        base_result = await asyncio.to_thread(
            self._list_workspace_files_sync, files_dir, include_dirs
        )

        # Collect repo-relative prefixes for disabled mounts so their files
        # are hidden from the file tree while the mount is unmounted.
        disabled_prefixes = await self._list_disabled_workspace_mount_target_repo_paths(
            workspace_id
        )

        if disabled_prefixes:
            base_result = [
                f
                for f in base_result
                if not any(
                    self._workspace_path_matches_mount_prefix(f.path, prefix)
                    for prefix in disabled_prefixes
                )
            ]
            # Inject synthetic directory entries for disabled mounts so the
            # tree UI still shows mount-point folders (greyed out / UNMOUNTED).
            existing_paths = {f.path for f in base_result}
            for prefix in disabled_prefixes:
                if prefix not in existing_paths:
                    base_result.append(
                        UserSpaceFileInfo(
                            path=prefix,
                            size_bytes=0,
                            updated_at=_utc_now(),
                            entry_type="directory",
                        )
                    )
                    existing_paths.add(prefix)

        mount_specs = await self.resolve_workspace_mounts_for_runtime(workspace_id)
        if mount_specs:
            mount_prefixes = self._deduplicate_ancestor_paths(
                [
                    repo_rel
                    for spec in mount_specs
                    if (
                        repo_rel := self._workspace_mount_target_repo_relative_path(
                            str(spec.get("target_path", "") or "")
                        )
                    )
                ]
            )
            if mount_prefixes:
                base_result = [
                    entry
                    for entry in base_result
                    if not any(
                        self._workspace_path_matches_mount_prefix(
                            entry.path,
                            prefix,
                        )
                        for prefix in mount_prefixes
                    )
                ]

            # Augment with files from mount source directories so they appear in
            # the file tree even though they live outside the workspace files dir.
            existing_paths = {f.path for f in base_result}
            mount_entries = await asyncio.to_thread(
                self._list_mount_source_files_sync,
                workspace_id,
                mount_specs,
                include_dirs,
            )
            for entry in mount_entries:
                if entry.path not in existing_paths:
                    base_result.append(entry)
                    existing_paths.add(entry.path)
            base_result.sort(key=lambda item: item.path)

        self._file_list_cache[workspace_id] = (
            base_result,
            include_dirs,
            _time.monotonic(),
        )
        return base_result

    async def upsert_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        request: UpsertWorkspaceFileRequest,
        user_id: str,
        skip_live_data_enforcement: bool = False,
    ) -> UserSpaceFileResponse:
        workspace = await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role="editor",
        )
        await self._ensure_workspace_git_repo(workspace_id)
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if self._is_reserved_internal_path(normalized_path):
            raise HTTPException(status_code=400, detail="Invalid file path")
        normalized_path = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_path,
        )

        if _requires_entrypoint_wiring(normalized_path, request.artifact_type):
            main_path = self._resolve_workspace_file_path(
                workspace_id,
                _USERSPACE_PREVIEW_ENTRY_PATH,
            )
            if not main_path.exists() or not main_path.is_file():
                main_path.parent.mkdir(parents=True, exist_ok=True)
                main_path.write_text(
                    self._build_dashboard_entrypoint_content(normalized_path),
                    encoding="utf-8",
                )

            try:
                main_content = main_path.read_text(encoding="utf-8")
            except OSError:
                main_content = ""

            candidates = _entrypoint_module_specifier_candidates(normalized_path)
            if not _entrypoint_references_module(main_content, candidates):
                main_path.write_text(
                    self._append_dashboard_entrypoint_reference(
                        main_content,
                        normalized_path,
                    ),
                    encoding="utf-8",
                )

        parsed_live_data_connections = request.live_data_connections or []
        parsed_live_data_checks = request.live_data_checks or []
        workspace_has_tools = bool(workspace.selected_tool_ids)
        _sqlite_include = (
            getattr(workspace, "sqlite_persistence_mode", "exclude") == "include"
        )
        _sqlite_suffix = (
            " Note: this workspace has SQLite local persistence enabled -- "
            "live data wiring is still required for dashboard datasets; "
            "use .ragtime/db/app.sqlite3 with numbered migrations for local app state."
            if _sqlite_include
            else ""
        )
        if not skip_live_data_enforcement and _requires_live_data_contract(
            normalized_path,
            request.live_data_requested,
            workspace_has_tools=workspace_has_tools,
        ):
            if not parsed_live_data_connections:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing required live_data_connections contract metadata. "
                        "For live-data-requested module source writes in dashboard/* or with artifact_type=module_ts, "
                        "provide at least one connection with component_kind=tool_config, "
                        "component_id, and request. Set live_data_requested=false for scaffolding without live wiring."
                        + _sqlite_suffix
                    ),
                )
            if not parsed_live_data_checks:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing required live_data_checks verification metadata. "
                        "For live-data-requested module source writes, include checks proving successful "
                        "connection and transformation for each live_data_connections component_id."
                        + _sqlite_suffix
                    ),
                )

        if parsed_live_data_connections:
            allowed_component_ids = set(workspace.selected_tool_ids)
            for connection in parsed_live_data_connections:
                if connection.component_id not in allowed_component_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Invalid live_data_connections component_id: "
                            f"{connection.component_id}. It must match a tool selected "
                            "for this workspace."
                        ),
                    )

        if parsed_live_data_checks:
            allowed_component_ids = set(workspace.selected_tool_ids)
            for check in parsed_live_data_checks:
                if check.component_id not in allowed_component_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Invalid live_data_checks component_id: "
                            f"{check.component_id}. It must match a tool selected "
                            "for this workspace."
                        ),
                    )
                if not skip_live_data_enforcement and (
                    not check.connection_check_passed
                    or not check.transformation_check_passed
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "live_data_checks must indicate successful connection and transformation "
                            f"for component_id={check.component_id}."
                        ),
                    )

        if not skip_live_data_enforcement and _requires_live_data_contract(
            normalized_path,
            request.live_data_requested,
            workspace_has_tools=workspace_has_tools,
        ):
            connected_ids = {
                connection.component_id for connection in parsed_live_data_connections
            }
            verified_ids = {
                check.component_id
                for check in parsed_live_data_checks
                if check.connection_check_passed and check.transformation_check_passed
            }
            missing_verified_ids = sorted(connected_ids - verified_ids)
            if missing_verified_ids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing successful live_data_checks verification for component_id(s): "
                        + ", ".join(missing_verified_ids)
                    ),
                )

            # Verify server-side execution proofs for declared connections.
            unproven_ids = self.verify_execution_proofs(workspace_id, connected_ids)
            if unproven_ids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No server-verified execution proof for component_id(s): "
                        + ", ".join(unproven_ids)
                        + ". Execute a successful query via the workspace tool or "
                        "execute-component endpoint before persisting live data connections."
                        + _sqlite_suffix
                    ),
                )

        file_path = await self._resolve_workspace_tree_file_path(
            workspace_id,
            normalized_path,
        )
        stat = await asyncio.to_thread(
            self._write_workspace_file_sync,
            file_path,
            request.content,
            request.artifact_type,
            request.live_data_connections,
            request.live_data_checks,
        )
        await self.clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
            workspace_id,
            [normalized_path],
        )
        await self._touch_workspace(workspace_id)

        # Invalidate entrypoint status cache when the entrypoint config is written
        normalized = normalized_path.strip("/")
        if normalized == ".ragtime/runtime-entrypoint.json":
            self.invalidate_entrypoint_cache(workspace_id)

        return UserSpaceFileResponse(
            path=normalized_path,
            content=request.content,
            artifact_type=request.artifact_type,
            live_data_connections=request.live_data_connections,
            live_data_checks=request.live_data_checks,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    async def get_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        user_id: str,
        decode_errors: Literal["strict", "replace"] = "strict",
    ) -> UserSpaceFileResponse:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if self._is_reserved_internal_path(normalized_path):
            raise HTTPException(status_code=404, detail="File not found")
        normalized_path = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_path,
        )

        file_path = await self._resolve_workspace_tree_file_path(
            workspace_id,
            normalized_path,
        )
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            artifact_type, live_data_connections, live_data_checks, content, stat = (
                await asyncio.to_thread(
                    self._read_workspace_file_sync,
                    file_path,
                    decode_errors,
                )
            )
        except _NonUtf8WorkspaceFileError as exc:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Workspace file is not UTF-8 text and cannot be opened in the text editor. "
                    f"Path: {normalized_path}"
                ),
            ) from exc

        return UserSpaceFileResponse(
            path=normalized_path,
            content=content,
            artifact_type=artifact_type,
            live_data_connections=live_data_connections,
            live_data_checks=live_data_checks,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    async def delete_workspace_file(
        self, workspace_id: str, relative_path: str, user_id: str
    ) -> None:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_workspace_git_repo(workspace_id)
        normalized_path = self._normalize_workspace_relative_path(relative_path)
        if self._is_reserved_internal_path(normalized_path):
            raise HTTPException(status_code=400, detail="Invalid file path")
        normalized_path = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_path,
        )

        file_path = await self._resolve_workspace_tree_file_path(
            workspace_id,
            normalized_path,
        )
        await asyncio.to_thread(self._delete_workspace_file_sync, file_path)

        await self.clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
            workspace_id,
            [normalized_path],
        )

        await self._touch_workspace(workspace_id)

        # Invalidate entrypoint status cache when the entrypoint config is deleted
        normalized = normalized_path.strip("/")
        if normalized == ".ragtime/runtime-entrypoint.json":
            self.invalidate_entrypoint_cache(workspace_id)

    async def move_workspace_file(
        self,
        workspace_id: str,
        old_relative_path: str,
        new_relative_path: str,
        user_id: str,
    ) -> dict[str, str]:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_workspace_git_repo(workspace_id)

        normalized_old = (
            (old_relative_path or "").strip().replace("\\", "/").lstrip("/")
        )
        normalized_new = (
            (new_relative_path or "").strip().replace("\\", "/").lstrip("/")
        )
        if not normalized_old or not normalized_new:
            raise HTTPException(status_code=400, detail="Invalid file path")
        if normalized_old == normalized_new:
            raise HTTPException(
                status_code=400,
                detail="Source and destination paths must be different",
            )
        if self._is_reserved_internal_path(
            normalized_old
        ) or self._is_reserved_internal_path(normalized_new):
            raise HTTPException(status_code=400, detail="Invalid file path")

        normalized_old = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_old,
        )
        normalized_new = await self.ensure_workspace_path_not_in_disabled_mount(
            workspace_id,
            normalized_new,
        )

        source_path = await self._resolve_workspace_tree_file_path(
            workspace_id,
            normalized_old,
        )
        target_path = await self._resolve_workspace_tree_file_path(
            workspace_id,
            normalized_new,
        )

        try:
            await asyncio.to_thread(
                self._move_workspace_file_sync,
                source_path,
                target_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409, detail="Target file already exists"
            ) from exc

        await self.clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
            workspace_id,
            [normalized_old, normalized_new],
        )

        await self._touch_workspace(workspace_id)
        return {"old_path": normalized_old, "new_path": normalized_new}

    @staticmethod
    def _sql_quote(value: str | None) -> str:
        if value is None:
            return "NULL"
        return "'" + value.replace("'", "''") + "'"

    def _branch_ref_name(self, branch_id: str) -> str:
        return f"{_SNAPSHOT_BRANCH_REF_PREFIX}{branch_id}"

    def _snapshot_from_row(
        self,
        row: dict[str, Any],
        current_snapshot_id: str | None,
        branch_name_by_id: dict[str, str],
        *,
        can_delete: bool = False,
    ) -> UserSpaceSnapshot:
        created_at_raw = row.get("created_at")
        created_at = _coerce_utc_datetime(created_at_raw)
        branch_id = str(row.get("branch_id") or "")
        snapshot_id = str(row.get("id") or "")
        return UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=str(row.get("workspace_id") or ""),
            branch_id=branch_id,
            branch_name=branch_name_by_id.get(branch_id, "Branch"),
            parent_snapshot_id=(
                str(row.get("parent_snapshot_id"))
                if row.get("parent_snapshot_id")
                else None
            ),
            is_current=snapshot_id == current_snapshot_id,
            can_rename=True,
            can_delete=can_delete,
            git_commit_hash=(
                str(row.get("git_commit_hash")) if row.get("git_commit_hash") else None
            ),
            remote_commit_hash=(
                str(row.get("remote_commit_hash"))
                if row.get("remote_commit_hash")
                else None
            ),
            message=(
                str(row.get("message")) if row.get("message") is not None else None
            ),
            created_at=created_at,
            file_count=int(row.get("file_count") or 0),
        )

    async def _set_current_snapshot_cursor(
        self,
        workspace_id: str,
        snapshot_id: str | None,
        branch_id: str | None,
    ) -> None:
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE workspaces
            SET current_snapshot_id = {self._sql_quote(snapshot_id)},
                current_snapshot_branch_id = {self._sql_quote(branch_id)},
                updated_at = NOW()
            WHERE id = {self._sql_quote(workspace_id)}
            """
        )

    async def _activate_branch(self, workspace_id: str, branch_id: str) -> None:
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE userspace_snapshot_branches
            SET is_active = CASE WHEN id = {self._sql_quote(branch_id)} THEN TRUE ELSE FALSE END,
                updated_at = NOW()
            WHERE workspace_id = {self._sql_quote(workspace_id)}
            """
        )

    async def _ensure_snapshot_timeline(
        self,
        workspace_id: str,
        user_id: str,
    ) -> None:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        async with self._snapshot_operation_semaphore:
            db = await get_db()
            existing = await db.query_raw(
                f"""
                SELECT id
                FROM userspace_snapshot_branches
                WHERE workspace_id = {self._sql_quote(workspace_id)}
                LIMIT 1
                """
            )
            if existing:
                cursor_rows = await db.query_raw(
                    f"""
                    SELECT current_snapshot_id, current_snapshot_branch_id
                    FROM workspaces
                    WHERE id = {self._sql_quote(workspace_id)}
                    LIMIT 1
                    """
                )
                if cursor_rows:
                    cursor = cursor_rows[0]
                    if not cursor.get("current_snapshot_id") or not cursor.get(
                        "current_snapshot_branch_id"
                    ):
                        latest_rows = await db.query_raw(
                            f"""
                            SELECT s.id AS snapshot_id, s.branch_id
                            FROM userspace_snapshots s
                            WHERE s.workspace_id = {self._sql_quote(workspace_id)}
                            ORDER BY s.created_at DESC
                            LIMIT 1
                            """
                        )
                        if latest_rows:
                            latest = latest_rows[0]
                            await self._set_current_snapshot_cursor(
                                workspace_id,
                                str(latest.get("snapshot_id") or ""),
                                str(latest.get("branch_id") or ""),
                            )
                return

            git_branch = (
                await self._run_git(
                    workspace_id, ["branch", "--show-current"], check=False
                )
            ).stdout.strip() or "main"
            branch_id = str(uuid4())
            await db.execute_raw(
                f"""
                INSERT INTO userspace_snapshot_branches
                (id, workspace_id, name, git_ref_name, is_active, created_at, updated_at)
                VALUES
                (
                    {self._sql_quote(branch_id)},
                    {self._sql_quote(workspace_id)},
                    'Main',
                    {self._sql_quote(git_branch)},
                    TRUE,
                    NOW(),
                    NOW()
                )
                """
            )

            git_log_result = await self._run_git(
                workspace_id,
                ["log", "--reverse", "--pretty=format:%H%x1f%ct%x1f%s"],
                check=False,
            )
            if git_log_result.returncode != 0 or not git_log_result.stdout.strip():
                await self._set_current_snapshot_cursor(workspace_id, None, branch_id)
                return

            parent_snapshot_id: str | None = None
            last_snapshot_id: str | None = None
            for line in git_log_result.stdout.splitlines():
                parts = line.split("\x1f")
                if len(parts) != 3:
                    continue
                commit_hash, commit_ts, commit_subject = parts
                snapshot_id = str(uuid4())
                last_snapshot_id = snapshot_id
                created_at = datetime.fromtimestamp(
                    int(commit_ts), tz=timezone.utc
                ).isoformat()
                await db.execute_raw(
                    f"""
                    INSERT INTO userspace_snapshots
                    (
                        id,
                        workspace_id,
                        branch_id,
                        git_commit_hash,
                        message,
                        file_count,
                        parent_snapshot_id,
                        created_by_user_id,
                        created_at,
                        updated_at
                    )
                    VALUES
                    (
                        {self._sql_quote(snapshot_id)},
                        {self._sql_quote(workspace_id)},
                        {self._sql_quote(branch_id)},
                        {self._sql_quote(commit_hash)},
                        {self._sql_quote(commit_subject[:200])},
                        0,
                        {self._sql_quote(parent_snapshot_id)},
                        {self._sql_quote(user_id)},
                        {self._sql_quote(created_at)},
                        {self._sql_quote(created_at)}
                    )
                    ON CONFLICT (workspace_id, branch_id, git_commit_hash)
                    DO NOTHING
                    """
                )
                parent_snapshot_id = snapshot_id

            await self._set_current_snapshot_cursor(
                workspace_id, last_snapshot_id, branch_id
            )

    async def _get_snapshot_timeline_data(
        self,
        workspace_id: str,
    ) -> tuple[
        list[UserSpaceSnapshot], list[UserSpaceSnapshotBranch], str | None, str | None
    ]:
        db = await get_db()
        cursor_rows = await db.query_raw(
            f"""
            SELECT current_snapshot_id, current_snapshot_branch_id
            FROM workspaces
            WHERE id = {self._sql_quote(workspace_id)}
            LIMIT 1
            """
        )
        current_snapshot_id = None
        current_branch_id = None
        if cursor_rows:
            first = cursor_rows[0]
            current_snapshot_id = (
                str(first.get("current_snapshot_id"))
                if first.get("current_snapshot_id")
                else None
            )
            current_branch_id = (
                str(first.get("current_snapshot_branch_id"))
                if first.get("current_snapshot_branch_id")
                else None
            )

        branch_rows = await db.query_raw(
            f"""
            SELECT id, workspace_id, name, git_ref_name, base_snapshot_id,
                   branched_from_snapshot_id, is_active, created_at
            FROM userspace_snapshot_branches
            WHERE workspace_id = {self._sql_quote(workspace_id)}
            ORDER BY created_at ASC
            """
        )
        branch_name_by_id: dict[str, str] = {}
        branches: list[UserSpaceSnapshotBranch] = []
        for row in branch_rows:
            branch_id = str(row.get("id") or "")
            branch_name = str(row.get("name") or "Branch")
            branch_name_by_id[branch_id] = branch_name
            branches.append(
                UserSpaceSnapshotBranch(
                    id=branch_id,
                    workspace_id=str(row.get("workspace_id") or ""),
                    name=branch_name,
                    git_ref_name=str(row.get("git_ref_name") or ""),
                    base_snapshot_id=(
                        str(row.get("base_snapshot_id"))
                        if row.get("base_snapshot_id")
                        else None
                    ),
                    branched_from_snapshot_id=(
                        str(row.get("branched_from_snapshot_id"))
                        if row.get("branched_from_snapshot_id")
                        else None
                    ),
                    is_active=bool(row.get("is_active")),
                    created_at=_coerce_utc_datetime(row.get("created_at")),
                )
            )

        snapshot_rows = await db.query_raw(
            f"""
            SELECT id, workspace_id, branch_id, git_commit_hash, message,
                 remote_commit_hash, file_count, parent_snapshot_id, created_at
            FROM userspace_snapshots
            WHERE workspace_id = {self._sql_quote(workspace_id)}
            ORDER BY created_at DESC
            """
        )
        # Compute which snapshots are heads (no other snapshot points to them as parent)
        child_parent_ids: set[str] = {
            str(row.get("parent_snapshot_id"))
            for row in snapshot_rows
            if row.get("parent_snapshot_id")
        }
        snapshots = [
            self._snapshot_from_row(
                row,
                current_snapshot_id,
                branch_name_by_id,
                can_delete=str(row.get("id") or "") not in child_parent_ids,
            )
            for row in snapshot_rows
        ]
        return snapshots, branches, current_snapshot_id, current_branch_id

    async def get_snapshot_timeline(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceSnapshotTimelineResponse:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        snapshots, branches, current_snapshot_id, current_branch_id = (
            await self._get_snapshot_timeline_data(workspace_id)
        )

        has_previous = False
        has_next = False
        if current_snapshot_id and current_branch_id:
            current_snapshot = next(
                (
                    snapshot
                    for snapshot in snapshots
                    if snapshot.id == current_snapshot_id
                ),
                None,
            )
            if current_snapshot is not None:
                has_previous = bool(current_snapshot.parent_snapshot_id)
                has_next = any(
                    snapshot.branch_id == current_branch_id
                    and snapshot.parent_snapshot_id == current_snapshot_id
                    for snapshot in snapshots
                )
                if not has_next:
                    has_next = any(
                        branch.branched_from_snapshot_id == current_snapshot_id
                        for branch in branches
                    )

        return UserSpaceSnapshotTimelineResponse(
            workspace_id=workspace_id,
            current_snapshot_id=current_snapshot_id,
            current_branch_id=current_branch_id,
            has_previous=has_previous,
            has_next=has_next,
            snapshots=snapshots,
            branches=branches,
        )

    async def list_snapshots(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceSnapshot]:
        timeline = await self.get_snapshot_timeline(workspace_id, user_id)
        return timeline.snapshots

    async def _restore_snapshot_by_id(
        self,
        workspace_id: str,
        snapshot_id: str,
        user_id: str,
    ) -> UserSpaceSnapshot:
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT s.id, s.workspace_id, s.branch_id, s.git_commit_hash, s.message,
                   s.remote_commit_hash, s.file_count, s.parent_snapshot_id,
                   s.created_at,
                                     b.git_ref_name, b.name AS branch_name,
                                     NOT EXISTS (
                                             SELECT 1
                                             FROM userspace_snapshots child
                                             WHERE child.workspace_id = s.workspace_id
                                                 AND child.branch_id = s.branch_id
                                                 AND child.parent_snapshot_id = s.id
                                     ) AS is_branch_tip
            FROM userspace_snapshots s
            JOIN userspace_snapshot_branches b ON b.id = s.branch_id
            WHERE s.workspace_id = {self._sql_quote(workspace_id)}
              AND s.id = {self._sql_quote(snapshot_id)}
            LIMIT 1
            """
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        row = rows[0]
        commit_hash = str(row.get("git_commit_hash") or "")
        branch_id = str(row.get("branch_id") or "")
        branch_ref_name = str(row.get("git_ref_name") or "")
        is_branch_tip_raw = row.get("is_branch_tip")
        is_branch_tip = is_branch_tip_raw is True or str(is_branch_tip_raw).lower() in {
            "1",
            "t",
            "true",
        }

        async with self._snapshot_operation_semaphore:
            # Reset the worktree first so restore is resilient to platform-managed
            # files being rewritten outside Git between snapshots.
            await self._run_git(workspace_id, ["reset", "--hard"], check=False)
            await self._run_git(workspace_id, ["clean", "-fd"], check=False)
            if branch_ref_name and is_branch_tip:
                checkout_result = await self._run_git(
                    workspace_id,
                    ["checkout", "-f", branch_ref_name],
                    check=False,
                )
                if checkout_result.returncode != 0:
                    await self._run_git(
                        workspace_id,
                        ["checkout", "--detach", commit_hash],
                    )
            else:
                await self._run_git(workspace_id, ["checkout", "--detach", commit_hash])
            await self._run_git(workspace_id, ["reset", "--hard", commit_hash])
            await self._run_git(workspace_id, ["clean", "-fd"])
            await self._activate_branch(workspace_id, branch_id)
            await self._set_current_snapshot_cursor(
                workspace_id, snapshot_id, branch_id
            )

        await self.clear_workspace_changed_file_acknowledgements_for_all_users(
            workspace_id
        )
        await self._touch_workspace(workspace_id)

        return UserSpaceSnapshot(
            id=str(row.get("id") or ""),
            workspace_id=str(row.get("workspace_id") or ""),
            branch_id=branch_id,
            branch_name=str(row.get("branch_name") or "Branch"),
            parent_snapshot_id=(
                str(row.get("parent_snapshot_id"))
                if row.get("parent_snapshot_id")
                else None
            ),
            is_current=True,
            can_rename=True,
            git_commit_hash=commit_hash,
            remote_commit_hash=(
                str(row.get("remote_commit_hash"))
                if row.get("remote_commit_hash")
                else None
            ),
            message=(
                str(row.get("message")) if row.get("message") is not None else None
            ),
            created_at=_coerce_utc_datetime(row.get("created_at")),
            file_count=int(row.get("file_count") or 0),
        )

    async def restore_snapshot(
        self, workspace_id: str, snapshot_id: str, user_id: str
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        return await self._restore_snapshot_by_id(workspace_id, snapshot_id, user_id)

    async def create_snapshot(
        self,
        workspace_id: str,
        user_id: str,
        message: str | None = None,
        *,
        auto_sync_to_scm: bool = True,
    ) -> UserSpaceSnapshot:
        """Create a snapshot of the workspace.

        When a remote is configured (SCM connected):
        - Snapshot is created locally
        - Snapshot is automatically pushed to remote (commit + push)
        - Sync status is tracked in workspace.scm metadata
        - If push fails, error is logged and reported in sync metadata

        When no remote is configured:
        - Snapshot is created locally only
        - auto_sync_to_scm parameter is ignored

        Args:
            workspace_id: Workspace ID
            user_id: User creating the snapshot
            message: Optional snapshot message
            auto_sync_to_scm: If True (default) and remote configured, push to remote.
                              Set to False only for internal operations
                              (import/export backups, etc.)

        Returns:
            The created UserSpaceSnapshot object
        """
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        normalized_message = (message or "Snapshot").strip()
        commit_subject = (
            normalized_message.splitlines()[0][:200]
            if normalized_message
            else "Snapshot"
        )

        async with self._snapshot_operation_semaphore:
            db = await get_db()
            cursor_rows = await db.query_raw(
                f"""
                SELECT current_snapshot_id, current_snapshot_branch_id
                FROM workspaces
                WHERE id = {self._sql_quote(workspace_id)}
                LIMIT 1
                """
            )
            current_snapshot_id = None
            current_branch_id = None
            if cursor_rows:
                cursor = cursor_rows[0]
                current_snapshot_id = (
                    str(cursor.get("current_snapshot_id"))
                    if cursor.get("current_snapshot_id")
                    else None
                )
                current_branch_id = (
                    str(cursor.get("current_snapshot_branch_id"))
                    if cursor.get("current_snapshot_branch_id")
                    else None
                )

            if not current_branch_id:
                branch_rows = await db.query_raw(
                    f"""
                    SELECT id
                    FROM userspace_snapshot_branches
                    WHERE workspace_id = {self._sql_quote(workspace_id)}
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                )
                current_branch_id = (
                    str(branch_rows[0].get("id")) if branch_rows else str(uuid4())
                )

            tip_rows = await db.query_raw(
                f"""
                SELECT s.id
                FROM userspace_snapshots s
                WHERE s.workspace_id = {self._sql_quote(workspace_id)}
                  AND s.branch_id = {self._sql_quote(current_branch_id)}
                  AND NOT EXISTS (
                    SELECT 1
                    FROM userspace_snapshots child
                    WHERE child.workspace_id = s.workspace_id
                      AND child.branch_id = s.branch_id
                      AND child.parent_snapshot_id = s.id
                  )
                ORDER BY s.created_at DESC, s.id DESC
                LIMIT 1
                """
            )
            branch_tip_id = str(tip_rows[0].get("id")) if tip_rows else None

            if (
                current_snapshot_id
                and branch_tip_id
                and current_snapshot_id != branch_tip_id
            ):
                branch_count_rows = await db.query_raw(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM userspace_snapshot_branches
                    WHERE workspace_id = {self._sql_quote(workspace_id)}
                    """
                )
                branch_count = (
                    int(branch_count_rows[0].get("count") or 0)
                    if branch_count_rows
                    else 0
                )
                new_branch_id = str(uuid4())
                branch_name = f"Branch {branch_count + 1}"
                branch_ref_name = self._branch_ref_name(new_branch_id)
                await self._run_git(workspace_id, ["checkout", "-b", branch_ref_name])
                await db.execute_raw(
                    f"""
                    INSERT INTO userspace_snapshot_branches
                    (id, workspace_id, name, git_ref_name, base_snapshot_id, branched_from_snapshot_id, is_active, created_at, updated_at)
                    VALUES
                    (
                        {self._sql_quote(new_branch_id)},
                        {self._sql_quote(workspace_id)},
                        {self._sql_quote(branch_name)},
                        {self._sql_quote(branch_ref_name)},
                        {self._sql_quote(current_snapshot_id)},
                        {self._sql_quote(current_snapshot_id)},
                        TRUE,
                        NOW(),
                        NOW()
                    )
                    """
                )
                await self._activate_branch(workspace_id, new_branch_id)
                current_branch_id = new_branch_id

            await self._stage_workspace_snapshot_files(workspace_id)
            await self._run_git(
                workspace_id,
                ["commit", "--allow-empty", "-m", commit_subject],
            )

            commit_hash = (
                await self._run_git(workspace_id, ["rev-parse", "HEAD"])
            ).stdout.strip()
            commit_ts = (
                await self._run_git(
                    workspace_id,
                    ["show", "-s", "--format=%ct", commit_hash],
                )
            ).stdout.strip()
            created_at = datetime.fromtimestamp(int(commit_ts), tz=timezone.utc)

            tracked_files = (
                await self._run_git(
                    workspace_id,
                    ["ls-tree", "-r", "--name-only", commit_hash],
                )
            ).stdout.splitlines()
            file_count = sum(
                1
                for file_name in tracked_files
                if not self._is_reserved_internal_path(file_name)
            )

            snapshot_id = str(uuid4())
            await db.execute_raw(
                f"""
                INSERT INTO userspace_snapshots
                (id, workspace_id, branch_id, git_commit_hash, message, file_count, parent_snapshot_id, created_by_user_id, created_at, updated_at)
                VALUES
                (
                    {self._sql_quote(snapshot_id)},
                    {self._sql_quote(workspace_id)},
                    {self._sql_quote(current_branch_id)},
                    {self._sql_quote(commit_hash)},
                    {self._sql_quote(commit_subject)},
                    {file_count},
                    {self._sql_quote(current_snapshot_id)},
                    {self._sql_quote(user_id)},
                    {self._sql_quote(created_at.isoformat())},
                    {self._sql_quote(created_at.isoformat())}
                )
                """
            )
            await self._activate_branch(workspace_id, current_branch_id)
            await self._set_current_snapshot_cursor(
                workspace_id,
                snapshot_id,
                current_branch_id,
            )

        await self.clear_workspace_changed_file_acknowledgements_for_all_users(
            workspace_id
        )
        await self._touch_workspace(workspace_id, ts=created_at)

        timeline = await self.get_snapshot_timeline(workspace_id, user_id)
        created = next((s for s in timeline.snapshots if s.id == snapshot_id), None)
        if created is None:
            created = UserSpaceSnapshot(
                id=snapshot_id,
                workspace_id=workspace_id,
                branch_id=current_branch_id or "",
                branch_name="Branch",
                parent_snapshot_id=current_snapshot_id,
                is_current=True,
                can_rename=True,
                git_commit_hash=commit_hash,
                remote_commit_hash=None,
                message=commit_subject,
                created_at=created_at,
                file_count=file_count,
            )
        if auto_sync_to_scm:
            await self._maybe_auto_push_snapshot_to_scm(workspace_id, created)
        return created

    async def update_snapshot(
        self,
        workspace_id: str,
        snapshot_id: str,
        request: UpdateSnapshotRequest,
        user_id: str,
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE userspace_snapshots
            SET message = {self._sql_quote(request.message.strip()[:200])},
                updated_at = NOW()
            WHERE workspace_id = {self._sql_quote(workspace_id)}
              AND id = {self._sql_quote(snapshot_id)}
            """
        )
        timeline = await self.get_snapshot_timeline(workspace_id, user_id)
        snapshot = next(
            (item for item in timeline.snapshots if item.id == snapshot_id), None
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return snapshot

    async def navigate_snapshot_previous(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        timeline = await self.get_snapshot_timeline(workspace_id, user_id)
        if not timeline.current_snapshot_id or not timeline.current_branch_id:
            raise HTTPException(status_code=409, detail="No current snapshot selected")

        snapshots_by_id = {snapshot.id: snapshot for snapshot in timeline.snapshots}
        current_snapshot = snapshots_by_id.get(timeline.current_snapshot_id)
        if current_snapshot is None:
            raise HTTPException(status_code=409, detail="Current snapshot not found")

        # Primary behavior: move to the snapshot's parent pointer.
        parent_snapshot_id = current_snapshot.parent_snapshot_id
        if parent_snapshot_id and parent_snapshot_id in snapshots_by_id:
            return await self._restore_snapshot_by_id(
                workspace_id,
                parent_snapshot_id,
                user_id,
            )

        # Secondary fallback using branch metadata for older migrated timelines.
        current_branch = next(
            (
                branch
                for branch in timeline.branches
                if branch.id == timeline.current_branch_id
            ),
            None,
        )
        if (
            current_branch
            and current_branch.branched_from_snapshot_id
            and current_branch.branched_from_snapshot_id in snapshots_by_id
        ):
            return await self._restore_snapshot_by_id(
                workspace_id,
                current_branch.branched_from_snapshot_id,
                user_id,
            )

        raise HTTPException(status_code=409, detail="No previous snapshot available")

    async def navigate_snapshot_next(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        timeline = await self.get_snapshot_timeline(workspace_id, user_id)
        if not timeline.current_snapshot_id or not timeline.current_branch_id:
            raise HTTPException(status_code=409, detail="No current snapshot selected")

        snapshots_by_id = {snapshot.id: snapshot for snapshot in timeline.snapshots}

        # Primary behavior at branch points: move to a child-branch head that
        # branched from the current snapshot.
        child_branches = [
            branch
            for branch in timeline.branches
            if branch.branched_from_snapshot_id == timeline.current_snapshot_id
        ]
        if child_branches:
            current_branch = next(
                (
                    branch
                    for branch in child_branches
                    if branch.id == timeline.current_branch_id
                ),
                None,
            )
            preferred_children = [
                branch
                for branch in child_branches
                if current_branch is None or branch.id != current_branch.id
            ]
            if not preferred_children:
                preferred_children = child_branches

            # Deterministic order: active child first, then newest branch.
            preferred_children.sort(
                key=lambda branch: (
                    0 if branch.is_active else 1,
                    -branch.created_at.timestamp(),
                )
            )
            target_branch_id = preferred_children[0].id
            target_branch_snapshots = [
                snapshot
                for snapshot in timeline.snapshots
                if snapshot.branch_id == target_branch_id
            ]
            if target_branch_snapshots:
                target_branch_snapshots.sort(
                    key=lambda snapshot: snapshot.created_at.timestamp(),
                    reverse=True,
                )
                target_snapshot = target_branch_snapshots[0]
                if target_snapshot.id in snapshots_by_id:
                    return await self._restore_snapshot_by_id(
                        workspace_id,
                        target_snapshot.id,
                        user_id,
                    )

        # Fallback: move to the direct child in the current branch.
        same_branch_children = [
            snapshot
            for snapshot in timeline.snapshots
            if snapshot.branch_id == timeline.current_branch_id
            and snapshot.parent_snapshot_id == timeline.current_snapshot_id
        ]
        if same_branch_children:
            same_branch_children.sort(
                key=lambda snapshot: (
                    snapshot.created_at.timestamp(),
                    snapshot.id,
                ),
                reverse=True,
            )
            return await self._restore_snapshot_by_id(
                workspace_id,
                same_branch_children[0].id,
                user_id,
            )

        raise HTTPException(status_code=409, detail="No next snapshot available")

    async def switch_snapshot_branch(
        self,
        workspace_id: str,
        request: SwitchSnapshotBranchRequest,
        user_id: str,
    ) -> UserSpaceSnapshotTimelineResponse:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT id, git_ref_name
            FROM userspace_snapshot_branches
            WHERE workspace_id = {self._sql_quote(workspace_id)}
              AND id = {self._sql_quote(request.branch_id)}
            LIMIT 1
            """
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Snapshot branch not found")

        branch = rows[0]
        branch_id = str(branch.get("id") or "")
        branch_ref_name = str(branch.get("git_ref_name") or "")
        head_rows = await db.query_raw(
            f"""
            SELECT s.id
            FROM userspace_snapshots s
            WHERE s.workspace_id = {self._sql_quote(workspace_id)}
              AND s.branch_id = {self._sql_quote(branch_id)}
              AND NOT EXISTS (
                SELECT 1
                FROM userspace_snapshots child
                WHERE child.workspace_id = s.workspace_id
                  AND child.branch_id = s.branch_id
                  AND child.parent_snapshot_id = s.id
              )
            ORDER BY s.created_at DESC, s.id DESC
            LIMIT 1
            """
        )
        target_snapshot_id = str(head_rows[0].get("id")) if head_rows else None

        async with self._snapshot_operation_semaphore:
            if branch_ref_name:
                await self._run_git(
                    workspace_id, ["checkout", branch_ref_name], check=False
                )
            await self._activate_branch(workspace_id, branch_id)
            await self._set_current_snapshot_cursor(
                workspace_id, target_snapshot_id, branch_id
            )

        await self._touch_workspace(workspace_id)
        return await self.get_snapshot_timeline(workspace_id, user_id)

    async def create_snapshot_branch(
        self,
        workspace_id: str,
        user_id: str,
        name: str | None = None,
    ) -> UserSpaceSnapshotTimelineResponse:
        """Create a new branch from the current snapshot."""
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        db = await get_db()
        cursor_rows = await db.query_raw(
            f"""
            SELECT current_snapshot_id, current_snapshot_branch_id
            FROM workspaces
            WHERE id = {self._sql_quote(workspace_id)}
            LIMIT 1
            """
        )
        if not cursor_rows:
            raise HTTPException(status_code=404, detail="Workspace not found")

        current_snapshot_id = (
            str(cursor_rows[0].get("current_snapshot_id"))
            if cursor_rows[0].get("current_snapshot_id")
            else None
        )
        if not current_snapshot_id:
            raise HTTPException(
                status_code=400,
                detail="No snapshot to branch from; create a snapshot first",
            )

        branch_count_rows = await db.query_raw(
            f"""
            SELECT COUNT(*) AS count
            FROM userspace_snapshot_branches
            WHERE workspace_id = {self._sql_quote(workspace_id)}
            """
        )
        branch_count = (
            int(branch_count_rows[0].get("count") or 0) if branch_count_rows else 0
        )
        branch_name = (name or "").strip() or f"Branch {branch_count + 1}"
        new_branch_id = str(uuid4())
        branch_ref_name = self._branch_ref_name(new_branch_id)

        # Fetch source snapshot details so the new branch has a visible snapshot
        source_rows = await db.query_raw(
            f"""
            SELECT git_commit_hash, remote_commit_hash, message, file_count
            FROM userspace_snapshots
            WHERE id = {self._sql_quote(current_snapshot_id)}
            LIMIT 1
            """
        )
        if not source_rows:
            raise HTTPException(status_code=404, detail="Source snapshot not found")
        source = source_rows[0]
        source_commit = str(source.get("git_commit_hash") or "")
        source_remote_commit = source.get("remote_commit_hash")
        source_message = str(source.get("message") or "")
        source_file_count = int(source.get("file_count") or 0)

        new_snapshot_id = str(uuid4())

        async with self._snapshot_operation_semaphore:
            await self._run_git(workspace_id, ["checkout", "-b", branch_ref_name])
            await db.execute_raw(
                f"""
                INSERT INTO userspace_snapshot_branches
                (id, workspace_id, name, git_ref_name, base_snapshot_id,
                 branched_from_snapshot_id, is_active, created_at, updated_at)
                VALUES (
                    {self._sql_quote(new_branch_id)},
                    {self._sql_quote(workspace_id)},
                    {self._sql_quote(branch_name)},
                    {self._sql_quote(branch_ref_name)},
                    {self._sql_quote(current_snapshot_id)},
                    {self._sql_quote(current_snapshot_id)},
                    TRUE,
                    NOW(),
                    NOW()
                )
                """
            )
            # Create an initial snapshot on the new branch so it appears in the timeline
            remote_col = (
                self._sql_quote(str(source_remote_commit))
                if source_remote_commit
                else "NULL"
            )
            await db.execute_raw(
                f"""
                INSERT INTO userspace_snapshots
                (id, workspace_id, branch_id, git_commit_hash, remote_commit_hash,
                 message, file_count, parent_snapshot_id, created_by_user_id,
                 created_at, updated_at)
                VALUES (
                    {self._sql_quote(new_snapshot_id)},
                    {self._sql_quote(workspace_id)},
                    {self._sql_quote(new_branch_id)},
                    {self._sql_quote(source_commit)},
                    {remote_col},
                    {self._sql_quote(source_message)},
                    {source_file_count},
                    NULL,
                    {self._sql_quote(user_id)},
                    NOW(),
                    NOW()
                )
                """
            )
            await self._activate_branch(workspace_id, new_branch_id)
            await self._set_current_snapshot_cursor(
                workspace_id, new_snapshot_id, new_branch_id
            )

        await self._touch_workspace(workspace_id)
        return await self.get_snapshot_timeline(workspace_id, user_id)

    async def delete_snapshot(
        self,
        workspace_id: str,
        snapshot_id: str,
        user_id: str,
    ) -> UserSpaceSnapshotTimelineResponse:
        """Delete a snapshot, only allowed at the branch head.

        If the branch becomes empty afterward, the branch record and its git
        ref are also removed and the workspace switches to the most-recently
        active remaining branch.
        """
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_snapshot_timeline(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        db = await get_db()

        # Load the snapshot and verify it belongs to this workspace.
        snap_rows = await db.query_raw(
            f"""
            SELECT s.id, s.branch_id, s.parent_snapshot_id,
                   b.git_ref_name, b.name AS branch_name
            FROM userspace_snapshots s
            JOIN userspace_snapshot_branches b ON b.id = s.branch_id
            WHERE s.workspace_id = {self._sql_quote(workspace_id)}
              AND s.id = {self._sql_quote(snapshot_id)}
            LIMIT 1
            """
        )
        if not snap_rows:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        snap = snap_rows[0]
        branch_id = str(snap.get("branch_id") or "")
        branch_ref_name = str(snap.get("git_ref_name") or "")
        parent_snapshot_id = (
            str(snap.get("parent_snapshot_id"))
            if snap.get("parent_snapshot_id")
            else None
        )

        # Reject if this snapshot has children (not a head).
        child_rows = await db.query_raw(
            f"""
            SELECT id FROM userspace_snapshots
            WHERE workspace_id = {self._sql_quote(workspace_id)}
              AND parent_snapshot_id = {self._sql_quote(snapshot_id)}
            LIMIT 1
            """
        )
        if child_rows:
            raise HTTPException(
                status_code=409,
                detail="Only the branch head may be deleted",
            )

        # Load workspace cursor so we know if this is the current snapshot.
        cursor_rows = await db.query_raw(
            f"""
            SELECT current_snapshot_id, current_snapshot_branch_id
            FROM workspaces
            WHERE id = {self._sql_quote(workspace_id)}
            LIMIT 1
            """
        )
        cursor = cursor_rows[0] if cursor_rows else {}
        current_snapshot_id = (
            str(cursor.get("current_snapshot_id"))
            if cursor.get("current_snapshot_id")
            else None
        )

        async with self._snapshot_operation_semaphore:
            # Delete the snapshot record.
            await db.execute_raw(
                f"""
                DELETE FROM userspace_snapshots
                WHERE workspace_id = {self._sql_quote(workspace_id)}
                  AND id = {self._sql_quote(snapshot_id)}
                """
            )

            # Count remaining snapshots on this branch.
            count_rows = await db.query_raw(
                f"""
                SELECT COUNT(*) AS cnt
                FROM userspace_snapshots
                WHERE workspace_id = {self._sql_quote(workspace_id)}
                  AND branch_id = {self._sql_quote(branch_id)}
                """
            )
            remaining_on_branch = int(
                count_rows[0].get("cnt") or 0 if count_rows else 0
            )

            if remaining_on_branch == 0:
                # Branch is now empty — delete it.
                await db.execute_raw(
                    f"""
                    DELETE FROM userspace_snapshot_branches
                    WHERE workspace_id = {self._sql_quote(workspace_id)}
                      AND id = {self._sql_quote(branch_id)}
                    """
                )
                # Delete the git branch ref if it exists.
                if branch_ref_name:
                    await self._run_git(
                        workspace_id,
                        ["branch", "-D", branch_ref_name],
                        check=False,
                    )

                # Find another branch to become active.
                other_rows = await db.query_raw(
                    f"""
                    SELECT b.id AS branch_id, b.git_ref_name,
                           s.id AS head_snapshot_id
                    FROM userspace_snapshot_branches b
                    LEFT JOIN LATERAL (
                        SELECT id FROM userspace_snapshots
                        WHERE workspace_id = b.workspace_id
                          AND branch_id = b.id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) s ON TRUE
                    WHERE b.workspace_id = {self._sql_quote(workspace_id)}
                    ORDER BY b.created_at ASC
                    LIMIT 1
                    """
                )
                if other_rows:
                    new_branch_id = str(other_rows[0].get("branch_id") or "")
                    new_ref = str(other_rows[0].get("git_ref_name") or "")
                    new_head = (
                        str(other_rows[0].get("head_snapshot_id"))
                        if other_rows[0].get("head_snapshot_id")
                        else None
                    )
                    if new_ref:
                        await self._run_git(
                            workspace_id,
                            ["checkout", new_ref],
                            check=False,
                        )
                    await self._activate_branch(workspace_id, new_branch_id)
                    await self._set_current_snapshot_cursor(
                        workspace_id, new_head, new_branch_id
                    )
                else:
                    # No branches remain at all.
                    await self._set_current_snapshot_cursor(workspace_id, None, None)
            elif current_snapshot_id == snapshot_id:
                # Snapshot was current — move cursor back to parent.
                await self._set_current_snapshot_cursor(
                    workspace_id, parent_snapshot_id, branch_id
                )

        await self._touch_workspace(workspace_id)
        return await self.get_snapshot_timeline(workspace_id, user_id)

    # ──────────────────────────────────────────────────────────────
    # Component execution bridge (preview live data)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_effective_timeout(
        requested_timeout: int,
        timeout_max_seconds: int,
    ) -> int:
        requested = max(0, int(requested_timeout))
        maximum = max(0, int(timeout_max_seconds))
        if maximum == 0:
            return requested
        return min(requested, maximum)

    @staticmethod
    def _extract_query_text(payload: dict[str, Any] | str) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            value = payload.get("query") or payload.get("sql") or payload.get("command")
            return str(value or "")
        return ""

    async def _lookup_sidecar_query(
        self,
        workspace_id: str,
        component_id: str,
    ) -> str:
        """Look up the default query for a component from the sidecar metadata."""
        entry_file = (
            self._workspace_files_dir(workspace_id) / _USERSPACE_PREVIEW_ENTRY_PATH
        )
        try:
            _, connections, _ = await asyncio.to_thread(
                self._read_artifact_sidecar, entry_file
            )
        except Exception:
            return ""
        if not connections:
            return ""
        cid_lower = component_id.strip().lower()
        for conn in connections:
            if conn.component_id.strip().lower() == cid_lower:
                return str(conn.request or "")
            cname = str(getattr(conn, "component_name", "") or "").strip()
            if cname.lower().replace(" ", "_") == cid_lower.replace(" ", "_"):
                return str(conn.request or "")
        return ""

    async def _load_workspace_for_component_execution(
        self,
        workspace_id: str,
        user_id: str | None = None,
    ) -> UserSpaceWorkspace:
        if user_id is not None:
            return await self._enforce_workspace_access(workspace_id, user_id)

        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return self._workspace_from_record(workspace_record)

    async def execute_component(
        self,
        workspace_id: str,
        request: ExecuteComponentRequest,
        user_id: str,
    ) -> ExecuteComponentResponse:
        """Execute a live data query against a workspace-selected tool.

        Validates that the component_id belongs to the workspace's selected
        tools, loads the tool config, dispatches the query through the
        appropriate database driver, and returns structured rows.
        """
        workspace = await self._load_workspace_for_component_execution(
            workspace_id,
            user_id=user_id,
        )
        return await self._execute_component_for_workspace(
            workspace,
            request,
            error_log_prefix="Component execution failed",
        )

    async def execute_shared_component(
        self,
        share_token: str,
        request: ExecuteComponentRequest,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> ExecuteComponentResponse:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        return await self._execute_shared_component_for_workspace_id(
            workspace_id,
            request,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def execute_shared_component_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        request: ExecuteComponentRequest,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> ExecuteComponentResponse:
        workspace_id = await self._resolve_workspace_id_from_share_slug(
            owner_username,
            share_slug,
        )
        return await self._execute_shared_component_for_workspace_id(
            workspace_id,
            request,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )

    async def execute_component_from_authorized_shared_preview(
        self,
        workspace_id: str,
        request: ExecuteComponentRequest,
    ) -> ExecuteComponentResponse:
        workspace = await self._load_workspace_for_component_execution(workspace_id)
        return await self._execute_component_for_workspace(
            workspace,
            request,
            error_log_prefix="Shared component execution failed",
        )

    async def _execute_shared_component_for_workspace_id(
        self,
        workspace_id: str,
        request: ExecuteComponentRequest,
        *,
        current_user: Any | None = None,
        password: str | None = None,
        share_auth_token: str | None = None,
    ) -> ExecuteComponentResponse:
        await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
            share_auth_token=share_auth_token,
        )
        workspace = await self._load_workspace_for_component_execution(workspace_id)

        return await self._execute_component_for_workspace(
            workspace,
            request,
            error_log_prefix="Shared component execution failed",
        )

    async def _execute_component_for_workspace(
        self,
        workspace: UserSpaceWorkspace,
        request: ExecuteComponentRequest,
        *,
        error_log_prefix: str,
    ) -> ExecuteComponentResponse:
        tool_type, conn_config, tool_config = (
            await self._resolve_component_execution_config(
                workspace, request.component_id
            )
        )

        query = self._extract_query_text(request.request)
        if not query.strip():
            # Fallback: look up default query from live_data_connections sidecar.
            # Try the resolved tool config ID first, then the original component_id.
            resolved_tool_id = str(getattr(tool_config, "id", "") or "").strip()
            sidecar_query = await self._lookup_sidecar_query(
                workspace.id, resolved_tool_id
            )
            if not sidecar_query and resolved_tool_id != request.component_id:
                sidecar_query = await self._lookup_sidecar_query(
                    workspace.id, request.component_id
                )
            if sidecar_query:
                query = sidecar_query
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="No query/command found in request payload.",
            )

        try:
            raw_output = await self._dispatch_tool_query(
                tool_type,
                conn_config,
                tool_config,
                query,
            )
        except Exception as exc:
            logger.error(
                "%s for %s: %s",
                error_log_prefix,
                request.component_id,
                exc,
            )
            return ExecuteComponentResponse(
                component_id=request.component_id,
                rows=[],
                columns=[],
                row_count=0,
                error=str(exc),
            )
        response = self._build_execute_component_response(
            component_id=request.component_id,
            raw_output=raw_output,
        )

        # Mint server-side execution proof for live-data contract verification.
        if not response.error:
            query = self._extract_query_text(request.request)
            self.record_execution_proof(
                workspace.id,
                request.component_id,
                response.row_count,
                query,
            )

        return response

    async def _resolve_component_execution_config(
        self,
        workspace: UserSpaceWorkspace,
        component_id: str,
    ) -> tuple[str, dict[str, Any], Any]:
        resolved_id = component_id

        if resolved_id not in workspace.selected_tool_ids:
            # Fallback: try to match by tool name among selected tools.
            matched_id = await self._resolve_component_id_by_name(
                workspace, component_id
            )
            if matched_id is None:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Component {component_id} is not selected "
                        "for this workspace."
                    ),
                )
            resolved_id = matched_id

        tool_config = await repository.get_tool_config(resolved_id)
        if tool_config is None or not tool_config.enabled:
            raise HTTPException(
                status_code=404,
                detail="Tool configuration not found or disabled.",
            )

        tool_type = tool_config.tool_type.value
        if tool_type not in _USPACE_EXEC_SUPPORTED_SQL_TOOLS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Live preview execution supports SQL tools only "
                    "(postgres, mysql, mssql, influxdb)."
                ),
            )

        conn_config = tool_config.connection_config or {}
        return tool_type, conn_config, tool_config

    @staticmethod
    async def _resolve_component_id_by_name(
        workspace: UserSpaceWorkspace,
        name: str,
    ) -> str | None:
        """Try to find a tool config by name among the workspace's selected tools."""
        name_lower = name.strip().lower().replace(" ", "_")
        if not name_lower:
            return None
        for tool_id in workspace.selected_tool_ids:
            try:
                tool_config = await repository.get_tool_config(tool_id)
            except Exception:
                continue
            if tool_config is None or not tool_config.enabled:
                continue
            tool_name = str(getattr(tool_config, "name", "") or "").strip()
            if tool_name.lower().replace(" ", "_") == name_lower:
                return tool_id
        return None

    def _build_execute_component_response(
        self,
        component_id: str,
        raw_output: str,
    ) -> ExecuteComponentResponse:
        if raw_output.startswith("Error:"):
            return ExecuteComponentResponse(
                component_id=component_id,
                rows=[],
                columns=[],
                row_count=0,
                error=raw_output,
            )

        rows, columns = self._parse_query_output(raw_output)
        return ExecuteComponentResponse(
            component_id=component_id,
            rows=rows,
            columns=columns,
            row_count=len(rows),
        )

    async def _dispatch_tool_query(
        self,
        tool_type: str,
        conn_config: dict,
        tool_config: Any,
        query: str,
    ) -> str:
        """Route a query to the correct database executor."""
        host = conn_config.get("host", "")
        port = conn_config.get("port", 5432)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")
        timeout = self._resolve_effective_timeout(
            tool_config.timeout or 30,
            getattr(tool_config, "timeout_max_seconds", 300) or 300,
        )
        max_results = tool_config.max_results or 100

        if tool_type == "postgres":
            return await self._execute_postgres_query(
                conn_config, query, timeout, max_results
            )
        elif tool_type == "mssql":
            from ragtime.tools.mssql import execute_mssql_query_async

            ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
            return await execute_mssql_query_async(
                query=query,
                host=host,
                port=int(conn_config.get("port", 1433)),
                user=user,
                password=password,
                database=database,
                timeout=timeout,
                max_results=max_results,
                allow_write=False,
                require_result_limit=False,
                description="Preview component execution",
                ssh_tunnel_config=ssh_tunnel_config,
                include_metadata=True,
            )
        elif tool_type == "mysql":
            from ragtime.tools.mysql import execute_mysql_query_async

            ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
            return await execute_mysql_query_async(
                query=query,
                host=host,
                port=int(conn_config.get("port", 3306)),
                user=user,
                password=password,
                database=database,
                timeout=timeout,
                max_results=max_results,
                allow_write=False,
                require_result_limit=False,
                description="Preview component execution",
                ssh_tunnel_config=ssh_tunnel_config,
                include_metadata=True,
            )
        elif tool_type == "influxdb":
            from ragtime.tools.influxdb import execute_influxdb_query_async

            use_https = bool(conn_config.get("use_https", False))
            token = conn_config.get("token", "")
            org = conn_config.get("org", "")
            influx_host = conn_config.get("host", "")
            influx_port = int(conn_config.get("port", 8086))
            ssh_tunnel_config = build_ssh_tunnel_config(
                conn_config,
                influx_host,
                influx_port,
            )

            return await execute_influxdb_query_async(
                query=query,
                host=influx_host,
                port=influx_port,
                use_https=use_https,
                token=token,
                org=org,
                timeout=timeout,
                max_results=max_results,
                allow_write=False,
                require_result_limit=False,
                description="Preview component execution",
                ssh_tunnel_config=ssh_tunnel_config,
                include_metadata=True,
            )
        else:
            raise ValueError(
                f"Unsupported tool type for preview execution: {tool_type}"
            )

    async def _execute_postgres_query(
        self,
        conn_config: dict,
        query: str,
        timeout: int,
        max_results: int,
    ) -> str:
        """Execute a read-only postgres query using shared SQL utility helpers."""
        is_safe, reason = validate_sql_query(
            query,
            enable_write=False,
            db_type=DB_TYPE_POSTGRES,
            require_result_limit=False,
        )
        if not is_safe:
            return f"Error: {reason}"

        query = enforce_max_results(query, max_results, db_type=DB_TYPE_POSTGRES)

        host = conn_config.get("host", "")
        port = conn_config.get("port", 5432)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")
        container = conn_config.get("container", "")

        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)

        if ssh_tunnel_config:

            def run_tunnel_query() -> str:
                try:
                    import psycopg2  # type: ignore[import-untyped]
                    import psycopg2.extras  # type: ignore[import-untyped]
                except ImportError:
                    return "Error: psycopg2 not available"

                tunnel: SSHTunnel | None = None
                conn = None
                try:
                    tunnel_cfg = ssh_tunnel_config_from_dict(
                        ssh_tunnel_config, default_remote_port=5432
                    )
                    if not tunnel_cfg:
                        return "Error: Invalid SSH tunnel configuration"
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    conn = psycopg2.connect(
                        host="127.0.0.1",
                        port=local_port,
                        user=user,
                        password=password,
                        dbname=database,
                        connect_timeout=timeout if timeout > 0 else 30,
                    )
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cursor.execute(query)
                    if cursor.description:
                        rows = [dict(row) for row in cursor.fetchall()]
                        columns = [
                            col.name if getattr(col, "name", None) else str(col[0])
                            for col in cursor.description
                        ]
                        return format_query_result(
                            rows,
                            columns,
                            include_metadata=True,
                        )
                    return "Query executed successfully (no results)"
                except Exception as e:
                    return f"Error: {e}"
                finally:
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    if tunnel:
                        try:
                            tunnel.stop()
                        except Exception:
                            pass

            if timeout > 0:
                return await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, run_tunnel_query),
                    timeout=timeout + 5,
                )
            return await asyncio.get_event_loop().run_in_executor(
                None, run_tunnel_query
            )

        # Direct host or docker container
        escaped_query = query.replace("'", "'\\''")
        if host:
            cmd = [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                database,
                "-c",
                query,
            ]
            env = dict(os.environ)
            env["PGPASSWORD"] = password
        elif container:
            cmd = [
                "docker",
                "exec",
                "-i",
                container,
                "bash",
                "-c",
                f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \'{escaped_query}\'',
            ]
            env = None
        else:
            return "Error: No connection configured"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            if timeout > 0:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    return (
                        f"Error: Query timed out after {timeout}s. "
                        "An admin can increase the tool timeout in "
                        "Settings > Tools."
                    )
            else:
                stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return f"Error: {stderr.decode('utf-8', errors='replace').strip()}"

            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return "Query executed successfully (no results)"

            return add_table_metadata_to_psql_output(output, include_metadata=True)
        except asyncio.TimeoutError:
            return (
                f"Error: Query timed out after {timeout}s. "
                "An admin can increase the tool timeout in "
                "Settings > Tools."
            )
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _parse_query_output(output: str) -> tuple[list[dict[str, Any]], list[str]]:
        parsed = UserSpaceService._extract_table_metadata(output)
        if parsed is not None:
            return parsed
        return UserSpaceService._parse_tabular_output(output)

    @staticmethod
    def _extract_table_metadata(
        output: str,
    ) -> tuple[list[dict[str, Any]], list[str]] | None:
        start_marker = "<!--TABLEDATA:"
        start = output.find(start_marker)
        if start == -1:
            return None

        end = output.find("-->", start + len(start_marker))
        if end == -1:
            return None

        json_str = output[start + len(start_marker) : end]
        try:
            table_data = json.loads(json_str)
        except Exception:
            return None

        columns = table_data.get("columns")
        raw_rows = table_data.get("rows")
        if not isinstance(columns, list):
            return None

        normalized_columns = [str(col) for col in columns]
        if not isinstance(raw_rows, list):
            return [], normalized_columns

        parsed_rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            if isinstance(raw_row, dict):
                parsed_rows.append({str(k): v for k, v in raw_row.items()})
                continue

            if isinstance(raw_row, list):
                row_dict: dict[str, Any] = {}
                for index, column_name in enumerate(normalized_columns):
                    row_dict[column_name] = (
                        raw_row[index] if index < len(raw_row) else None
                    )
                parsed_rows.append(row_dict)

        return parsed_rows, normalized_columns

    @staticmethod
    def _parse_tabular_output(output: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Parse psql/tabular output into a list of dicts and column names.

        Handles the standard format:
            col1 | col2 | col3
            -----+------+-----
            val1 | val2 | val3
            (N rows)
        """
        lines = output.strip().splitlines()
        if len(lines) < 3:
            return [], []

        # First line is column headers
        header_line = lines[0]
        columns = [col.strip() for col in header_line.split("|")]

        # Skip separator line (index 1)
        rows: list[dict[str, Any]] = []
        for line in lines[2:]:
            stripped = line.strip()
            # Stop at the row count line
            if stripped.startswith("(") and stripped.endswith(")"):
                break
            if not stripped:
                continue
            values = [val.strip() for val in line.split("|")]
            row: dict[str, Any] = {}
            for i, col in enumerate(columns):
                if i < len(values):
                    # Try to parse numeric values
                    raw = values[i]
                    try:
                        if "." in raw:
                            row[col] = float(raw)
                        else:
                            row[col] = int(raw)
                    except (ValueError, TypeError):
                        row[col] = raw
                else:
                    row[col] = None
            rows.append(row)

        return rows, columns

    def _list_workspace_files_sync(
        self, files_dir: Path, include_dirs: bool = False
    ) -> list[UserSpaceFileInfo]:
        files: list[UserSpaceFileInfo] = []
        for path in files_dir.rglob("*"):
            is_file = path.is_file()
            is_dir = path.is_dir()
            if not is_file and not (include_dirs and is_dir):
                continue

            relative = str(path.relative_to(files_dir))
            if self._is_reserved_internal_path(relative):
                continue

            stat = path.stat()
            files.append(
                UserSpaceFileInfo(
                    path=relative,
                    size_bytes=stat.st_size if is_file else 0,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    entry_type="directory" if is_dir else "file",
                )
            )
        files.sort(key=lambda item: item.path)
        return files

    def _list_mount_source_files_sync(
        self,
        _workspace_id: str,
        mount_specs: list[dict[str, Any]],
        include_dirs: bool = False,
    ) -> list[UserSpaceFileInfo]:
        """Walk mount source directories and return entries mapped to their
        workspace-relative target paths so they appear in the file tree."""
        entries_by_path: dict[str, UserSpaceFileInfo] = {}
        for spec in mount_specs:
            source_local_path = spec.get("source_local_path", "")
            target_path = spec.get("target_path", "")
            repo_rel = self._workspace_mount_target_repo_relative_path(target_path)
            if not repo_rel:
                continue

            source_dirs: list[Path] = []
            if source_local_path:
                source_dir = Path(source_local_path)
                if source_dir.is_dir():
                    source_dirs.append(source_dir)

            if not source_dirs:
                continue

            if include_dirs:
                for source_dir in source_dirs:
                    try:
                        stat = source_dir.stat()
                        entries_by_path.setdefault(
                            repo_rel,
                            UserSpaceFileInfo(
                                path=repo_rel,
                                size_bytes=0,
                                updated_at=datetime.fromtimestamp(
                                    stat.st_mtime, tz=timezone.utc
                                ),
                                entry_type="directory",
                            ),
                        )
                        break
                    except OSError:
                        continue

            for source_dir in source_dirs:
                for path in source_dir.rglob("*"):
                    is_file = path.is_file()
                    is_dir = path.is_dir()
                    if not is_file and not (include_dirs and is_dir):
                        continue
                    try:
                        relative = str(path.relative_to(source_dir))
                        mapped_path = f"{repo_rel}/{relative}"
                        stat = path.stat()
                        entries_by_path.setdefault(
                            mapped_path,
                            UserSpaceFileInfo(
                                path=mapped_path,
                                size_bytes=stat.st_size if is_file else 0,
                                updated_at=datetime.fromtimestamp(
                                    stat.st_mtime, tz=timezone.utc
                                ),
                                entry_type="directory" if is_dir else "file",
                            ),
                        )
                    except (OSError, ValueError):
                        continue
        return sorted(entries_by_path.values(), key=lambda item: item.path)

    def _write_workspace_file_sync(
        self,
        file_path: Path,
        content: str,
        artifact_type: ArtifactType | None,
        live_data_connections: list[UserSpaceLiveDataConnection] | None,
        live_data_checks: list[UserSpaceLiveDataCheck] | None,
    ) -> os.stat_result:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        sidecar_payload: dict[str, Any] = {}
        if artifact_type is not None:
            sidecar_payload["artifact_type"] = artifact_type

        if live_data_connections is not None:
            sidecar_payload["live_data_connections"] = [
                connection.model_dump(mode="json")
                for connection in live_data_connections
            ]

        if live_data_checks is not None:
            sidecar_payload["live_data_checks"] = [
                check.model_dump(mode="json") for check in live_data_checks
            ]

        if sidecar_payload:
            sidecar.write_text(
                json.dumps(sidecar_payload),
                encoding="utf-8",
            )
        elif sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

        return file_path.stat()

    def _read_workspace_file_sync(
        self,
        file_path: Path,
        decode_errors: Literal["strict", "replace"] = "strict",
    ) -> tuple[
        ArtifactType | None,
        list[UserSpaceLiveDataConnection] | None,
        list[UserSpaceLiveDataCheck] | None,
        str,
        os.stat_result,
    ]:
        artifact_type, live_data_connections, live_data_checks = (
            self._read_artifact_sidecar(file_path)
        )
        try:
            content = file_path.read_text(encoding="utf-8", errors=decode_errors)
        except UnicodeDecodeError as exc:
            raise _NonUtf8WorkspaceFileError(file_path) from exc
        stat = file_path.stat()
        return artifact_type, live_data_connections, live_data_checks, content, stat

    def _delete_workspace_file_sync(self, file_path: Path) -> None:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

    def _move_workspace_file_sync(self, source_path: Path, target_path: Path) -> None:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        if target_path.exists():
            raise FileExistsError(str(target_path))

        source_sidecar = source_path.with_suffix(source_path.suffix + ".artifact.json")
        target_sidecar = target_path.with_suffix(target_path.suffix + ".artifact.json")
        if source_sidecar.exists() and target_sidecar.exists():
            raise FileExistsError(str(target_sidecar))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)
        if source_sidecar.exists() and source_sidecar.is_file():
            target_sidecar.parent.mkdir(parents=True, exist_ok=True)
            source_sidecar.rename(target_sidecar)

    @staticmethod
    def _extract_local_module_specifiers(source: str) -> list[str]:
        pattern = re.compile(
            r"(?:import|export)\s+(?:[^\"']*?\s+from\s+)?[\"']([^\"']+)[\"']|"
            r"import\(\s*[\"']([^\"']+)[\"']\s*\)"
        )
        specifiers: list[str] = []
        for match in pattern.findall(source):
            value = match[0] or match[1]
            if not value:
                continue
            if value.startswith(".") or value.startswith("/"):
                specifiers.append(value)
        return specifiers

    def _resolve_workspace_module_path(
        self,
        files_dir: Path,
        importer_relative_path: str,
        specifier: str,
    ) -> str | None:
        importer = PurePosixPath(importer_relative_path)
        base_dir = importer.parent
        raw_candidate = (
            PurePosixPath(specifier.lstrip("/"))
            if specifier.startswith("/")
            else base_dir / specifier
        )
        normalized = posixpath.normpath(str(raw_candidate).replace("\\", "/"))
        if normalized.startswith("../") or normalized == "..":
            return None

        candidates: list[str] = [normalized]
        if not normalized.lower().endswith(_MODULE_SOURCE_EXTENSIONS):
            for extension in _MODULE_SOURCE_EXTENSIONS:
                candidates.append(f"{normalized}{extension}")
                candidates.append(f"{normalized}/index{extension}")

        for candidate in candidates:
            if not candidate or candidate.startswith("../"):
                continue
            if self._is_reserved_internal_path(candidate):
                continue
            target_path = files_dir / candidate
            if target_path.exists() and target_path.is_file():
                return candidate
        return None

    def _collect_preview_workspace_files(
        self,
        files_dir: Path,
        entry_path: str,
    ) -> dict[str, str]:
        entry_relative_path = entry_path.strip().replace("\\", "/")
        entry_file_path = files_dir / entry_relative_path
        if not entry_file_path.exists() or not entry_file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Shared workspace preview entry not found",
            )

        queue: list[str] = [entry_relative_path]
        workspace_files: dict[str, str] = {}
        total_bytes = 0

        while queue:
            current_relative = queue.pop(0)
            if current_relative in workspace_files:
                continue

            file_path = files_dir / current_relative
            if not file_path.exists() or not file_path.is_file():
                continue
            if self._is_reserved_internal_path(current_relative):
                continue

            content = file_path.read_text(encoding="utf-8")
            encoded_size = len(content.encode("utf-8"))

            if len(workspace_files) >= _USERSPACE_PREVIEW_MAX_FILES:
                raise HTTPException(
                    status_code=413,
                    detail="Shared workspace preview exceeds file count limit",
                )
            if total_bytes + encoded_size > _USERSPACE_PREVIEW_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Shared workspace preview exceeds payload size limit",
                )

            workspace_files[current_relative] = content
            total_bytes += encoded_size

            for specifier in self._extract_local_module_specifiers(content):
                dependency = self._resolve_workspace_module_path(
                    files_dir,
                    current_relative,
                    specifier,
                )
                if dependency and dependency not in workspace_files:
                    queue.append(dependency)

        if entry_relative_path not in workspace_files:
            raise HTTPException(
                status_code=404,
                detail="Shared workspace preview entry not found",
            )

        return workspace_files

    # ------------------------------------------------------------------
    # SQL dump → workspace SQLite import
    # ------------------------------------------------------------------

    async def import_sql_to_workspace_sqlite(
        self,
        workspace_id: str,
        user_id: str,
        file_bytes: bytes,
        filename: str,
    ) -> "SqliteImportResponse":
        workspace = await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )

        if workspace.sqlite_persistence_mode != "include":
            raise HTTPException(
                status_code=400,
                detail=(
                    "SQLite persistence mode must be enabled (set to 'include') "
                    "on this workspace before importing a SQL dump."
                ),
            )

        if len(file_bytes) > _MAX_IMPORT_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"SQL dump exceeds the {_MAX_IMPORT_SIZE_BYTES // (1024 * 1024)} MB "
                    "size limit. Consider splitting the dump into smaller files."
                ),
            )

        if detect_binary_pg_dump(file_bytes):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Binary PostgreSQL dump format detected. "
                    "Please re-export using: pg_dump --format=plain"
                ),
            )

        # Decode bytes — try UTF-8 first, fall back to Latin-1 (lossless for arbitrary bytes).
        try:
            sql_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            sql_text = file_bytes.decode("latin-1")

        dialect = detect_sql_dialect(sql_text)
        sqlite_path = (
            self._workspace_files_dir(workspace_id) / ".ragtime" / "db" / "app.sqlite3"
        )
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        result: SqlImportResult = import_sql_to_sqlite(sqlite_path, sql_text, dialect)

        if result.success:
            summary_parts = [
                f"Imported {result.tables_created} table(s)",
                f"{result.rows_inserted} row(s)",
                f"{result.statements_executed} statement(s) executed",
            ]
            if result.errors:
                summary_parts.append(f"{len(result.errors)} non-fatal error(s)")
            message = "; ".join(summary_parts) + f" (dialect: {dialect})."
        else:
            message = (
                f"Import failed with {len(result.errors)} error(s) "
                f"after {result.statements_executed} statement(s)."
            )

        return SqliteImportResponse(
            success=result.success,
            dialect_detected=dialect,
            tables_created=result.tables_created,
            rows_inserted=result.rows_inserted,
            statements_executed=result.statements_executed,
            errors=result.errors,
            warnings=result.warnings,
            message=message,
        )


userspace_service = UserSpaceService()
userspace_service = UserSpaceService()
