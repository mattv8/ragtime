from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import re
import secrets
import shutil
import subprocess
import time as _time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException
from prisma import fields as prisma_fields

from ragtime.config import settings
from ragtime.core.app_settings import SettingsCache
from ragtime.core.auth import _get_ldap_connection, get_ldap_config
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.entrypoint_status import (EntrypointStatus,
                                            parse_entrypoint_config)
from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (DB_TYPE_POSTGRES,
                                    add_table_metadata_to_psql_output,
                                    enforce_max_results, format_query_result,
                                    validate_sql_query)
from ragtime.core.ssh import (SSHTunnel, build_ssh_tunnel_config,
                              ssh_tunnel_config_from_dict)
from ragtime.indexer.repository import repository
from ragtime.userspace.models import (ArtifactType, CreateWorkspaceRequest,
                                      DeleteWorkspaceEnvVarResponse,
                                      ExecuteComponentRequest,
                                      ExecuteComponentResponse,
                                      PaginatedWorkspacesResponse,
                                      ShareAccessMode, SqlitePersistenceMode,
                                      SwitchSnapshotBranchRequest,
                                      UpdateSnapshotRequest,
                                      UpdateWorkspaceMembersRequest,
                                      UpdateWorkspaceRequest,
                                      UpdateWorkspaceShareAccessRequest,
                                      UpsertWorkspaceEnvVarRequest,
                                      UpsertWorkspaceFileRequest,
                                      UserSpaceFileInfo, UserSpaceFileResponse,
                                      UserSpaceLiveDataCheck,
                                      UserSpaceLiveDataConnection,
                                      UserSpaceSharedPreviewResponse,
                                      UserSpaceSnapshot,
                                      UserSpaceSnapshotBranch,
                                      UserSpaceSnapshotDiffFileSummary,
                                      UserSpaceSnapshotDiffSummaryResponse,
                                      UserSpaceSnapshotFileDiffResponse,
                                      UserSpaceSnapshotTimelineResponse,
                                      UserSpaceWorkspace,
                                      UserSpaceWorkspaceEnvVar,
                                      UserSpaceWorkspaceShareLink,
                                      UserSpaceWorkspaceShareLinkStatus,
                                      WorkspaceMember,
                                      WorkspaceShareSlugAvailabilityResponse)

logger = get_logger(__name__)

_FILE_LIST_CACHE_TTL_SECONDS = 2
_ENTRYPOINT_STATUS_CACHE_TTL_SECONDS = 300  # 5-minute TTL for entrypoint status
_CHANGED_FILE_ACK_MAX_ROWS_PER_WORKSPACE_USER = (
    2000  # Threshold to bound growth of UserSpaceChangedFileAcknowledgement table
)


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


class _GitCommandResult:
    """Async git command result payload."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


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
_WORKSPACE_ENV_VAR_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORKSPACE_ENV_VAR_MAX_COUNT = 200

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
_RUNTIME_BOOTSTRAP_TEMPLATE_VERSION = 5
_RUNTIME_BRIDGE_PATH = ".ragtime/bridge.js"
_RUNTIME_BRIDGE_VERSION = 7
_RUNTIME_BRIDGE_VERSION_TAG = f"@ragtime/bridge v{_RUNTIME_BRIDGE_VERSION}"
_RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS = 310_000  # 300s + 10s buffer


def _build_bridge_content(timeout_ms: int = _RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS) -> str:
    """Build the bridge.js content with a workspace-specific timeout."""
    timeout_seconds = max(timeout_ms // 1000, 60)
    return (
        f"// {_RUNTIME_BRIDGE_VERSION_TAG} — platform-managed, do not edit\n"
        "(function () {\n"
        "  var B = 'userspace-exec-v1';\n"
        "  var E = 'ragtime-execute';\n"
        "  var R = 'ragtime-execute-result';\n"
        f"  var T = {timeout_ms};\n"
        f"  var T_LABEL = '{timeout_seconds}s';\n"
        "  var CHART_URL = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';\n"
        "  var JQUERY_URL = 'https://code.jquery.com/jquery-3.7.1.min.js';\n"
        "  var DATATABLES_JS_URL = 'https://cdn.datatables.net/1.13.8/js/dataTables.min.js';\n"
        "  var DATATABLES_CSS_URL = 'https://cdn.datatables.net/1.13.8/css/dataTables.dataTables.min.css';\n"
        "\n"
        "  function hasDataTables() {\n"
        "    return !!(window.jQuery && window.jQuery.fn && window.jQuery.fn.DataTable);\n"
        "  }\n"
        "\n"
        "  function ensureStylesheet(href) {\n"
        "    if (document.querySelector('link[href=\"' + href + '\"]')) return;\n"
        "    var link = document.createElement('link');\n"
        "    link.rel = 'stylesheet';\n"
        "    link.href = href;\n"
        "    document.head.appendChild(link);\n"
        "  }\n"
        "\n"
        "  function loadScript(src) {\n"
        "    return new Promise(function (resolve, reject) {\n"
        "      var existing = document.querySelector('script[src=\"' + src + '\"]');\n"
        "      if (existing) {\n"
        "        if (existing.getAttribute('data-ragtime-loaded') === '1') {\n"
        "          resolve();\n"
        "          return;\n"
        "        }\n"
        "        existing.addEventListener('load', function () { resolve(); }, { once: true });\n"
        "        existing.addEventListener('error', function () { reject(new Error('Failed to load ' + src)); }, { once: true });\n"
        "        return;\n"
        "      }\n"
        "      var script = document.createElement('script');\n"
        "      script.src = src;\n"
        "      script.async = false;\n"
        "      script.onload = function () {\n"
        "        script.setAttribute('data-ragtime-loaded', '1');\n"
        "        resolve();\n"
        "      };\n"
        "      script.onerror = function () { reject(new Error('Failed to load ' + src)); };\n"
        "      document.head.appendChild(script);\n"
        "    });\n"
        "  }\n"
        "\n"
        "  function bootstrapVizLibs() {\n"
        "    if (window.__ragtime_viz_bootstrap_promise) return window.__ragtime_viz_bootstrap_promise;\n"
        "\n"
        "    ensureStylesheet(DATATABLES_CSS_URL);\n"
        "\n"
        "    var canUseDocumentWrite = document.readyState === 'loading' && !!document.currentScript;\n"
        "    if (canUseDocumentWrite) {\n"
        "      if (!window.Chart) {\n"
        "        document.write('<script src=\"' + CHART_URL + '\"><\\/script>');\n"
        "      }\n"
        "      if (!window.jQuery) {\n"
        "        document.write('<script src=\"' + JQUERY_URL + '\"><\\/script>');\n"
        "      }\n"
        "      if (!hasDataTables()) {\n"
        "        document.write('<script src=\"' + DATATABLES_JS_URL + '\"><\\/script>');\n"
        "      }\n"
        "      window.__ragtime_viz_bootstrap_promise = Promise.resolve();\n"
        "      return window.__ragtime_viz_bootstrap_promise;\n"
        "    }\n"
        "\n"
        "    var chain = Promise.resolve();\n"
        "    if (!window.Chart) {\n"
        "      chain = chain.then(function () { return loadScript(CHART_URL); });\n"
        "    }\n"
        "    if (!window.jQuery) {\n"
        "      chain = chain.then(function () { return loadScript(JQUERY_URL); });\n"
        "    }\n"
        "    if (!hasDataTables()) {\n"
        "      chain = chain.then(function () { return loadScript(DATATABLES_JS_URL); });\n"
        "    }\n"
        "\n"
        "    window.__ragtime_viz_bootstrap_promise = chain.catch(function (error) {\n"
        "      console.warn('[ragtime bridge] visualization bootstrap failed:', error);\n"
        "    });\n"
        "    return window.__ragtime_viz_bootstrap_promise;\n"
        "  }\n"
        "\n"
        "  bootstrapVizLibs();\n"
        "\n"
        "  function getDirectExecuteUrl() {\n"
        "    var pathname = window.location && window.location.pathname ? window.location.pathname : '/';\n"
        "    var parts = pathname.split('/').filter(Boolean);\n"
        "\n"
        "    if (parts.length >= 6 &&\n"
        "        parts[0] === 'indexes' &&\n"
        "        parts[1] === 'userspace' &&\n"
        "        parts[2] === 'shared' &&\n"
        "        parts[5] === 'preview') {\n"
        "      return '/indexes/userspace/shared/' +\n"
        "        encodeURIComponent(parts[3]) + '/' +\n"
        "        encodeURIComponent(parts[4]) +\n"
        "        '/execute-component';\n"
        "    }\n"
        "\n"
        "    if (parts.length >= 5 &&\n"
        "        parts[0] === 'indexes' &&\n"
        "        parts[1] === 'userspace' &&\n"
        "        parts[2] === 'shared' &&\n"
        "        parts[4] === 'preview') {\n"
        "      return '/indexes/userspace/shared/' +\n"
        "        encodeURIComponent(parts[3]) +\n"
        "        '/execute-component';\n"
        "    }\n"
        "\n"
        "    var reserved = {\n"
        "      auth: true,\n"
        "      authorize: true,\n"
        "      docs: true,\n"
        "      health: true,\n"
        "      indexes: true,\n"
        "      mcp: true,\n"
        "      openapi: true,\n"
        "      shared: true,\n"
        "      v1: true\n"
        "    };\n"
        "\n"
        "    if (parts.length >= 2 && !reserved[parts[0]]) {\n"
        "      return '/indexes/userspace/shared/' +\n"
        "        encodeURIComponent(parts[0]) + '/' +\n"
        "        encodeURIComponent(parts[1]) +\n"
        "        '/execute-component';\n"
        "    }\n"
        "\n"
        "    return null;\n"
        "  }\n"
        "\n"
        "  function executeDirect(componentId, request, resolve) {\n"
        "    var executeUrl = getDirectExecuteUrl();\n"
        "    if (!executeUrl) {\n"
        "      resolve({ rows: [], columns: [], row_count: 0, error: 'Live data host unavailable in this context' });\n"
        "      return;\n"
        "    }\n"
        "\n"
        "    fetch(executeUrl, {\n"
        "      method: 'POST',\n"
        "      credentials: 'include',\n"
        "      headers: { 'Content-Type': 'application/json' },\n"
        "      body: JSON.stringify({ component_id: componentId, request: request || {} }),\n"
        "    })\n"
        "      .then(function (response) {\n"
        "        return response\n"
        "          .json()\n"
        "          .catch(function () { return {}; })\n"
        "          .then(function (payload) { return { ok: response.ok, payload: payload }; });\n"
        "      })\n"
        "      .then(function (result) {\n"
        "        if (result.ok) {\n"
        "          resolve(result.payload || { rows: [], columns: [], row_count: 0 });\n"
        "          return;\n"
        "        }\n"
        "        var detail = result.payload && (result.payload.detail || result.payload.error);\n"
        "        resolve({\n"
        "          rows: [],\n"
        "          columns: [],\n"
        "          row_count: 0,\n"
        "          error: detail || 'Failed to execute live data component',\n"
        "        });\n"
        "      })\n"
        "      .catch(function (error) {\n"
        "        resolve({\n"
        "          rows: [],\n"
        "          columns: [],\n"
        "          row_count: 0,\n"
        "          error: error && error.message ? error.message : String(error),\n"
        "        });\n"
        "      });\n"
        "  }\n"
        "\n"
        "  function makeExecute(componentId) {\n"
        "    return function execute(request) {\n"
        "      var hasParentHost = !!(window.parent && window.parent !== window);\n"
        "      var callId = '__exec_' + Math.random().toString(36).slice(2) + '_' + Date.now();\n"
        "      return new Promise(function (resolve) {\n"
        "        if (!hasParentHost) {\n"
        "          executeDirect(componentId, request, resolve);\n"
        "          return;\n"
        "        }\n"
        "\n"
        "        var timer = setTimeout(function () {\n"
        "          window.removeEventListener('message', handler);\n"
        "          resolve({ rows: [], columns: [], row_count: 0, error: 'Execute timed out after ' + T_LABEL + '. An admin can increase the tool timeout in Settings > Tools.' });\n"
        "        }, T);\n"
        "        function handler(event) {\n"
        "          if (event.source !== window.parent) return;\n"
        "          if (\n"
        "            event.data &&\n"
        "            event.data.bridge === B &&\n"
        "            event.data.type === R &&\n"
        "            event.data.callId === callId\n"
        "          ) {\n"
        "            window.removeEventListener('message', handler);\n"
        "            clearTimeout(timer);\n"
        "            resolve(event.data.result || { rows: [], columns: [], row_count: 0, error: 'Empty response' });\n"
        "          }\n"
        "        }\n"
        "        window.addEventListener('message', handler);\n"
        "        window.parent.postMessage(\n"
        "          { bridge: B, type: E, callId: callId, component_id: componentId, request: request || {} },\n"
        "          '*'\n"
        "        );\n"
        "      });\n"
        "    };\n"
        "  }\n"
        "\n"
        "  var componentsProxy = new Proxy({}, {\n"
        "    get: function (_, prop) {\n"
        "      if (typeof prop !== 'string') return undefined;\n"
        "      return Object.freeze({ component_id: prop, execute: makeExecute(prop) });\n"
        "    },\n"
        "    has: function () { return true; },\n"
        "  });\n"
        "\n"
        "  window.__ragtime_context = Object.freeze({\n"
        "    components: Object.freeze(componentsProxy),\n"
        "  });\n"
        "  if (!window.context) { window.context = window.__ragtime_context; }\n"
        "})();\n"
    )


# Keep a static reference for the runtime-bridge.js endpoint (no workspace
# context available there, so use the default timeout).
_RUNTIME_BRIDGE_CONTENT = _build_bridge_content()
_SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
_SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})

_HIDDEN_DIRS = frozenset({".git", "node_modules", "__pycache__", ".ragtime", "dist"})
_AGENT_WRITABLE_RAGTIME_FILES = frozenset({"runtime-entrypoint.json"})
_AGENT_WRITABLE_RAGTIME_PREFIXES = ("db/migrations/",)


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
        # Serialize snapshot mutations to keep timeline metadata consistent.
        self._snapshot_operation_semaphore = asyncio.Semaphore(1)

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
                "requirements.txt",
            ],
            "commands": [
                {
                    "name": "npm_ci",
                    "when_exists": "package-lock.json",
                    "run": "npm ci",
                },
                {
                    "name": "npm_install",
                    "when_exists": "package.json",
                    "unless_exists": "node_modules",
                    "run": "npm install",
                },
                {
                    "name": "npm_tailwind_tooling",
                    "when_exists": "package.json",
                    "unless_exists": "node_modules/.bin/tailwindcss",
                    "run": "npm install -D tailwindcss @tailwindcss/cli",
                },
                {
                    "name": "pip_requirements",
                    "when_exists": "requirements.txt",
                    "run": "python3 -m pip install -r requirements.txt",
                },
            ],
        }

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
        self._sync_runtime_bridge_script(workspace_id)

    def _sync_runtime_bridge_script(
        self,
        workspace_id: str,
        selected_tool_ids: list[str] | None = None,
    ) -> None:
        """Write or update the platform-managed ``.ragtime/bridge.js`` file.

        The bridge provides ``window.__ragtime_context`` with a
        ``components[id].execute()`` data bridge using ``postMessage``
        to the parent preview host.

        The bridge timeout is derived from the workspace's selected tool
        configs so that complex dashboard queries are not prematurely
        cancelled by the client-side timeout.
        """
        files_dir = self._workspace_files_dir(workspace_id)
        bridge_path = files_dir / _RUNTIME_BRIDGE_PATH

        # Compute per-workspace timeout from selected tool configs.
        timeout_ms = self._compute_bridge_timeout_ms(selected_tool_ids)

        # Re-generate if version changed OR if timeout changed.
        if bridge_path.exists():
            try:
                existing = bridge_path.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            if _RUNTIME_BRIDGE_VERSION_TAG in existing:
                # Check if the timeout value embedded in the file matches.
                if f"var T = {timeout_ms};" in existing:
                    return
        bridge_path.parent.mkdir(parents=True, exist_ok=True)
        bridge_path.write_text(_build_bridge_content(timeout_ms), encoding="utf-8")

    @staticmethod
    def _compute_bridge_timeout_ms(
        selected_tool_ids: list[str] | None = None,
    ) -> int:
        """Derive the bridge postMessage timeout from workspace tool configs.

        Uses the maximum ``timeout_max_seconds`` across the workspace's
        selected tools (with a 300 s floor) plus a 10 s buffer, converted
        to milliseconds.
        """
        if not selected_tool_ids:
            return _RUNTIME_BRIDGE_DEFAULT_TIMEOUT_MS

        max_timeout = 300
        cached_configs = SettingsCache.get_instance()._tool_configs
        if cached_configs:
            id_set = set(selected_tool_ids)
            for cfg in cached_configs:
                if cfg.get("id") in id_set:
                    t = cfg.get("timeout_max_seconds", 300) or 300
                    if t > max_timeout:
                        max_timeout = t
        return (max(max_timeout, 300) + 10) * 1000

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

    async def _run_git_raw(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
    ) -> tuple[int, bytes, bytes]:
        files_dir = self._workspace_files_dir(workspace_id)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(files_dir),
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
    ) -> _GitCommandResult:
        returncode, stdout_bytes, stderr_bytes = await self._run_git_raw(
            workspace_id,
            args,
            check=check,
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
    ) -> tuple[int, bytes, bytes]:
        return await self._run_git_raw(workspace_id, args, check=check)

    async def _ensure_workspace_git_repo(self, workspace_id: str) -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        git_dir = self._workspace_git_dir(workspace_id)
        if git_dir.exists() and git_dir.is_dir():
            # Ensure .gitignore exists even for pre-existing repos
            gitignore = files_dir / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "node_modules/\ndist/\n__pycache__/\n",
                    encoding="utf-8",
                )
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

        gitignore = files_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "node_modules/\ndist/\n__pycache__/\n",
                encoding="utf-8",
            )

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
                   s.file_count, s.parent_snapshot_id, s.created_at,
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
            matched: _WorkspaceSnapshotFileDiff | None = (
                summary_by_path.get(numstat_path) or old_path_lookup.get(numstat_path)
            )
            if matched is None:
                # Try normalizing in case of path format differences
                normalized = self._normalize_workspace_relative_path(numstat_path)
                if normalized:
                    matched = summary_by_path.get(
                        normalized
                    ) or old_path_lookup.get(normalized)
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
                    np = self._normalize_workspace_relative_path(
                        candidate_path or ""
                    )
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
                    np = self._normalize_workspace_relative_path(
                        candidate_path or ""
                    )
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

    async def _resolve_workspace_id_from_share_token(self, share_token: str) -> str:
        token = (share_token or "").strip()
        if not token:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        db = await get_db()
        workspace = await db.workspace.find_first(
            where={"shareToken": token},
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
            },
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        return str(workspace.id)

    async def _resolve_workspace_id_from_share_slug(
        self,
        owner_username: str,
        share_slug: str,
    ) -> str:
        normalized_owner = _normalize_owner_username_for_share_path(owner_username)
        normalized_slug = _normalize_share_slug_for_uniqueness(share_slug)
        if not normalized_owner or not normalized_slug:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        db = await get_db()
        candidate_usernames = [normalized_owner]
        if not normalized_owner.startswith("local:"):
            candidate_usernames.append(f"local:{normalized_owner}")

        users = await db.user.find_many(
            where={"username": {"in": candidate_usernames, "mode": "insensitive"}},
            take=5,
        )
        owner_ids = [
            str(getattr(user, "id", ""))
            for user in users
            if _normalize_owner_username_for_share_path(
                str(getattr(user, "username", "") or "")
            )
            == normalized_owner
        ]
        if not owner_ids:
            raise HTTPException(status_code=404, detail="Shared workspace not found")

        workspace = await db.workspace.find_first(
            where={
                "ownerUserId": {"in": owner_ids},
                "shareSlug": normalized_slug,
                "shareToken": {"not": None},
            },
            include={
                "members": True,
                "toolSelections": True,
                "toolGroupSelections": True,
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
            if decrypt_secret(password_encrypted) != provided:
                raise HTTPException(status_code=401, detail="Invalid password")
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
        return (
            "import React from 'react';\n"
            "import { createRoot } from 'react-dom/client';\n"
            f"import App from '{specifier}';\n\n"
            "const rootElement = document.getElementById('root') || (() => {\n"
            "  const element = document.createElement('div');\n"
            "  element.id = 'root';\n"
            "  document.body.appendChild(element);\n"
            "  return element;\n"
            "})();\n\n"
            "createRoot(rootElement).render(\n"
            "  <React.StrictMode>\n"
            "    <App />\n"
            "  </React.StrictMode>\n"
            ");\n"
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

    async def _apply_snapshot_sqlite_policy(self, workspace_id: str) -> None:
        db = await get_db()
        workspace = await db.workspace.find_unique(where={"id": workspace_id})
        mode = _normalize_sqlite_persistence_mode(
            str(getattr(workspace, "sqlitePersistenceMode", "exclude") or "exclude")
            if workspace
            else "exclude"
        )
        if mode != "exclude":
            return
        for pattern in _SQLITE_EXCLUDE_GLOBS:
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
            created_at=record.createdAt,
            updated_at=record.updatedAt,
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

        # Extract selected tool IDs early so we can pass them to the bridge
        # sync for per-workspace timeout computation.
        _selected_tool_ids = self._selected_tool_ids_from_workspace_record(workspace)

        if is_admin:
            self._sync_runtime_bootstrap_config(workspace_id)
            self._sync_runtime_bridge_script(workspace_id, _selected_tool_ids)
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
                self._sync_runtime_bridge_script(workspace_id, _selected_tool_ids)
                return self._workspace_from_record(workspace)
            raise HTTPException(status_code=404, detail="Workspace not found")

        if required_role == "editor" and user_role not in {"owner", "editor"}:
            raise HTTPException(status_code=403, detail="Editor access required")
        if required_role == "owner" and user_role != "owner":
            raise HTTPException(status_code=403, detail="Owner access required")

        self._sync_runtime_bootstrap_config(workspace_id)
        self._sync_runtime_bridge_script(workspace_id, _selected_tool_ids)
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
            "shareSelectedUserIds": selected_user_ids,
            "shareSelectedLdapGroups": selected_ldap_groups,
            "updatedAt": _utc_now(),
        }

        if mode == "password":
            password = (request.password or "").strip()
            if not password:
                raise HTTPException(status_code=400, detail="Password is required")
            update_data["sharePassword"] = encrypt_secret(password)
        elif request.password is not None and request.password.strip():
            update_data["sharePassword"] = encrypt_secret(request.password.strip())
        elif mode != "password":
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
        try:
            await db.workspace.update(where={"id": workspace_id}, data=update_data)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

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
    ) -> UserSpaceSharedPreviewResponse:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        return await self._build_shared_preview_response(
            workspace_id,
            current_user=current_user,
            password=password,
        )

    async def _get_authorized_shared_workspace_record(
        self,
        workspace_id: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> Any:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        await self._enforce_share_access(workspace_record, current_user, password)
        tool_ids = self._selected_tool_ids_from_workspace_record(workspace_record)
        self._sync_runtime_bridge_script(workspace_id, tool_ids)
        return workspace_record

    async def resolve_shared_workspace_id(
        self,
        share_token: str,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> str:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
        )
        return workspace_id

    async def get_shared_preview_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_id = await self._resolve_workspace_id_from_share_slug(
            owner_username,
            share_slug,
        )
        return await self._build_shared_preview_response(
            workspace_id,
            current_user=current_user,
            password=password,
        )

    async def resolve_shared_workspace_id_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> str:
        workspace_id = await self._resolve_workspace_id_from_share_slug(
            owner_username,
            share_slug,
        )
        await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
        )
        return workspace_id

    async def _build_shared_preview_response(
        self,
        workspace_id: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_record = await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
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
        return self._sanitize_workspace_env_map(list(rows))

    def invalidate_file_list_cache(self, workspace_id: str) -> None:
        self._file_list_cache.pop(workspace_id, None)

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
        self, workspace_id: str, user_id: str, include_dirs: bool = False
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

        result = await asyncio.to_thread(
            self._list_workspace_files_sync, files_dir, include_dirs
        )
        self._file_list_cache[workspace_id] = (result, include_dirs, _time.monotonic())
        return result

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
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if _requires_entrypoint_wiring(relative_path, request.artifact_type):
            main_path = self._resolve_workspace_file_path(
                workspace_id,
                _USERSPACE_PREVIEW_ENTRY_PATH,
            )
            if not main_path.exists() or not main_path.is_file():
                main_path.parent.mkdir(parents=True, exist_ok=True)
                main_path.write_text(
                    self._build_dashboard_entrypoint_content(relative_path),
                    encoding="utf-8",
                )

            try:
                main_content = main_path.read_text(encoding="utf-8")
            except OSError:
                main_content = ""

            candidates = _entrypoint_module_specifier_candidates(relative_path)
            if not _entrypoint_references_module(main_content, candidates):
                main_path.write_text(
                    self._append_dashboard_entrypoint_reference(
                        main_content,
                        relative_path,
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
            relative_path,
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
            relative_path,
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

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
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
            [relative_path],
        )
        await self._touch_workspace(workspace_id)

        # Invalidate entrypoint status cache when the entrypoint config is written
        normalized = (relative_path or "").strip("/")
        if normalized == ".ragtime/runtime-entrypoint.json":
            self.invalidate_entrypoint_cache(workspace_id)

        return UserSpaceFileResponse(
            path=relative_path,
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
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
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
                    f"Path: {relative_path}"
                ),
            ) from exc

        return UserSpaceFileResponse(
            path=relative_path,
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
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        await asyncio.to_thread(self._delete_workspace_file_sync, file_path)

        await self.clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
            workspace_id,
            [relative_path],
        )

        await self._touch_workspace(workspace_id)

        # Invalidate entrypoint status cache when the entrypoint config is deleted
        normalized = (relative_path or "").strip("/")
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

        source_path = self._resolve_workspace_file_path(workspace_id, normalized_old)
        target_path = self._resolve_workspace_file_path(workspace_id, normalized_new)

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
            git_commit_hash=(
                str(row.get("git_commit_hash")) if row.get("git_commit_hash") else None
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
                   file_count, parent_snapshot_id, created_at
            FROM userspace_snapshots
            WHERE workspace_id = {self._sql_quote(workspace_id)}
            ORDER BY created_at DESC
            """
        )
        snapshots = [
            self._snapshot_from_row(
                row,
                current_snapshot_id,
                branch_name_by_id,
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
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT s.id, s.workspace_id, s.branch_id, s.git_commit_hash, s.message,
                   s.file_count, s.parent_snapshot_id, s.created_at,
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
            if branch_ref_name and is_branch_tip:
                await self._run_git(workspace_id, ["checkout", branch_ref_name])
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
    ) -> UserSpaceSnapshot:
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

            await self._run_git(workspace_id, ["add", "-A"])
            await self._apply_snapshot_sqlite_policy(workspace_id)
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
        if created is not None:
            return created
        return UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            branch_id=current_branch_id or "",
            branch_name="Branch",
            parent_snapshot_id=current_snapshot_id,
            is_current=True,
            can_rename=True,
            git_commit_hash=commit_hash,
            message=commit_subject,
            created_at=created_at,
            file_count=file_count,
        )

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
    ) -> ExecuteComponentResponse:
        workspace_id = await self._resolve_workspace_id_from_share_token(share_token)
        return await self._execute_shared_component_for_workspace_id(
            workspace_id,
            request,
            current_user=current_user,
            password=password,
        )

    async def execute_shared_component_by_slug(
        self,
        owner_username: str,
        share_slug: str,
        request: ExecuteComponentRequest,
        current_user: Any | None = None,
        password: str | None = None,
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
        )

    async def _execute_shared_component_for_workspace_id(
        self,
        workspace_id: str,
        request: ExecuteComponentRequest,
        *,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> ExecuteComponentResponse:
        await self._get_authorized_shared_workspace_record(
            workspace_id,
            current_user=current_user,
            password=password,
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


userspace_service = UserSpaceService()
