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
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.auth import _get_ldap_connection, get_ldap_config
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (DB_TYPE_POSTGRES,
                                    add_table_metadata_to_psql_output,
                                    enforce_max_results, format_query_result,
                                    validate_sql_query)
from ragtime.core.ssh import (SSHTunnel, build_ssh_tunnel_config,
                              ssh_tunnel_config_from_dict)
from ragtime.indexer.repository import repository
from ragtime.userspace.models import (ArtifactType, CreateWorkspaceRequest,
                                      ExecuteComponentRequest,
                                      ExecuteComponentResponse,
                                      PaginatedWorkspacesResponse,
                                      ShareAccessMode, SqlitePersistenceMode,
                                      UpdateWorkspaceMembersRequest,
                                      UpdateWorkspaceRequest,
                                      UpdateWorkspaceShareAccessRequest,
                                      UpsertWorkspaceFileRequest,
                                      UserSpaceFileInfo, UserSpaceFileResponse,
                                      UserSpaceLiveDataCheck,
                                      UserSpaceLiveDataConnection,
                                      UserSpaceSharedPreviewResponse,
                                      UserSpaceSnapshot, UserSpaceWorkspace,
                                      UserSpaceWorkspaceShareLink,
                                      UserSpaceWorkspaceShareLinkStatus,
                                      WorkspaceMember,
                                      WorkspaceShareSlugAvailabilityResponse)

logger = get_logger(__name__)


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


_EXECUTION_PROOF_MAX_AGE_SECONDS = 3600  # 1 hour

_USPACE_EXEC_SUPPORTED_SQL_TOOLS = {"postgres", "mysql", "mssql"}
_USERSPACE_PREVIEW_ENTRY_PATH = "dashboard/main.ts"
_DEFAULT_SHARE_SLUG_PREFIX = "share"
_USERSPACE_PREVIEW_MAX_FILES = 200
_USERSPACE_PREVIEW_MAX_BYTES = 3_000_000
_SQLITE_EXCLUDE_GLOBS = (
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.db3",
)

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
_RUNTIME_BOOTSTRAP_TEMPLATE_VERSION = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    artifact_type: ArtifactType | None,
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
        return True

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
    mode = (value or "include").strip().lower()
    return mode if mode in {"include", "exclude"} else "include"


class UserSpaceService:
    def __init__(self) -> None:
        self._base_dir = Path(settings.index_data_path) / "_userspace"
        self._workspaces_dir = self._base_dir / "workspaces"
        self._workspaces_dir.mkdir(parents=True, exist_ok=True)
        self._execution_proofs: dict[str, dict[str, _ExecutionProofRecord]] = {}

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
    def _default_runtime_bootstrap_config() -> dict[str, Any]:
        return {
            "version": 1,
            "managed_by": "ragtime",
            "template_version": _RUNTIME_BOOTSTRAP_TEMPLATE_VERSION,
            "auto_update": True,
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

    def _workspace_git_dir(self, workspace_id: str) -> Path:
        return self._workspace_files_dir(workspace_id) / ".git"

    async def _run_git(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
    ) -> _GitCommandResult:
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
            result = _GitCommandResult(
                returncode=process.returncode if process.returncode is not None else 1,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
            )
            if check and result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise HTTPException(
                    status_code=500,
                    detail=f"Git snapshot operation failed: {stderr or 'unknown error'}",
                )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="Git binary not available for User Space snapshots",
            ) from exc

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

        files_dir = self._workspace_files_dir(workspace_id)
        target = (files_dir / relative_path).resolve()
        if files_dir.resolve() not in target.parents and target != files_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid file path")
        return target

    def resolve_workspace_file_path(
        self, workspace_id: str, relative_path: str
    ) -> Path:
        return self._resolve_workspace_file_path(workspace_id, relative_path)

    _HIDDEN_DIRS = frozenset(
        {".git", "node_modules", "__pycache__", ".ragtime", "dist"}
    )

    _AGENT_WRITABLE_RAGTIME_FILES = frozenset({"runtime-entrypoint.json"})

    def _is_reserved_internal_path(self, relative_path: str) -> bool:
        normalized = relative_path.strip("/")
        if normalized.endswith(".artifact.json"):
            return True
        parts = Path(normalized).parts
        # Allow agent access to specific .ragtime/ files (e.g. runtime-entrypoint.json)
        if (
            len(parts) == 2
            and parts[0] == ".ragtime"
            and parts[1] in self._AGENT_WRITABLE_RAGTIME_FILES
        ):
            return False
        return bool(self._HIDDEN_DIRS.intersection(parts))

    def is_reserved_internal_path(self, relative_path: str) -> bool:
        return self._is_reserved_internal_path(relative_path)

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
            include={"members": True, "toolSelections": True},
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
            where={"username": {"in": candidate_usernames}},
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
            include={"members": True, "toolSelections": True},
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
            str(getattr(workspace, "sqlitePersistenceMode", "include") or "include")
            if workspace
            else "include"
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

        return UserSpaceWorkspace(
            id=record.id,
            name=record.name,
            description=record.description,
            sqlite_persistence_mode=cast(
                SqlitePersistenceMode,
                _normalize_sqlite_persistence_mode(
                    str(
                        getattr(record, "sqlitePersistenceMode", "include") or "include"
                    )
                ),
            ),
            owner_user_id=record.ownerUserId,
            selected_tool_ids=selected_tool_ids,
            conversation_ids=[],
            members=members,
            created_at=record.createdAt,
            updated_at=record.updatedAt,
        )

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
    ) -> UserSpaceWorkspace:
        db = await get_db()

        workspace = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

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
            include={"members": True, "toolSelections": True},
        )

    async def list_workspaces(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> PaginatedWorkspacesResponse:
        db = await get_db()

        where_clause: dict[str, Any] = {
            "OR": [
                {"ownerUserId": user_id},
                {"members": {"some": {"userId": user_id}}},
            ]
        }

        rows = await db.workspace.find_many(
            where=where_clause,
            include={"members": True, "toolSelections": True},
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
                include={"members": True, "toolSelections": True},
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

        self._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        self._seed_runtime_bootstrap_config(workspace_id)
        self._seed_runtime_entrypoint_config(workspace_id)
        await self._ensure_workspace_git_repo(workspace_id)

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
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

    async def _build_shared_preview_response(
        self,
        workspace_id: str,
        *,
        current_user: Any | None = None,
        password: str | None = None,
    ) -> UserSpaceSharedPreviewResponse:
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        await self._enforce_share_access(workspace_record, current_user, password)

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

    async def delete_workspace(self, workspace_id: str, user_id: str) -> None:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="owner"
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
    ) -> UserSpaceWorkspace:
        return await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role=required_role,
        )

    async def update_workspace(
        self,
        workspace_id: str,
        request: UpdateWorkspaceRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        db = await get_db()

        update_data: dict[str, Any] = {"updatedAt": _utc_now()}
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

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
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
            include={"members": True, "toolSelections": True},
        )
        if not refreshed:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return self._workspace_from_record(refreshed)

    async def list_workspace_files(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceFileInfo]:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)
        files_dir = self._workspace_files_dir(workspace_id)
        if not files_dir.exists():
            return []

        return await asyncio.to_thread(self._list_workspace_files_sync, files_dir)

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
        if not skip_live_data_enforcement and _requires_live_data_contract(
            relative_path,
            request.artifact_type,
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
                    ),
                )
            if not parsed_live_data_checks:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing required live_data_checks verification metadata. "
                        "For live-data-requested module source writes, include checks proving successful "
                        "connection and transformation for each live_data_connections component_id."
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
            request.artifact_type,
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
        await self._touch_workspace(workspace_id)

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

        await self._touch_workspace(workspace_id)

    async def create_snapshot(
        self,
        workspace_id: str,
        user_id: str,
        message: str | None = None,
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_workspace_git_repo(workspace_id)
        await self._run_git(workspace_id, ["add", "-A"])
        await self._apply_snapshot_sqlite_policy(workspace_id)

        normalized_message = (message or "Snapshot").strip()
        commit_subject = (
            normalized_message.splitlines()[0][:200]
            if normalized_message
            else "Snapshot"
        )
        await self._run_git(
            workspace_id,
            ["commit", "--allow-empty", "-m", commit_subject],
        )

        snapshot_id = (
            await self._run_git(workspace_id, ["rev-parse", "HEAD"])
        ).stdout.strip()
        commit_ts = (
            await self._run_git(
                workspace_id,
                ["show", "-s", "--format=%ct", snapshot_id],
            )
        ).stdout.strip()
        created_at = datetime.fromtimestamp(int(commit_ts), tz=timezone.utc)

        tracked_files = (
            await self._run_git(
                workspace_id,
                ["ls-tree", "-r", "--name-only", snapshot_id],
            )
        ).stdout.splitlines()
        file_count = sum(
            1
            for file_name in tracked_files
            if not self._is_reserved_internal_path(file_name)
        )

        await self._touch_workspace(workspace_id, ts=created_at)

        return UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            message=commit_subject,
            created_at=created_at,
            file_count=file_count,
        )

    async def _get_snapshot_retention_cutoff(self) -> float | None:
        """Return a UNIX timestamp cutoff based on app_settings snapshot_retention_days.

        Returns None if retention is disabled (0 or unset).
        """
        try:
            app_settings = await repository.get_settings()

            if not app_settings:
                return None
            retention_days = getattr(app_settings, "snapshot_retention_days", 0) or 0
            if retention_days <= 0:
                return None
            return _utc_now().timestamp() - (retention_days * 86400)
        except Exception:
            return None

    async def list_snapshots(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceSnapshot]:
        await self._enforce_workspace_access(workspace_id, user_id)
        await self._ensure_workspace_git_repo(workspace_id)

        head_check = await self._run_git(
            workspace_id, ["rev-parse", "--verify", "HEAD"], check=False
        )
        if head_check.returncode != 0:
            return []

        cutoff = await self._get_snapshot_retention_cutoff()

        snapshots: list[UserSpaceSnapshot] = []
        git_log = (
            await self._run_git(
                workspace_id,
                ["log", "--pretty=format:%H%x1f%ct%x1f%s"],
            )
        ).stdout

        for line in git_log.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) != 3:
                continue

            commit_id, commit_ts, commit_subject = parts
            ts = int(commit_ts)
            if cutoff is not None and ts < cutoff:
                continue
            tracked_files = (
                await self._run_git(
                    workspace_id,
                    ["ls-tree", "-r", "--name-only", commit_id],
                )
            ).stdout.splitlines()
            file_count = sum(
                1
                for file_name in tracked_files
                if not self._is_reserved_internal_path(file_name)
            )
            snapshots.append(
                UserSpaceSnapshot(
                    id=commit_id,
                    workspace_id=workspace_id,
                    message=commit_subject,
                    created_at=datetime.fromtimestamp(ts, tz=timezone.utc),
                    file_count=file_count,
                )
            )
        return snapshots

    async def restore_snapshot(
        self, workspace_id: str, snapshot_id: str, user_id: str
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        await self._ensure_workspace_git_repo(workspace_id)

        commit_check = await self._run_git(
            workspace_id,
            ["cat-file", "-e", f"{snapshot_id}^{{commit}}"],
            check=False,
        )
        if commit_check.returncode != 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        # Enforce retention window on restore
        cutoff = await self._get_snapshot_retention_cutoff()
        if cutoff is not None:
            commit_ts = (
                await self._run_git(
                    workspace_id,
                    ["show", "-s", "--format=%ct", snapshot_id],
                )
            ).stdout.strip()
            if int(commit_ts) < cutoff:
                raise HTTPException(
                    status_code=410,
                    detail="Snapshot has expired and cannot be restored",
                )

        await self._run_git(workspace_id, ["reset", "--hard", snapshot_id])
        await self._run_git(workspace_id, ["clean", "-fd"])

        commit_ts = (
            await self._run_git(
                workspace_id,
                ["show", "-s", "--format=%ct", snapshot_id],
            )
        ).stdout.strip()
        commit_subject = (
            await self._run_git(
                workspace_id,
                ["show", "-s", "--format=%s", snapshot_id],
            )
        ).stdout.strip()
        tracked_files = (
            await self._run_git(
                workspace_id,
                ["ls-tree", "-r", "--name-only", snapshot_id],
            )
        ).stdout.splitlines()
        file_count = sum(
            1
            for file_name in tracked_files
            if not self._is_reserved_internal_path(file_name)
        )

        snapshot = UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            message=commit_subject,
            created_at=datetime.fromtimestamp(int(commit_ts), tz=timezone.utc),
            file_count=file_count,
        )

        await self._touch_workspace(workspace_id)

        return snapshot

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
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        await self._enforce_share_access(workspace_record, current_user, password)
        workspace = await self._load_workspace_for_component_execution(workspace_id)

        return await self._execute_component_for_workspace(
            workspace,
            request,
            error_log_prefix="Shared component execution failed",
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
        workspace_record = await self._get_workspace_record(workspace_id)
        if not workspace_record:
            raise HTTPException(status_code=404, detail="Shared workspace not found")
        await self._enforce_share_access(workspace_record, current_user, password)
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
        if component_id not in workspace.selected_tool_ids:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Component {component_id} is not selected " "for this workspace."
                ),
            )

        tool_config = await repository.get_tool_config(component_id)
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
                    "(postgres, mysql, mssql)."
                ),
            )

        conn_config = tool_config.connection_config or {}
        return tool_type, conn_config, tool_config

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
                    return f"Error: Query timed out after {timeout}s"
            else:
                stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return f"Error: {stderr.decode('utf-8', errors='replace').strip()}"

            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return "Query executed successfully (no results)"

            return add_table_metadata_to_psql_output(output, include_metadata=True)
        except asyncio.TimeoutError:
            return f"Error: Query timed out after {timeout}s"
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

    def _list_workspace_files_sync(self, files_dir: Path) -> list[UserSpaceFileInfo]:
        files: list[UserSpaceFileInfo] = []
        for file_path in files_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = str(file_path.relative_to(files_dir))
            if self._is_reserved_internal_path(relative):
                continue
            stat = file_path.stat()
            files.append(
                UserSpaceFileInfo(
                    path=relative,
                    size_bytes=stat.st_size,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
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
