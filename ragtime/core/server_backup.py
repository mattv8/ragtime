from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import http.client
import json
import os
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterator, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ragtime.config.settings import ENCRYPTION_KEY_FILE, settings
from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream, encrypt_stream, is_encrypted_backup
from ragtime.core.file_lock import backup_restore_lock
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(os.environ.get("INDEX_DATA_PATH", settings.index_data_path))
PRISMA_DIR = Path("/ragtime/prisma")
LOCK_PATH = Path(tempfile.gettempdir()) / "ragtime-server-backup.lock"
MANIFEST_NAME = "backup-meta.json"
DEPLOYMENT_ENV_NAME = "deployment-env.json"
RESTORE_CONFIRMATION_PREFIX = "RESTORE"
_DEPLOYMENT_ENV_IGNORED_NAMES = frozenset(
    {
        "_",
        "HOME",
        "HOSTNAME",
        "PATH",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "TERM",
        "LANG",
        "GPG_KEY",
        "VIRTUAL_ENV",
        "RAGTIME_VERSION",
        "RAGTIME_LEGACY_CPU",
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_TOKEN",
        "GITLAB_CI",
        "CI_JOB_TOKEN",
        "SSH_AUTH_SOCK",
        "AWS_SESSION_TOKEN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)
_DEPLOYMENT_ENV_IGNORED_PREFIXES = (
    "LC_",
    "PYTHON",
    "PIP_",
    "UV_",
    "RUNNER_",
    "KUBERNETES_",
    "RAGTIME_BRIDGE_",
)
_DEPLOYMENT_ENV_MAX_BYTES = 1024 * 1024


class BackupError(RuntimeError):
    pass


class BackupValidationError(BackupError):
    pass


class BackupMutationError(BackupError):
    pass


class BackupRollbackError(BackupMutationError):
    pass


class BackupPasswordRequiredError(BackupError):
    pass


class LegacyBackupKeyConfirmationRequiredError(BackupValidationError):
    pass


class BackupScope(str, Enum):
    FULL = "full"
    DATABASE = "database"
    FILES = "files"


@dataclass
class BackupOptions:
    scope: BackupScope = BackupScope.FULL
    output_path: Optional[Path] = None
    output_stream: Optional[BinaryIO] = None
    encrypt: bool = False
    password_fd: Optional[int] = None


@dataclass
class RestoreOptions:
    archive_path: Optional[Path] = None
    archive_stream: Optional[BinaryIO] = None
    scope_override: Optional[BackupScope] = None
    skip_migrations: bool = False
    pg_data_only: bool = False
    replace_data: bool = False
    acknowledge_legacy_key: bool = False
    restore_confirmation: Optional[str] = None
    mirror_local_admin_access: bool = False
    mirror_local_admin_from: str = "auto"
    local_admin_username: Optional[str] = None
    password_fd: Optional[int] = None


@dataclass
class BackupManifest:
    format: str
    version: int
    created_at: str
    scope: BackupScope
    ragtime_version: str
    schema_version: str
    encrypted: bool
    includes_managed_key: bool
    deployment_environment_variables: list[str] = field(default_factory=list)
    legacy_embedded_key: bool = False
    sqlite_history: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BackupManifest":
        version_value = payload.get("version", 1)
        if not isinstance(version_value, int):
            raise BackupValidationError("Backup manifest version is invalid")
        environment_variable_names = payload.get("deployment_environment_variables", [])
        if not isinstance(environment_variable_names, list):
            environment_variable_names = []
        sqlite_history_payload = payload.get("sqlite_history")
        sqlite_history: dict[str, object] | None = None
        if isinstance(sqlite_history_payload, dict) and all(isinstance(key, str) for key in sqlite_history_payload):
            sqlite_history = {key: value for key, value in sqlite_history_payload.items() if isinstance(key, str)}
        return cls(
            format=str(payload.get("format", "tar.gz")),
            version=version_value,
            created_at=str(payload.get("created_at", "")),
            scope=BackupScope(str(payload.get("scope", BackupScope.FULL.value))),
            ragtime_version=str(payload.get("ragtime_version", "unknown")),
            schema_version=str(payload.get("schema_version", "unknown")),
            encrypted=bool(payload.get("encrypted", False)),
            includes_managed_key=bool(payload.get("includes_managed_key", False)),
            deployment_environment_variables=sorted(str(name) for name in environment_variable_names if isinstance(name, str)),
            legacy_embedded_key=bool(payload.get("legacy_embedded_key", False)),
            sqlite_history=sqlite_history,
        )


@dataclass
class DeploymentEnvironmentRecovery:
    variables: dict[str, str]
    variable_names: list[str]
    warnings: list[str]


ProgressCallback = Callable[[dict[str, object]], None]


def _emit_progress(callback: Optional[ProgressCallback], phase: str, **details: object) -> None:
    if callback is None:
        return
    event: dict[str, object] = {"phase": phase}
    event.update(details)
    try:
        callback(event)
    except Exception as exc:
        logger.warning("Backup progress callback failed during %s: %s", phase, exc)


class _ItemProgressReporter:
    def __init__(
        self,
        callback: Optional[ProgressCallback],
        phase: str,
        *,
        start: int,
        end: int,
        total: int,
        message: str,
        min_interval: float = 0.25,
    ) -> None:
        self._callback = callback
        self._phase = phase
        self._start = start
        self._end = end
        self._total = total
        self._message = message
        self._min_interval = min_interval
        self._last_emitted_at: Optional[float] = None

    def update(self, item: str, processed: int, *, force: bool = False) -> None:
        if self._callback is None or self._total <= 0:
            return
        now = time.monotonic()
        should_emit = self._last_emitted_at is None or force
        if not should_emit and self._last_emitted_at is not None:
            should_emit = (now - self._last_emitted_at) >= self._min_interval
        if not should_emit:
            return

        safe_item = item[2:] if item.startswith("./") else item
        bounded_processed = max(0, min(processed, self._total))
        if bounded_processed >= self._total:
            progress_value = self._end
        else:
            span = max(self._end - self._start, 0)
            progress_value = self._start + int((span * bounded_processed) / self._total)
            if progress_value >= self._end:
                progress_value = max(self._start, self._end - 1)

        _emit_progress(
            self._callback,
            self._phase,
            progress=progress_value,
            message=self._message,
            current_item=safe_item,
            processed_items=bounded_processed,
            total_items=self._total,
        )
        self._last_emitted_at = now


def _iter_tree_entries(root: Path) -> list[Path]:
    return sorted(root.rglob("*"))


def _get_ragtime_version() -> str:
    return os.environ.get("RAGTIME_VERSION", "unknown")


def _get_current_schema_version() -> str:
    migrations_dir = PRISMA_DIR / "migrations"
    if not migrations_dir.exists():
        return "unknown"
    versions = [child.name[:14] for child in migrations_dir.iterdir() if child.is_dir() and len(child.name) >= 15 and child.name[:14].isdigit()]
    return max(versions) if versions else "unknown"


def _database_env() -> dict[str, str]:
    database_url = os.environ["DATABASE_URL"] if "DATABASE_URL" in os.environ else settings.database_url
    parsed = urlparse(database_url)
    if parsed.scheme and parsed.username and parsed.password and parsed.path:
        return {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 5432),
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/"),
        }
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "user": os.environ.get("POSTGRES_USER", ""),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
        "database": os.environ.get("POSTGRES_DB", ""),
    }


def _pg_env() -> dict[str, str]:
    env = os.environ.copy()
    password = _database_env().get("password", "")
    if password:
        env["PGPASSWORD"] = password
    return env


def _run_command(command: list[str], *, stdout: BinaryIO | int | None = None) -> None:
    result = subprocess.run(command, env=_pg_env(), stdout=stdout, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise BackupError(result.stderr.decode("utf-8", errors="replace") or f"Command failed: {' '.join(command[:2])}")


def _run_psql(sql: str, *, variables: Optional[dict[str, str]] = None, capture: bool = False) -> str:
    db = _database_env()
    if not db["database"] or not db["user"]:
        raise BackupError("Database settings are not configured")
    command = [
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        db["host"],
        "-p",
        db["port"],
        "-U",
        db["user"],
        "-d",
        db["database"],
        "-At",
    ]
    for key, value in (variables or {}).items():
        command.extend(["-v", f"{key}={value}"])
    command.extend(["-c", sql])
    result = subprocess.run(command, env=_pg_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise BackupError(result.stderr.strip() or "psql command failed")
    return result.stdout.strip() if capture else ""


def _dump_database(destination: Path) -> None:
    db = _database_env()
    if not db["database"] or not db["user"]:
        raise BackupError("Database settings are not configured")
    with destination.open("wb") as handle:
        _run_command(["pg_dump", "-Fc", "-h", db["host"], "-p", db["port"], "-U", db["user"], db["database"]], stdout=handle)


def _create_database_safety_dump(destination: Path) -> None:
    _dump_database(destination)


def _restore_database(archive_path: Path, *, data_only: bool) -> None:
    db = _database_env()
    if not db["database"] or not db["user"]:
        raise BackupError("Database settings are not configured")
    command = [
        "pg_restore",
        "-h",
        db["host"],
        "-p",
        db["port"],
        "-U",
        db["user"],
        "-d",
        db["database"],
        "--exit-on-error",
        "--no-owner",
        "--no-privileges",
        "--single-transaction",
    ]
    if data_only:
        command.extend(["--data-only", "--disable-triggers"])
    else:
        command.extend(["--clean", "--if-exists"])
    command.append(str(archive_path))
    _run_command(command)


def _restore_database_from_safety_dump(archive_path: Path) -> None:
    _restore_database(archive_path, data_only=False)


def _run_migrations() -> None:
    result = subprocess.run([sys.executable, "-m", "prisma", "migrate", "deploy"], cwd="/ragtime", stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise BackupError(result.stderr.decode("utf-8", errors="replace") or "Prisma migrations failed")


def _terminate_other_database_connections() -> None:
    try:
        _run_psql("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()")
    except BackupError:
        logger.warning("Could not terminate concurrent database sessions before restore")


def _invalidate_restored_runtime_sessions() -> None:
    try:
        exists = _run_psql("SELECT to_regclass('public.userspace_runtime_sessions') IS NOT NULL;", capture=True)
        if exists != "t":
            return
        _run_psql(
            """
UPDATE userspace_runtime_sessions
SET state = 'stopped'::"RuntimeSessionState",
    provider_session_id = NULL,
    preview_internal_url = NULL,
    launch_port = NULL,
    last_heartbeat_at = now(),
    last_error = 'Backup restore invalidated active runtime state',
    updated_at = now()
WHERE state IN ('starting'::"RuntimeSessionState", 'running'::"RuntimeSessionState", 'stopping'::"RuntimeSessionState");
"""
        )
    except BackupError:
        logger.warning("Could not invalidate restored runtime sessions")


def _invalidate_restored_workspace_runtime_artifacts() -> None:
    workspaces_root = DATA_DIR / "_userspace" / "workspaces"
    if not workspaces_root.exists():
        return
    for workspace_dir in workspaces_root.iterdir():
        bootstrap_stamp = workspace_dir / "files" / ".ragtime" / ".runtime-bootstrap.done"
        if bootstrap_stamp.exists():
            bootstrap_stamp.unlink()
        rootfs_dir = workspace_dir / "rootfs"
        if rootfs_dir.exists():
            shutil.rmtree(rootfs_dir, ignore_errors=True)


def _normalize_local_admin_username(username: str) -> str:
    return username if username.startswith("local:") else f"local:{username}"


def _select_admin_access_mirror_source(target_username: str) -> str:
    output = _run_psql(
        """
WITH candidate_scores AS (
    SELECT
        u.username,
        COUNT(DISTINCT owned_workspaces.id) AS owned_workspace_count,
        COUNT(DISTINCT workspace_access.workspace_id) AS workspace_access_count,
        COUNT(DISTINCT owned_conversations.id) AS owned_conversation_count,
        COUNT(DISTINCT conversation_access.conversation_id) AS conversation_access_count
    FROM users u
    LEFT JOIN workspaces owned_workspaces ON owned_workspaces.owner_user_id = u.id
    LEFT JOIN workspace_members workspace_access ON workspace_access.user_id = u.id
    LEFT JOIN conversations owned_conversations ON owned_conversations.user_id = u.id
    LEFT JOIN conversation_members conversation_access ON conversation_access.user_id = u.id
    WHERE u.role = 'admin'
      AND u.username <> :'target_username'
      AND u.username NOT LIKE 'local:%'
    GROUP BY u.id, u.username
)
SELECT username
FROM candidate_scores
ORDER BY
    (owned_workspace_count + workspace_access_count + owned_conversation_count + conversation_access_count) DESC,
    owned_conversation_count DESC,
    workspace_access_count DESC,
    username ASC
LIMIT 1;
""",
        variables={"target_username": target_username},
        capture=True,
    )
    lines = output.splitlines()
    return lines[0] if lines else ""


def _mirror_local_admin_access(source: str, target: Optional[str]) -> None:
    target_username = _normalize_local_admin_username(target or settings.local_admin_user)
    _run_psql(
        """
INSERT INTO users (
    id,
    username,
    auth_provider,
    cached_groups,
    display_name,
    role,
    role_manually_set,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid()::text,
    :'target_username',
    'local'::"AuthProvider",
    '[]'::jsonb,
    'Local Admin',
    'admin'::"UserRole",
    false,
    now(),
    now()
)
ON CONFLICT (username) DO UPDATE
SET role = 'admin'::"UserRole",
    updated_at = now();
""",
        variables={"target_username": target_username},
    )

    selected_source = source if source and source != "auto" else _select_admin_access_mirror_source(target_username)
    if not selected_source or selected_source == target_username:
        return

    exists = _run_psql(
        "SELECT 1 FROM users WHERE username = :'source_username' LIMIT 1;",
        variables={"source_username": selected_source},
        capture=True,
    )
    if exists != "1":
        raise BackupError(f"Cannot mirror local admin access: source user not found: {selected_source}")

    _run_psql(
        """
WITH source_user AS (
    SELECT id FROM users WHERE username = :'source_username'
),
target_user AS (
    SELECT id FROM users WHERE username = :'target_username'
),
workspace_source_access AS (
    SELECT w.id AS workspace_id, 'owner'::"WorkspaceRole" AS role
    FROM workspaces w
    JOIN source_user source ON source.id = w.owner_user_id
    UNION ALL
    SELECT wm.workspace_id, wm.role
    FROM workspace_members wm
    JOIN source_user source ON source.id = wm.user_id
),
workspace_ranked_access AS (
    SELECT DISTINCT ON (workspace_id)
        workspace_id,
        role
    FROM workspace_source_access
    ORDER BY workspace_id,
        CASE role
            WHEN 'owner'::"WorkspaceRole" THEN 3
            WHEN 'editor'::"WorkspaceRole" THEN 2
            ELSE 1
        END DESC
),
workspace_upsert AS (
    INSERT INTO workspace_members (id, workspace_id, user_id, role, created_at, updated_at)
    SELECT gen_random_uuid()::text, access.workspace_id, target.id, access.role, now(), now()
    FROM workspace_ranked_access access
    CROSS JOIN target_user target
    ON CONFLICT (workspace_id, user_id) DO UPDATE
    SET role = CASE
            WHEN CASE EXCLUDED.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
               > CASE workspace_members.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
            THEN EXCLUDED.role
            ELSE workspace_members.role
        END,
        updated_at = now()
),
conversation_source_access AS (
    SELECT c.id AS conversation_id, 'owner'::"WorkspaceRole" AS role
    FROM conversations c
    JOIN source_user source ON source.id = c.user_id
    UNION ALL
    SELECT cm.conversation_id, cm.role
    FROM conversation_members cm
    JOIN source_user source ON source.id = cm.user_id
),
conversation_ranked_access AS (
    SELECT DISTINCT ON (conversation_id)
        conversation_id,
        role
    FROM conversation_source_access
    ORDER BY conversation_id,
        CASE role
            WHEN 'owner'::"WorkspaceRole" THEN 3
            WHEN 'editor'::"WorkspaceRole" THEN 2
            ELSE 1
        END DESC
),
conversation_upsert AS (
    INSERT INTO conversation_members (id, conversation_id, user_id, role, created_at, updated_at)
    SELECT gen_random_uuid()::text, access.conversation_id, target.id, access.role, now(), now()
    FROM conversation_ranked_access access
    CROSS JOIN target_user target
    ON CONFLICT (conversation_id, user_id) DO UPDATE
    SET role = CASE
            WHEN CASE EXCLUDED.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
               > CASE conversation_members.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
            THEN EXCLUDED.role
            ELSE conversation_members.role
        END,
        updated_at = now()
),
auth_group_upsert AS (
    INSERT INTO auth_group_memberships (id, user_id, group_id, source_provider, source_synced_at, created_at, updated_at)
    SELECT gen_random_uuid()::text, target.id, membership.group_id, 'local_managed'::"AuthProvider", now(), now(), now()
    FROM auth_group_memberships membership
    JOIN source_user source ON source.id = membership.user_id
    CROSS JOIN target_user target
    ON CONFLICT (user_id, group_id) DO UPDATE
    SET updated_at = now()
)
SELECT 1;
""",
        variables={"source_username": selected_source, "target_username": target_username},
    )


def _confirmation_phrase() -> str:
    override = os.environ.get("RESTORE_CONFIRMATION_PHRASE", "")
    if override:
        return override
    database = _database_env().get("database") or "RAGTIME"
    return f"{RESTORE_CONFIRMATION_PREFIX} {database}"


def _read_password(fd: Optional[int], *, prompt: str) -> str:
    if fd is not None:
        with os.fdopen(os.dup(fd), "rb") as handle:
            password = handle.read().decode("utf-8").rstrip("\r\n")
        if not password:
            raise BackupPasswordRequiredError("Backup password fd did not provide a password")
        return password
    if os.path.exists("/dev/tty") and os.access("/dev/tty", os.R_OK):
        with open("/dev/tty", "r+", encoding="utf-8") as tty:
            return getpass.getpass(prompt, stream=tty)
    raise BackupPasswordRequiredError("A backup password is required and no password fd or /dev/tty is available")


def _read_tty_line(prompt: str) -> str:
    if os.path.exists("/dev/tty") and os.access("/dev/tty", os.R_OK):
        with open("/dev/tty", "r+", encoding="utf-8") as tty:
            tty.write(prompt)
            tty.flush()
            return tty.readline().rstrip("\r\n")
    raise BackupValidationError("Interactive confirmation requires /dev/tty")


@contextmanager
def locked_operation() -> Iterator[None]:
    """Compatibility wrapper which preserves the patchable local lock path."""
    with backup_restore_lock(LOCK_PATH):
        yield


# Compatibility for internal callers; new filesystem users import the public helper.
_locked_operation = locked_operation


def _safe_member_path(name: str) -> Path:
    normalized_name = name
    while normalized_name.startswith("./"):
        normalized_name = normalized_name[2:]
    normalized = PurePosixPath(normalized_name)
    if not normalized.parts:
        raise BackupValidationError("Backup archive contains an empty path")
    if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
        raise BackupValidationError(f"Backup archive contains unsafe path: {name}")
    return Path(*normalized.parts)


def _resolve_link_target(member_name: str, link_name: str) -> None:
    if PurePosixPath(link_name).is_absolute():
        raise BackupValidationError(f"Backup archive contains unsafe link target: {link_name}")
    base = PurePosixPath(member_name).parent
    resolved = base.joinpath(link_name)
    if any(part == ".." for part in resolved.parts):
        raise BackupValidationError(f"Backup archive contains unsafe link target: {link_name}")


def _validate_member(member: tarfile.TarInfo) -> Path:
    path = _safe_member_path(member.name)
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        raise BackupValidationError(f"Backup archive contains unsupported special entry: {member.name}")
    if member.islnk():
        raise BackupValidationError(f"Backup archive contains unsupported hard link entry: {member.name}")
    if member.issym():
        _resolve_link_target(member.name, member.linkname or "")
    return path


def _is_relative_subpath(relative: Path, root_name: str) -> bool:
    parts = relative.parts
    return bool(parts) and parts[0] == root_name


def _should_skip_data_path(relative: Path, *, skip_runtime_history: bool = False) -> bool:
    path = relative.as_posix()
    history_path = relative.parts[:2] == ("_userspace", "_sqlite_history")
    if history_path and len(relative.parts) >= 3 and relative.parts[2] in {"secrets", "cache", "scratch", "transfers"}:
        # These are never generic backup inputs, even when runtime is offline.
        # Portable repository keys enter only the encrypted runtime bundle.
        return True
    if path in {".encryption_key", ".jwt_secret"}:
        return True
    if path.startswith("_tmp"):
        return True
    if _is_relative_subpath(relative, "_server_backups"):
        return True
    # Once runtime history is managed, portable runtime export/import owns its
    # transfer.  Inactive legacy history remains part of generic backups.
    if skip_runtime_history and (
        history_path
        or _is_relative_subpath(relative, "_sqlite_history")
        or (len(relative.parts) >= 4 and relative.parts[:2] == ("_userspace", "workspaces") and "sqlite_backups" in relative.parts[2:])
    ):
        return True
    if path.startswith("_userspace/workspaces/") and "/rootfs" in path:
        return True
    if path.startswith("_userspace/workspaces/") and path.endswith("/.runtime-bootstrap.done"):
        return True
    # Bulk legacy staging is intentionally outside user-visible storage and can
    # be incomplete. Published generations and durable receipts remain backed
    # up so a restart can resume them.
    parts = relative.parts
    if len(parts) >= 5 and parts[:3] == ("_userspace", "_object_storage", "_legacy_imports") and parts[4].startswith("tmp-"):
        return True
    return False


def _write_manifest(tempdir: Path, manifest: BackupManifest) -> None:
    (tempdir / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")


def _collect_deployment_environment_variables() -> dict[str, str]:
    return {name: value for name, value in sorted(os.environ.items()) if _should_include_deployment_environment_name(name)}


def _is_valid_environment_name(name: str) -> bool:
    if not name or not name.isascii():
        return False
    allowed_start = string.ascii_letters + "_"
    allowed_rest = allowed_start + string.digits
    return name[0] in allowed_start and all(character in allowed_rest for character in name[1:])


def _should_include_deployment_environment_name(name: str) -> bool:
    return _is_valid_environment_name(name) and name not in _DEPLOYMENT_ENV_IGNORED_NAMES and not name.startswith(_DEPLOYMENT_ENV_IGNORED_PREFIXES)


def _write_deployment_environment(tempdir: Path, variables: dict[str, str]) -> None:
    payload = {"version": 1, "variables": variables}
    (tempdir / DEPLOYMENT_ENV_NAME).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _build_tarball(
    tempdir: Path,
    archive_path: Path,
    *,
    progress: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 0,
    progress_message: str = "",
) -> int:
    eligible_children = [child for child in _iter_tree_entries(tempdir) if (child.is_symlink() or not child.is_dir()) and child != archive_path]
    reporter = _ItemProgressReporter(
        progress,
        "archive_build_start",
        start=progress_start,
        end=progress_end,
        total=len(eligible_children),
        message=progress_message,
    )
    item_count = 0
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in eligible_children:
            archive.add(child, arcname=child.relative_to(tempdir).as_posix(), recursive=False)
            item_count += 1
            reporter.update(child.relative_to(tempdir).as_posix(), item_count, force=item_count == len(eligible_children))
    return item_count


def _extract_archive(
    archive_path: Path,
    destination_dir: Path,
    *,
    progress: Optional[ProgressCallback] = None,
    progress_phase: str = "archive_extract_start",
    progress_start: int = 0,
    progress_end: int = 0,
    progress_message: str = "",
) -> tuple[BackupManifest, set[str], dict[str, int]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        reporter = _ItemProgressReporter(
            progress,
            progress_phase,
            start=progress_start,
            end=progress_end,
            total=len(members),
            message=progress_message,
        )
        names: set[str] = set()
        member_count = 0
        data_item_count = 0
        for member in members:
            relative = _validate_member(member)
            names.add(relative.as_posix())
            member_count += 1
            if relative.parts and relative.parts[0] in {"data", "faiss"} and not member.isdir():
                data_item_count += 1
            target = destination_dir / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                reporter.update(relative.as_posix(), member_count, force=member_count == len(members))
                continue
            if member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(member.linkname)
                reporter.update(relative.as_posix(), member_count, force=member_count == len(members))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                target.touch()
                reporter.update(relative.as_posix(), member_count, force=member_count == len(members))
                continue
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            reporter.update(relative.as_posix(), member_count, force=member_count == len(members))

    manifest_path = destination_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = BackupManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    else:
        legacy_key = "data/.encryption_key" in names or ".encryption_key" in names
        scope = BackupScope.FULL
        if "database.dump" in names and not any(name.startswith(("data/", "faiss/")) for name in names):
            scope = BackupScope.DATABASE
        elif "database.dump" not in names:
            scope = BackupScope.FILES
        manifest = BackupManifest(
            format="tar.gz",
            version=1,
            created_at="",
            scope=scope,
            ragtime_version="unknown",
            schema_version="unknown",
            encrypted=False,
            includes_managed_key=legacy_key,
            deployment_environment_variables=[],
            legacy_embedded_key=legacy_key,
        )

    legacy_key = "data/.encryption_key" in names or ".encryption_key" in names or "faiss/.encryption_key" in names
    if not manifest.encrypted and legacy_key:
        manifest.legacy_embedded_key = True
        manifest.includes_managed_key = True
    return manifest, names, {"member_count": member_count, "data_item_count": data_item_count}


def _spool_stream(stream: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as handle:
        shutil.copyfileobj(stream, handle)


def _prepare_archive(source: Optional[Path], stream: Optional[BinaryIO], password: Optional[str]):
    tempdir = tempfile.TemporaryDirectory(prefix="ragtime-server-backup-")
    staged_input = Path(tempdir.name) / "input.bin"
    archive_path = Path(tempdir.name) / "archive.tar.gz"
    try:
        if source is not None:
            shutil.copyfile(source, staged_input)
        elif stream is not None:
            _spool_stream(stream, staged_input)
        else:
            raise BackupValidationError("Restore archive is required")

        with staged_input.open("rb") as handle:
            encrypted = is_encrypted_backup(handle)

        if encrypted:
            if password is None:
                password = _read_password(None, prompt="Backup password: ")
            with staged_input.open("rb") as encrypted_source, archive_path.open("wb") as decrypted_destination:
                decrypt_stream(encrypted_source, decrypted_destination, password)
        else:
            shutil.move(staged_input, archive_path)
        return archive_path, tempdir
    except Exception:
        tempdir.cleanup()
        raise


def _snapshot_directory(
    target: Path,
    snapshot_root: Path,
    *,
    exclude_paths: set[Path] | None = None,
    progress: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 0,
    progress_message: str = "",
) -> tuple[Path, int]:
    snapshot_path = snapshot_root / "data-snapshot"
    snapshot_path.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        return snapshot_path, 0
    item_count = _copy_tree_contents(
        target,
        snapshot_path,
        replace=False,
        exclude_paths=exclude_paths,
        progress=progress,
        progress_phase="data_snapshot_start",
        progress_start=progress_start,
        progress_end=progress_end,
        progress_message=progress_message,
    )
    return snapshot_path, item_count


def _restore_snapshot(snapshot_path: Path, target: Path, *, exclude_paths: set[Path] | None = None) -> None:
    excluded = exclude_paths or set()
    if target.exists():
        for child in list(target.iterdir()):
            _remove_restore_path(child, Path(child.name), excluded)
    else:
        target.mkdir(parents=True, exist_ok=True)
    for child in snapshot_path.iterdir():
        destination = target / child.name
        if _is_excluded_restore_path(Path(child.name), excluded):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _copy_tree_contents(
    source_dir: Path,
    destination_dir: Path,
    *,
    replace: bool,
    preserve_paths: set[Path] | None = None,
    preserve_runtime_history: bool = False,
    exclude_paths: set[Path] | None = None,
    progress: Optional[ProgressCallback] = None,
    progress_phase: str = "files_restore_start",
    progress_start: int = 0,
    progress_end: int = 0,
    progress_message: str = "",
) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    all_children = _iter_tree_entries(source_dir)
    total_items = sum(1 for child in all_children if not child.is_dir() or child.is_symlink())
    reporter = _ItemProgressReporter(
        progress,
        progress_phase,
        start=progress_start,
        end=progress_end,
        total=total_items,
        message=progress_message,
    )
    item_count = 0
    preserved = (preserve_paths or set()) | (exclude_paths or set())
    if replace:
        for child in list(destination_dir.iterdir()):
            _remove_restore_path(child, Path(child.name), preserved, preserve_runtime_history)
    for child in all_children:
        relative = child.relative_to(source_dir)
        if _is_excluded_restore_path(relative, preserved) or (preserve_runtime_history and _is_runtime_history_path(relative)):
            continue
        target = destination_dir / relative
        if child.is_dir() and not child.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.readlink(child))
            item_count += 1
            reporter.update(relative.as_posix(), item_count, force=item_count == total_items)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            item_count += 1
            reporter.update(relative.as_posix(), item_count, force=item_count == total_items)
    return item_count


def _is_runtime_history_path(relative: Path) -> bool:
    parts = relative.parts
    return (len(parts) >= 2 and parts[:2] == ("_userspace", "_sqlite_history")) or (
        len(parts) >= 4 and parts[:2] == ("_userspace", "workspaces") and parts[3] == "sqlite_backups"
    )


def _is_excluded_restore_path(relative: Path, excluded: set[Path]) -> bool:
    # Probe the (short) ancestor chain against the set instead of scanning every
    # managed workspace exclusion for every restored file.
    return any(candidate in excluded for candidate in (relative, *relative.parents))


def _remove_restore_path(path: Path, relative: Path, excluded: set[Path], preserve_runtime_history: bool = False) -> None:
    if _is_excluded_restore_path(relative, excluded) or (preserve_runtime_history and _is_runtime_history_path(relative)):
        return
    if not path.is_dir() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return
    for child in list(path.iterdir()):
        child_relative = relative / child.name
        _remove_restore_path(child, child_relative, excluded, preserve_runtime_history)
    if not any(path.iterdir()):
        path.rmdir()


def _copy_backup_data_tree(
    source_dir: Path,
    destination_dir: Path,
    *,
    progress: Optional[ProgressCallback] = None,
    progress_start: int = 0,
    progress_end: int = 0,
    progress_message: str = "",
    skip_runtime_history: bool = False,
) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    all_children = [
        child
        for child in _iter_tree_entries(source_dir)
        if not _should_skip_data_path(child.relative_to(source_dir), skip_runtime_history=skip_runtime_history)
    ]
    total_items = sum(1 for child in all_children if not child.is_dir() or child.is_symlink())
    reporter = _ItemProgressReporter(
        progress,
        "files_collect_start",
        start=progress_start,
        end=progress_end,
        total=total_items,
        message=progress_message,
    )
    item_count = 0
    for child in all_children:
        relative = child.relative_to(source_dir)
        target = destination_dir / relative
        if child.is_dir() and not child.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.readlink(child))
            item_count += 1
            reporter.update(relative.as_posix(), item_count, force=item_count == total_items)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            item_count += 1
            reporter.update(relative.as_posix(), item_count, force=item_count == total_items)
    return item_count


def _runtime_history_is_managed() -> bool:
    history = DATA_DIR / "_userspace" / "_sqlite_history"
    return history.is_dir() and not history.is_symlink() and (history / "restic").is_dir()


def _runtime_history_restore_exclusions() -> set[Path]:
    """Return the actual runtime-owned paths which generic restore must not touch."""
    if not _runtime_history_is_managed():
        return set()
    root = DATA_DIR / "_userspace"
    exclusions = {Path("_userspace") / "_sqlite_history"}
    workspaces = root / "workspaces"
    if workspaces.is_dir() and not workspaces.is_symlink():
        for workspace in workspaces.iterdir():
            history = workspace / "sqlite_backups"
            if history.is_dir() and not history.is_symlink():
                exclusions.add(Path("_userspace") / "workspaces" / workspace.name / "sqlite_backups")
    return exclusions


def _object_storage_root() -> Path:
    return Path(os.environ.get("STORAGE_ROOT", str(DATA_DIR / "_userspace" / "_object_storage")))


def _object_storage_has_state() -> bool:
    root = _object_storage_root()
    try:
        return root.is_dir() and any(root.iterdir())
    except OSError:
        return False


def _object_storage_control_request(path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    key = settings.encryption_key.strip().encode("utf-8")
    if not key:
        raise BackupError("Object storage consistency check requires the managed encryption key")
    token = hmac.new(key, b"ragtime-object-storage-control-v1", hashlib.sha256).hexdigest()
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    endpoint = os.environ.get("OBJECT_STORAGE_CONTROL_URL", "http://runtime-s3:9001").rstrip("/")
    request = Request(
        f"{endpoint}{path}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=15) as response:  # nosec B310: internal Compose control plane
            decoded = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise BackupError("Object storage is initialized but could not be quiesced for a consistent backup") from exc
    if not isinstance(decoded, dict):
        raise BackupError("Object storage returned an invalid backup consistency response")
    return decoded


@contextmanager
def _object_storage_backup_lease() -> Iterator[None]:
    if not _object_storage_has_state():
        yield
        return
    prepared = _object_storage_control_request("/v1/backup/prepare")
    lease_id = prepared.get("lease_id")
    if prepared.get("consistent") is not True or not isinstance(lease_id, str) or not lease_id:
        raise BackupError("Object storage could not provide a consistent backup lease")
    try:
        yield
    finally:
        try:
            _object_storage_control_request("/v1/backup/release", {"lease_id": lease_id})
        except BackupError:
            logger.error("Object storage backup lease %s could not be released", lease_id)


_RUNTIME_HISTORY_BUNDLE_PATH = Path("runtime-sqlite-history") / "export.bundle"


def _is_repository_identity(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in string.hexdigits.lower() for character in value)


def _runtime_history_endpoint() -> tuple[str, str] | None:
    """Return the authenticated manager endpoint only when it is configured."""
    token = (getattr(settings, "userspace_runtime_auth_token", "") or "").strip()
    base_url = (getattr(settings, "userspace_runtime_manager_url", "") or "").strip().rstrip("/")
    return (base_url, token) if base_url and token else None


def _runtime_history_request(path: str, *, method: str = "GET", body: bytes | None = None, allow_not_found: bool = False):
    endpoint = _runtime_history_endpoint()
    if endpoint is None:
        return None
    base_url, token = endpoint
    request = Request(
        f"{base_url}/sqlite-history{path}",
        data=body,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        return urlopen(request, timeout=30)  # nosec B310: authenticated internal runtime manager
    except HTTPError as exc:
        if allow_not_found and exc.code == 404:
            return None
        raise BackupError("Runtime SQLite history status could not be determined") from exc
    except Exception as exc:
        # Transport errors and 503 responses are ambiguous.  Never substitute
        # a generic data-tree copy for an unavailable active history backend.
        raise BackupError("Runtime SQLite history status could not be determined") from exc


def _runtime_history_status() -> dict[str, object] | None:
    observed_activation = DATA_DIR / "_userspace" / "sqlite-history-runtime-activation-v2.json"
    activation_response = _runtime_history_request("/activation", allow_not_found=True)
    if activation_response is None:
        if os.path.lexists(observed_activation):
            raise BackupError("Runtime SQLite history must be available after activation")
        return None
    with activation_response:
        try:
            activation = json.loads(activation_response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError("Runtime SQLite history returned invalid status metadata") from exc
    if not isinstance(activation, dict) or not isinstance(activation.get("active"), bool) or not isinstance(activation.get("capability"), bool):
        raise BackupError("Runtime SQLite history returned invalid status metadata")
    # An old or deliberately inactive runtime has no history to export.  Once
    # it advertises the v2 capability, every status/export failure is fatal.
    if not activation["active"]:
        if os.path.lexists(observed_activation):
            raise BackupError("Runtime SQLite history activation cannot be downgraded")
        return None
    response = _runtime_history_request("/exports/status")
    if response is None:  # Defensive: status is never allowed to 404 here.
        raise BackupError("Runtime SQLite history status is unavailable")
    with response:
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError("Runtime SQLite history returned invalid status metadata") from exc
    if not isinstance(payload, dict) or payload.get("active") is not True:
        raise BackupError("Runtime SQLite history is active but export status is invalid")
    return payload if payload["active"] else None


def _runtime_history_export(destination: Path, *, include_repository_key: bool) -> dict[str, object] | None:
    """Stage an immutable runtime-created export without opening its repository."""
    status = _runtime_history_status()
    if status is None:
        return None
    response = _runtime_history_request("/exports", method="POST", body=json.dumps({"include_repository_key": include_repository_key}).encode("utf-8"))
    if response is None:
        raise BackupError("Runtime SQLite history became unavailable while preparing export")
    with response:
        try:
            receipt = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError("Runtime SQLite history returned an invalid export receipt") from exc
    if not isinstance(receipt, dict) or not isinstance(receipt.get("export_id"), str) or not receipt["export_id"]:
        raise BackupError("Runtime SQLite history returned an invalid export receipt")
    export_id = receipt["export_id"]
    # Runtime must not respond with a downloadable path until its private
    # repository/catalog barrier has completed.  A bounded poll preserves the
    # existing backup-job timeout behavior while avoiding a live-file copy.
    deadline = time.monotonic() + 3600
    while receipt.get("status") not in {"completed", "failed"}:
        if time.monotonic() >= deadline:
            raise BackupError("Runtime SQLite history export did not complete in time")
        time.sleep(1)
        response = _runtime_history_request(f"/exports/{export_id}")
        if response is None:
            raise BackupError("Runtime SQLite history became unavailable while exporting")
        with response:
            try:
                receipt = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackupError("Runtime SQLite history returned an invalid export receipt") from exc
    if receipt.get("status") != "completed":
        raise BackupError("Runtime SQLite history export failed")
    repository_id = receipt.get("repository_id")
    if not _is_repository_identity(repository_id):
        raise BackupError("Runtime SQLite history export lacks repository identity")
    includes_key = receipt.get("includes_repository_key")
    if includes_key is not include_repository_key:
        raise BackupError("Runtime SQLite history export key inclusion did not match the requested backup")
    response = _runtime_history_request(f"/exports/{export_id}/download")
    if response is None:
        raise BackupError("Runtime SQLite history export download is unavailable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = receipt.get("bundle_size_bytes")
    expected_digest = receipt.get("bundle_sha256")
    if (expected_size is None) != (expected_digest is None):
        raise BackupError("Runtime SQLite history export has incomplete bundle integrity metadata")
    if expected_size is not None and (
        not isinstance(expected_size, int) or expected_size <= 0 or not isinstance(expected_digest, str) or len(expected_digest) != 64
    ):
        raise BackupError("Runtime SQLite history export has invalid bundle integrity metadata")
    digest = hashlib.sha256()
    downloaded = 0
    with response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            downloaded += len(chunk)
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if expected_size is not None:
        assert isinstance(expected_digest, str)
        if downloaded != expected_size or not hmac.compare_digest(digest.hexdigest(), expected_digest):
            destination.unlink(missing_ok=True)
            raise BackupError("Runtime SQLite history export bundle integrity verification failed")
    return {
        "version": int(receipt.get("export_version", 1)),
        "repository_id": repository_id,
        "includes_repository_key": includes_key,
        "requires_original_repository_key": not includes_key,
        "bundle": _RUNTIME_HISTORY_BUNDLE_PATH.as_posix(),
        **({"bundle_size_bytes": expected_size, "bundle_sha256": expected_digest} if expected_size is not None else {}),
    }


def _runtime_history_import(bundle: Path, metadata: dict[str, object]) -> None:
    """Ask runtime to validate and atomically activate an immutable bundle."""
    if bundle.is_symlink() or not bundle.is_file():
        raise BackupValidationError("Backup archive contains an invalid runtime SQLite history bundle")
    endpoint = _runtime_history_endpoint()
    if endpoint is None:
        raise BackupValidationError("Runtime SQLite history is required to restore this backup")
    base_url, token = endpoint
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BackupValidationError("Runtime SQLite history endpoint is invalid")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=600)
    path = f"{parsed.path.rstrip('/')}/sqlite-history/imports" or "/sqlite-history/imports"
    try:
        connection.putrequest("POST", path)
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(bundle.stat().st_size))
        connection.putheader("X-Ragtime-History-Metadata", json.dumps(metadata, separators=(",", ":")))
        connection.endheaders()
        with bundle.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        if response.status >= 400:
            raise OSError(f"runtime returned {response.status}")
        receipt = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise BackupValidationError("Runtime SQLite history import could not be accepted") from exc
    finally:
        connection.close()
    if not isinstance(receipt, dict) or not isinstance(receipt.get("import_id"), str) or not receipt["import_id"]:
        raise BackupValidationError("Runtime SQLite history import returned an invalid receipt")
    import_id = receipt["import_id"]
    deadline = time.monotonic() + 3600
    while receipt.get("status") not in {"completed", "failed"}:
        if time.monotonic() >= deadline:
            raise BackupValidationError("Runtime SQLite history import did not complete in time")
        time.sleep(1)
        try:
            response = _runtime_history_request(f"/imports/{import_id}")
        except BackupError as exc:
            raise BackupValidationError("Runtime SQLite history import status is unavailable") from exc
        if response is None:
            raise BackupValidationError("Runtime SQLite history import status is unavailable")
        with response:
            try:
                receipt = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackupValidationError("Runtime SQLite history import returned an invalid receipt") from exc
    if receipt.get("status") != "completed":
        raise BackupValidationError("Runtime SQLite history import did not validate and activate")


def _stream_sha256(path: Path) -> tuple[int, str]:
    """Hash a potentially large bundle without materializing it in memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _runtime_history_bundle_from_archive(extract_dir: Path, manifest: BackupManifest, restore_scope: BackupScope) -> tuple[Path, dict[str, object]] | None:
    metadata = manifest.sqlite_history
    if metadata is None:
        return None  # Legacy formats predate runtime-owned history exports.
    included = metadata.get("included", True)
    if included is False:
        return None
    if restore_scope == BackupScope.DATABASE:
        raise BackupValidationError("Database-only restore cannot activate runtime SQLite history")
    repository_id = metadata.get("repository_id")
    bundle_name = metadata.get("bundle")
    if not _is_repository_identity(repository_id):
        raise BackupValidationError("Backup archive contains invalid runtime SQLite history metadata")
    if bundle_name != _RUNTIME_HISTORY_BUNDLE_PATH.as_posix():
        raise BackupValidationError("Backup archive contains invalid runtime SQLite history bundle reference")
    bundle = extract_dir / _RUNTIME_HISTORY_BUNDLE_PATH
    if bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size == 0:
        raise BackupValidationError("Backup archive does not contain a valid runtime SQLite history bundle")
    expected_size = metadata.get("bundle_size_bytes")
    expected_digest = metadata.get("bundle_sha256")
    if (expected_size is None) != (expected_digest is None):
        raise BackupValidationError("Backup archive contains incomplete runtime SQLite history bundle integrity metadata")
    if expected_size is not None:
        if not isinstance(expected_size, int) or expected_size <= 0 or not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise BackupValidationError("Backup archive contains invalid runtime SQLite history bundle integrity metadata")
        actual_size, digest = _stream_sha256(bundle)
        if actual_size != expected_size or not hmac.compare_digest(digest, expected_digest):
            raise BackupValidationError("Backup archive runtime SQLite history bundle integrity verification failed")
    return bundle, metadata


def _preflight_runtime_history_import(metadata: dict[str, object]) -> None:
    """Reject unavailable/keyless runtime history before app mutation begins."""
    if _runtime_history_endpoint() is None:
        raise BackupValidationError("Runtime SQLite history is required to restore this backup")
    if metadata.get("includes_repository_key"):
        return
    status = _runtime_history_status()
    if status is None or status.get("repository_id") != metadata.get("repository_id"):
        raise BackupValidationError("Keyless runtime SQLite history restore requires the original active repository key")


def _resolve_data_source(extract_dir: Path) -> Optional[Path]:
    if (extract_dir / "data").exists():
        return extract_dir / "data"
    if (extract_dir / "faiss").exists():
        return extract_dir / "faiss"
    return None


def _parse_deployment_environment_payload(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise BackupValidationError("Deployment environment payload is invalid")
    version = payload.get("version")
    if version != 1:
        raise BackupValidationError("Deployment environment payload version is invalid")
    variables = payload.get("variables")
    if not isinstance(variables, dict):
        raise BackupValidationError("Deployment environment variables are invalid")
    parsed: dict[str, str] = {}
    for name, value in variables.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise BackupValidationError("Deployment environment variables are invalid")
        if not _is_valid_environment_name(name):
            raise BackupValidationError("Deployment environment variable has an invalid name")
        parsed[name] = value
    return {name: parsed[name] for name in sorted(parsed)}


def _load_deployment_environment_payload(payload_bytes: bytes) -> dict[str, str]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupValidationError("Backup archive contains an invalid deployment environment payload") from exc
    return _parse_deployment_environment_payload(payload)


def _managed_key_source_path(extract_dir: Path) -> Optional[Path]:
    for candidate in (extract_dir / "data" / ".encryption_key", extract_dir / ".encryption_key", extract_dir / "faiss" / ".encryption_key"):
        if candidate.exists():
            return candidate
    return None


_PRESERVABLE_DATA_KEY_NAMES = (".encryption_key", ".jwt_secret")


def _validate_regular_key(path: Path, *, location: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise BackupValidationError(f"Backup archive contains an invalid {location} key")


def _archive_data_key_paths(data_source: Path) -> dict[str, Path]:
    key_paths: dict[str, Path] = {}
    for name in _PRESERVABLE_DATA_KEY_NAMES:
        path = data_source / name
        if path.exists() or path.is_symlink():
            _validate_regular_key(path, location="archived")
            key_paths[name] = path
    return key_paths


def _preserved_destination_key_paths(archived_keys: dict[str, Path]) -> set[Path]:
    preserved: set[Path] = set()
    for name in _PRESERVABLE_DATA_KEY_NAMES:
        if name in archived_keys:
            continue
        path = DATA_DIR / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise BackupValidationError("Backup restore cannot preserve an unsafe existing key path")
            preserved.add(Path(name))
    return preserved


def _directory_has_entries(path: Optional[Path]) -> bool:
    try:
        return path is not None and path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _validate_files_restore_key_policy(
    *,
    manifest: BackupManifest,
    data_source: Path,
    restored_storage: Optional[Path],
    replace_data: bool,
) -> set[Path]:
    archived_keys = _archive_data_key_paths(data_source)
    archived_managed_key = archived_keys.get(".encryption_key")
    if manifest.includes_managed_key and archived_managed_key is None:
        raise BackupValidationError("Backup archive does not contain a managed encryption key")
    if archived_managed_key is not None and not ((manifest.encrypted and manifest.includes_managed_key) or manifest.legacy_embedded_key):
        raise BackupValidationError("Backup archive managed encryption key is not validated")

    preserved = _preserved_destination_key_paths(archived_keys) if replace_data else set()
    destination_key = DATA_DIR / ".encryption_key"
    has_authoritative_key = archived_managed_key is not None or Path(".encryption_key") in preserved
    if not has_authoritative_key and destination_key.exists() and not destination_key.is_symlink():
        has_authoritative_key = destination_key.is_file() and destination_key.stat().st_size > 0
    if _directory_has_entries(restored_storage) and not has_authoritative_key:
        raise BackupValidationError("Keyless initialized object storage restore requires an authoritative key")
    return preserved


def _install_database_only_managed_key(extract_dir: Path) -> None:
    source = _managed_key_source_path(extract_dir)
    if source is None:
        raise BackupValidationError("Backup archive does not contain a managed encryption key")
    if source.is_symlink() or not source.is_file():
        raise BackupValidationError("Backup archive contains an invalid managed encryption key")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / ".encryption_key"
    with tempfile.NamedTemporaryFile(dir=DATA_DIR, prefix=".encryption_key.", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copyfile(source, temp_path)
        temp_path.chmod(0o600)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def create_backup(options: BackupOptions, progress: Optional[ProgressCallback] = None) -> BackupManifest:
    password = _read_password(options.password_fd, prompt="Backup password: ") if options.encrypt else None
    storage_lease = _object_storage_backup_lease if options.scope in {BackupScope.FULL, BackupScope.FILES} else nullcontext
    with _locked_operation(), storage_lease(), tempfile.TemporaryDirectory(prefix="ragtime-server-backup-build-") as tmpdir:
        root = Path(tmpdir)
        _emit_progress(progress, "start", progress=5, message="Preparing backup", scope=options.scope.value, encrypted=options.encrypt)

        if options.scope in {BackupScope.FULL, BackupScope.DATABASE}:
            _emit_progress(progress, "database_dump_start", progress=15, message="Dumping database")
            _dump_database(root / "database.dump")
            _emit_progress(progress, "database_dump_complete", progress=25, message="Database dump complete", item_count=1)

        sqlite_history: dict[str, object] | None = None
        skip_runtime_history = False
        if options.scope in {BackupScope.FULL, BackupScope.FILES}:
            if _runtime_history_status() is not None:
                _emit_progress(progress, "sqlite_history_export_start", progress=30, message="Staging runtime SQLite history export")
                sqlite_history = _runtime_history_export(
                    root / _RUNTIME_HISTORY_BUNDLE_PATH,
                    include_repository_key=options.encrypt,
                )
                if sqlite_history is None:
                    raise BackupError("Runtime SQLite history became unavailable while preparing export")
                skip_runtime_history = sqlite_history is not None
                _emit_progress(progress, "sqlite_history_export_complete", progress=34, message="Runtime SQLite history export staged")
        elif _runtime_history_status() is not None:
            # Database-only artifacts deliberately do not carry history bytes.
            sqlite_history = {"version": 1, "included": False, "reason": "database_scope"}

        includes_managed_key = options.encrypt and ENCRYPTION_KEY_FILE.exists()
        copied_data_items = 0
        if options.scope in {BackupScope.FULL, BackupScope.FILES} or includes_managed_key:
            data_root = root / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            if DATA_DIR.exists() and options.scope in {BackupScope.FULL, BackupScope.FILES}:
                files_collect_message = "Copying data files"
                _emit_progress(progress, "files_collect_start", progress=35, message=files_collect_message)
                copied_data_items = _copy_backup_data_tree(
                    DATA_DIR,
                    data_root,
                    progress=progress,
                    progress_start=35,
                    progress_end=55,
                    progress_message=files_collect_message,
                    skip_runtime_history=skip_runtime_history,
                )
                _emit_progress(
                    progress,
                    "files_collect_complete",
                    progress=55,
                    message=f"Copied {copied_data_items} data item{'s' if copied_data_items != 1 else ''}",
                    item_count=copied_data_items,
                )
            if includes_managed_key:
                shutil.copy2(ENCRYPTION_KEY_FILE, data_root / ".encryption_key")

        deployment_environment_variables = []
        if options.encrypt and options.scope == BackupScope.FULL:
            deployment_environment = _collect_deployment_environment_variables()
            deployment_environment_variables = sorted(deployment_environment)
            _write_deployment_environment(root, deployment_environment)

        manifest = BackupManifest(
            format="ragbak" if options.encrypt else "tar.gz",
            version=2 if sqlite_history is not None else 1,
            created_at=datetime.now(timezone.utc).isoformat(),
            scope=options.scope,
            ragtime_version=_get_ragtime_version(),
            schema_version=_get_current_schema_version(),
            encrypted=options.encrypt,
            includes_managed_key=includes_managed_key,
            deployment_environment_variables=deployment_environment_variables,
            legacy_embedded_key=False,
            sqlite_history=sqlite_history,
        )
        _emit_progress(progress, "manifest_write", progress=65, message="Backup manifest written")
        _write_manifest(root, manifest)

        archive_path = root / "backup.tar.gz"
        archive_build_message = "Compressing staged database dump, data files, and manifest into a gzip-compressed tar archive"
        _emit_progress(progress, "archive_build_start", progress=75, message=archive_build_message)
        archive_item_count = _build_tarball(
            root,
            archive_path,
            progress=progress,
            progress_start=75,
            progress_end=85,
            progress_message=archive_build_message,
        )
        _emit_progress(progress, "archive_build_complete", progress=85, message="Backup archive built", item_count=archive_item_count)

        destination: BinaryIO
        close_destination = False
        if options.output_stream is not None:
            destination = options.output_stream
        elif options.output_path is not None:
            destination = options.output_path.open("wb")
            close_destination = True
        else:
            destination = sys.stdout.buffer

        try:
            with archive_path.open("rb") as source_handle:
                _emit_progress(progress, "output_write_start", progress=90, message="Writing backup artifact")
                if options.encrypt:
                    encrypt_stream(source_handle, destination, password or "")
                else:
                    shutil.copyfileobj(source_handle, destination)
                _emit_progress(progress, "output_write_complete", progress=95, message="Backup artifact written")
        finally:
            if close_destination:
                destination.close()

        _emit_progress(progress, "complete", progress=100, message="Backup ready for download", format=manifest.format)
        return manifest


def inspect_backup(path, password: Optional[str] = None, progress: Optional[ProgressCallback] = None) -> BackupManifest:
    try:
        _emit_progress(progress, "archive_prepare", progress=5, message="Preparing restore archive")
        archive_path, tempdir = _prepare_archive(Path(path), None, password)
        try:
            extract_dir = Path(tempdir.name) / "inspect"
            extract_dir.mkdir()
            archive_extract_message = "Extracting restore archive"
            _emit_progress(progress, "archive_extract_start", progress=15, message=archive_extract_message)
            manifest, _, stats = _extract_archive(
                archive_path,
                extract_dir,
                progress=progress,
                progress_phase="archive_extract_start",
                progress_start=15,
                progress_end=20,
                progress_message=archive_extract_message,
            )
            _emit_progress(
                progress,
                "archive_extract_complete",
                progress=20,
                message="Restore archive extracted",
                item_count=stats["member_count"],
                data_item_count=stats["data_item_count"],
            )
            _emit_progress(progress, "validation_complete", progress=40, message="Restore validated; confirmation required")
            return manifest
        finally:
            tempdir.cleanup()
    except (BackupCryptoError, tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise BackupValidationError(str(exc)) from exc


def recover_deployment_environment(path: Path, password: str) -> DeploymentEnvironmentRecovery:
    archive_path, tempdir = _prepare_archive(path, None, password)
    try:
        deployment_payload: Optional[dict[str, str]] = None
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                relative = _validate_member(member)
                if relative.as_posix() != DEPLOYMENT_ENV_NAME:
                    continue
                if deployment_payload is not None:
                    raise BackupValidationError("Backup archive contains duplicate deployment environment payloads")
                if not member.isreg():
                    raise BackupValidationError("Backup archive contains an invalid deployment environment payload")
                if member.size > _DEPLOYMENT_ENV_MAX_BYTES:
                    raise BackupValidationError("Deployment environment payload exceeds the maximum supported size")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BackupValidationError("Backup archive contains an invalid deployment environment payload")
                with extracted:
                    payload_bytes = extracted.read(_DEPLOYMENT_ENV_MAX_BYTES + 1)
                if len(payload_bytes) > _DEPLOYMENT_ENV_MAX_BYTES:
                    raise BackupValidationError("Deployment environment payload exceeds the maximum supported size")
                deployment_payload = _load_deployment_environment_payload(payload_bytes)
        if deployment_payload is None:
            raise BackupValidationError("Backup archive does not contain a deployment environment payload")
        variable_names = sorted(deployment_payload)
        warnings = []
        if "DATABASE_URL" in deployment_payload:
            warnings.append("Recovered deployment environment includes DATABASE_URL; verify it before applying to another deployment.")
        if "POSTGRES_PASSWORD" in deployment_payload:
            warnings.append("Recovered deployment environment includes POSTGRES_PASSWORD; verify it before applying to another deployment.")
        return DeploymentEnvironmentRecovery(variables=deployment_payload, variable_names=variable_names, warnings=warnings)
    except (BackupCryptoError, tarfile.TarError, OSError) as exc:
        raise BackupValidationError(str(exc)) from exc
    finally:
        tempdir.cleanup()


def restore_backup(options: RestoreOptions, progress: Optional[ProgressCallback] = None) -> BackupManifest:
    password = _read_password(options.password_fd, prompt="Backup password: ") if options.password_fd is not None else None
    try:
        _emit_progress(progress, "archive_prepare", progress=41, message="Preparing restore archive")
        archive_path, tempdir = _prepare_archive(options.archive_path, options.archive_stream, password)
        extract_dir = Path(tempdir.name) / "extract"
        extract_dir.mkdir()
        archive_extract_message = "Extracting restore archive"
        _emit_progress(progress, "archive_extract_start", progress=42, message=archive_extract_message)
        manifest, names, extract_stats = _extract_archive(
            archive_path,
            extract_dir,
            progress=progress,
            progress_phase="archive_extract_start",
            progress_start=42,
            progress_end=43,
            progress_message=archive_extract_message,
        )
        _emit_progress(
            progress,
            "archive_extract_complete",
            progress=44,
            message="Restore archive extracted",
            item_count=extract_stats["member_count"],
            data_item_count=extract_stats["data_item_count"],
        )
        restore_scope = options.scope_override or manifest.scope
        runtime_history_bundle = _runtime_history_bundle_from_archive(extract_dir, manifest, restore_scope)
        if runtime_history_bundle is not None:
            _preflight_runtime_history_import(runtime_history_bundle[1])
        data_source = _resolve_data_source(extract_dir)
        restored_storage = data_source / "_userspace" / "_object_storage" if data_source is not None else None
        destination_storage = _object_storage_root()
        archive_includes_storage = restored_storage is not None and restored_storage.exists()
        destination_has_storage = destination_storage.exists()
        if (
            (restore_scope in {BackupScope.FULL, BackupScope.FILES} and (archive_includes_storage or destination_has_storage))
            or (restore_scope == BackupScope.DATABASE and manifest.includes_managed_key and destination_has_storage)
        ) and os.environ.get("OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED") != "true":
            raise BackupValidationError(
                "Restoring object storage requires the gateway to be stopped; set OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED=true only after it is offline"
            )
        if manifest.legacy_embedded_key and not options.acknowledge_legacy_key:
            raise LegacyBackupKeyConfirmationRequiredError("Legacy plaintext backups containing an embedded key require explicit acknowledgement")
        expected_confirmation = _confirmation_phrase()
        confirmation = (
            options.restore_confirmation if options.restore_confirmation is not None else _read_tty_line(f"Type '{expected_confirmation}' to continue: ")
        )
        if confirmation != expected_confirmation:
            raise BackupValidationError(f"Restore confirmation did not match {expected_confirmation!r}")
        if restore_scope in {BackupScope.FULL, BackupScope.DATABASE} and "database.dump" not in names:
            raise BackupValidationError("Backup archive does not contain a database dump")
        if restore_scope in {BackupScope.FULL, BackupScope.FILES} and data_source is None:
            raise BackupValidationError("Backup archive does not contain data files")
        preserved_data_keys: set[Path] = set()
        if restore_scope in {BackupScope.FULL, BackupScope.FILES}:
            if data_source is None:
                raise BackupValidationError("Backup archive does not contain data files")
            preserved_data_keys = _validate_files_restore_key_policy(
                manifest=manifest,
                data_source=data_source,
                restored_storage=restored_storage,
                replace_data=options.replace_data,
            )
        _emit_progress(progress, "validation_complete", progress=44, message="Restore validated; confirmation required", scope=restore_scope.value)
    except (BackupCryptoError, tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise BackupValidationError(str(exc)) from exc

    snapshot_path: Optional[Path] = None
    database_safety_dump_path: Optional[Path] = None
    db_mutated = False
    runtime_history_exclusions = _runtime_history_restore_exclusions()
    try:
        with _locked_operation():
            try:
                _emit_progress(progress, "start", progress=45, message="Starting restore", scope=restore_scope.value)
                if restore_scope in {BackupScope.FULL, BackupScope.FILES}:
                    snapshot_message = "Creating data safety snapshot"
                    _emit_progress(progress, "data_snapshot_start", progress=48, message=snapshot_message)
                    snapshot_path, snapshot_count = _snapshot_directory(
                        DATA_DIR,
                        Path(tempdir.name),
                        exclude_paths=runtime_history_exclusions,
                        progress=progress,
                        progress_start=48,
                        progress_end=50,
                        progress_message=snapshot_message,
                    )
                    _emit_progress(progress, "data_snapshot_complete", progress=50, message="Created data safety snapshot", item_count=snapshot_count)
                if restore_scope in {BackupScope.FULL, BackupScope.DATABASE}:
                    database_safety_dump_path = Path(tempdir.name) / "database-safety.dump"
                    _emit_progress(progress, "database_safety_dump_start", progress=55, message="Creating database safety dump")
                    _create_database_safety_dump(database_safety_dump_path)
                    _emit_progress(progress, "database_safety_dump_complete", progress=60, message="Created database safety dump", item_count=1)

                if restore_scope in {BackupScope.FULL, BackupScope.DATABASE}:
                    _emit_progress(progress, "database_restore_start", progress=65, message="Restoring database")
                    _terminate_other_database_connections()
                    _restore_database(extract_dir / "database.dump", data_only=options.pg_data_only)
                    _emit_progress(progress, "database_restore_complete", progress=70, message="Database restore complete")
                    db_mutated = True
                    if not options.skip_migrations and not options.pg_data_only:
                        _emit_progress(progress, "database_migrations_start", progress=75, message="Applying database migrations")
                        _run_migrations()
                        _emit_progress(progress, "database_migrations_complete", progress=80, message="Database migrations applied")
                    if options.mirror_local_admin_access:
                        _mirror_local_admin_access(options.mirror_local_admin_from, options.local_admin_username)
                        _emit_progress(progress, "admin_access_mirror", progress=85, message="Local admin access mirrored")
                    _invalidate_restored_runtime_sessions()
                    _emit_progress(progress, "runtime_sessions_invalidate", progress=88, message="Restored runtime sessions invalidated")
                    if restore_scope == BackupScope.DATABASE and manifest.includes_managed_key:
                        _install_database_only_managed_key(extract_dir)

                if restore_scope in {BackupScope.FULL, BackupScope.FILES}:
                    files_restore_message = "Restoring data files"
                    _emit_progress(progress, "files_restore_start", progress=90, message=files_restore_message)
                    if data_source is None:
                        raise BackupValidationError("Backup archive does not contain data files")
                    restored_items = _copy_tree_contents(
                        data_source,
                        DATA_DIR,
                        replace=options.replace_data,
                        preserve_paths=preserved_data_keys,
                        exclude_paths=runtime_history_exclusions,
                        progress=progress,
                        progress_phase="files_restore_start",
                        progress_start=90,
                        progress_end=95,
                        progress_message=files_restore_message,
                        preserve_runtime_history=False,
                    )
                    _emit_progress(
                        progress,
                        "files_restore_complete",
                        progress=95,
                        message=f"Restored {restored_items} data item{'s' if restored_items != 1 else ''}",
                        item_count=restored_items,
                    )
                    _invalidate_restored_workspace_runtime_artifacts()
                    _emit_progress(progress, "workspace_runtime_invalidate", progress=97, message="Workspace runtime artifacts invalidated")
                    key_path = DATA_DIR / ".encryption_key"
                    if key_path.exists():
                        key_path.chmod(0o600)

                if runtime_history_bundle is not None:
                    _emit_progress(progress, "sqlite_history_import_start", progress=98, message="Validating and activating runtime SQLite history")
                    _runtime_history_import(*runtime_history_bundle)
                    _emit_progress(progress, "sqlite_history_import_complete", progress=99, message="Runtime SQLite history activated")

                _emit_progress(progress, "complete", progress=100, message="Restore completed", scope=restore_scope.value)
                if options.scope_override is not None:
                    manifest.scope = options.scope_override
                return manifest
            except Exception as exc:
                rollback_errors: list[str] = []
                if snapshot_path is not None:
                    try:
                        _restore_snapshot(snapshot_path, DATA_DIR, exclude_paths=runtime_history_exclusions)
                    except Exception as rollback_exc:
                        logger.error("Data rollback failed after restore error: %s", rollback_exc)
                        rollback_errors.append(f"data rollback failed: {rollback_exc}")
                if db_mutated and database_safety_dump_path is not None and database_safety_dump_path.exists():
                    try:
                        _restore_database_from_safety_dump(database_safety_dump_path)
                    except Exception as rollback_exc:
                        logger.error("Database rollback failed after restore error: %s", rollback_exc)
                        rollback_errors.append(f"database rollback failed: {rollback_exc}")
                if rollback_errors:
                    raise BackupRollbackError(f"{exc}; {'; '.join(rollback_errors)}") from exc
                if isinstance(exc, BackupMutationError):
                    raise
                raise BackupMutationError(str(exc)) from exc
    except (BackupMutationError, BackupRollbackError):
        raise
    except Exception as exc:
        raise BackupMutationError(str(exc)) from exc
    finally:
        tempdir.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ragtime server backup and restore")
    subparsers = parser.add_subparsers(dest="command")

    backup = subparsers.add_parser("backup", help="Create a server backup")
    backup_scope = backup.add_mutually_exclusive_group()
    backup_scope.add_argument("--db-only", action="store_true")
    backup_scope.add_argument("--files-only", "--data-dir-only", "--faiss-only", action="store_true")
    backup.add_argument("--encrypt", action="store_true")
    backup.add_argument("--include-secret", action="store_true", help="Compatibility alias for portable encrypted backups")
    backup.add_argument("--password-fd", type=int)
    backup.add_argument("output", nargs="?", default="-")

    restore = subparsers.add_parser("restore", help="Restore a server backup")
    restore_scope = restore.add_mutually_exclusive_group()
    restore_scope.add_argument("--db-only", action="store_true")
    restore_scope.add_argument("--files-only", "--data-dir-only", "--faiss-only", action="store_true")
    restore.add_argument("--skip-migrations", action="store_true")
    restore.add_argument("--data-only", "--pg-data-only", action="store_true")
    restore.add_argument("--replace-existing-data", "--replace-data", action="store_true")
    restore.add_argument("--include-secret", action="store_true", help="Compatibility alias for acknowledging a legacy embedded key")
    restore.add_argument("--acknowledge-legacy-key", action="store_true")
    restore.add_argument("--confirm-restore")
    restore.add_argument("--mirror-local-admin-access", action="store_true")
    restore.add_argument("--mirror-local-admin-from", default="auto")
    restore.add_argument("--local-admin-username")
    restore.add_argument("--password-fd", type=int)
    restore.add_argument("archive")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "backup":
        scope = BackupScope.FULL
        if args.db_only:
            scope = BackupScope.DATABASE
        elif args.files_only:
            scope = BackupScope.FILES
        create_backup(
            BackupOptions(
                scope=scope,
                output_path=None if args.output == "-" else Path(args.output),
                output_stream=sys.stdout.buffer if args.output == "-" else None,
                encrypt=bool(args.encrypt or args.include_secret),
                password_fd=args.password_fd,
            )
        )
        return 0

    if args.command == "restore":
        scope_override = None
        if args.db_only:
            scope_override = BackupScope.DATABASE
        elif args.files_only:
            scope_override = BackupScope.FILES
        restore_backup(
            RestoreOptions(
                archive_path=None if args.archive == "-" else Path(args.archive),
                archive_stream=sys.stdin.buffer if args.archive == "-" else None,
                scope_override=scope_override,
                skip_migrations=args.skip_migrations,
                pg_data_only=args.data_only,
                replace_data=args.replace_existing_data,
                acknowledge_legacy_key=bool(args.acknowledge_legacy_key or args.include_secret),
                restore_confirmation=args.confirm_restore,
                mirror_local_admin_access=args.mirror_local_admin_access,
                mirror_local_admin_from=args.mirror_local_admin_from,
                local_admin_username=args.local_admin_username,
                password_fd=args.password_fd,
            )
        )
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
