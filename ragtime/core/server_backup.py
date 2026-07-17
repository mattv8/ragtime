from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterator, Optional
from urllib.parse import urlparse

from ragtime.config.settings import ENCRYPTION_KEY_FILE, settings
from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream, encrypt_stream, is_encrypted_backup
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(os.environ.get("INDEX_DATA_PATH", settings.index_data_path))
PRISMA_DIR = Path("/ragtime/prisma")
LOCK_PATH = Path(tempfile.gettempdir()) / "ragtime-server-backup.lock"
MANIFEST_NAME = "backup-meta.json"
RESTORE_CONFIRMATION_PREFIX = "RESTORE"


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
    legacy_embedded_key: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BackupManifest":
        version_value = payload.get("version", 1)
        if not isinstance(version_value, int):
            raise BackupValidationError("Backup manifest version is invalid")
        return cls(
            format=str(payload.get("format", "tar.gz")),
            version=version_value,
            created_at=str(payload.get("created_at", "")),
            scope=BackupScope(str(payload.get("scope", BackupScope.FULL.value))),
            ragtime_version=str(payload.get("ragtime_version", "unknown")),
            schema_version=str(payload.get("schema_version", "unknown")),
            encrypted=bool(payload.get("encrypted", False)),
            includes_managed_key=bool(payload.get("includes_managed_key", False)),
            legacy_embedded_key=bool(payload.get("legacy_embedded_key", False)),
        )


ProgressCallback = Callable[[dict[str, object]], None]


def _emit_progress(progress: Optional[ProgressCallback], phase: str, **details: object) -> None:
    if progress is None:
        return
    event: dict[str, object] = {"phase": phase}
    event.update(details)
    progress(event)


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
def _locked_operation() -> Iterator[None]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _should_skip_data_path(relative: Path) -> bool:
    path = relative.as_posix()
    if path in {".encryption_key", ".jwt_secret"}:
        return True
    if path.startswith("_tmp"):
        return True
    if path.startswith("_userspace/workspaces/") and "/rootfs" in path:
        return True
    if path.startswith("_userspace/workspaces/") and path.endswith("/.runtime-bootstrap.done"):
        return True
    return False


def _write_manifest(tempdir: Path, manifest: BackupManifest) -> None:
    (tempdir / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")


def _build_tarball(tempdir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(tempdir.rglob("*")):
            if child.is_dir():
                continue
            archive.add(child, arcname=child.relative_to(tempdir).as_posix(), recursive=False)


def _extract_archive(archive_path: Path, destination_dir: Path) -> tuple[BackupManifest, set[str]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        names: set[str] = set()
        for member in archive.getmembers():
            relative = _validate_member(member)
            names.add(relative.as_posix())
            target = destination_dir / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(member.linkname)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                target.touch()
                continue
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)

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
            legacy_embedded_key=legacy_key,
        )

    legacy_key = "data/.encryption_key" in names or ".encryption_key" in names or "faiss/.encryption_key" in names
    if not manifest.encrypted and legacy_key:
        manifest.legacy_embedded_key = True
        manifest.includes_managed_key = True
    return manifest, names


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


def _snapshot_directory(target: Path, snapshot_root: Path) -> Path:
    snapshot_path = snapshot_root / "data-snapshot"
    snapshot_path.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        return snapshot_path
    for child in target.iterdir():
        destination = snapshot_path / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)
    return snapshot_path


def _restore_snapshot(snapshot_path: Path, target: Path) -> None:
    if target.exists():
        for child in list(target.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        target.mkdir(parents=True, exist_ok=True)
    for child in snapshot_path.iterdir():
        destination = target / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _copy_tree_contents(source_dir: Path, destination_dir: Path, *, replace: bool) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        for child in list(destination_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    for child in source_dir.rglob("*"):
        relative = child.relative_to(source_dir)
        target = destination_dir / relative
        if child.is_dir() and not child.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.readlink(child))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def _resolve_data_source(extract_dir: Path) -> Optional[Path]:
    if (extract_dir / "data").exists():
        return extract_dir / "data"
    if (extract_dir / "faiss").exists():
        return extract_dir / "faiss"
    return None


def create_backup(options: BackupOptions, progress: Optional[ProgressCallback] = None) -> BackupManifest:
    password = _read_password(options.password_fd, prompt="Backup password: ") if options.encrypt else None
    with _locked_operation(), tempfile.TemporaryDirectory(prefix="ragtime-server-backup-build-") as tmpdir:
        root = Path(tmpdir)
        _emit_progress(progress, "start", scope=options.scope.value, encrypted=options.encrypt)

        if options.scope in {BackupScope.FULL, BackupScope.DATABASE}:
            _emit_progress(progress, "database_dump")
            _dump_database(root / "database.dump")

        includes_managed_key = options.encrypt and ENCRYPTION_KEY_FILE.exists()
        if options.scope in {BackupScope.FULL, BackupScope.FILES} or includes_managed_key:
            data_root = root / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            if DATA_DIR.exists() and options.scope in {BackupScope.FULL, BackupScope.FILES}:
                for child in DATA_DIR.rglob("*"):
                    relative = child.relative_to(DATA_DIR)
                    if _should_skip_data_path(relative):
                        continue
                    target = data_root / relative
                    if child.is_dir() and not child.is_symlink():
                        target.mkdir(parents=True, exist_ok=True)
                    elif child.is_symlink():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists() or target.is_symlink():
                            target.unlink()
                        target.symlink_to(os.readlink(child))
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(child, target)
            if includes_managed_key:
                shutil.copy2(ENCRYPTION_KEY_FILE, data_root / ".encryption_key")

        manifest = BackupManifest(
            format="ragbak" if options.encrypt else "tar.gz",
            version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            scope=options.scope,
            ragtime_version=_get_ragtime_version(),
            schema_version=_get_current_schema_version(),
            encrypted=options.encrypt,
            includes_managed_key=includes_managed_key,
            legacy_embedded_key=False,
        )
        _write_manifest(root, manifest)

        archive_path = root / "backup.tar.gz"
        _emit_progress(progress, "archive")
        _build_tarball(root, archive_path)

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
                if options.encrypt:
                    encrypt_stream(source_handle, destination, password or "")
                else:
                    shutil.copyfileobj(source_handle, destination)
        finally:
            if close_destination:
                destination.close()

        _emit_progress(progress, "complete", format=manifest.format)
        return manifest


def inspect_backup(path, password: Optional[str] = None) -> BackupManifest:
    try:
        archive_path, tempdir = _prepare_archive(Path(path), None, password)
        try:
            extract_dir = Path(tempdir.name) / "inspect"
            extract_dir.mkdir()
            manifest, _ = _extract_archive(archive_path, extract_dir)
            return manifest
        finally:
            tempdir.cleanup()
    except (BackupCryptoError, tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise BackupValidationError(str(exc)) from exc


def restore_backup(options: RestoreOptions, progress: Optional[ProgressCallback] = None) -> BackupManifest:
    password = _read_password(options.password_fd, prompt="Backup password: ") if options.password_fd is not None else None
    try:
        archive_path, tempdir = _prepare_archive(options.archive_path, options.archive_stream, password)
        extract_dir = Path(tempdir.name) / "extract"
        extract_dir.mkdir()
        manifest, names = _extract_archive(archive_path, extract_dir)
        restore_scope = options.scope_override or manifest.scope
        data_source = _resolve_data_source(extract_dir)
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
    except (BackupCryptoError, tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise BackupValidationError(str(exc)) from exc

    snapshot_path: Optional[Path] = None
    database_safety_dump_path: Optional[Path] = None
    db_mutated = False
    try:
        with _locked_operation():
            _emit_progress(progress, "start", scope=restore_scope.value)
            if restore_scope in {BackupScope.FULL, BackupScope.FILES}:
                snapshot_path = _snapshot_directory(DATA_DIR, Path(tempdir.name))
            if restore_scope in {BackupScope.FULL, BackupScope.DATABASE}:
                database_safety_dump_path = Path(tempdir.name) / "database-safety.dump"
                _create_database_safety_dump(database_safety_dump_path)

            if restore_scope in {BackupScope.FULL, BackupScope.DATABASE}:
                _emit_progress(progress, "database_restore")
                _terminate_other_database_connections()
                _restore_database(extract_dir / "database.dump", data_only=options.pg_data_only)
                db_mutated = True
                if not options.skip_migrations and not options.pg_data_only:
                    _run_migrations()
                if options.mirror_local_admin_access:
                    _mirror_local_admin_access(options.mirror_local_admin_from, options.local_admin_username)
                _invalidate_restored_runtime_sessions()

            if restore_scope in {BackupScope.FULL, BackupScope.FILES}:
                _emit_progress(progress, "files_restore")
                _copy_tree_contents(data_source, DATA_DIR, replace=options.replace_data)  # type: ignore[arg-type]
                _invalidate_restored_workspace_runtime_artifacts()
                key_path = DATA_DIR / ".encryption_key"
                if key_path.exists():
                    key_path.chmod(0o600)

            _emit_progress(progress, "complete", scope=restore_scope.value)
            if options.scope_override is not None:
                manifest.scope = options.scope_override
            return manifest
    except Exception as exc:
        rollback_errors: list[str] = []
        if snapshot_path is not None:
            try:
                _restore_snapshot(snapshot_path, DATA_DIR)
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
