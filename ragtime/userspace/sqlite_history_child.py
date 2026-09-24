"""Landlock-confined SQLite source capture child.

This is deliberately a tiny exec boundary: SQLite receives descriptor-rooted
``/proc/self/fd`` names only after Landlock has limited it to the already-pinned
workspace database directory and private history destination directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

from runtime.core.sqlite_capture_state import pinned_source_state, source_state_token
from runtime.worker import mount_sync_launcher as _landlock
from runtime.worker.sqlite_history.capture_child import _confined_sqlite_temp


def _load_sqlite_recovery() -> None:
    """Initialize SQLite only after its confined temporary directory is set."""
    import runtime.core.sqlite_recovery  # inline-import: keep


# Keep these names patchable for legacy callers/tests without importing the
# SQLite engine before its confined temporary directory is established.
def _connect_readonly(path: Path) -> sqlite3.Connection:
    from runtime.core.sqlite_recovery import _connect_readonly as implementation  # inline-import: keep

    return implementation(path)


def capture_database(source: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
    from runtime.core.sqlite_recovery import capture_database as implementation  # inline-import: keep

    return implementation(source, destination, **kwargs)


def database_fingerprint(path: Path) -> str:
    from runtime.core.sqlite_recovery import database_fingerprint as implementation  # inline-import: keep

    return implementation(path)


def migration_fingerprint(path: Path) -> str:
    from runtime.core.sqlite_recovery import migration_fingerprint as implementation  # inline-import: keep

    return implementation(path)


def prepare_restore(backup_path: Path, current_path: Path | None, migrations_dir: Path, output_path: Path, **kwargs: Any) -> dict[str, Any]:
    from runtime.core.sqlite_recovery import prepare_restore as implementation  # inline-import: keep

    return implementation(backup_path, current_path, migrations_dir, output_path, **kwargs)


def _install_sqlite_landlock(source_fd: int, destination_fd: int) -> None:
    """Minimal Landlock policy; unlike rsync it grants no global `/usr` tree."""
    _landlock.trial_ruleset()
    create, _add, restrict = _landlock._syscall_numbers()
    attr = _landlock._RulesetAttr(_landlock._HANDLED_RIGHTS)
    ruleset_fd = _landlock._libc().syscall(create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        _landlock._add_path_rule(ruleset_fd, source_fd, _landlock._SOURCE_RIGHTS)
        _landlock._add_path_rule(ruleset_fd, destination_fd, _landlock._DESTINATION_RIGHTS)
        if _landlock._libc().prctl(_landlock._PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if _landlock._libc().syscall(restrict, ruleset_fd, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(ruleset_fd)


def _name(value: str) -> str:
    if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("invalid confined SQLite filename")
    return value


def _open_relative(root_fd: int, relative: str, *, directory: bool = False) -> int:
    """Open a confined path without accepting symlink components."""
    current_fd = os.dup(root_fd)
    try:
        parts = _name(relative).split("/")
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        flags = os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else os.O_NONBLOCK)
        result = os.open(parts[-1], flags, dir_fd=current_fd)
        os.close(current_fd)
        current_fd = -1
        return result
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _require_regular(root_fd: int, relative: str) -> None:
    fd = _open_relative(root_fd, relative)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("SQLite source is not a regular file")
    finally:
        os.close(fd)


def _freeze_migrations(source_fd: int, scratch: Path) -> Path:
    """Copy migration SQL through no-follow FDs before any restore work.

    The engine reads this private immutable-for-the-operation copy, so a
    workspace edit cannot change the migration digest after preparation.
    """
    frozen = scratch / "migrations"
    frozen.mkdir(mode=0o700)
    try:
        migrations_fd = _open_relative(source_fd, "migrations", directory=True)
    except FileNotFoundError:
        return frozen
    try:
        for name in sorted(os.listdir(migrations_fd)):
            try:
                source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=migrations_fd)
            except OSError as exc:
                raise ValueError("Migration directory contains an unsafe entry") from exc
            try:
                if not stat.S_ISREG(os.fstat(source).st_mode):
                    raise ValueError("Migration directory contains a non-regular entry")
                if Path(name).suffix.lower() != ".sql":
                    continue
                frozen_fd = os.open(frozen, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    destination = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=frozen_fd,
                    )
                    try:
                        while block := os.read(source, 1024 * 1024):
                            view = memoryview(block)
                            while view:
                                view = view[os.write(destination, view) :]
                    finally:
                        os.close(destination)
                finally:
                    os.close(frozen_fd)
            finally:
                os.close(source)
    finally:
        os.close(migrations_fd)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fd", type=int, required=True)
    parser.add_argument("--destination-fd", type=int, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--destination-name", required=True)
    parser.add_argument("--preview-backup")
    parser.add_argument("--preview-current")
    parser.add_argument("--preview-migrations")
    parser.add_argument("--preview-candidate")
    parser.add_argument("--mode")
    parser.add_argument("--conflict-policy")
    parser.add_argument("--table-policies", default="{}")
    parser.add_argument("--fingerprint-current")
    parser.add_argument("--fingerprint-migrations")
    parser.add_argument("--probe-source", action="store_true")
    args = parser.parse_args()
    source_name = _name(args.source_name)
    destination_name = _name(args.destination_name)
    previous_cwd_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        source_root = f"/proc/self/fd/{args.source_fd}"
        destination_root = f"/proc/self/fd/{args.destination_fd}"
        with _confined_sqlite_temp(args.destination_fd, previous_cwd_fd) as scratch:
            # sqlite3 chooses its Unix temporary directory when its library is
            # initialized. Set the confined directory before lazily importing
            # recovery helpers, then install Landlock before database access.
            _load_sqlite_recovery()
            _install_sqlite_landlock(args.source_fd, args.destination_fd)
            if args.probe_source:
                print(json.dumps({"source_token": source_state_token(args.source_fd, source_name)}, sort_keys=True))
            elif args.fingerprint_current:
                # SQLite fingerprints must use an online backup too: reading
                # the live main file directly can race a committed WAL state.
                current_name = _name(args.fingerprint_current)
                current_copy = scratch / "current.sqlite3"
                try:
                    _require_regular(args.source_fd, current_name)
                except FileNotFoundError:
                    current_fingerprint = None
                else:
                    capture_database(Path(os.path.join(source_root, current_name)), current_copy, include_fingerprint=False)
                    current_fingerprint = database_fingerprint(current_copy)
                migrations = _freeze_migrations(args.source_fd, scratch)
                print(json.dumps({"current_fingerprint": current_fingerprint, "migration_fingerprint": migration_fingerprint(migrations)}, sort_keys=True))
            elif args.preview_backup:
                backup = Path(os.path.join(destination_root, _name(args.preview_backup)))
                current_name = _name(args.preview_current or "")
                candidate = Path(os.path.join(destination_root, _name(args.preview_candidate or "")))
                current = Path(os.path.join(source_root, current_name))
                current_copy = scratch / "current.sqlite3"
                current_fingerprint = None
                _require_regular(args.destination_fd, _name(args.preview_backup))
                try:
                    _require_regular(args.source_fd, current_name)
                except FileNotFoundError:
                    pass
                else:
                    capture_database(current, current_copy, include_fingerprint=False)
                    current_fingerprint = database_fingerprint(current_copy)
                migrations = _freeze_migrations(args.source_fd, scratch)
                result = prepare_restore(
                    backup,
                    current_copy if current_copy.exists() else None,
                    migrations,
                    candidate,
                    mode=args.mode or "",
                    conflict_policy=args.conflict_policy or "",
                    table_policies=json.loads(args.table_policies),
                )
                result["current_fingerprint"] = current_fingerprint
                result["migration_fingerprint"] = migration_fingerprint(migrations)
                print(json.dumps(result, sort_keys=True))
            else:
                destination = os.path.join(destination_root, destination_name)
                with pinned_source_state(args.source_fd, source_name) as state:
                    if state.main_fd is None:
                        raise ValueError("SQLite source is not a regular file")
                    source = Path(f"/proc/self/fd/{state.main_fd}")
                    source_connection = _connect_readonly(source)
                    try:
                        source_connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                        if not state.refresh_after_warmup():
                            raise ValueError("SQLite source changed during readonly warmup")
                        before = state.token()
                        result = capture_database(source, Path(destination), include_fingerprint=False, source_connection=source_connection)
                    finally:
                        source_connection.close()
                    after = state.token()
                if before is not None and before == after:
                    result["source_token"] = after
                print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        print(f"confined SQLite capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(77)
    finally:
        os.close(previous_cwd_fd)


if __name__ == "__main__":
    main()
