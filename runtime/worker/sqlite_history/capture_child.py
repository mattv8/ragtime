"""Landlock-confined runtime SQLite capture child (no control-plane imports)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from runtime.core.sqlite_capture_state import pinned_source_state, source_state_token
from runtime.core.sqlite_history_scratch import SCRATCH_PREFIX, scratch_owner_lock
from runtime.worker import mount_sync_launcher as _landlock


def _name(value: str) -> str:
    if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("invalid confined SQLite filename")
    return value


def _open_relative(root_fd: int, relative: str, *, directory: bool = False) -> int:
    current = os.dup(root_fd)
    try:
        parts = _name(relative).split("/")
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = next_fd
        result = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else os.O_NONBLOCK),
            dir_fd=current,
        )
        os.close(current)
        current = -1
        return result
    finally:
        if current >= 0:
            os.close(current)


def _require_regular(root_fd: int, relative: str) -> None:
    fd = _open_relative(root_fd, relative)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("SQLite source is not a regular file")
    finally:
        os.close(fd)


def _freeze_migrations(source_fd: int, scratch: Path) -> Path:
    frozen = scratch / "migrations"
    frozen.mkdir(mode=0o700)
    try:
        migrations_fd = _open_relative(source_fd, "migrations", directory=True)
    except FileNotFoundError:
        return frozen
    try:
        for name in sorted(os.listdir(migrations_fd)):
            source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=migrations_fd)
            try:
                if not stat.S_ISREG(os.fstat(source).st_mode) or Path(name).suffix.lower() != ".sql":
                    if Path(name).suffix.lower() == ".sql":
                        raise ValueError("Migration directory contains a non-regular entry")
                    continue
                destination = frozen / name
                out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    while block := os.read(source, 1024 * 1024):
                        os.write(out, block)
                finally:
                    os.close(out)
            finally:
                os.close(source)
    finally:
        os.close(migrations_fd)
    return frozen


def _landlock_install(source_fd: int, destination_fd: int) -> None:
    _landlock.trial_ruleset()
    create, _add, restrict = _landlock._syscall_numbers()
    attr = _landlock._RulesetAttr(_landlock._HANDLED_RIGHTS)
    ruleset = _landlock._libc().syscall(create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "cannot create Landlock ruleset")
    try:
        _landlock._add_path_rule(ruleset, source_fd, _landlock._SOURCE_RIGHTS)
        _landlock._add_path_rule(ruleset, destination_fd, _landlock._DESTINATION_RIGHTS)
        if _landlock._libc().prctl(_landlock._PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) or _landlock._libc().syscall(restrict, ruleset, 0):
            raise OSError(ctypes.get_errno(), "cannot restrict Landlock ruleset")
    finally:
        os.close(ruleset)


@contextmanager
def _confined_sqlite_temp(destination_fd: int, previous_cwd_fd: int) -> Iterator[Path]:
    """Keep every SQLite/Python temporary file beneath the pinned destination."""
    previous_environment = {name: os.environ.get(name) for name in ("TMPDIR", "SQLITE_TMPDIR")}
    previous_tempdir = tempfile.tempdir
    scratch: Path | None = None
    try:
        os.fchdir(destination_fd)
        scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=".")).resolve()
        with scratch_owner_lock(scratch):
            scratch_name = os.fspath(scratch)
            os.environ["TMPDIR"] = scratch_name
            os.environ["SQLITE_TMPDIR"] = scratch_name
            tempfile.tempdir = scratch_name
            yield scratch
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        os.fchdir(previous_cwd_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fd", type=int, required=True)
    parser.add_argument("--destination-fd", type=int, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--destination-name", required=True)
    parser.add_argument("--probe-source", action="store_true")
    parser.add_argument("--preview-backup")
    parser.add_argument("--preview-current")
    parser.add_argument("--preview-candidate")
    parser.add_argument("--preview-migrations", default="migrations")
    parser.add_argument("--fingerprint-current")
    parser.add_argument("--fingerprint-migrations")
    parser.add_argument("--mode")
    parser.add_argument("--conflict-policy")
    parser.add_argument("--table-policies", default="{}")
    args = parser.parse_args()
    source_name = _name(args.source_name)
    destination_name = _name(args.destination_name)
    previous_cwd_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        source_root = Path(f"/proc/self/fd/{args.source_fd}")
        destination_root = Path(f"/proc/self/fd/{args.destination_fd}")
        with _confined_sqlite_temp(args.destination_fd, previous_cwd_fd) as scratch:
            # sqlite3 selects its Unix temp directory during process/library
            # initialization. Establish the confined path before importing the
            # recovery module, then install Landlock before any database access.
            from runtime.core.sqlite_recovery import (
                _connect_readonly,
                capture_database,
                database_fingerprint,
                migration_fingerprint,
                prepare_restore,
            )

            _landlock_install(args.source_fd, args.destination_fd)
            if args.probe_source:
                print(json.dumps({"source_token": source_state_token(args.source_fd, source_name)}, sort_keys=True))
                return
            if args.fingerprint_current:
                current_name = _name(args.fingerprint_current)
                current = scratch / "current.sqlite3"
                try:
                    _require_regular(args.source_fd, current_name)
                    capture_database(source_root / current_name, current, include_fingerprint=False)
                    current_fingerprint = database_fingerprint(current)
                except FileNotFoundError:
                    current_fingerprint = None
                migrations = _freeze_migrations(args.source_fd, scratch)
                print(
                    json.dumps(
                        {
                            "current_fingerprint": current_fingerprint,
                            "migration_fingerprint": migration_fingerprint(migrations),
                        },
                        sort_keys=True,
                    )
                )
                return
            if args.preview_backup:
                backup = destination_root / _name(args.preview_backup)
                candidate = destination_root / _name(args.preview_candidate or "")
                current_name = _name(args.preview_current or "")
                _require_regular(args.destination_fd, _name(args.preview_backup))
                current_copy = scratch / "current.sqlite3"
                current = source_root / current_name
                try:
                    _require_regular(args.source_fd, current_name)
                    capture_database(current, current_copy, include_fingerprint=False)
                    current_fingerprint = database_fingerprint(current_copy)
                except FileNotFoundError:
                    current_fingerprint = None
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
                result.update(
                    current_fingerprint=current_fingerprint,
                    migration_fingerprint=migration_fingerprint(migrations),
                )
                print(json.dumps(result, sort_keys=True))
                return
            with pinned_source_state(args.source_fd, source_name) as state:
                if state.main_fd is None:
                    raise ValueError("SQLite source is not a regular file")
                connection = _connect_readonly(Path(f"/proc/self/fd/{state.main_fd}"))
                try:
                    connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                    if not state.refresh_after_warmup():
                        raise ValueError("SQLite source changed during readonly warmup")
                    before = state.token()
                    result = capture_database(
                        Path(f"/proc/self/fd/{state.main_fd}"),
                        destination_root / destination_name,
                        include_fingerprint=False,
                        source_connection=connection,
                    )
                finally:
                    connection.close()
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
