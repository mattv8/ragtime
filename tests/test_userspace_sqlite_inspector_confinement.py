"""Regression coverage for SQLite inspector managed-path confinement."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException

from ragtime.userspace import sqlite_inspector as si


def _make_victim(path: Path) -> bytes:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE confidential (value TEXT)")
        connection.execute("INSERT INTO confidential VALUES ('CONFIDENTIAL-B')")
    return path.read_bytes()


def _assert_rejected(files_dir: Path, victim: Path, original_bytes: bytes) -> None:
    with pytest.raises(HTTPException):
        si._resolve_database_path(files_dir, "app.sqlite3")
    assert si.list_databases(files_dir) == ([], 0)
    with pytest.raises(HTTPException):
        si.execute_readonly_query(files_dir, "app.sqlite3", "SELECT value FROM confidential")
    with pytest.raises(HTTPException):
        si.create_table(files_dir, "app.sqlite3", "changed", [si.ColumnDefinition(name="id", type="INTEGER")])
    assert victim.read_bytes() == original_bytes


@pytest.mark.parametrize("unsafe_component", [".ragtime", ".ragtime/db"])
def test_inspector_rejects_symlinked_managed_parent(tmp_path: Path, unsafe_component: str) -> None:
    files_dir = tmp_path / "workspace" / "files"
    database_dir = files_dir / ".ragtime" / "db"
    database_dir.mkdir(parents=True)
    victim = tmp_path / "victim.sqlite3"
    original_bytes = _make_victim(victim)

    if unsafe_component == ".ragtime":
        (files_dir / ".ragtime").rename(files_dir / ".ragtime-safe")
        (files_dir / ".ragtime").symlink_to(tmp_path)
    else:
        database_dir.rename(files_dir / ".ragtime" / "db-safe")
        database_dir.symlink_to(tmp_path)

    _assert_rejected(files_dir, victim, original_bytes)


def test_inspector_rejects_symlinked_database_leaf(tmp_path: Path) -> None:
    files_dir = tmp_path / "workspace" / "files"
    database_dir = files_dir / ".ragtime" / "db"
    database_dir.mkdir(parents=True)
    victim = tmp_path / "victim.sqlite3"
    original_bytes = _make_victim(victim)
    (database_dir / "app.sqlite3").symlink_to(victim)

    _assert_rejected(files_dir, victim, original_bytes)


@pytest.mark.parametrize("sidecar", ["-wal", "-shm", "-journal"])
def test_inspector_rejects_symlinked_sqlite_sidecar(tmp_path: Path, sidecar: str) -> None:
    files_dir = tmp_path / "workspace" / "files"
    database_dir = files_dir / ".ragtime" / "db"
    database_dir.mkdir(parents=True)
    managed_db = database_dir / "app.sqlite3"
    _make_victim(managed_db)
    victim = tmp_path / "victim-sidecar"
    original_bytes = _make_victim(victim)
    (database_dir / f"app.sqlite3{sidecar}").symlink_to(victim)

    _assert_rejected(files_dir, victim, original_bytes)


def test_inspector_uses_normal_managed_database(tmp_path: Path) -> None:
    files_dir = tmp_path / "workspace" / "files"
    files_dir.mkdir(parents=True)

    si.initialize_database(files_dir)
    si.create_table(files_dir, "app.sqlite3", "items", [si.ColumnDefinition(name="value", type="TEXT")])
    si.insert_row(files_dir, "app.sqlite3", "items", {"value": "workspace-only"})

    summaries, _ = si.list_databases(files_dir)
    assert [summary.name for summary in summaries] == ["app.sqlite3"]
    assert si.execute_readonly_query(files_dir, "app.sqlite3", "SELECT value FROM items").rows == [{"value": "workspace-only"}]


def test_inspector_read_remains_pinned_after_parent_swap(tmp_path: Path) -> None:
    files_dir = tmp_path / "workspace" / "files"
    files_dir.mkdir(parents=True)
    si.initialize_database(files_dir)
    si.create_table(files_dir, "app.sqlite3", "safe", [si.ColumnDefinition(name="value", type="TEXT")])
    si.insert_row(files_dir, "app.sqlite3", "safe", {"value": "workspace-only"})
    database_dir = files_dir / ".ragtime" / "db"
    victim = tmp_path / "victim" / "app.sqlite3"
    victim.parent.mkdir()
    original_bytes = _make_victim(victim)
    original_connect = si._connect

    @contextmanager
    def swap_parent_before_connect(db_path: Path | si._ManagedDatabasePath):
        database_dir.rename(files_dir / ".ragtime" / "db-safe")
        database_dir.symlink_to(victim.parent)
        with original_connect(db_path) as connection:
            yield connection

    with mock.patch.object(si, "_connect", swap_parent_before_connect):
        result = si.execute_readonly_query(files_dir, "app.sqlite3", "SELECT value FROM safe")

    assert result.rows == [{"value": "workspace-only"}]
    assert victim.read_bytes() == original_bytes


def test_inspector_mutation_remains_pinned_after_parent_swap(tmp_path: Path) -> None:
    files_dir = tmp_path / "workspace" / "files"
    files_dir.mkdir(parents=True)
    si.initialize_database(files_dir)
    database_dir = files_dir / ".ragtime" / "db"
    victim = tmp_path / "victim" / "app.sqlite3"
    victim.parent.mkdir()
    original_bytes = _make_victim(victim)
    original_connect = si._connect

    @contextmanager
    def swap_parent_before_connect(db_path: Path | si._ManagedDatabasePath):
        database_dir.rename(files_dir / ".ragtime" / "db-safe")
        database_dir.symlink_to(victim.parent)
        with original_connect(db_path) as connection:
            yield connection

    with mock.patch.object(si, "_connect", swap_parent_before_connect):
        si.create_table(files_dir, "app.sqlite3", "safe", [si.ColumnDefinition(name="value", type="TEXT")])

    with sqlite3.connect(files_dir / ".ragtime" / "db-safe" / "app.sqlite3") as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'safe'").fetchone() == ("safe",)
    assert victim.read_bytes() == original_bytes


def test_inspector_wal_database_works_through_pinned_directory_fd(tmp_path: Path) -> None:
    files_dir = tmp_path / "workspace" / "files"
    files_dir.mkdir(parents=True)
    summary = si.initialize_database(files_dir)
    database_path = files_dir / si.MANAGED_DB_DIRNAME / summary.name

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE items (value TEXT)")
    si.insert_row(files_dir, summary.name, "items", {"value": "wal-safe"})

    assert si.execute_readonly_query(files_dir, summary.name, "SELECT value FROM items").rows == [{"value": "wal-safe"}]


@pytest.mark.parametrize("mutation", [False, True])
def test_inspector_leaf_swap_after_validation_cannot_access_victim(tmp_path: Path, mutation: bool) -> None:
    files = tmp_path / "files"
    si.initialize_database(files)
    victim = tmp_path / "victim.sqlite3"
    before = _make_victim(victim)
    leaf = files / si.MANAGED_DB_DIRNAME / "app.sqlite3"
    connect = si._connect

    @contextmanager
    def swap_leaf(handle):
        leaf.unlink()
        leaf.symlink_to(victim)
        with connect(handle) as connection:
            yield connection

    with mock.patch.object(si, "_connect", swap_leaf):
        if mutation:
            # The pinned original inode is mutated, not the symlink target.
            si.create_table(files, "app.sqlite3", "leaked", [si.ColumnDefinition(name="id", type="INTEGER")])
        else:
            with pytest.raises((sqlite3.Error, HTTPException)):
                si.execute_readonly_query(files, "app.sqlite3", "SELECT value FROM confidential")
    assert victim.read_bytes() == before
