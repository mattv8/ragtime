"""Tests for the workspace SQLite inspector helpers.

Exercises the pure synchronous primitives in
`ragtime.userspace.sqlite_inspector`, which the async service layer wraps
with `asyncio.to_thread`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from ragtime.userspace import sqlite_inspector as si

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_identifier_accepts_basic_names() -> None:
    assert si.validate_identifier("users") == "users"
    assert si.validate_identifier(" Users_1 ") == "Users_1"


@pytest.mark.parametrize(
    "name",
    ["", "1users", "users-1", "users.id", "drop;table", "sqlite_master"],
)
def test_validate_identifier_rejects_invalid(name: str) -> None:
    with pytest.raises(HTTPException):
        si.validate_identifier(name)


def test_validate_column_type_normalises_case() -> None:
    assert si.validate_column_type("text") == "TEXT"
    assert si.validate_column_type("Integer") == "INTEGER"
    with pytest.raises(HTTPException):
        si.validate_column_type("VARCHAR(50)")


def test_default_literal_allowlist() -> None:
    assert si._validate_default_literal(None) is None
    assert si._validate_default_literal("NULL") == "NULL"
    assert si._validate_default_literal("0") == "0"
    assert si._validate_default_literal("-1.5") == "-1.5"
    assert si._validate_default_literal("'hello'") == "'hello'"
    assert si._validate_default_literal("CURRENT_TIMESTAMP") == "CURRENT_TIMESTAMP"
    with pytest.raises(HTTPException):
        si._validate_default_literal("(SELECT 1)")
    with pytest.raises(HTTPException):
        si._validate_default_literal("'em\\bedded'")


def test_resolve_database_path_rejects_traversal(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    with pytest.raises(HTTPException):
        si._resolve_database_path(files_dir, "../escape.sqlite3")
    with pytest.raises(HTTPException):
        si._resolve_database_path(files_dir, "no_extension")
    with pytest.raises(HTTPException):
        si._resolve_database_path(files_dir, "")


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------


def test_list_databases_empty(tmp_path: Path) -> None:
    summaries, total = si.list_databases(tmp_path / "files")
    assert summaries == []
    assert total == 0


def test_initialize_and_delete_database(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    summary = si.initialize_database(files_dir)
    assert summary.name == si.DEFAULT_DATABASE_NAME
    assert summary.table_count == 0
    assert (files_dir / si.MANAGED_DB_DIRNAME / si.DEFAULT_DATABASE_NAME).exists()

    summaries, total = si.list_databases(files_dir)
    assert len(summaries) == 1
    assert total == summaries[0].size_bytes

    si.delete_database(files_dir, si.DEFAULT_DATABASE_NAME)
    assert not (files_dir / si.MANAGED_DB_DIRNAME / si.DEFAULT_DATABASE_NAME).exists()
    with pytest.raises(HTTPException):
        si.delete_database(files_dir, si.DEFAULT_DATABASE_NAME)


# ---------------------------------------------------------------------------
# Table + row CRUD
# ---------------------------------------------------------------------------


def _init(tmp_path: Path) -> Path:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    si.initialize_database(files_dir)
    return files_dir


def test_create_and_drop_table(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    summary = si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
            si.ColumnDefinition(name="title", type="TEXT", not_null=True),
            si.ColumnDefinition(name="created_at", type="TEXT", default_value="CURRENT_TIMESTAMP"),
        ],
    )
    assert summary.name == "items"

    tables = si.list_tables(files_dir, si.DEFAULT_DATABASE_NAME)
    assert [t.name for t in tables] == ["items"]

    schema = si.get_table_schema(files_dir, si.DEFAULT_DATABASE_NAME, "items")
    assert [c.name for c in schema.columns] == ["id", "title", "created_at"]
    pk = [c for c in schema.columns if c.primary_key]
    assert pk and pk[0].name == "id"

    # Duplicate fails
    with pytest.raises(HTTPException):
        si.create_table(
            files_dir,
            si.DEFAULT_DATABASE_NAME,
            "items",
            [si.ColumnDefinition(name="id", type="INTEGER", primary_key=True)],
        )

    si.drop_table(files_dir, si.DEFAULT_DATABASE_NAME, "items")
    assert si.list_tables(files_dir, si.DEFAULT_DATABASE_NAME) == []


def test_create_table_rejects_duplicate_columns(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    with pytest.raises(HTTPException):
        si.create_table(
            files_dir,
            si.DEFAULT_DATABASE_NAME,
            "dupes",
            [
                si.ColumnDefinition(name="id", type="INTEGER", primary_key=True),
                si.ColumnDefinition(name="ID", type="INTEGER"),
            ],
        )


def test_row_crud_with_pagination(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
            si.ColumnDefinition(name="title", type="TEXT"),
        ],
    )

    for i in range(5):
        si.insert_row(
            files_dir,
            si.DEFAULT_DATABASE_NAME,
            "items",
            {"id": i + 1, "title": f"item-{i + 1}"},
        )

    page = si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "items", limit=2, offset=0)
    assert page.total == 5
    assert len(page.rows) == 2
    assert page.rows[0]["id"] == 1

    page2 = si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "items", limit=2, offset=2)
    assert [r["id"] for r in page2.rows] == [3, 4]

    desc = si.list_rows(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        limit=10,
        order_by="id",
        order_direction="desc",
    )
    assert [r["id"] for r in desc.rows] == [5, 4, 3, 2, 1]

    # Update via PK
    updated = si.update_row(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        {"id": 1},
        {"title": "renamed"},
    )
    assert updated["title"] == "renamed"

    # Delete via PK
    si.delete_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {"id": 5})
    page_final = si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "items", limit=10)
    assert page_final.total == 4

    # Missing PK rejected
    with pytest.raises(HTTPException):
        si.update_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {}, {"title": "x"})

    # Unknown order_by rejected
    with pytest.raises(HTTPException):
        si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "items", order_by="nope")


def test_rows_without_primary_key_use_rowid(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "log",
        [si.ColumnDefinition(name="message", type="TEXT")],
    )
    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "log", {"message": "a"})
    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "log", {"message": "b"})

    page = si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "log")
    assert page.total == 2
    assert all("_rowid" in r for r in page.rows)

    target_rowid = page.rows[0]["_rowid"]
    updated = si.update_row(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "log",
        {"_rowid": target_rowid},
        {"message": "edited"},
    )
    assert updated["message"] == "edited"


def test_alter_table_add_and_rename(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True)],
    )

    schema = si.apply_table_alterations(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.TableAlteration(op="add_column", column=si.ColumnDefinition(name="note", type="TEXT")),
            si.TableAlteration(op="rename_table", new_table_name="things"),
        ],
    )
    assert schema.name == "things"
    assert any(c.name == "note" for c in schema.columns)
    assert [t.name for t in si.list_tables(files_dir, si.DEFAULT_DATABASE_NAME)] == ["things"]


def test_alter_table_add_primary_key_column_rejected(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True)],
    )
    with pytest.raises(HTTPException):
        si.apply_table_alterations(
            files_dir,
            si.DEFAULT_DATABASE_NAME,
            "items",
            [
                si.TableAlteration(
                    op="add_column",
                    column=si.ColumnDefinition(name="alt_pk", type="INTEGER", primary_key=True),
                ),
            ],
        )


def test_change_column_type_only_allowed_when_empty(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
            si.ColumnDefinition(name="quantity", type="INTEGER"),
        ],
    )
    schema = si.apply_table_alterations(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.TableAlteration(
                op="change_column_type",
                column_name="quantity",
                column=si.ColumnDefinition(name="quantity", type="REAL"),
            ),
        ],
    )
    assert next(c for c in schema.columns if c.name == "quantity").type == "REAL"

    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {"id": 1, "quantity": 5})
    with pytest.raises(HTTPException):
        si.apply_table_alterations(
            files_dir,
            si.DEFAULT_DATABASE_NAME,
            "items",
            [
                si.TableAlteration(
                    op="change_column_type",
                    column_name="quantity",
                    column=si.ColumnDefinition(name="quantity", type="TEXT"),
                ),
            ],
        )


def test_export_and_import_table_csv(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(
        files_dir,
        si.DEFAULT_DATABASE_NAME,
        "items",
        [
            si.ColumnDefinition(name="id", type="INTEGER", primary_key=True, not_null=True),
            si.ColumnDefinition(name="title", type="TEXT"),
        ],
    )
    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {"id": 1, "title": "Ada"})
    csv_text = si.export_table_csv(files_dir, si.DEFAULT_DATABASE_NAME, "items")
    assert "id,title" in csv_text
    assert "Ada" in csv_text

    summary = si.import_table_csv(files_dir, si.DEFAULT_DATABASE_NAME, "imported", csv_text)
    assert summary.name == "imported"
    page = si.list_rows(files_dir, si.DEFAULT_DATABASE_NAME, "imported")
    assert page.rows == [{"id": "1", "title": "Ada", "_rowid": 1}]


def test_execute_readonly_query_rejects_mutations(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(files_dir, si.DEFAULT_DATABASE_NAME, "items", [si.ColumnDefinition(name="title", type="TEXT")])
    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {"title": "Ada"})

    result = si.execute_readonly_query(files_dir, si.DEFAULT_DATABASE_NAME, "SELECT title FROM items")
    assert result.columns == ["title"]
    assert result.rows == [{"title": "Ada"}]

    with pytest.raises(HTTPException):
        si.execute_readonly_query(files_dir, si.DEFAULT_DATABASE_NAME, "DELETE FROM items")
    with pytest.raises(HTTPException):
        si.execute_readonly_query(files_dir, si.DEFAULT_DATABASE_NAME, "SELECT 1; SELECT 2")


def test_export_and_import_database_file(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    si.create_table(files_dir, si.DEFAULT_DATABASE_NAME, "items", [si.ColumnDefinition(name="title", type="TEXT")])
    si.insert_row(files_dir, si.DEFAULT_DATABASE_NAME, "items", {"title": "Ada"})

    exported = si.export_database_copy(files_dir, si.DEFAULT_DATABASE_NAME)
    try:
        summary = si.import_database_file(files_dir, "copy.sqlite3", exported)
    finally:
        exported.unlink(missing_ok=True)

    assert summary.name == "copy.sqlite3"
    page = si.list_rows(files_dir, "copy.sqlite3", "items")
    assert page.rows[0]["title"] == "Ada"


def test_get_schema_missing_table_404(tmp_path: Path) -> None:
    files_dir = _init(tmp_path)
    with pytest.raises(HTTPException) as exc:
        si.get_table_schema(files_dir, si.DEFAULT_DATABASE_NAME, "missing")
    assert exc.value.status_code == 404
