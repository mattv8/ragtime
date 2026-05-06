import sqlite3
from pathlib import Path

from ragtime.userspace.sqlite_import import (
    decode_sql_dump_bytes,
    detect_sql_dialect,
    import_sql_to_sqlite,
)


def _import(tmp_path: Path, sql: str, dialect: str | None = None):
    sqlite_path = tmp_path / "app.sqlite3"
    result = import_sql_to_sqlite(sqlite_path, sql, dialect)  # type: ignore[arg-type]
    conn = sqlite3.connect(str(sqlite_path))
    return result, conn


def test_detects_headerless_postgres_copy_dump() -> None:
    sql = 'CREATE TABLE "public"."users" ("id" integer);\nCOPY "public"."users" ("id") FROM stdin;\n1\n\\.\n'

    assert detect_sql_dialect(sql) == "postgresql"


def test_imports_postgres_schema_qualified_copy(tmp_path: Path) -> None:
    sql = """-- PostgreSQL database dump
CREATE TABLE public.users (
    id integer NOT NULL,
    name text,
    note text
);
COPY public.users (id, name, note) FROM stdin;
1	Ada	hello\\nworld
2	Bob	\\tTabbed
\\.
"""

    result, conn = _import(tmp_path, sql)

    assert result.success is True
    assert result.dialect == "postgresql"
    assert result.tables_created == 1
    assert result.rows_inserted == 2
    assert conn.execute("SELECT * FROM users ORDER BY id").fetchall() == [
        (1, "Ada", "hello\nworld"),
        (2, "Bob", "\tTabbed"),
    ]
    conn.close()


def test_imports_postgres_quoted_schema_copy(tmp_path: Path) -> None:
    sql = """CREATE TABLE "public"."users" ("id" integer, "name" text);
COPY "public"."users" ("id", "name") FROM stdin;
1	Ada
\\.
"""

    result, conn = _import(tmp_path, sql)

    assert result.success is True
    assert conn.execute("SELECT * FROM users").fetchall() == [(1, "Ada")]
    conn.close()


def test_skips_common_postgres_restore_noise(tmp_path: Path) -> None:
    sql = """CREATE TABLE users (id integer);
ALTER TABLE public.users OWNER TO app;
ALTER TABLE ONLY public.users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
SELECT pg_catalog.setval('users_id_seq', 1, false);
INSERT INTO users VALUES (1);
"""

    result, conn = _import(tmp_path, sql, "postgresql")

    assert result.success is True
    assert result.errors == []
    assert conn.execute("SELECT * FROM users").fetchall() == [(1,)]
    conn.close()


def test_imports_mysql_auto_increment_and_on_duplicate(tmp_path: Path) -> None:
    sql = """-- MySQL dump
CREATE TABLE `users` (`id` int NOT NULL AUTO_INCREMENT, `name` varchar(255), PRIMARY KEY (`id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `users` (`id`, `name`) VALUES (1, 'Ada') ON DUPLICATE KEY UPDATE `name`=VALUES(`name`);
INSERT INTO `users` (`id`, `name`) VALUES (1, 'Grace') ON DUPLICATE KEY UPDATE `name`=VALUES(`name`);
"""

    result, conn = _import(tmp_path, sql)

    assert result.success is True
    assert result.dialect == "mysql"
    assert result.tables_created == 1
    assert conn.execute("SELECT * FROM users").fetchall() == [(1, "Grace")]
    conn.close()


def test_statement_splitter_preserves_semicolons_inside_strings(tmp_path: Path) -> None:
    sql = """CREATE TABLE notes (body text);
INSERT INTO notes VALUES ('first; second');
"""

    result, conn = _import(tmp_path, sql, "sqlite")

    assert result.success is True
    assert conn.execute("SELECT body FROM notes").fetchall() == [("first; second",)]
    conn.close()


def test_rolls_back_when_statement_fails(tmp_path: Path) -> None:
    sql = """CREATE TABLE users (id integer);
INSERT INTO users VALUES (1);
INSERT INTO missing_table VALUES (2);
"""

    result, conn = _import(tmp_path, sql, "sqlite")

    assert result.success is False
    assert result.tables_created == 0
    assert any("Rolled back" in warning for warning in result.warnings)
    assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    conn.close()


def test_decodes_utf16_sql_dump_bytes() -> None:
    sql = "CREATE TABLE users (id integer);"

    assert decode_sql_dump_bytes(sql.encode("utf-16")).startswith("CREATE TABLE")