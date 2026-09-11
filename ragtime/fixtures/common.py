"""Database and JSON helpers shared by fixture handlers."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_MILLISECONDS = 30_000


class FixtureError(Exception):
    """A safe, user-facing fixture error."""


class DatabaseConfigurationError(FixtureError):
    """DATABASE_URL was unavailable."""


def database_url_with_schema(database_url: str) -> tuple[str, str]:
    """Remove Prisma's schema URL argument while retaining libpq options."""
    parts = urlsplit(database_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    schemas = [value for key, value in query if key == "schema"]
    dsn = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode([(k, v) for k, v in query if k != "schema"]), parts.fragment))
    return dsn, schemas[-1] if schemas else "public"


def search_path(schema: str) -> str:
    return f'"{schema.replace(chr(34), chr(34) * 2)}"'


@contextlib.contextmanager
def database_cursor(*, read_only: bool, connect=None) -> Iterator[Any]:
    """Yield a configured cursor and commit only successful write transactions."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise DatabaseConfigurationError("Fixture database configuration is missing.")
    dsn, schema = database_url_with_schema(database_url)
    if connect is None:
        connect = importlib.import_module("psycopg2").connect
    connection = connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    cursor = None
    try:
        cursor = connection.cursor()
        if read_only:
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        else:
            cursor.execute("BEGIN")
        cursor.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MILLISECONDS,))
        cursor.execute("SELECT set_config('search_path', %s, true)", (search_path(schema),))
        yield cursor
        if read_only:
            try:
                connection.rollback()
            except Exception:
                pass
        else:
            connection.commit()
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


def json_parameter(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def read_json(source: str | None, stdin) -> object:
    """Read a UTF-8 JSON document from stdin (default/-) or a file."""
    try:
        if source is None or source == "-":
            # Use raw stdin when available so fixture encoding is independent
            # of the invoking shell's locale. StringIO remains useful to callers.
            content = getattr(stdin, "buffer", stdin).read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return json.loads(content)
        with open(source, encoding="utf-8") as fixture_file:
            return json.load(fixture_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureError("Fixture input must be valid UTF-8 JSON.") from error
