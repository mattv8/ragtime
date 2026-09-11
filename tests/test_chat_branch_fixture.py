import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.fixtures import chat, cli
from ragtime.fixtures.common import FixtureError, database_url_with_schema, read_json


def _document():
    timestamp = "2026-09-11T12:00:00+00:00"
    return {
        "version": 1,
        "conversation": {
            "id": "source-chat",
            "title": "Original",
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "total_tokens": 3,
            "active_branch_id": "branch-a",
            "tool_output_mode": "show",
            "disabled_builtin_tool_ids": ["debug"],
        },
        "branches": [
            {
                "id": "branch-a",
                "conversation_id": "source-chat",
                "parent_branch_id": None,
                "branch_point_index": 0,
                "branch_kind": None,
                "preserved_messages": [{"role": "assistant", "content": "reply"}],
                "base_messages": [],
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        ],
        "completed_tasks": [
            {
                "id": "task-a",
                "conversation_id": "source-chat",
                "status": "completed",
                "user_message": "hello",
                "response_content": "reply",
                "streaming_state": {"events": []},
                "created_at": timestamp,
                "completed_at": timestamp,
                "last_update_at": timestamp,
            }
        ],
    }


class ChatFixtureValidationTests(unittest.TestCase):
    def test_accepts_legacy_v1_null_kind_and_null_base(self):
        document = _document()
        document.pop("fixture_type", None)
        document["branches"][0]["base_messages"] = None
        self.assertIs(chat.validate_document(document), document)

    def test_rejects_boolean_version_foreign_source_ids_and_task_status(self):
        document = _document()
        document["version"] = True
        with self.assertRaisesRegex(FixtureError, "Unsupported"):
            chat.validate_document(document)
        document = _document()
        document["branches"][0]["conversation_id"] = "other"
        with self.assertRaisesRegex(FixtureError, "branch conversation"):
            chat.validate_document(document)
        document = _document()
        document["completed_tasks"][0]["status"] = "pending"
        with self.assertRaisesRegex(FixtureError, "task status"):
            chat.validate_document(document)

    def test_rejects_cycle_and_duplicate_branch_ids(self):
        document = _document()
        document["branches"][0]["parent_branch_id"] = "branch-a"
        with self.assertRaisesRegex(FixtureError, "cycle"):
            chat.validate_document(document)
        document = _document()
        document["branches"].append(dict(document["branches"][0]))
        with self.assertRaisesRegex(FixtureError, "branch IDs"):
            chat.validate_document(document)


class ChatFixtureExportTests(unittest.TestCase):
    def test_resolve_exact_id_precedes_title_then_literal_substring(self):
        cursor = _Cursor(results=[[("chat-id", "chat-id", "owner", None, None)]])
        self.assertEqual(chat.resolve_chat(cursor, "CHAT-ID")["id"], "chat-id")
        self.assertIn("lower(c.id)", cursor.calls[0][0])
        cursor = _Cursor(results=[[], [("title-id", "Report", "owner", None, None)]])
        self.assertEqual(chat.resolve_chat(cursor, "report")["id"], "title-id")
        cursor = _Cursor(results=[[], [], [("partial-id", "Monthly Report", "owner", None, None)]])
        self.assertEqual(chat.resolve_chat(cursor, "monthly")["id"], "partial-id")

    def test_duplicate_exact_title_and_literal_matches_cap_candidates(self):
        cursor = _Cursor(results=[[], [("one", "Report", "owner", None, None), ("two", "Report", "owner", None, None)]])
        with self.assertRaises(chat.AmbiguousChatError):
            chat.resolve_chat(cursor, "report")
        candidates = [(str(index), f"report {index}", "owner", None, None) for index in range(21)]
        cursor = _Cursor(results=[[], [], candidates])
        with self.assertRaises(chat.AmbiguousChatError) as raised:
            chat.resolve_chat(cursor, "report")
        self.assertEqual(len(raised.exception.candidates), 20)
        self.assertTrue(raised.exception.has_more)

    def test_blank_no_match_and_wildcards_are_safe_literal_bindings(self):
        with self.assertRaises(chat.ChatNotFoundError):
            chat.resolve_chat(_Cursor(), " ")
        with self.assertRaises(chat.ChatNotFoundError):
            chat.resolve_chat(_Cursor(results=[[], [], []]), "missing")
        cursor = _Cursor(results=[[], [], [("id", "100%_done", "owner", None, None)]])
        chat.resolve_chat(cursor, r"100%_\done")
        self.assertEqual(cursor.calls[2][1], (r"%100\%\_\\done%", r"%100\%\_\\done%"))
        self.assertIn("ESCAPE '\\'", cursor.calls[2][0])

    def test_export_uses_one_readonly_snapshot_and_closes_resources(self):
        connection = _Connection(results=[[("id", "title", "owner", None, None)], [({"version": 1},)]], owner=None)
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://host/db?schema=tenant&sslmode=require"}):
            self.assertEqual(chat.export_chat("id", connect=lambda *_args, **_kwargs: connection), {"version": 1})
        queries = [query for query, _params in connection.cursor_instance.calls]
        self.assertIn("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY", queries)
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)
        self.assertTrue(connection.cursor_instance.closed)

    def test_dsn_retains_options_and_cli_never_leaks_dsn(self):
        dsn, schema = database_url_with_schema("postgresql://user:secret@example/db?schema=tenant&sslmode=require")
        self.assertEqual(schema, "tenant")
        self.assertNotIn("schema=", dsn)
        self.assertIn("sslmode=require", dsn)
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(chat, "export_chat", side_effect=RuntimeError("postgres://secret@host")):
            self.assertEqual(cli.main(["export", "chat", "id"], stdout=output, stderr=errors), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("secret", errors.getvalue())


class ChatFixtureImportTests(unittest.TestCase):
    def test_import_remaps_ids_and_persists_completed_historical_tasks(self):
        connection = _Connection(owner=("owner-id",))
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://host/db?schema=public"}):
            result = chat.import_chat(_document(), "local:admin", connect=lambda *_args, **_kwargs: connection)
        self.assertEqual(result["completed_task_count"], 1)
        self.assertTrue(connection.committed)
        inserts = [(query, params) for query, params in connection.cursor_instance.calls if query.lstrip().startswith("INSERT")]
        self.assertNotEqual(inserts[0][1][0], "source-chat")
        self.assertNotEqual(inserts[1][1][0], "branch-a")
        self.assertIn("'completed'", inserts[2][0])
        self.assertNotEqual(inserts[2][1][0], "task-a")

    def test_missing_owner_and_insert_failure_roll_back(self):
        for connection in (_Connection(owner=None), _Connection(owner=("owner",), fail_insert=True)):
            with self.subTest(connection=connection):
                with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://host/db"}):
                    with self.assertRaises(FixtureError if connection.owner is None else RuntimeError):
                        chat.import_chat(_document(), "owner", connect=lambda *_args, **_kwargs: connection)
                self.assertTrue(connection.rolled_back)
                self.assertTrue(connection.closed)
                self.assertFalse(connection.committed)

    def test_commit_failure_rolls_back_and_closes(self):
        connection = _Connection(owner=("owner",), fail_commit=True)
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://host/db"}):
            with self.assertRaises(RuntimeError):
                chat.import_chat(_document(), "owner", connect=lambda *_args, **_kwargs: connection)
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)


class ChatFixtureCliTests(unittest.TestCase):
    def test_stdin_decodes_utf8_bytes_without_relying_on_locale(self):
        self.assertEqual(read_json("-", io.BytesIO('{"title":"café"}'.encode("utf-8"))), {"title": "café"})
        with self.assertRaises(FixtureError):
            read_json("-", io.BytesIO(b'{"title":"\xff"}'))

    def test_chat_cli_success_stdout_and_invalid_input_stderr(self):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(chat, "export_chat", return_value={"version": 1}):
            self.assertEqual(cli.main(["export", "chat", "id"], stdout=output, stderr=errors), 0)
        self.assertEqual(output.getvalue(), '{"version":1}\n')
        with contextlib.redirect_stderr(errors):
            self.assertEqual(cli.main(["import", "chat", "-", "--owner", "owner"], stdout=output, stderr=errors, stdin=io.StringIO("not json")), 1)
        self.assertIn("valid UTF-8 JSON", errors.getvalue())

    def test_registry_allows_non_chat_options_and_binary_output(self):
        class BinaryFixture:
            @staticmethod
            def register_export(parser):
                parser.add_argument("--flavor", required=True)
                parser.set_defaults(run=BinaryFixture.export)

            @staticmethod
            def register_import(parser):
                parser.add_argument("--ignored", action="store_true")
                parser.set_defaults(run=lambda _args: 0)

            @staticmethod
            def export(args):
                args.stdout.write(b"binary:" + args.flavor.encode())
                return 0

        output = io.BytesIO()
        self.assertEqual(cli.main(["export", "binary", "--flavor", "zip"], stdout=output, handlers={"binary": BinaryFixture}), 0)
        self.assertEqual(output.getvalue(), b"binary:zip")

    def test_wrapper_argv_uses_module_cli(self):
        root = Path(__file__).parents[1]
        wrapper = root / "docker/scripts/export.sh"
        if not wrapper.is_file():
            wrapper = Path("/docker-scripts/export.sh")
        if wrapper.is_file():
            with tempfile.TemporaryDirectory() as temporary_directory:
                capture = Path(temporary_directory) / "arguments"
                fake_python = Path(temporary_directory) / "python"
                fake_python.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{capture}"\n')
                fake_python.chmod(0o755)
                subprocess.run(["sh", wrapper, "chat", "selector"], env={"PATH": f"{temporary_directory}:{os.environ['PATH']}"}, check=True)
                self.assertEqual(capture.read_text().splitlines(), ["-m", "ragtime.fixtures.cli", "export", "chat", "selector"])
        else:
            self.skipTest("export wrapper is unavailable in this environment")

    def test_production_stage_installs_import_and_export(self):
        dockerfile = Path(__file__).parents[1] / "docker/Dockerfile"
        if not dockerfile.is_file():
            self.skipTest("production Dockerfile is unavailable in this environment")
        production = dockerfile.read_text()
        self.assertIn("COPY docker/scripts/import.sh /docker-scripts/import.sh", production)
        self.assertIn("/usr/local/bin/import", production)

    def test_development_stage_installs_import_and_export(self):
        dockerfile = Path(__file__).parents[1] / "docker/Dockerfile.dev"
        if not dockerfile.is_file():
            self.skipTest("development Dockerfile is unavailable in this environment")
        development = dockerfile.read_text()
        self.assertIn("COPY docker/scripts/import.sh /docker-scripts/import.sh", development)
        self.assertIn("/usr/local/bin/import", development)


@unittest.skipUnless(os.environ.get("CHAT_FIXTURE_LIVE_TEST_SELECTOR"), "set CHAT_FIXTURE_LIVE_TEST_SELECTOR to run against DATABASE_URL")
class ChatFixtureLiveDatabaseTests(unittest.TestCase):
    def test_export_returns_a_v1_fixture_from_the_configured_database(self):
        document = chat.export_chat(os.environ["CHAT_FIXTURE_LIVE_TEST_SELECTOR"])
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["fixture_type"], "chat")
        self.assertIn("conversation", document)


@unittest.skipUnless(
    os.environ.get("CHAT_FIXTURE_CONTAINER_TESTS") == "1" and os.environ.get("CHAT_FIXTURE_LIVE_TEST_SELECTOR"),
    "local container CLI integration needs CHAT_FIXTURE_LIVE_TEST_SELECTOR",
)
class ChatFixtureContainerTests(unittest.TestCase):
    def test_installed_export_command_returns_selected_fixture(self):
        selector = os.environ["CHAT_FIXTURE_LIVE_TEST_SELECTOR"]
        result = subprocess.run(["/usr/local/bin/export", "chat", selector], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout)["fixture_type"], "chat")


class _Cursor:
    def __init__(self, results=None, owner=None, fail_insert=False):
        self.results = list(results or [])
        self.owner = owner
        self.fail_insert = fail_insert
        self.calls = []
        self.current = []
        self.closed = False

    def execute(self, query, params=None):
        self.calls.append((query, params))
        if self.fail_insert and query.lstrip().startswith("INSERT"):
            raise RuntimeError("insert failed")
        if "FROM conversations c" in query or "jsonb_build_object" in query:
            self.current = self.results.pop(0) if self.results else []
        elif "SELECT id FROM users" in query:
            self.current = [self.owner] if self.owner else []

    def fetchall(self):
        return self.current

    def fetchone(self):
        return self.current[0] if self.current else None

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, results=None, owner=None, fail_insert=False, fail_commit=False):
        self.owner = owner
        self.cursor_instance = _Cursor(results, owner, fail_insert)
        self.fail_commit = fail_commit
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True
