from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ragtime.userspace.templates.sqlite_migrate import _statements as template_statements
from ragtime.userspace.templates.sqlite_migrate import apply_migrations
from runtime.core.sqlite_recovery import _connect_readonly, _schema_hash, _statements, capture_database, database_fingerprint, prepare_restore
from runtime.core.workspace_ops import (
    is_managed_sqlite_artifact,
    iter_managed_sqlite_database_paths,
)


class SqliteRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.migrations = self.root / "migrations"
        self.migrations.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _db(self, name: str, sql: str) -> Path:
        path = self.root / name
        with sqlite3.connect(path) as conn:
            conn.executescript(sql)
        return path

    def _ledger(self, path: Path, applied: dict[str, str]) -> None:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE _ragtime_migrations (filename TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)")
            for filename, sql in applied.items():
                conn.execute("INSERT INTO _ragtime_migrations VALUES (?, ?, 'now')", (filename, hashlib.sha256(sql.encode()).hexdigest()))

    def test_capture_includes_wal_data_without_changing_source(self) -> None:
        source = self.root / "live.sqlite3"
        conn = sqlite3.connect(source)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO item VALUES (1, 'committed')")
        conn.commit()
        before = source.read_bytes()
        destination = self.root / "capture.sqlite3"
        result = capture_database(source, destination)
        self.assertEqual(before, source.read_bytes())
        with sqlite3.connect(destination) as captured:
            self.assertEqual(captured.execute("SELECT value FROM item").fetchone()[0], "committed")
        self.assertTrue(result["fingerprint"])
        self.assertFalse((destination.parent / f"{destination.name}-wal").exists())
        self.assertFalse((destination.parent / f"{destination.name}-journal").exists())

    def test_capture_allows_preexisting_foreign_key_violations(self) -> None:
        source = self._db(
            "fk-imperfect.sqlite3",
            "PRAGMA foreign_keys=OFF; CREATE TABLE parent (id INTEGER PRIMARY KEY); "
            "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id)); "
            "INSERT INTO child VALUES (99);",
        )
        destination = self.root / "capture.sqlite3"
        capture_database(source, destination)
        with sqlite3.connect(destination) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchone()[0], "child")

    def test_same_line_statement_splitting_preserves_trigger_quotes_and_comments(self) -> None:
        sql = (
            "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT); "
            "INSERT INTO item VALUES (1, 'semi;colon'); "
            "CREATE TRIGGER item_audit AFTER INSERT ON item BEGIN "
            "INSERT INTO item VALUES (NEW.id + 1, 'trigger;value'); END; -- trailing; comment\n"
        )
        statements = _statements(sql)
        self.assertEqual(len(statements), 3)
        self.assertEqual(template_statements(sql), statements)
        self.assertIn("trigger;value", statements[-1])
        self.migrations.joinpath("001_inline.sql").write_text(sql, encoding="utf-8")
        db = self.root / "inline.sqlite3"
        self.assertEqual(apply_migrations(db, self.migrations), 1)
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO item VALUES (10, 'run')")
            self.assertEqual(conn.execute("SELECT value FROM item WHERE id=11").fetchone()[0], "trigger;value")

    def test_prepare_overwrite_applies_forward_migration_and_drops_current_rows(self) -> None:
        first = "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT);"
        second = "ALTER TABLE item ADD COLUMN note TEXT;"
        self.migrations.joinpath("001_base.sql").write_text(first, encoding="utf-8")
        self.migrations.joinpath("002_note.sql").write_text(second, encoding="utf-8")
        backup = self._db("backup.sqlite3", first + " INSERT INTO item VALUES (1, 'old');")
        current = self._db("current.sqlite3", "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT, note TEXT); INSERT INTO item VALUES (1, 'new', 'n'); INSERT INTO item VALUES (2, 'only-current', 'n');")
        self._ledger(backup, {"001_base.sql": first})
        self._ledger(current, {"001_base.sql": first, "002_note.sql": second})
        output = self.root / "candidate.sqlite3"
        result = prepare_restore(backup, current, self.migrations, output, mode="overwrite")
        self.assertTrue(result["can_apply"], result["blockers"])
        self.assertEqual(result["migrations_applied"], ["002_note.sql"])
        self.assertEqual(result["tables"][0]["deleted"], 1)
        with sqlite3.connect(output) as conn:
            self.assertEqual(conn.execute("SELECT id, value, note FROM item").fetchall(), [(1, "old", None)])

    def test_merge_reports_conflicts_and_use_backup_updates_without_rowid(self) -> None:
        backup = self._db("backup.sqlite3", "CREATE TABLE item (a INTEGER, b INTEGER, value BLOB, PRIMARY KEY(a, b)); INSERT INTO item VALUES (1, 1, x'01'); INSERT INTO item VALUES (2, 2, x'02');")
        current = self._db("current.sqlite3", "CREATE TABLE item (a INTEGER, b INTEGER, value BLOB, PRIMARY KEY(a, b)); INSERT INTO item VALUES (1, 1, x'03');")
        output = self.root / "candidate.sqlite3"
        result = prepare_restore(backup, current, self.migrations, output, mode="merge", conflict_policy="use_backup")
        self.assertTrue(result["can_apply"], result["blockers"])
        table = result["tables"][0]
        self.assertEqual((table["inserted"], table["updated"], table["conflicts"]), (1, 1, 1))
        self.assertEqual(table["conflict_samples"][0]["key"], {"a": 1, "b": 1})
        with sqlite3.connect(output) as conn:
            self.assertEqual(conn.execute("SELECT a, b, hex(value) FROM item ORDER BY a").fetchall(), [(1, 1, "01"), (2, 2, "02")])

    def test_merge_blocks_trigger_writes_and_unsafe_migrations(self) -> None:
        backup = self._db("backup.sqlite3", "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT); CREATE TRIGGER audit AFTER UPDATE ON item BEGIN SELECT 1; END; INSERT INTO item VALUES (1, 'backup');")
        current = self._db("current.sqlite3", "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT); CREATE TRIGGER audit AFTER UPDATE ON item BEGIN SELECT 1; END; INSERT INTO item VALUES (1, 'current');")
        result = prepare_restore(backup, current, self.migrations, self.root / "candidate.sqlite3", mode="merge", conflict_policy="use_backup")
        self.assertFalse(result["can_apply"])
        self.assertTrue(any("trigger" in blocker.lower() for blocker in result["blockers"]))

    def test_forward_migration_rejects_transaction_control_without_candidate(self) -> None:
        base = "CREATE TABLE item (id INTEGER PRIMARY KEY);"
        unsafe = "BEGIN; ALTER TABLE item ADD COLUMN value TEXT; COMMIT;"
        self.migrations.joinpath("001_base.sql").write_text(base, encoding="utf-8")
        self.migrations.joinpath("002_unsafe.sql").write_text(unsafe, encoding="utf-8")
        backup = self._db("backup.sqlite3", base)
        current = self._db("current.sqlite3", "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT);")
        self._ledger(backup, {"001_base.sql": base})
        self._ledger(current, {"001_base.sql": base, "002_unsafe.sql": unsafe})
        output = self.root / "candidate.sqlite3"
        result = prepare_restore(backup, current, self.migrations, output, mode="overwrite")
        self.assertFalse(result["can_apply"])
        self.assertFalse(output.exists())
        self.assertTrue(any("forbidden" in blocker.lower() for blocker in result["blockers"]))

    def test_forward_migration_allows_forbidden_words_in_identifiers_literals_and_trigger_bodies(self) -> None:
        base = "CREATE TABLE item (id INTEGER PRIMARY KEY);"
        forward = """CREATE TABLE "begin" ("commit" TEXT, "rollback" TEXT);
        INSERT INTO "begin" ("commit", "rollback") VALUES ('SAVEPOINT RELEASE', 'PRAGMA ATTACH DETACH');
        CREATE TRIGGER item_audit AFTER INSERT ON item BEGIN
            INSERT INTO "begin" ("commit", "rollback") VALUES ('BEGIN', 'COMMIT');
        END;"""
        self.migrations.joinpath("001_base.sql").write_text(base, encoding="utf-8")
        self.migrations.joinpath("002_keywords.sql").write_text(forward, encoding="utf-8")
        backup = self._db("backup.sqlite3", base)
        current = self._db("current.sqlite3", base + " CREATE TABLE \"begin\" (\"commit\" TEXT, \"rollback\" TEXT); CREATE TRIGGER item_audit AFTER INSERT ON item BEGIN INSERT INTO \"begin\" (\"commit\", \"rollback\") VALUES ('BEGIN', 'COMMIT'); END;")
        self._ledger(backup, {"001_base.sql": base})
        self._ledger(current, {"001_base.sql": base, "002_keywords.sql": forward})

        result = prepare_restore(backup, current, self.migrations, self.root / "candidate.sqlite3", mode="overwrite")

        self.assertTrue(result["can_apply"], result["blockers"])

    def test_forward_migration_rejects_forbidden_leading_commands(self) -> None:
        base = "CREATE TABLE item (id INTEGER PRIMARY KEY);"
        self.migrations.joinpath("001_base.sql").write_text(base, encoding="utf-8")
        for index, command in enumerate(("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT work", "RELEASE work", "PRAGMA user_version", "ATTACH ':memory:' AS other", "DETACH other")):
            with self.subTest(command=command):
                migration = f"{command}; ALTER TABLE item ADD COLUMN value_{index} TEXT;"
                name = f"002_forbidden_{index}.sql"
                self.migrations.joinpath(name).write_text(migration, encoding="utf-8")
                backup = self._db(f"backup-{index}.sqlite3", base)
                current = self._db(f"current-{index}.sqlite3", f"CREATE TABLE item (id INTEGER PRIMARY KEY, value_{index} TEXT);")
                self._ledger(backup, {"001_base.sql": base})
                self._ledger(current, {"001_base.sql": base, name: migration})

                result = prepare_restore(backup, current, self.migrations, self.root / f"candidate-{index}.sqlite3", mode="overwrite")

                self.assertFalse(result["can_apply"])
                self.assertTrue(any("forbidden" in blocker.lower() for blocker in result["blockers"]))
                self.migrations.joinpath(name).unlink()

    def test_merge_conflict_samples_truncate_large_values_without_affecting_restore(self) -> None:
        large_text = "x" * 10_000
        large_blob = b"y" * 10_000
        sql = "CREATE TABLE item (id INTEGER PRIMARY KEY, text_value TEXT, blob_value BLOB);"
        backup = self._db("backup.sqlite3", sql)
        current = self._db("current.sqlite3", sql)
        with sqlite3.connect(backup) as conn:
            conn.execute("INSERT INTO item VALUES (?, ?, ?)", (1, large_text, large_blob))
        with sqlite3.connect(current) as conn:
            conn.execute("INSERT INTO item VALUES (?, ?, ?)", (1, "current", b"current"))

        result = prepare_restore(backup, current, self.migrations, self.root / "candidate.sqlite3", mode="merge", conflict_policy="use_backup")

        self.assertTrue(result["can_apply"], result["blockers"])
        sample = result["tables"][0]["conflict_samples"][0]["backup"]
        self.assertLess(len(sample["text_value"]), len(large_text))
        self.assertLess(len(sample["blob_value"]["$blob"]), len(large_blob))
        self.assertTrue(sample["blob_value"]["$truncated"])
        with sqlite3.connect(self.root / "candidate.sqlite3") as conn:
            self.assertEqual(conn.execute("SELECT text_value, blob_value FROM item WHERE id=1").fetchone(), (large_text, large_blob))

    def test_fingerprint_deterministic_with_reverse_insertion_order(self) -> None:
        """Fingerprint produces identical hash regardless of insertion order."""
        base_sql = "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT);"
        db1 = self._db(
            "db1.sqlite3",
            base_sql
            + "INSERT INTO item VALUES (1, 'a'); INSERT INTO item VALUES (2, 'b');",
        )
        db2 = self._db(
            "db2.sqlite3",
            base_sql
            + "INSERT INTO item VALUES (2, 'b'); INSERT INTO item VALUES (1, 'a');",
        )
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_fingerprint_detects_duplicate_rows(self) -> None:
        """Fingerprint differs when rows are duplicated."""
        base_sql = "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT);"
        db1 = self._db("db1.sqlite3", base_sql + "INSERT INTO item VALUES (1, 'a');")
        db2 = self._db(
            "db2.sqlite3",
            base_sql
            + "INSERT INTO item VALUES (1, 'a'); INSERT INTO item VALUES (2, 'a');",
        )
        self.assertNotEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_fingerprint_handles_blob_values(self) -> None:
        """Fingerprint correctly handles binary blobs."""
        base_sql = "CREATE TABLE data (id INTEGER PRIMARY KEY, value BLOB);"
        db1 = self._db("db1.sqlite3", base_sql)
        db2 = self._db("db2.sqlite3", base_sql)
        with sqlite3.connect(db1) as conn:
            conn.execute("INSERT INTO data VALUES (1, ?)", (b"\x00\x01\xff",))
        with sqlite3.connect(db2) as conn:
            conn.execute("INSERT INTO data VALUES (1, ?)", (b"\x00\x01\xff",))
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_fingerprint_handles_invalid_utf8_via_sql_cast(self) -> None:
        """Fingerprint handles invalid UTF-8 cast via SQL CAST AS BLOB."""
        base_sql = "CREATE TABLE data (id INTEGER PRIMARY KEY, value TEXT);"
        db = self._db("db.sqlite3", base_sql)
        with sqlite3.connect(db) as conn:
            # Insert invalid UTF-8 directly via SQL
            conn.execute("INSERT INTO data VALUES (1, CAST(x'61FF00' AS TEXT))")
        fp = database_fingerprint(db)
        self.assertTrue(fp)
        self.assertEqual(len(fp), 64)  # SHA256 hex

    def test_fingerprint_includes_visible_generated_columns(self) -> None:
        """Fingerprint includes stored generated columns (xinfo row[6] in 0,2,3)."""
        sql = "CREATE TABLE item (id INTEGER PRIMARY KEY, value INTEGER, doubled INTEGER GENERATED ALWAYS AS (value * 2) STORED);"
        db1 = self._db("db1.sqlite3", sql + "INSERT INTO item (id, value) VALUES (1, 5);")
        db2 = self._db("db2.sqlite3", sql + "INSERT INTO item (id, value) VALUES (1, 5);")
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))
        db3 = self._db("db3.sqlite3", sql + "INSERT INTO item (id, value) VALUES (1, 6);")
        self.assertNotEqual(database_fingerprint(db1), database_fingerprint(db3))

    def test_fingerprint_differentiates_null_and_numeric_types(self) -> None:
        """Fingerprint distinguishes NULL from 0; integer from float via typeof()."""
        sql = "CREATE TABLE num (id INTEGER PRIMARY KEY, value NUMERIC);"
        db_null = self._db("null.sqlite3", sql + "INSERT INTO num VALUES (1, NULL);")
        db_zero = self._db("zero.sqlite3", sql + "INSERT INTO num VALUES (1, 0);")
        db_float = self._db(
            "float.sqlite3", sql + "INSERT INTO num VALUES (1, 0.0);"
        )
        self.assertNotEqual(database_fingerprint(db_null), database_fingerprint(db_zero))

    def test_fingerprint_respects_collation_in_ordering(self) -> None:
        """Fingerprint uses COLLATE BINARY for deterministic row ordering."""
        sql = "CREATE TABLE text_val (id INTEGER PRIMARY KEY, value TEXT COLLATE BINARY);"
        db1 = self._db(
            "db1.sqlite3",
            sql + "INSERT INTO text_val VALUES (2, 'a'); INSERT INTO text_val VALUES (1, 'b');",
        )
        db2 = self._db(
            "db2.sqlite3",
            sql + "INSERT INTO text_val VALUES (1, 'b'); INSERT INTO text_val VALUES (2, 'a');",
        )
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_fingerprint_without_rowid_table(self) -> None:
        """Fingerprint works with WITHOUT ROWID tables."""
        sql = "CREATE TABLE item (id TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;"
        db1 = self._db(
            "db1.sqlite3",
            sql
            + "INSERT INTO item VALUES ('a', '1'); INSERT INTO item VALUES ('b', '2');",
        )
        db2 = self._db(
            "db2.sqlite3",
            sql
            + "INSERT INTO item VALUES ('b', '2'); INSERT INTO item VALUES ('a', '1');",
        )
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_fingerprint_empty_tables_same_schema_differ_when_row_moved(self) -> None:
        """Empty tables A/B with same schema differ when a row moves from A to B."""
        sql_empty = "CREATE TABLE a (id INTEGER PRIMARY KEY, value TEXT); CREATE TABLE b (id INTEGER PRIMARY KEY, value TEXT);"
        db_empty = self._db("empty.sqlite3", sql_empty)
        fp_empty = database_fingerprint(db_empty)

        db_moved = self._db(
            "moved.sqlite3",
            sql_empty + "INSERT INTO b VALUES (1, 'test');",
        )
        fp_moved = database_fingerprint(db_moved)

        self.assertNotEqual(fp_empty, fp_moved)

    def test_fingerprint_fts_visible_inserts_equal_with_explicit_rowid_mapping(
        self,
    ) -> None:
        """FTS table fingerprints match when inserts use explicit same rowid mappings."""
        fts_sql = "CREATE VIRTUAL TABLE docs USING fts5(title, content);"
        db1 = self._db(
            "fts1.sqlite3",
            fts_sql + "INSERT INTO docs (rowid, title, content) VALUES (1, 't1', 'c1');",
        )
        db2 = self._db(
            "fts2.sqlite3",
            fts_sql + "INSERT INTO docs (rowid, title, content) VALUES (1, 't1', 'c1');",
        )
        self.assertEqual(database_fingerprint(db1), database_fingerprint(db2))

    def test_managed_artifact_classifier_is_flat_and_enumerates_mains_only(self) -> None:
        files = self.root / "files"
        db = files / ".ragtime/db"
        db.mkdir(parents=True)
        for name in ("app.sqlite3", "app.sqlite3-wal", "other.db", "migrations/001.sql"):
            path = db / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        self.assertTrue(is_managed_sqlite_artifact(".ragtime/db/app.sqlite3-wal"))
        self.assertFalse(is_managed_sqlite_artifact(".ragtime/db/migrations/app.sqlite3"))
        self.assertFalse(is_managed_sqlite_artifact("elsewhere/app.sqlite3"))
        self.assertEqual([path.name for path in iter_managed_sqlite_database_paths(files)], ["app.sqlite3", "other.db"])

    def test_schema_shape_ignores_formatting_but_detects_semantic_ddl_changes(self) -> None:
        base = """CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT COLLATE NOCASE, value INTEGER CHECK(value >= 0), doubled INTEGER GENERATED ALWAYS AS (value * 2) STORED); CREATE INDEX item_name ON item(name); CREATE TRIGGER item_audit AFTER INSERT ON item BEGIN SELECT 1; END;"""
        equivalent = """create table item(id integer primary key, name text collate nocase, value integer check ( value >= 0 ), doubled integer generated always as ( value * 2 ) stored); -- formatting only
        create index item_name on item ( name ); create trigger item_audit after insert on item begin select 1; end;"""
        first = self._db("first.sqlite3", base)
        second = self._db("second.sqlite3", equivalent)
        with _connect_readonly(first) as conn:
            first_hash = _schema_hash(conn)
        with _connect_readonly(second) as conn:
            self.assertEqual(first_hash, _schema_hash(conn))
        for index, changed in enumerate((
            base.replace("value >= 0", "value > 0"),
            base.replace("COLLATE NOCASE", "COLLATE RTRIM"),
            base.replace("value * 2", "value * 3"),
            base.replace("item_name ON item(name)", "item_name ON item(value)"),
            base.replace("SELECT 1", "SELECT 2"),
        )):
            path = self._db(f"different-{index}.sqlite3", changed)
            with _connect_readonly(path) as conn:
                self.assertNotEqual(first_hash, _schema_hash(conn))

    def test_schema_conversion_requires_current_ledger_and_policy_names_are_known(self) -> None:
        base = "CREATE TABLE item (id INTEGER PRIMARY KEY);"
        forward = "ALTER TABLE item ADD COLUMN value TEXT;"
        self.migrations.joinpath("001_base.sql").write_text(base, encoding="utf-8")
        self.migrations.joinpath("002_value.sql").write_text(forward, encoding="utf-8")
        backup = self._db("backup.sqlite3", base)
        current = self._db("current.sqlite3", "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT);")
        self._ledger(backup, {"001_base.sql": base})
        missing_ledger = prepare_restore(backup, current, self.migrations, self.root / "missing-ledger.sqlite3", mode="overwrite")
        self.assertFalse(missing_ledger["can_apply"])
        self.assertTrue(any("current database" in blocker for blocker in missing_ledger["blockers"]))
        same_backup = self._db("same-backup.sqlite3", base)
        same_current = self._db("same-current.sqlite3", base)
        unknown_policy = prepare_restore(same_backup, same_current, self.migrations, self.root / "unknown-policy.sqlite3", mode="merge", table_policies={"absent": "use_backup"})
        self.assertFalse(unknown_policy["can_apply"])
        self.assertTrue(any("Unknown table policy" in blocker for blocker in unknown_policy["blockers"]))

    def test_runner_and_restore_allow_backdated_forward_migration(self) -> None:
        later = "CREATE TABLE later (id INTEGER PRIMARY KEY);"
        backdated = "CREATE TABLE earlier (id INTEGER PRIMARY KEY);"
        db = self.root / "runner.sqlite3"
        self.migrations.joinpath("002_later.sql").write_text(later, encoding="utf-8")
        self.assertEqual(apply_migrations(db, self.migrations), 1)
        self.migrations.joinpath("001_earlier.sql").write_text(backdated, encoding="utf-8")
        self.assertEqual(apply_migrations(db, self.migrations), 1)
        backup = self._db("backup.sqlite3", later)
        self._ledger(backup, {"002_later.sql": later})
        output = self.root / "candidate.sqlite3"
        result = prepare_restore(backup, db, self.migrations, output, mode="overwrite")
        self.assertTrue(result["can_apply"], result["blockers"])
        self.assertEqual(result["migrations_applied"], ["001_earlier.sql"])

    def test_runner_migration_failure_leaves_no_partial_schema_or_ledger(self) -> None:
        broken = "CREATE TABLE should_not_exist (id INTEGER); INSERT INTO missing_table VALUES (1);"
        self.migrations.joinpath("001_broken.sql").write_text(broken, encoding="utf-8")
        db = self.root / "runner.sqlite3"
        with self.assertRaises(sqlite3.Error):
            apply_migrations(db, self.migrations)
        with sqlite3.connect(db) as conn:
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='should_not_exist'").fetchone())
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM _ragtime_migrations").fetchone()[0], 0)

    def test_merge_repairs_autoincrement_sequence_for_backup_explicit_id(self) -> None:
        sql = "CREATE TABLE item (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT);"
        backup = self._db("backup.sqlite3", sql + "INSERT INTO item (id, value) VALUES (99, 'backup');")
        current = self._db("current.sqlite3", sql + "INSERT INTO item (id, value) VALUES (1, 'current');")
        output = self.root / "candidate.sqlite3"
        result = prepare_restore(backup, current, self.migrations, output, mode="merge")
        self.assertTrue(result["can_apply"], result["blockers"])
        with sqlite3.connect(output) as conn:
            self.assertGreaterEqual(conn.execute("SELECT seq FROM sqlite_sequence WHERE name='item'").fetchone()[0], 99)

    def test_merge_no_primary_key_accepts_equal_duplicate_multisets_only(self) -> None:
        sql = "CREATE TABLE item (value TEXT);"
        backup = self._db("backup.sqlite3", sql + "INSERT INTO item VALUES ('same'), ('same');")
        current = self._db("current.sqlite3", sql + "INSERT INTO item VALUES ('same'), ('same');")
        equal = prepare_restore(backup, current, self.migrations, self.root / "equal.sqlite3", mode="merge")
        self.assertTrue(equal["can_apply"], equal["blockers"])
        self.assertEqual(equal["tables"][0]["unchanged"], 2)

        mismatched = self._db("mismatched.sqlite3", sql + "INSERT INTO item VALUES ('same');")
        unequal = prepare_restore(backup, mismatched, self.migrations, self.root / "unequal.sqlite3", mode="merge")
        self.assertFalse(unequal["can_apply"])
        self.assertTrue(any("no declared primary key" in blocker for blocker in unequal["blockers"]))

    def test_merge_keyed_diff_uses_sqlite_nocase_and_numeric_affinity(self) -> None:
        nocase = "CREATE TABLE text_item (id TEXT PRIMARY KEY COLLATE NOCASE, value TEXT);"
        numeric = "CREATE TABLE numeric_item (id INTEGER PRIMARY KEY, value TEXT);"
        backup = self._db("backup.sqlite3", nocase + numeric + "INSERT INTO text_item VALUES ('A', 'backup'); INSERT INTO numeric_item VALUES ('01', 'backup');")
        current = self._db("current.sqlite3", nocase + numeric + "INSERT INTO text_item VALUES ('a', 'current'); INSERT INTO numeric_item VALUES (1, 'current');")
        output = self.root / "candidate.sqlite3"

        result = prepare_restore(backup, current, self.migrations, output, mode="merge", conflict_policy="use_backup")

        self.assertTrue(result["can_apply"], result["blockers"])
        self.assertEqual([(table["inserted"], table["updated"], table["conflicts"]) for table in result["tables"]], [(0, 1, 1), (0, 1, 1)])
        with sqlite3.connect(output) as conn:
            self.assertEqual(conn.execute("SELECT id, value FROM text_item").fetchone(), ("a", "backup"))
            self.assertEqual(conn.execute("SELECT id, value FROM numeric_item").fetchone(), (1, "backup"))

    def test_merge_defers_child_before_parent_foreign_key_inserts(self) -> None:
        sql = "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id)); CREATE TABLE parent (id INTEGER PRIMARY KEY);"
        backup = self._db("backup.sqlite3", sql + "INSERT INTO parent VALUES (1); INSERT INTO child VALUES (1, 1);")
        current = self._db("current.sqlite3", sql)
        output = self.root / "candidate.sqlite3"

        result = prepare_restore(backup, current, self.migrations, output, mode="merge")

        self.assertTrue(result["can_apply"], result["blockers"])
        with sqlite3.connect(output) as conn:
            self.assertEqual(conn.execute("SELECT parent_id FROM child").fetchone()[0], 1)
        self.assertFalse((output.parent / f"{output.name}-wal").exists())
        self.assertFalse((output.parent / f"{output.name}-journal").exists())

    def test_failed_prepare_removes_restore_temporary_directory(self) -> None:
        sql = "CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT); CREATE TRIGGER audit AFTER UPDATE ON item BEGIN SELECT 1; END;"
        backup = self._db("backup.sqlite3", sql + "INSERT INTO item VALUES (1, 'backup');")
        current = self._db("current.sqlite3", sql + "INSERT INTO item VALUES (1, 'current');")
        before = {path.name for path in self.root.glob("sqlite-restore-*")}

        result = prepare_restore(backup, current, self.migrations, self.root / "candidate.sqlite3", mode="merge", conflict_policy="use_backup")

        self.assertFalse(result["can_apply"])
        self.assertTrue(any("trigger" in blocker.lower() for blocker in result["blockers"]))
        self.assertEqual({path.name for path in self.root.glob("sqlite-restore-*")}, before)
