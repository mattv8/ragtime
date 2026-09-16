import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "worktree_migrations.py"
SPEC = importlib.util.spec_from_file_location("worktree_migrations", MODULE_PATH)
assert SPEC and SPEC.loader
wm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wm
SPEC.loader.exec_module(wm)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class FakeCursor:
    def __init__(self, rows, unfinished=()):
        self.rows = rows
        self.unfinished = unfinished
        self.last_sql = ""

    def execute(self, sql, params=None):
        self.last_sql = sql

    def fetchall(self):
        return self.unfinished if "finished_at IS NULL" in self.last_sql else self.rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeConnection:
    def __init__(self, rows, unfinished=()):
        self.cursor_value = FakeCursor(rows, unfinished)

    def cursor(self):
        return self.cursor_value


class WorktreeMigrationUnitTests(unittest.TestCase):
    def test_baseline_alias_extends_prefix_and_preserves_raw_initial_history(self):
        current = "a" * 64
        raw = "manual-applied"
        aliases = {("A", raw): current}
        applied = wm.read_active_history(FakeConnection([("1", "A", raw)]), set())
        prefix, outgoing, forward = wm.common_prefix(applied, [wm.Migration("A", current, "")], aliases)
        self.assertEqual(applied, [{"id": "1", "name": "A", "sha256": raw}])
        self.assertEqual(prefix, ["A"])
        self.assertEqual(outgoing, [])
        self.assertEqual(forward, [])

    def test_prefix_compares_historical_rows_in_migration_name_order(self):
        current_a = "a" * 64
        applied = [
            {"id": "2", "name": "B", "sha256": "b" * 64},
            {"id": "1", "name": "A", "sha256": "manual-applied"},
        ]
        prefix, outgoing, forward = wm.common_prefix(
            applied,
            [wm.Migration("A", current_a, ""), wm.Migration("B", "b" * 64, "")],
            {("A", "manual-applied"): current_a},
        )
        self.assertEqual(prefix, ["A", "B"])
        self.assertEqual(outgoing, [])
        self.assertEqual(forward, [])

    def test_active_baseline_alias_requires_matching_primary_and_target_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = self._prisma_package(root / "primary", {"A": "CREATE TABLE a (id int);\n"})
            target = self._prisma_package(root / "target", {"A": "CREATE TABLE a (id int);\n"})
            migration_sha = wm.migrations(target)[0].sha256
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "legacy_applied": [],
                        "down_migrations": [],
                        "baseline_aliases": [{"name": "A", "applied_checksum": "manual-applied", "migration_sha256": migration_sha}],
                    }
                )
            )
            primary_migration = primary / "migrations" / "A" / "migration.sql"
            primary_migration.write_text("CREATE TABLE changed (id int);\n")
            args = self._plan_args(root, primary, target, registry)
            with self.assertRaisesRegex(wm.Refusal, "baseline alias is not present in primary"):
                wm.build_plan(args, FakeConnection([("1", "A", "manual-applied")]))
            primary_migration.write_text("CREATE TABLE a (id int);\n")
            (target / "migrations" / "A" / "migration.sql").write_text("CREATE TABLE changed (id int);\n")
            with self.assertRaisesRegex(wm.Refusal, "baseline alias is not present in target"):
                wm.build_plan(args, FakeConnection([("1", "A", "manual-applied")]))

    def test_alias_that_would_be_outgoing_is_refused_without_reversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            migrations = {"0": "CREATE TABLE current (id int);\n", "A": "CREATE TABLE a (id int);\n"}
            primary = self._prisma_package(root / "primary", migrations)
            target = self._prisma_package(root / "target", migrations)
            a_sha = next(item.sha256 for item in wm.migrations(target) if item.name == "A")
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "legacy_applied": [],
                        "down_migrations": [],
                        "baseline_aliases": [{"name": "A", "applied_checksum": "manual-applied", "migration_sha256": a_sha}],
                    }
                )
            )
            args = self._plan_args(root, primary, target, registry)
            with self.assertRaisesRegex(wm.Refusal, "aliased migration would be outgoing: A"):
                wm.build_plan(args, FakeConnection([("1", "0", sha("CREATE TABLE old (id int);\n")), ("2", "A", "manual-applied")]))

    def test_normalized_alias_comparison_accepts_only_the_pinned_raw_checksum(self):
        current = "a" * 64
        aliases = {("A", "manual-applied"): current}
        self.assertEqual(
            wm.normalize_history([{"id": "1", "name": "A", "sha256": "manual-applied"}], aliases),
            [{"name": "A", "sha256": current}],
        )
        self.assertEqual(
            wm.normalize_history([{"id": "1", "name": "A", "sha256": "unexpected"}], aliases),
            [{"name": "A", "sha256": "unexpected"}],
        )

    def test_registry_rejects_duplicate_invalid_and_legacy_conflicting_baseline_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            base = {"version": 1, "legacy_applied": [], "down_migrations": []}
            for aliases, message in (
                ([{"name": "A", "applied_checksum": "manual-applied", "migration_sha256": "a" * 64}] * 2, "duplicate baseline alias"),
                ([{"name": "A", "applied_checksum": "manual-applied", "migration_sha256": "invalid"}], "invalid baseline alias"),
                ([{"name": "A", "applied_checksum": "manual-applied", "migration_sha256": "a" * 64}], "conflicts with legacy"),
            ):
                with self.subTest(message=message):
                    legacy = [{"name": "A", "migration_sha256": "a" * 64}] if "conflicts" in message else []
                    path.write_text(json.dumps({**base, "legacy_applied": legacy, "baseline_aliases": aliases}))
                    with self.assertRaisesRegex(wm.Refusal, message):
                        wm.load_registry(path)

    @staticmethod
    def _prisma_package(root, migration_sql):
        (root / "migrations").mkdir(parents=True)
        (root / "schema.prisma").write_text("generator client {}\n")
        for name, sql in migration_sql.items():
            migration = root / "migrations" / name
            migration.mkdir()
            (migration / "migration.sql").write_text(sql)
        return root

    @staticmethod
    def _plan_args(root, primary, target, registry):
        state = root / "state"
        state.mkdir()
        plan = state / "runs" / "plan.json"
        plan.parent.mkdir(parents=True)
        return type(
            "Args", (), {"state_root": str(state), "plan": str(plan), "target": str(target), "primary": str(primary), "registry": str(registry), "source": []}
        )()

    def test_active_history_orders_by_started_at_and_filters_only_pinned_legacy(self):
        rows = [("b", "B", "b" * 64), ("a", "A", "a" * 64)]
        result = wm.read_active_history(FakeConnection(rows), {("A", "a" * 64)})
        self.assertEqual(result, [{"id": "b", "name": "B", "sha256": "b" * 64}])

    def test_unfinished_row_is_refused_but_rolled_back_tombstone_is_not(self):
        with self.assertRaisesRegex(wm.Refusal, "unfinished"):
            wm.read_active_history(FakeConnection([], [("failed",)]), set())
        self.assertEqual(wm.read_active_history(FakeConnection([]), set()), [])

    def test_common_prefix_handles_insertions_and_reverses_actual_suffix(self):
        applied = [{"id": "1", "name": "A", "sha256": "a"}, {"id": "2", "name": "B", "sha256": "b"}, {"id": "3", "name": "C", "sha256": "c"}]
        target = [wm.Migration("A", "a", ""), wm.Migration("X", "x", ""), wm.Migration("B", "b", ""), wm.Migration("C", "c", "")]
        prefix, outgoing, forward = wm.common_prefix(applied, target)
        self.assertEqual(prefix, ["A"])
        self.assertEqual([item["name"] for item in reversed(outgoing)], ["C", "B"])
        self.assertEqual([item.name for item in forward], ["X", "B", "C"])

    def test_same_name_different_sql_is_not_common(self):
        prefix, outgoing, forward = wm.common_prefix([{"id": "1", "name": "A", "sha256": "old"}], [wm.Migration("A", "new", "")])
        self.assertEqual(prefix, [])
        self.assertEqual([item["name"] for item in outgoing], ["A"])
        self.assertEqual([item.name for item in forward], ["A"])

    def test_down_sql_allows_only_static_reversal_ast(self):
        wm.validate_down_sql('ALTER TABLE "items" DROP COLUMN "value"; DROP TABLE "items"; DELETE FROM "items";')
        for sql in (
            "",
            "-- no inverse statements\n",
            "BEGIN; DROP TABLE items;",
            "DO $$ BEGIN END $$;",
            "CREATE TABLE items (id int);",
            "UPDATE items SET id = 1;",
        ):
            with self.subTest(sql=sql), self.assertRaises(wm.Refusal):
                wm.validate_down_sql(sql)

    def test_cached_definition_is_immutable_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            migration_dir = root / "source" / "migrations" / "20260101000000_branch"
            migration_dir.mkdir(parents=True)
            sql = "DROP TABLE branch_table;\n"
            migration = migration_dir / "migration.sql"
            migration.write_text("CREATE TABLE branch_table (id int);\n")
            (migration_dir / "down.sql").write_text(sql)
            item = wm.Migration(migration_dir.name, hashlib.sha256(migration.read_bytes()).hexdigest(), str(migration))
            self.assertEqual(wm._cache_definition(root, item), (sql, sha(sql)))
            cache = wm._cache_path(root, (item.name, item.sha256))
            payload = json.loads(cache.read_text())
            payload["migration_sha256"] = "a" * 64
            cache.write_text(json.dumps(payload))
            with self.assertRaisesRegex(wm.Refusal, "corrupt cached"):
                wm._load_cached_definition(root, (item.name, item.sha256))

    def test_package_manifest_detects_schema_lock_and_extra_migration_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            (target / "migrations" / "20260101000000_base" / "extra").mkdir(parents=True)
            (target / "schema.prisma").write_text("generator client {}\n")
            (target / "migration_lock.toml").write_text('provider = "postgresql"\n')
            (target / "migrations" / "20260101000000_base" / "migration.sql").write_text("CREATE TABLE x (id int);\n")
            (target / "migrations" / "20260101000000_base" / "extra" / "seed.sql").write_text("SELECT 1;\n")
            package = Path(directory) / "state" / "package"
            wm.package_target(target, package)
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertIn("schema.prisma", manifest["files"])
            self.assertIn("migration_lock.toml", manifest["files"])
            self.assertIn("migrations/20260101000000_base/extra/seed.sql", manifest["files"])
            self.assertIn("migrations/20260101000000_base/extra", manifest["directories"])
            plan = {
                "initial_history": [],
                "legacy_applied": [],
                "package": str(package),
                "manifest": str(package / "manifest.json"),
                "expected_final_history": [],
                "outgoing": [],
            }
            wm._assert_unchanged(plan, FakeConnection([]))
            (package / "migration_lock.toml").write_text("changed")
            with self.assertRaisesRegex(wm.Refusal, "packaged migration files changed"):
                wm._assert_unchanged(plan, FakeConnection([]))

    def test_registry_requires_exact_down_sql_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "legacy_applied": [],
                        "baseline_aliases": [],
                        "down_migrations": [{"name": "x", "migration_sha256": "a" * 64, "down_sha256": "b" * 64, "down_sql": "DROP TABLE x;"}],
                    }
                )
            )
            with self.assertRaisesRegex(wm.Refusal, "digest"):
                wm.load_registry(path)

    def test_plan_paths_cannot_escape_its_declared_state_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            root.mkdir()
            plan = root / "runs" / "run" / "plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "state_root": str(root),
                        "package": "/tmp/package",
                        "manifest": "",
                        "expected_final_history": [],
                        "outgoing": [],
                        "baseline_aliases": [],
                    }
                )
            )
            with self.assertRaisesRegex(wm.Refusal, "escapes"):
                wm._load_plan(plan)

    def test_registry_pins_external_agent_reliability_sql(self):
        registry = wm.load_registry(Path(__file__).parents[1] / "scripts" / "worktree_down_migrations.json")
        item = registry["down_migrations"][0]
        self.assertEqual(item["migration_sha256"], "def13c6ad0d78cc8f588007fc13c6c99501e93ba19f9a3ecf82d0abdb5b2e25b")
        self.assertEqual(item["down_sha256"], sha(item["down_sql"]))


@unittest.skipUnless(os.getenv("RUN_WORKTREE_MIGRATION_INTEGRATION") == "1", "set RUN_WORKTREE_MIGRATION_INTEGRATION=1")
class WorktreeMigrationPostgresTests(unittest.TestCase):
    """Uses a unique disposable PostgreSQL container; never the dev database."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = f"ragtime-worktree-migrations-{os.getpid()}-{time.time_ns()}"
        self.addCleanup(subprocess.run, ["docker", "rm", "-f", self.name], check=False, capture_output=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "-e",
                "POSTGRES_PASSWORD=test",
                "-e",
                "POSTGRES_DB=test",
                "-p",
                "127.0.0.1::5432",
                "postgres:16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        port = subprocess.run(["docker", "port", self.name, "5432/tcp"], check=True, capture_output=True, text=True).stdout.strip().rsplit(":", 1)[1]
        self.env = {**os.environ, "DATABASE_URL": f"postgresql://postgres:test@127.0.0.1:{port}/test"}
        for _ in range(30):
            if subprocess.run(["docker", "exec", self.name, "pg_isready", "-U", "postgres"], capture_output=True).returncode == 0:
                break
            time.sleep(1)
        else:
            self.fail("disposable PostgreSQL did not become ready")

    def tearDown(self):
        pass

    def _package(self, name, branch, *, branch_sql=None, down_sql=None):
        prisma = self.root / name / "prisma"
        (prisma / "migrations" / "20260101000000_base").mkdir(parents=True)
        (prisma / "schema.prisma").write_text(
            'generator client {\n  provider = "prisma-client-py"\n}\n\n'
            'datasource db {\n  provider = "postgresql"\n  url = env("DATABASE_URL")\n}\n\n'
            "model Keep {\n  id Int @id\n  value String\n}\n"
        )
        (prisma / "migrations" / "20260101000000_base" / "migration.sql").write_text(
            'CREATE TABLE "Keep" ("id" INTEGER NOT NULL, "value" TEXT NOT NULL, CONSTRAINT "Keep_pkey" PRIMARY KEY ("id"));\n'
        )
        if branch:
            migration = prisma / "migrations" / "20260102000000_branch"
            migration.mkdir()
            (migration / "migration.sql").write_text(branch_sql or 'ALTER TABLE "Keep" ADD COLUMN "branch_value" TEXT;\n')
            (migration / "down.sql").write_text(down_sql or 'ALTER TABLE "Keep" DROP COLUMN "branch_value";\n')
        return prisma

    def _helper(self, *args):
        return subprocess.run([sys.executable, str(MODULE_PATH), *args], env=self.env, check=True, capture_output=True, text=True)

    def test_actual_prisma_up_down_up_preserves_unaffected_data(self):
        primary = self._package("primary", False)
        target = self._package("target", True)
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"version": 1, "legacy_applied": [], "baseline_aliases": [], "down_migrations": []}))
        subprocess.run(
            [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(primary / "schema.prisma")],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )
        import psycopg2

        with psycopg2.connect(self.env["DATABASE_URL"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute('INSERT INTO "Keep" ("id", "value") VALUES (1, \'preserve\')')
        state = self.root / "state"
        state.mkdir()
        forward = state / "runs" / "forward" / "plan.json"
        forward.parent.mkdir(parents=True)
        self._helper(
            "check", "--target", str(target), "--primary", str(primary), "--state-root", str(state), "--plan", str(forward), "--registry", str(registry)
        )
        self._helper("apply", "--plan", str(forward))
        reverse = state / "runs" / "reverse" / "plan.json"
        reverse.parent.mkdir(parents=True)
        self._helper(
            "check",
            "--target",
            str(primary),
            "--primary",
            str(primary),
            "--source",
            str(target),
            "--state-root",
            str(state),
            "--plan",
            str(reverse),
            "--registry",
            str(registry),
        )
        self._helper("apply", "--plan", str(reverse))
        repeat_forward = state / "runs" / "repeat-forward" / "plan.json"
        repeat_forward.parent.mkdir(parents=True)
        self._helper(
            "check", "--target", str(target), "--primary", str(primary), "--state-root", str(state), "--plan", str(repeat_forward), "--registry", str(registry)
        )
        self._helper("apply", "--plan", str(repeat_forward))
        with psycopg2.connect(self.env["DATABASE_URL"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT "value", "branch_value" FROM "Keep" WHERE "id" = 1')
                self.assertEqual(cursor.fetchone(), ("preserve", None))

    def test_cached_down_survives_deleted_source_checkout(self):
        primary = self._package("primary", False)
        target = self._package("target", True)
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"version": 1, "legacy_applied": [], "baseline_aliases": [], "down_migrations": []}))
        subprocess.run(
            [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(primary / "schema.prisma")],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )
        state = self.root / "state"
        state.mkdir()
        forward = state / "runs" / "forward" / "plan.json"
        forward.parent.mkdir(parents=True)
        self._helper(
            "check", "--target", str(target), "--primary", str(primary), "--state-root", str(state), "--plan", str(forward), "--registry", str(registry)
        )
        self._helper("apply", "--plan", str(forward))
        migration = target / "migrations" / "20260102000000_branch"
        for child in migration.iterdir():
            child.unlink()
        migration.rmdir()
        reverse = state / "runs" / "reverse" / "plan.json"
        reverse.parent.mkdir(parents=True)
        self._helper(
            "check",
            "--target",
            str(primary),
            "--primary",
            str(primary),
            "--source",
            str(target),
            "--state-root",
            str(state),
            "--plan",
            str(reverse),
            "--registry",
            str(registry),
        )
        self._helper("apply", "--plan", str(reverse))

    def test_reverse_failure_rolls_back_schema_and_prisma_metadata(self):
        primary = self._package("primary", False)
        target = self._package("target", True, down_sql='ALTER TABLE "Keep" DROP COLUMN "branch_value"; DROP TABLE "missing";\n')
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"version": 1, "legacy_applied": [], "baseline_aliases": [], "down_migrations": []}))
        subprocess.run(
            [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(primary / "schema.prisma")],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )
        state = self.root / "state"
        state.mkdir()
        forward = state / "runs" / "forward" / "plan.json"
        forward.parent.mkdir(parents=True)
        self._helper(
            "check", "--target", str(target), "--primary", str(primary), "--state-root", str(state), "--plan", str(forward), "--registry", str(registry)
        )
        self._helper("apply", "--plan", str(forward))
        reverse = state / "runs" / "reverse" / "plan.json"
        reverse.parent.mkdir(parents=True)
        self._helper(
            "check",
            "--target",
            str(primary),
            "--primary",
            str(primary),
            "--source",
            str(target),
            "--state-root",
            str(state),
            "--plan",
            str(reverse),
            "--registry",
            str(registry),
        )
        with self.assertRaises(subprocess.CalledProcessError):
            self._helper("apply", "--plan", str(reverse))
        import psycopg2

        with psycopg2.connect(self.env["DATABASE_URL"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM information_schema.columns WHERE table_name = %s AND column_name = %s", ("Keep", "branch_value"))
                row = cursor.fetchone()
                assert row is not None
                self.assertEqual(row[0], 1)
                cursor.execute(
                    'SELECT count(*) FROM "_prisma_migrations" WHERE migration_name = %s AND finished_at IS NOT NULL AND rolled_back_at IS NULL',
                    ("20260102000000_branch",),
                )
                row = cursor.fetchone()
                assert row is not None
                self.assertEqual(row[0], 1)

    def test_forward_failure_leaves_prisma_failed_record(self):
        primary = self._package("primary", False)
        target = self._package("target", True, branch_sql='ALTER TABLE "Keep" ADD COLUMN "branch_value" TEXT; ALTER TABLE "missing" ADD COLUMN "x" TEXT;\n')
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"version": 1, "legacy_applied": [], "baseline_aliases": [], "down_migrations": []}))
        subprocess.run(
            [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(primary / "schema.prisma")],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )
        state = self.root / "state"
        state.mkdir()
        plan = state / "runs" / "forward" / "plan.json"
        plan.parent.mkdir(parents=True)
        self._helper("check", "--target", str(target), "--primary", str(primary), "--state-root", str(state), "--plan", str(plan), "--registry", str(registry))
        result = subprocess.run([sys.executable, str(MODULE_PATH), "apply", "--plan", str(plan)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        import psycopg2

        with psycopg2.connect(self.env["DATABASE_URL"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT count(*) FROM "_prisma_migrations" WHERE migration_name = %s AND finished_at IS NULL AND rolled_back_at IS NULL',
                    ("20260102000000_branch",),
                )
                row = cursor.fetchone()
                assert row is not None
                self.assertEqual(row[0], 1)
