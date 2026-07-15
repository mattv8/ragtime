import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "prisma" / "schema.prisma"
MIGRATION = ROOT / "prisma" / "migrations" / "20260715180000_add_index_job_phase" / "migration.sql"


class IndexJobPhaseMigrationTests(unittest.TestCase):
    def test_schema_declares_non_null_phase_with_preparing_default(self) -> None:
        schema = SCHEMA.read_text(encoding="utf-8")
        self.assertIn("enum IndexJobPhase {", schema)
        self.assertRegex(schema, r"phase\s+IndexJobPhase\s+@default\(preparing\)")

    def test_migration_is_idempotent_and_backfills_terminal_rows(self) -> None:
        sql = MIGRATION.read_text(encoding="utf-8")
        self.assertIn('CREATE TYPE "IndexJobPhase" AS ENUM', sql)
        self.assertIn("WHEN duplicate_object THEN null", sql)
        self.assertIn('ADD COLUMN IF NOT EXISTS "phase"', sql)
        self.assertIn("ILIKE '%cancel%'", sql)
        self.assertIn("'completed'::\"IndexJobPhase\"", sql)
        self.assertIn("'cancelled'::\"IndexJobPhase\"", sql)
        self.assertIn("'failed'::\"IndexJobPhase\"", sql)
