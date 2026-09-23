from __future__ import annotations

import asyncio
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.worker.sqlite_history.catalog import HistoryCatalog
from runtime.worker.sqlite_history.storage import AsyncRepositoryGate, eligible_snapshot_ids, repository_gate


class RuntimeResticMaintenanceTests(unittest.TestCase):
    def test_workspace_charge_counts_aliases_once(self) -> None:
        manifest = {
            "backups": [
                {"status": "ready", "sha256": "a", "size_bytes": 10},
                {"status": "ready", "sha256": "a", "size_bytes": 10},
                {"status": "ready", "sha256": "b", "size_bytes": 3},
            ]
        }
        self.assertEqual(13, HistoryCatalog.logical_charge(manifest))

    def test_graph_filter_keeps_snapshot_referenced_by_another_workspace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for workspace in ("one", "two"):
                history = root / "workspaces" / workspace / "sqlite_backups"
                history.mkdir(parents=True)
                storage = {"kind": "restic", "repository_id": "a" * 64, "snapshot_id": "b" * 64, "path": "/database.sqlite3"}
                (history / "manifest-v1.json").write_text(__import__("json").dumps({"workspace_id": workspace, "backups": [{"storage": storage}]}))
            self.assertEqual([], eligible_snapshot_ids(root, ["b" * 64]))

    def test_repository_gate_reenters_for_nested_maintenance_work(self) -> None:
        """Maintenance holds the global exclusive gate while draining tombstones."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            async def drain() -> None:
                async with AsyncRepositoryGate(root, exclusive=True):
                    with ExitStack() as stack:
                        stack.enter_context(repository_gate(root, exclusive=True))
                        stack.enter_context(repository_gate(root, exclusive=True))

            asyncio.run(drain())
