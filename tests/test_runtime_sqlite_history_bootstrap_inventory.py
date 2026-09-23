from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from runtime.worker.sqlite_history.inventory import inventory_legacy_history


class LegacyHistoryInventoryTests(unittest.TestCase):
    digest = "a" * 64

    def _catalog(self, root: Path, workspace_id: str = "one") -> Path:
        catalog = root / "workspaces" / workspace_id / "sqlite_backups"
        (root / "workspaces" / workspace_id / "files").mkdir(parents=True)
        (catalog / "blobs").mkdir(parents=True)
        return catalog

    def _legacy_row(self, blob: str, *, digest: str | None = None, size: int = 6) -> dict:
        return {
            "id": blob,
            "status": "ready",
            "sha256": digest or self.digest,
            "size_bytes": size,
            "storage": {"kind": "legacy_file", "blob": f"blobs/{blob}"},
        }

    def _write_manifest(self, catalog: Path, workspace_id: str, backups: list[dict]) -> None:
        (catalog / "manifest-v1.json").write_text(
            json.dumps({"version": 1, "workspace_id": workspace_id, "backups": backups}),
            encoding="utf-8",
        )

    def _ledger_name(self, workspace_id: str, blob: str, digest: str) -> tuple[str, str]:
        operation = "legacy-v1-" + hashlib.sha256(f"{workspace_id}\0blobs/{blob}\0{digest}".encode()).hexdigest()
        return operation, hashlib.sha256(operation.encode()).hexdigest() + ".json"

    def test_inventory_is_read_only_and_counts_aliases_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            (catalog / "blobs" / "image.sqlite3").write_bytes(b"sqlite")
            self._write_manifest(
                catalog,
                "one",
                [
                    self._legacy_row("image.sqlite3"),
                    self._legacy_row("image.sqlite3"),
                ],
            )
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            root.chmod(0o555)
            try:
                report = inventory_legacy_history(root)
            finally:
                root.chmod(0o755)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(report["workspaces"][0]["legacy_records"], 2)
            self.assertEqual(report["workspaces"][0]["unique_legacy_blobs"], 1)
            self.assertEqual(report["totals"]["legacy_bytes"], 6)

    def test_inventory_rejects_symlinked_catalog_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            self._write_manifest(catalog, "one", [self._legacy_row("unsafe.sqlite3")])
            (catalog / "blobs" / "unsafe.sqlite3").symlink_to(root / "outside")
            report = inventory_legacy_history(root, ["one"])
            self.assertEqual(report["workspaces"][0]["legacy_records"], 1)
            self.assertIn("legacy blob path is missing or unsafe", report["workspaces"][0]["issues"])

    def test_empty_scope_does_not_expand_and_none_discovers_only_catalogs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root, "retained")
            (catalog / "blobs" / "image.sqlite3").write_bytes(b"sqlite")
            self._write_manifest(catalog, "retained", [self._legacy_row("image.sqlite3")])
            (root / "workspaces" / "ordinary" / "files").mkdir(parents=True)

            self.assertEqual(inventory_legacy_history(root, [])["workspaces"], [])
            self.assertEqual(
                [item["workspace_id"] for item in inventory_legacy_history(root)["workspaces"]],
                ["retained"],
            )

    def test_explicit_missing_workspace_reports_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = inventory_legacy_history(Path(temporary), ["missing"])
            self.assertEqual(report["workspaces"][0]["workspace_id"], "missing")
            self.assertIn("workspace files directory is missing or unsafe", report["workspaces"][0]["issues"])
            self.assertIn("history catalog directory is missing or unsafe", report["workspaces"][0]["issues"])

    def test_symlinked_private_repository_parent_is_not_traversed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            (outside / "restic").mkdir(parents=True)
            (outside / "restic" / "outside-data").write_bytes(b"outside")
            (root / "_sqlite_history").symlink_to(outside, target_is_directory=True)

            report = inventory_legacy_history(root, [])
            self.assertEqual(report["totals"]["repository_physical_bytes"], 0)

    def test_physical_inode_bytes_are_distinct_from_logical_rows_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            first = catalog / "blobs" / "first.sqlite3"
            first.write_bytes(b"sqlite")
            os.link(first, catalog / "blobs" / "second.sqlite3")
            self._write_manifest(
                catalog,
                "one",
                [
                    self._legacy_row("first.sqlite3", digest="a" * 64),
                    self._legacy_row("second.sqlite3", digest="b" * 64),
                ],
            )

            report = inventory_legacy_history(root)
            workspace = report["workspaces"][0]
            self.assertEqual(workspace["unique_legacy_blobs"], 1)
            self.assertEqual(workspace["legacy_bytes"], 6)
            self.assertEqual(workspace["logical_legacy_bytes"], 12)
            self.assertEqual(workspace["unique_legacy_contents"], 1)
            self.assertIn("legacy blob digest does not match catalog metadata", workspace["issues"])

    def test_malformed_catalog_and_ledger_are_issues_not_pending_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            (catalog / "manifest-v1.json").write_text("{", encoding="utf-8")
            report = inventory_legacy_history(root, ["one"])
            self.assertIn("history catalog is unreadable", report["workspaces"][0]["issues"])

            self._write_manifest(catalog, "one", [])
            ledger = catalog / "conversion-ledger"
            ledger.mkdir()
            (ledger / "bad.json").write_text("{}", encoding="utf-8")
            report = inventory_legacy_history(root, ["one"])
            self.assertEqual(report["workspaces"][0]["ledger_pending"], 0)
            self.assertIn("conversion ledger is unreadable", report["workspaces"][0]["issues"])

    def test_published_catalog_pending_cleanup_reports_unreferenced_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            source = catalog / "blobs" / "image.sqlite3"
            source.write_bytes(b"sqlite")
            self._write_manifest(
                catalog,
                "one",
                [
                    {
                        "id": "a",
                        "status": "ready",
                        "sha256": self.digest,
                        "size_bytes": 6,
                        "storage": {"kind": "restic", "repository_id": "repo", "snapshot_id": "snap", "path": "/image.sqlite3"},
                    }
                ],
            )
            ledger = catalog / "conversion-ledger"
            ledger.mkdir()
            operation, filename = self._ledger_name("one", "image.sqlite3", self.digest)
            (ledger / filename).write_text(
                json.dumps(
                    {
                        "version": 1,
                        "workspace_id": "one",
                        "legacy_blob": "blobs/image.sqlite3",
                        "sha256": self.digest,
                        "size_bytes": 6,
                        "operation_id": operation,
                        "backup_ids": ["a"],
                        "stage": "catalog_published",
                        "storage": {"kind": "restic", "repository_id": "repo", "snapshot_id": "snap", "path": "/image.sqlite3"},
                    }
                ),
                encoding="utf-8",
            )

            report = inventory_legacy_history(root)
            self.assertEqual(report["workspaces"][0]["legacy_records"], 0)
            self.assertEqual(report["workspaces"][0]["ledger_pending"], 1)
            self.assertEqual(report["workspaces"][0]["ledger_pending_bytes"], 6)
            self.assertEqual(report["required_headroom_bytes"], 6 + 64 * 1024 * 1024)

    def test_repository_physical_metrics_change_without_creating_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = inventory_legacy_history(root, [])
            repository = root / "_sqlite_history" / "restic"
            repository.mkdir(parents=True)
            payload = repository / "pack"
            payload.write_bytes(b"repository")
            os.link(payload, repository / "pack-alias")
            after = inventory_legacy_history(root, [])
            self.assertEqual(before["totals"]["repository_physical_bytes"], 0)
            self.assertEqual(after["totals"]["repository_physical_bytes"], len(b"repository"))
