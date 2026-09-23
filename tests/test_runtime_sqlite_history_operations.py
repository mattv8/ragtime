"""Durable runtime SQLite-history receipt contracts."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.worker.sqlite_history.operations import OperationStore

OPERATION_ID = "6b98acd2-b817-4199-8dc8-47ce1a61831b"


def _hold_liveness(root: str, ready: multiprocessing.synchronize.Event, release: multiprocessing.synchronize.Event) -> None:
    with OperationStore(Path(root)).hold_liveness(OPERATION_ID):
        ready.set()
        release.wait(10)


class OperationStoreTests(unittest.TestCase):
    def _accept(self, store: object, **extra: object) -> dict[str, object]:
        values: dict[str, object] = {
            "operation_id": OPERATION_ID,
            "workspace_id": "workspace-a",
            "creator_id": "user-a",
            "request_digest": "request-digest",
            "kind": "capture",
            "accepted_payload": {"databases": ["primary"], "trigger": "manual"},
        }
        values.update(extra)
        return store.accept(**values)  # type: ignore[attr-defined,arg-type]

    def test_matching_acceptance_replays_immutable_request(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            first = self._accept(store)
            self.assertEqual(self._accept(store), first)
            self.assertEqual(first["phase"], "accepted")
            self.assertEqual(first["accepted_payload"], {"databases": ["primary"], "trigger": "manual"})

    def test_conflicting_payload_cannot_reuse_an_operation_id(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationConflict, OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self._accept(store)
            with self.assertRaises(OperationConflict):
                self._accept(store, request_digest="different-digest")

    def test_transition_progress_cancel_and_acknowledgement_contract(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationConflict, OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self._accept(store)
            running = store.transition(OPERATION_ID, "running", database_outcomes={"primary": {"state": "capturing"}})
            self.assertEqual(running["phase"], "running")
            with self.assertRaises(OperationConflict):
                store.transition(OPERATION_ID, "running", workspace_id="other")
            self.assertEqual(store.request_cancel(OPERATION_ID)["phase"], "cancelling")
            self.assertEqual(store.request_cancel(OPERATION_ID)["phase"], "cancelling")
            cancelled = store.transition(OPERATION_ID, "cancelled")
            self.assertIsNone(cancelled["acknowledged_at"])
            self.assertIsNotNone(store.acknowledge(OPERATION_ID)["acknowledged_at"])

    def test_restart_reconciliation_edges_and_workspace_listing(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self._accept(store)
            store.transition(OPERATION_ID, "running")
            store.transition(OPERATION_ID, "repository_committed", repository_refs=[{"snapshot_id": "a"}])
            store.transition(OPERATION_ID, "reconciling")
            result = store.transition(OPERATION_ID, "catalog_committed", database_outcomes={"primary": {"state": "ready"}})
            self.assertEqual(result["phase"], "catalog_committed")
            self.assertEqual([item["operation_id"] for item in store.list_active(workspace_id="workspace-a")], [OPERATION_ID])
            store.transition(OPERATION_ID, "completed")
            self.assertEqual(store.list_active(), [])

    def test_unknown_get_does_not_create_directories(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationNotFound, OperationStore

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing"
            with self.assertRaises(OperationNotFound):
                OperationStore(root).get(OPERATION_ID)
            self.assertFalse(root.exists())

    def test_rejects_symlink_and_malformed_receipt(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationConflict, OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self._accept(store)
            receipt = Path(temporary) / "operations" / OPERATION_ID / "receipt.json"
            receipt.unlink()
            receipt.symlink_to("/etc/passwd")
            with self.assertRaises(OperationConflict):
                store.get(OPERATION_ID)

    def test_liveness_is_cross_process_and_not_ttl_based(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationConflict, OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self._accept(store)
            context = multiprocessing.get_context("spawn")
            ready, release = context.Event(), context.Event()
            process = context.Process(target=_hold_liveness, args=(temporary, ready, release))
            process.start()
            self.assertTrue(ready.wait(10))
            self.assertTrue(store.is_live(OPERATION_ID))
            with self.assertRaises(OperationConflict):
                with store.hold_liveness(OPERATION_ID):
                    pass
            release.set()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            self.assertFalse(store.is_live(OPERATION_ID))

    def test_suboperation_id_validates_names_and_tombstones_prevent_reacceptance(self) -> None:
        from runtime.worker.sqlite_history.operations import OperationConflict, OperationStore

        with TemporaryDirectory() as temporary:
            store = OperationStore(Path(temporary))
            self.assertEqual(store.suboperation_id(OPERATION_ID, "primary"), store.suboperation_id(OPERATION_ID, "primary"))
            with self.assertRaises(ValueError):
                store.suboperation_id(OPERATION_ID, "../primary")
            self._accept(store)
            store.transition(OPERATION_ID, "cancelled")
            retired = store.retire(OPERATION_ID)
            self.assertIsNotNone(retired["retired_at"])
            self.assertEqual(self._accept(store)["retired_at"], retired["retired_at"])
            with self.assertRaises(OperationConflict):
                store.transition(OPERATION_ID, "running")
