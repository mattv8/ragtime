import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class RuntimeRestartTests(unittest.TestCase):
    def test_restart_hash_is_stable_and_reason_scoped(self) -> None:
        self.assertEqual(
            UserSpaceRuntimeService._restart_request_hash("upgrade"),
            UserSpaceRuntimeService._restart_request_hash("upgrade"),
        )
        self.assertNotEqual(
            UserSpaceRuntimeService._restart_request_hash("upgrade"),
            UserSpaceRuntimeService._restart_request_hash("other"),
        )

    def test_operation_payload_omits_provider_identity(self) -> None:
        payload = UserSpaceRuntimeService._runtime_operation_payload(
            SimpleNamespace(id="op", workspaceId="workspace", state="accepted", operationId=None, error=None)
        )
        self.assertEqual(payload["id"], "op")
        self.assertNotIn("provider_session_id", payload)


class RuntimeRestartLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_provider_operation_identity_is_interrupted(self) -> None:
        service = UserSpaceRuntimeService()
        operation = SimpleNamespace(
            id="operation-id",
            workspaceId="workspace",
            userId="user",
            state="accepted",
            providerSessionId="session",
            operationId=None,
            error=None,
        )
        model = mock.AsyncMock()
        model.create.return_value = operation
        model.find_first.return_value = None
        model.find_unique.return_value = SimpleNamespace(**{**operation.__dict__, "state": "interrupted", "error": "missing"})
        session_model = mock.AsyncMock()
        session_model.find_first.return_value = SimpleNamespace(providerSessionId="session")
        tx = mock.MagicMock()
        tx.query_raw = mock.AsyncMock()
        tx.workspaceruntimeoperation = model
        tx.userspaceruntimesession = session_model
        transaction = mock.MagicMock()
        transaction.__aenter__ = mock.AsyncMock(return_value=tx)
        transaction.__aexit__ = mock.AsyncMock(return_value=False)
        db = mock.MagicMock()
        db.tx.return_value = transaction
        db.workspaceruntimeoperation = model

        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", new=mock.AsyncMock()),
            mock.patch.object(service, "_runtime_provider_restart_app", new=mock.AsyncMock(return_value={})),
        ):
            result = await service.request_app_restart("workspace", "user", "abcdefgh")

        self.assertEqual(result["state"], "interrupted")
        self.assertEqual(model.update_many.await_args_list[-1].kwargs["data"]["state"], "interrupted")

    async def test_mismatched_provider_operation_is_interrupted(self) -> None:
        service = UserSpaceRuntimeService()
        row = SimpleNamespace(id="op", workspaceId="workspace", userId="user", state="running", providerSessionId="session", operationId="expected", error=None)
        model = mock.AsyncMock()
        model.find_first.return_value = row
        model.update.return_value = SimpleNamespace(**{**row.__dict__, "state": "interrupted", "error": "mismatch"})
        db = SimpleNamespace(workspaceruntimeoperation=model)
        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", new=mock.AsyncMock()),
            mock.patch.object(service, "_get_active_session_row", new=mock.AsyncMock(return_value=SimpleNamespace(providerSessionId="session"))),
            mock.patch.object(
                service, "_runtime_provider_get_status", new=mock.AsyncMock(return_value={"runtime_operation_id": "other", "runtime_operation_phase": "failed"})
            ),
        ):
            result = await service.get_app_runtime_operation("workspace", "user", "op")

        self.assertEqual(result["state"], "interrupted")

    async def test_reconcile_provider_outage_marks_row_and_continues(self) -> None:
        service = UserSpaceRuntimeService()
        stale = SimpleNamespace(id="stale", workspaceId="workspace", state="running", providerSessionId="session", operationId="provider-op")
        completed = SimpleNamespace(id="done", workspaceId="workspace", state="running", providerSessionId="session", operationId="provider-op-2")
        model = mock.AsyncMock()
        model.find_many.return_value = [stale, completed]
        db = SimpleNamespace(workspaceruntimeoperation=model)
        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(
                service,
                "_runtime_provider_get_status",
                side_effect=[RuntimeError("offline"), {"runtime_operation_id": "provider-op-2", "runtime_operation_phase": "ready"}],
            ),
            mock.patch.object(service, "_get_active_session_row", new=mock.AsyncMock(return_value=SimpleNamespace(providerSessionId="session"))),
        ):
            await service.reconcile_stale_runtime_operations()

        self.assertEqual(model.update_many.await_count, 2)
        self.assertEqual(model.update_many.await_args_list[0].kwargs["data"]["state"], "interrupted")
        self.assertEqual(model.update_many.await_args_list[1].kwargs["data"]["state"], "completed")
