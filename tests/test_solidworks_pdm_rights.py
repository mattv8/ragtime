import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ragtime.core.security import require_admin
from ragtime.indexer.models import ToolType
from ragtime.solidworks.pdm_rights import PdmRightsService, PdmToolNotFound, _error
from ragtime.solidworks.routes import PdmFoldersRequest, PdmRightsRequest, router
from ragtime.solidworks.transport import SolidworksTransportError, TransportErrorKind


class FakeTransport:
    def __init__(self):
        self.requests = []

    async def call(self, request, *, timeout_seconds):
        self.requests.append(request)
        rights = {"read": True, "add_files": True, "check_out": True, "add_folders": False}
        if request["operation"] == "pdm.folder.rights":
            result = {
                "user": "user",
                "items": [{"vault_path": path, "checked": True, "exists": True, "rights": rights} for path in request["arguments"]["paths"]],
            }
        else:
            result = {"items": [{"name": "é", "vault_path": "\\é", "rights": rights}], "offset": 0, "next_offset": None, "total": 1, "snapshot": "a" * 64}
        return {"id": request["id"], "operation": request["operation"], "ok": True, "result": result}

    async def run_powershell(self, script, *, timeout_seconds):
        raise NotImplementedError

    async def upload_file(self, local_path, remote_path, *, timeout_seconds):
        raise NotImplementedError

    async def download_file(self, remote_path, local_path, *, max_bytes, timeout_seconds):
        raise NotImplementedError

    async def remove_file(self, remote_path, *, timeout_seconds):
        raise NotImplementedError


class PdmRightsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = FakeTransport()
        self.service = PdmRightsService(lambda config: self.transport)
        self.tool = SimpleNamespace(tool_type=ToolType.SOLIDWORKS_PDM, connection_config={"database": "vault"}, enabled=False)
        self.patch = patch("ragtime.solidworks.pdm_rights.repository.get_tool_config", new=AsyncMock(return_value=self.tool))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def test_root_normalization_dedup_and_context(self):
        response = await self.service.rights({"id": "bad id", "allow_write": True}, "pdm", ["/Folder", "\\folder", "\\"])
        self.assertTrue(response["checked"])
        self.assertEqual([item["path"] for item in response["folders"]], ["\\Folder"])
        self.assertTrue(response["root"]["write_ready"])
        context = self.transport.requests[0]["context"]
        self.assertFalse(context["allow_write"])
        self.assertEqual(context["pdm"], {"tool_id": "unsaved", "vault": "vault", "approved_folders": [], "comment_prefix": ""})

    async def test_invalid_vault_skips_transport(self):
        self.tool.connection_config["database"] = "a/b"
        result = await self.service.rights({}, "pdm", [])
        self.assertEqual(result["error"]["code"], "pdm_vault_invalid")
        self.assertEqual(self.transport.requests, [])

    async def test_updating_gate_maps_both_operations(self):
        from ragtime.solidworks.host_activity import HostUpdatingError  # inline-import: keep

        async def blocked(key, timeout, work):
            raise HostUpdatingError()

        with patch("ragtime.solidworks.pdm_rights.host_activity") as activity:
            activity.run_call = blocked
            rights = await self.service.rights({}, "pdm", ["\\folder"])
            folders = await self.service.folders({}, "pdm", "\\", 0, None)
        self.assertFalse(rights["checked"])
        self.assertFalse(rights["root"]["checked"])
        self.assertFalse(rights["folders"][0]["checked"])
        self.assertEqual(rights["error"]["code"], "host_updating")
        self.assertEqual(folders["error"]["code"], "host_updating")
        self.assertEqual(self.transport.requests, [])

    async def test_non_pdm_tool_is_not_found(self):
        self.tool.tool_type = ToolType.SOLIDWORKS
        with self.assertRaises(PdmToolNotFound):
            await self.service.rights({}, "pdm", [])

    async def test_batches_split_by_count_and_size(self):
        folders = ["\\" + str(i) + "x" * 245 for i in range(25)]
        response = await self.service.rights({}, "pdm", folders)
        self.assertTrue(response["checked"])
        self.assertGreater(len(self.transport.requests), 1)
        self.assertTrue(all(len(request["arguments"]["paths"]) <= 25 for request in self.transport.requests))

    async def test_all_batches_share_one_gate(self):
        from ragtime.solidworks.host_activity import GateTicket  # inline-import: keep

        entered = 0
        held = False
        original = self.transport.call

        async def gate(key, timeout, work):
            nonlocal entered, held
            entered += 1
            held = True
            try:
                return await work(GateTicket(False))
            finally:
                held = False

        async def call(request, *, timeout_seconds):
            self.assertTrue(held)
            return await original(request, timeout_seconds=timeout_seconds)

        self.transport.call = call
        with patch("ragtime.solidworks.pdm_rights.host_activity") as activity:
            activity.run_call = gate
            result = await self.service.rights({}, "pdm", ["\\" + str(i) + "x" * 245 for i in range(25)])
        self.assertTrue(result["checked"])
        self.assertGreater(len(self.transport.requests), 1)
        self.assertEqual(entered, 1)

    async def test_cancelled_rights_keeps_gate_across_batches(self):
        from ragtime.solidworks.host_activity import HostActivity, HostBusyError, host_key  # inline-import: keep

        activity = HostActivity(poll_seconds=0.001)
        config = {"connection_config": {"pairing_id": "host"}}
        key = host_key(config)
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        original = self.transport.call

        async def call(request, *, timeout_seconds):
            if not self.transport.requests:

                def block():
                    loop.call_soon_threadsafe(started.set)
                    release.wait(5)

                await asyncio.to_thread(block)
            self.assertEqual(activity.snapshot(key).in_flight, 1)
            return await original(request, timeout_seconds=timeout_seconds)

        self.transport.call = call
        with patch("ragtime.solidworks.pdm_rights.host_activity", activity):
            task = asyncio.create_task(self.service.rights(config, "pdm", ["\\" + str(i) + "x" * 245 for i in range(25)]))
            try:
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(activity.snapshot(key).in_flight, 1)
                with self.assertRaises(HostBusyError):
                    await activity.run_exclusive(key, 0.01, AsyncMock())
            finally:
                release.set()
            for _ in range(100):
                if activity.snapshot(key).in_flight == 0:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(activity.snapshot(key).in_flight, 0)
            self.assertGreater(len(self.transport.requests), 1)

    async def test_partial_failure_and_correlation(self):
        original = self.transport.call
        calls = 0

        async def call(request, *, timeout_seconds):
            nonlocal calls
            calls += 1
            result = await original(request, timeout_seconds=timeout_seconds)
            if calls == 2:
                result["id"] = "secret"
            return result

        self.transport.call = call
        response = await self.service.rights({}, "pdm", ["\\" + str(i) + "x" * 245 for i in range(25)])
        self.assertTrue(response["root"]["checked"])
        self.assertFalse(response["checked"])
        self.assertEqual(response["error"]["code"], "internal_error")
        self.assertNotIn("secret", str(response))

    async def test_unchecked_host_item_reports_timeout(self):
        original = self.transport.call

        async def call(request, *, timeout_seconds):
            response = await original(request, timeout_seconds=timeout_seconds)
            response["result"]["items"][1] = {"vault_path": "\\folder", "checked": False, "exists": None, "rights": None}
            return response

        self.transport.call = call
        response = await self.service.rights({}, "pdm", ["\\folder"])
        self.assertTrue(response["root"]["checked"])
        self.assertFalse(response["folders"][0]["checked"])
        self.assertEqual(response["error"], {"code": "operation_timeout", "message": "The SolidWorks agent did not answer in time. Try again."})

    def test_error_mapping_never_echoes_details(self):
        for source, expected in [
            ("ssh_auth_failed", "ssh_auth_failed"),
            ("ssh_unreachable", "ssh_unreachable"),
            ("timeout", "operation_timeout"),
            ("no_response", "host_not_running"),
            ("unknown_operation", "host_update_required"),
            ("host_not_running", "host_not_running"),
            ("no_interactive_session", "no_interactive_session"),
            ("pdm_not_found", "pdm_not_found"),
            ("stale_snapshot", "stale_snapshot"),
            ("invalid_arguments", "invalid_arguments"),
            ("secret", "internal_error"),
        ]:
            self.assertEqual(_error(source)["code"], expected)
            self.assertNotIn("secret", str(_error(source)))
        for reason in ("client_missing", "view_missing", "login_failed", "busy"):
            self.assertEqual(_error("pdm_unavailable", reason)["code"], "pdm_unavailable")

    async def test_browse_unicode_and_snapshot(self):
        result = await self.service.folders({}, "pdm", "\\é", 0, "a" * 64)
        self.assertTrue(result["items"][0]["write_ready"])
        self.assertEqual(self.transport.requests[0]["arguments"]["snapshot"], "a" * 64)
        self.assertEqual(PdmFoldersRequest(pdm_tool_id="p", path="\\é").path, "\\é")

    def test_request_limit(self):
        with self.assertRaises(ValidationError):
            PdmRightsRequest(pdm_tool_id="p", folders=["\\x"] * 26)

    async def test_host_error_envelopes_are_mapped_without_secrets(self):
        cases = [
            ("host_not_running", None, "host_not_running", "The SolidWorks agent is not running or did not respond."),
            ("no_interactive_session", None, "no_interactive_session", "The account has no interactive desktop session."),
            ("operation_timeout", None, "operation_timeout", "The SolidWorks agent did not answer in time. Try again."),
            ("unknown_operation", None, "host_update_required", "Update the SolidWorks agent to 0.4.0 or later to check vault access."),
            ("pdm_unavailable", "client_missing", "pdm_unavailable", "The PDM client is not installed on this machine."),
            ("pdm_unavailable", "view_missing", "pdm_unavailable", "This machine has no vault view for the linked vault."),
            ("pdm_unavailable", "login_failed", "pdm_unavailable", "Logging in to the vault view failed. Log in once in the PDM client on this machine."),
            ("pdm_unavailable", "busy", "pdm_unavailable", "The PDM client is busy with an earlier request. Try again in a moment."),
            ("pdm_not_found", None, "pdm_not_found", "That vault folder no longer exists."),
            ("stale_snapshot", None, "stale_snapshot", "The folder changed while you were browsing. Reload it."),
            ("invalid_arguments", None, "invalid_arguments", "The vault path is not valid."),
            ("unexpected", None, "internal_error", "The SolidWorks agent gave an unexpected answer."),
        ]
        for code, reason, expected, message in cases:

            async def call(request, *, timeout_seconds):
                return {
                    "id": request["id"],
                    "operation": request["operation"],
                    "ok": False,
                    "error": {"code": code, "message": "host-secret", "details": {"reason": reason, "secret": "host-secret"}},
                }

            with (
                self.subTest(code=code, reason=reason),
                patch.object(self.transport, "call", call),
                patch("ragtime.solidworks.pdm_rights.logger.warning") as warning,
            ):
                response = await self.service.rights({}, "pdm", [])
                self.assertEqual(response["error"], {"code": expected, "message": message})
                self.assertNotIn("host-secret", str(response))
                self.assertNotIn("host-secret", str(warning.call_args_list))

    async def test_transport_errors_are_mapped_without_secrets_in_logs(self):
        cases: list[tuple[TransportErrorKind, str, str]] = [
            ("ssh_auth_failed", "ssh_auth_failed", "SSH authentication failed; check the configured credentials."),
            ("ssh_unreachable", "ssh_unreachable", "SSH connection failed; check the SSH host and connection."),
            ("timeout", "operation_timeout", "The SolidWorks agent did not answer in time. Try again."),
            ("no_response", "host_not_running", "The SolidWorks agent is not running or did not respond."),
        ]
        for kind, expected, message in cases:
            with (
                self.subTest(kind=kind),
                patch.object(self.transport, "call", AsyncMock(side_effect=SolidworksTransportError(kind, "transport-secret"))),
                self.assertLogs("ragtime.solidworks.pdm_rights", level="WARNING") as logs,
            ):
                response = await self.service.rights({}, "pdm", [])
                self.assertEqual(response["error"], {"code": expected, "message": message})
                self.assertNotIn("transport-secret", str(response))
                self.assertNotIn("transport-secret", str(logs.output))

    async def test_malformed_envelopes_are_safe(self):
        for kind in ("not_dict", "id", "operation", "ok", "result"):

            async def call(request, *, timeout_seconds):
                if kind == "not_dict":
                    return "host-secret"
                response = {"id": request["id"], "operation": request["operation"], "ok": True, "result": "host-secret"}
                if kind == "id":
                    response["id"] = "host-secret"
                elif kind == "operation":
                    response["operation"] = "host-secret"
                elif kind == "ok":
                    response["ok"] = "host-secret"
                return response

            with self.subTest(kind=kind), patch.object(self.transport, "call", call), patch("ragtime.solidworks.pdm_rights.logger.warning") as warning:
                response = await self.service.rights({}, "pdm", [])
                self.assertEqual(response["error"]["code"], "internal_error")
                self.assertNotIn("host-secret", str(response))
                self.assertNotIn("host-secret", str(warning.call_args_list))


class PdmRightsHttpTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_admin] = lambda: object()
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.transport = FakeTransport()
        self.sw = SimpleNamespace(tool_type=ToolType.SOLIDWORKS, model_dump=lambda: {"id": "sw"})
        self.pdm = SimpleNamespace(tool_type=ToolType.SOLIDWORKS_PDM, connection_config={"database": "vault"}, enabled=False)
        tools = {"sw": self.sw, "pdm": self.pdm, "other": SimpleNamespace(tool_type=ToolType.SSH_SHELL)}
        self.repository = patch("ragtime.solidworks.routes.repository.get_tool_config", new=AsyncMock(side_effect=lambda tool_id: tools.get(tool_id)))
        self.repository.start()
        self.addCleanup(self.repository.stop)
        service = patch("ragtime.solidworks.routes.pdm_rights_service", PdmRightsService(lambda config: self.transport))
        service.start()
        self.addCleanup(service.stop)
        self.base = "/indexes/solidworks/tools"

    def test_http_validation(self):
        url = f"{self.base}/sw/pdm"
        self.assertEqual(self.client.post(f"{url}/rights", json={"pdm_tool_id": "pdm", "folders": ["\\a"] * 26}).status_code, 422)
        self.assertEqual(self.client.post(f"{url}/rights", json={"pdm_tool_id": "pdm", "folders": ["relative"]}).status_code, 422)
        self.assertEqual(self.client.post(f"{url}/folders", json={"pdm_tool_id": "pdm", "snapshot": "invalid"}).status_code, 422)

    def test_missing_tools_and_disabled_pdm(self):
        for endpoint, body in (("rights", {"pdm_tool_id": "pdm", "folders": []}), ("folders", {"pdm_tool_id": "pdm"})):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.client.post(f"{self.base}/other/pdm/{endpoint}", json=body).status_code, 404)
                for pdm_id in ("unknown", "other"):
                    response = self.client.post(f"{self.base}/sw/pdm/{endpoint}", json={**body, "pdm_tool_id": pdm_id})
                    self.assertEqual(response.status_code, 404)
                    self.assertEqual(response.json()["detail"], "PDM tool not found")
                self.assertEqual(self.client.post(f"{self.base}/sw/pdm/{endpoint}", json=body).status_code, 200)

    def test_malformed_folder_item_is_internal_error(self):
        original = self.transport.call

        async def call(request, *, timeout_seconds):
            response = await original(request, timeout_seconds=timeout_seconds)
            response["result"]["items"][0] = {"name": "host-secret"}
            return response

        with patch.object(self.transport, "call", call), patch("ragtime.solidworks.pdm_rights.logger.warning") as warning:
            response = self.client.post(f"{self.base}/sw/pdm/folders", json={"pdm_tool_id": "pdm"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["error"]["code"], "internal_error")
            self.assertNotIn("host-secret", response.text)
            self.assertNotIn("host-secret", str(warning.call_args_list))
