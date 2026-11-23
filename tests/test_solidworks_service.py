import json
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from ragtime.core.ssh import SSHResult
from ragtime.solidworks import pdm_link
from ragtime.solidworks.service import SolidworksService
from ragtime.solidworks.transport import SolidworksTransportError, TransportErrorKind


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeTransport:
    def __init__(self):
        self.calls: list[tuple[dict, float]] = []
        self.error: Exception | None = None
        self.operations: list[dict] = [{"name": "change", "access": "write"}]
        self.result: dict = {"value": "ok"}
        self.host_version = "1"
        self.response_overrides: dict = {}
        self.on_call: Callable[[dict], None] | None = None

    async def call(self, request: dict, *, timeout_seconds: float) -> dict:
        self.calls.append((request, timeout_seconds))
        if self.on_call:
            self.on_call(request)
        if self.error:
            raise self.error
        result = {"host_version": "1", "operations": self.operations} if request["operation"] == "describe" else self.result
        response = {
            "protocol": 1,
            "id": request["id"],
            "operation": request["operation"],
            "ok": True,
            "result": result,
            "host": {"host_version": self.host_version},
            "elapsed_ms": 0,
        }
        response.update(self.response_overrides)
        return response

    async def run_powershell(self, script: str, *, timeout_seconds: float) -> SSHResult:
        raise AssertionError("run_powershell is not used by the service tests")

    async def upload_file(self, local_path: Path, remote_path: str, *, timeout_seconds: float) -> None:
        raise AssertionError("upload_file is not used by the service tests")


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.clock = FakeClock()
        self.service = SolidworksService(lambda config: self.transport, clock=self.clock)
        self.config = {"id": "tool", "timeout_max_seconds": 100}
        self.settings_loader = patch.object(pdm_link, "load_stored_pdm_settings", AsyncMock(return_value=None))
        self.load_settings = self.settings_loader.start()
        self.addCleanup(self.settings_loader.stop)

    async def execute(self, operation, arguments=None, timeout=None):
        return json.loads(await self.service.execute(self.config, operation, arguments, timeout))

    def forwarded(self):
        return [request["operation"] for request, _ in self.transport.calls]

    async def test_lifecycle_delegates_with_transport_factory(self):
        status = MagicMock()
        status.host_status = AsyncMock(return_value={"running": True})
        status.test_connection = AsyncMock(return_value=(True, "ok", {}))
        installer = MagicMock()
        installer.install_host = AsyncMock(return_value={"installed": True})
        installer.uninstall_host = AsyncMock(return_value={"installed": False})
        status_module = ModuleType("ragtime.solidworks.host_status")
        status_class = MagicMock(return_value=status)
        setattr(status_module, "SolidworksHostStatusService", status_class)
        installer_module = ModuleType("ragtime.solidworks.host_installer")
        installer_class = MagicMock(return_value=installer)
        setattr(installer_module, "SolidworksHostInstaller", installer_class)
        with patch.dict(
            sys.modules,
            {
                "ragtime.solidworks.host_status": status_module,
                "ragtime.solidworks.host_installer": installer_module,
            },
        ):
            self.assertEqual(await self.service.host_status(self.config), {"running": True})
            self.assertEqual(await self.service.test_connection(self.config), (True, "ok", {}))
            self.assertEqual(await self.service.install_host(self.config), {"installed": True})
            self.assertEqual(await self.service.uninstall_host(self.config), {"installed": False})
        status_class.assert_called_with(transport_factory=self.service.transport_factory)
        installer_class.assert_called_with(transport_factory=self.service.transport_factory, status_service=status)
        status.host_status.assert_awaited_once_with(self.config)
        status.test_connection.assert_awaited_once_with(self.config)
        installer.install_host.assert_awaited_once_with(self.config)
        installer.uninstall_host.assert_awaited_once_with(self.config)

    async def test_guide_and_validation(self):
        self.assertEqual((await self.execute("guide"))["result"]["topic"], "overview")
        self.assertEqual((await self.execute("guide", {"topic": "unknown"}))["error"]["code"], "invalid_arguments")
        self.assertEqual((await self.execute(""))["error"]["code"], "invalid_arguments")
        self.assertEqual((await self.execute("status", []))["error"]["code"], "invalid_arguments")
        self.assertEqual(self.transport.calls, [])

    async def test_describe_cache_and_write_gate(self):
        describe = await self.execute("describe")
        self.assertIn("guide", [op["name"] for op in describe["result"]["operations"]])
        self.transport.error = SolidworksTransportError("no_response", "secret")
        self.assertTrue((await self.execute("describe"))["ok"])
        self.transport.error = None
        self.assertEqual((await self.execute("change"))["error"]["code"], "write_not_allowed")
        self.config["allow_write"] = True
        self.assertTrue((await self.execute("change"))["ok"])
        self.assertEqual(self.forwarded(), ["describe", "describe", "change"])

    async def test_context_timeout_and_oversized_request(self):
        self.config["connection_config"] = {"instance_mode": "attach", "max_instances": 3}
        await self.execute("status", {"allow_write": True}, timeout=30)
        request, timeout = self.transport.calls[0]
        self.assertEqual(timeout, 30)
        self.assertEqual(
            request["context"],
            {
                "solidworks_version": None,
                "instance_mode": "attach",
                "max_instances": 3,
                "idle_shutdown_seconds": 900,
                "allow_write": False,
                "allow_scripts": False,
                "deadline_seconds": 20,
                "max_result_bytes": 32768,
            },
        )
        self.assertEqual((await self.execute("status", {"data": "x" * 10000}))["error"]["code"], "invalid_arguments")
        self.assertEqual(len(self.transport.calls), 1)

    async def test_contract_is_fetched_once_per_tool_and_deadline_has_floor(self):
        await self.execute("change", timeout=3)
        self.assertEqual(self.forwarded(), ["describe"])
        self.assertEqual(self.transport.calls[0][0]["context"]["deadline_seconds"], 5)
        await self.execute("change")
        self.assertEqual(len(self.transport.calls), 1)
        self.config["id"] = "other-tool"
        await self.execute("change")
        self.assertEqual(len(self.transport.calls), 2)

    async def test_errors_and_large_response(self):
        cases: list[tuple[TransportErrorKind, str]] = [
            ("timeout", "operation_timeout"),
            ("no_response", "host_not_running"),
            ("ssh_unreachable", "internal_error"),
            ("ssh_auth_failed", "internal_error"),
        ]
        for kind, code in cases:
            self.transport.error = SolidworksTransportError(kind, "secret")
            response = await self.execute("status")
            self.assertEqual(response["error"]["code"], code)
            self.assertNotIn("secret", json.dumps(response))
        self.transport.error = RuntimeError("secret")
        self.assertNotIn("secret", json.dumps(await self.execute("status")))
        self.transport.error = None
        self.transport.result = {"value": "x" * 50000}
        self.assertEqual(len((await self.execute("status"))["result"]["value"]), 50000)

    async def test_one_budget_is_shared_by_contract_fetch_and_operation(self):
        self.config["allow_write"] = True
        self.transport.on_call = lambda request: setattr(self.clock, "now", self.clock.now + 4) if request["operation"] == "describe" else None
        await self.execute("change", timeout=30)
        describe_request, describe_timeout = self.transport.calls[0]
        operation_request, operation_timeout = self.transport.calls[1]
        self.assertEqual(describe_timeout, 30)
        self.assertEqual(operation_timeout, 26)
        self.assertEqual(operation_request["context"]["deadline_seconds"], 16)

    async def test_operation_is_refused_when_under_six_seconds_remain_after_contract_fetch(self):
        self.config["allow_write"] = True
        self.transport.on_call = lambda request: setattr(self.clock, "now", self.clock.now + 5) if request["operation"] == "describe" else None
        response = await self.execute("change", timeout=10)
        self.assertEqual(response["error"]["code"], "operation_timeout")
        self.assertEqual(self.forwarded(), ["describe"])

    async def test_nonpositive_timeout_is_rejected_without_host_call(self):
        for timeout in (0, -5, 0.0):
            with self.subTest(timeout=timeout):
                response = await self.execute("status", timeout=timeout)
                self.assertEqual(response["error"]["code"], "invalid_arguments")
                self.assertEqual(response["error"]["message"], "timeout must be a positive number of seconds")
        self.assertEqual(self.transport.calls, [])

    async def test_timeout_is_clamped_to_configured_cap_and_defaults_to_it(self):
        self.config["timeout_max_seconds"] = 45
        await self.execute("status")
        await self.execute("status", timeout=3000)
        self.assertEqual([timeout for _, timeout in self.transport.calls], [45, 45])

    async def test_missing_or_zero_cap_falls_back_to_default_positive_cap(self):
        for cap in (None, 0, -1, "45"):
            with self.subTest(cap=cap):
                self.transport.calls.clear()
                self.config["timeout_max_seconds"] = cap
                await self.execute("status")
                self.assertEqual(self.transport.calls[0][1], 100)

    async def test_operation_outside_host_contract_is_refused_without_forwarding(self):
        response = await self.execute("features.list")
        self.assertEqual(response["error"]["code"], "unknown_operation")
        self.assertEqual(self.forwarded(), ["describe"])

    async def test_host_namespace_is_refused_even_when_contract_lists_it(self):
        self.transport.operations = [{"name": "host.shutdown", "access": "read"}]
        for operation in ("host.shutdown", "host.restart"):
            with self.subTest(operation=operation):
                response = await self.execute(operation)
                self.assertEqual(response["error"]["code"], "unknown_operation")
        self.assertNotIn("host.shutdown", self.forwarded())
        self.assertNotIn("host.restart", self.forwarded())

    async def test_admin_pdm_operations_are_hidden_and_rejected(self):
        self.transport.operations = [{"name": name, "access": "read"} for name in pdm_link.PDM_ADMIN_OPERATIONS]
        with patch.object(pdm_link, "resolve_pdm_link", AsyncMock()) as resolve:
            for operation in pdm_link.PDM_ADMIN_OPERATIONS:
                response = await self.execute(operation)
                self.assertEqual(response["error"]["code"], "unknown_operation")
                self.assertEqual(response["error"]["message"], f"There is no operation named '{operation}'. Call describe for the list.")
            self.load_settings.assert_not_awaited()
            resolve.assert_not_awaited()
            self.assertEqual(self.transport.calls, [])
        description = await self.execute("describe")
        names = {item["name"] for item in description["result"]["operations"]}
        self.assertTrue(names.isdisjoint(pdm_link.PDM_ADMIN_OPERATIONS))

    async def test_contract_fetch_failure_without_cache_forwards_nothing(self):
        self.transport.error = SolidworksTransportError("ssh_unreachable", "secret")
        response = await self.execute("features.list")
        self.assertEqual(response["error"]["code"], "internal_error")
        self.assertNotIn("secret", json.dumps(response))
        self.assertEqual(self.forwarded(), ["describe"])

    async def test_mismatched_envelope_is_internal_error_and_not_cached(self):
        self.config["allow_write"] = True
        self.transport.response_overrides = {"id": "someone-else"}
        response = await self.execute("describe")
        self.assertEqual(response["error"]["code"], "internal_error")
        self.assertEqual(response["error"]["message"], "The host answered a different request.")
        self.transport.response_overrides = {"operation": "other"}
        self.assertEqual((await self.execute("status"))["error"]["message"], "The host answered a different request.")
        self.transport.response_overrides = {}
        self.assertTrue((await self.execute("change"))["ok"])
        self.assertEqual(self.forwarded(), ["describe", "status", "describe", "change"])

    async def test_cached_contract_is_dropped_when_host_version_changes(self):
        self.config["allow_write"] = True
        await self.execute("describe")
        self.transport.host_version = "2"
        await self.execute("status")
        self.assertTrue((await self.execute("change"))["ok"])
        self.assertEqual(self.forwarded(), ["describe", "status", "describe", "change"])

    async def test_tool_without_id_is_never_cached(self):
        self.config["allow_write"] = True
        del self.config["id"]
        await self.execute("change")
        await self.execute("change")
        self.assertEqual(self.forwarded(), ["describe", "change", "describe", "change"])

    async def test_contract_cache_is_bounded_and_drops_oldest_tool(self):
        self.config["allow_write"] = True
        with mock.patch("ragtime.solidworks.service.MAX_CACHED_TOOLS", 2):
            for tool_id in ("a", "b", "c"):
                self.config["id"] = tool_id
                await self.execute("describe")
            self.config["id"] = "c"
            await self.execute("change")
            self.assertEqual(self.forwarded().count("describe"), 3)
            self.config["id"] = "a"
            await self.execute("change")
        self.assertEqual(self.forwarded().count("describe"), 4)

    async def test_explicit_describe_always_refetches(self):
        await self.execute("describe")
        await self.execute("describe")
        self.assertEqual(self.forwarded(), ["describe", "describe"])

    async def test_pdm_link_reasons_and_status(self):
        for reason in ("no_link", "linked_tool_missing", "linked_tool_disabled"):
            with patch.object(pdm_link, "resolve_pdm_link", AsyncMock(side_effect=pdm_link.PdmLinkError(reason))):
                response = await self.execute("pdm.search", {"query": "x"})
                self.assertEqual(response["error"]["code"], "pdm_not_linked")
                self.assertEqual(response["error"]["details"]["reason"], reason)
                self.transport.operations = [{"name": "pdm.status", "access": "read"}]
                self.assertTrue((await self.execute("pdm.status"))["ok"])
                self.assertNotIn("pdm", self.transport.calls[-1][0]["context"])

    async def test_forwarded_context_and_describe_link_is_not_cached(self):
        link = pdm_link.PdmLink("id", "vault tool", "index", "PDM")
        self.config["name"] = "Machine"
        self.transport.operations = [{"name": "features.list", "access": "read"}]
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)) as resolve,
            patch.object(pdm_link, "resolve_actor_label", AsyncMock(return_value="alice")),
        ):
            response = await self.execute("describe")
            self.assertEqual([x["name"] for x in response["result"]["operations"]][-3:], ["guide", "pdm.search", "pdm.lookup"])
            self.assertTrue((await self.execute("features.list"))["ok"])
            self.assertEqual(self.transport.calls[-1][0]["context"]["pdm"]["comment_prefix"], "[Ragtime: Machine; user: alice] ")
            self.assertEqual(self.transport.calls[-1][0]["context"]["pdm"]["tool_id"], "tool")
            self.assertEqual(resolve.await_count, 2)
        with patch.object(pdm_link, "resolve_pdm_link", AsyncMock(side_effect=RuntimeError("broken"))):
            self.assertTrue((await self.execute("features.list"))["ok"])
            self.assertNotIn("pdm", self.transport.calls[-1][0]["context"])
            response = await self.execute("describe")
            self.assertNotIn("pdm.search", [x["name"] for x in response["result"]["operations"]])

    async def test_pdm_host_contract_and_write_gates(self):
        link = pdm_link.PdmLink("id", "tool", "index", "vault")
        self.config["allow_write"] = True
        self.config["connection_config"] = {"pdm_approved_folders": ["\\approved"]}
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)),
            patch.object(pdm_link, "resolve_actor_label", AsyncMock(return_value="alice")),
        ):
            response = await self.execute("pdm.file.checkout")
            self.assertEqual(response["error"]["message"], "Update the SolidWorks host to use PDM operations.")
            self.transport.operations = [{"name": "pdm.file.checkout", "access": "write"}]
            await self.execute("describe")
            self.transport.calls.clear()
            self.config["allow_write"] = False
            self.assertEqual((await self.execute("pdm.file.checkout"))["error"]["code"], "write_not_allowed")
            self.config["allow_write"] = True
            self.config["connection_config"]["pdm_approved_folders"] = []
            response = await self.execute("pdm.file.checkout")
            self.assertEqual(response["error"]["code"], "pdm_folder_not_approved")
            self.assertEqual(response["error"]["details"], {"approved_folders": []})
            self.assertEqual(self.transport.calls, [])

    async def test_pdm_write_refusals_do_not_contact_host_with_cold_cache(self):
        link = pdm_link.PdmLink("id", "tool", "index", "vault")
        settings = {"enabled": True, "allow_write": False, "pdm_tool_id": "id", "pdm_approved_folders": []}
        self.load_settings.return_value = settings
        with patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)):
            for operation in pdm_link.PDM_WRITE_OPERATIONS:
                self.assertEqual((await self.execute(operation))["error"]["code"], "write_not_allowed")
            settings["allow_write"] = True
            for operation in pdm_link.PDM_WRITE_OPERATIONS:
                response = await self.execute(operation)
                self.assertEqual(response["error"]["code"], "pdm_folder_not_approved")
                self.assertEqual(response["error"]["details"], {"approved_folders": []})
        self.assertEqual(self.transport.calls, [])

    async def test_pdm_stored_settings_override_stale_config(self):
        link = pdm_link.PdmLink("pdm", "tool", "index", "vault")
        self.config.update(allow_write=True, connection_config={"pdm_tool_id": "pdm", "pdm_approved_folders": ["\\old"]})
        self.transport.operations = [{"name": "pdm.file.checkout", "access": "write"}]
        await self.execute("describe")
        self.transport.calls.clear()
        settings = {"enabled": True, "allow_write": False, "pdm_tool_id": "pdm", "pdm_approved_folders": ["\\new"]}
        self.load_settings.return_value = settings
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)) as resolve,
            patch.object(pdm_link, "resolve_actor_label", AsyncMock(return_value="alice")),
        ):
            self.assertEqual((await self.execute("pdm.file.checkout"))["error"]["code"], "write_not_allowed")
            self.assertEqual(self.transport.calls, [])
            settings["allow_write"] = True
            settings["pdm_approved_folders"] = []
            self.assertEqual((await self.execute("pdm.file.checkout"))["error"]["code"], "pdm_folder_not_approved")
            self.assertEqual(self.transport.calls, [])
            settings["pdm_approved_folders"] = ["\\new"]
            self.assertTrue((await self.execute("pdm.file.checkout"))["ok"])
            context = self.transport.calls[-1][0]["context"]
            self.assertTrue(context["allow_write"])
            self.assertEqual(context["pdm"]["approved_folders"], ["\\new"])
            assert resolve.await_args is not None
            self.assertEqual(resolve.await_args.args[0]["connection_config"]["pdm_tool_id"], "pdm")
            self.assertEqual(self.config["connection_config"]["pdm_approved_folders"], ["\\old"])
        self.assertEqual(self.load_settings.await_count, 3)

    async def test_pdm_stored_settings_fail_closed(self):
        self.config["connection_config"] = {"pdm_tool_id": "old"}
        settings = {"enabled": True, "allow_write": True, "pdm_tool_id": None, "pdm_approved_folders": []}
        self.load_settings.return_value = settings
        response = await self.execute("pdm.search", {"query": "x"})
        self.assertEqual(response["error"]["code"], "pdm_not_linked")
        settings["enabled"] = False
        self.assertEqual((await self.execute("pdm.status"))["error"]["details"]["reason"], "no_link")
        self.load_settings.side_effect = pdm_link.PdmLinkError("no_link")
        self.assertEqual((await self.execute("pdm.status"))["error"]["code"], "pdm_not_linked")
        self.load_settings.side_effect = RuntimeError("secret")
        response = await self.execute("pdm.status")
        self.assertEqual(response["error"]["message"], "The operation failed; check the host configuration.")
        self.assertEqual(self.transport.calls, [])

    async def test_unsaved_pdm_config_uses_passed_settings(self):
        self.config.pop("id")
        self.config.update(allow_write=True, connection_config={"pdm_approved_folders": ["\\passed"]})
        self.transport.operations = [{"name": "pdm.file.checkout", "access": "write"}]
        link = pdm_link.PdmLink("pdm", "tool", "index", "vault")
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)),
            patch.object(pdm_link, "resolve_actor_label", AsyncMock(return_value="alice")),
        ):
            self.assertTrue((await self.execute("pdm.file.checkout"))["ok"])
        context = self.transport.calls[-1][0]["context"]
        self.assertTrue(context["allow_write"])
        self.assertEqual(context["pdm"]["approved_folders"], ["\\passed"])

    async def test_missing_cached_pdm_operation_refreshes_contract_once(self):
        link = pdm_link.PdmLink("id", "tool", "index", "vault")
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)),
            patch.object(pdm_link, "resolve_actor_label", AsyncMock(return_value="alice")),
        ):
            await self.execute("describe")
            self.transport.operations = [{"name": "pdm.file.info", "access": "read"}]
            response = await self.execute("pdm.file.info")
            self.assertTrue(response["ok"])
            self.assertEqual(self.forwarded(), ["describe", "describe", "pdm.file.info"])
            self.transport.calls.clear()
            response = await self.execute("pdm.folder.list")
            self.assertEqual(response["error"]["message"], "Update the SolidWorks host to use PDM operations.")
            self.assertEqual(self.forwarded(), ["describe"])
            self.transport.calls.clear()
            self.service._contracts.clear()
            response = await self.execute("pdm.folder.list")
            self.assertEqual(response["error"]["code"], "unknown_operation")
            self.assertEqual(self.forwarded(), ["describe"])

    async def test_pdm_index_envelope_and_failure(self):
        link = pdm_link.PdmLink("id", "tool", "index", "vault")
        with (
            patch.object(pdm_link, "resolve_pdm_link", AsyncMock(return_value=link)),
            patch.object(pdm_link, "execute_index_operation", AsyncMock(return_value={"items": []})) as execute_index,
        ):
            response = await self.execute("pdm.search", {"query": "x"})
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"], {"items": []})
            self.assertEqual(response["host"], {"host_version": None, "solidworks_revision": None, "solidworks_version": None, "instance_id": None})
            execute_index.side_effect = pdm_link.PdmArgumentError("Bad input.")
            self.assertEqual((await self.execute("pdm.search"))["error"]["code"], "invalid_arguments")
            execute_index.side_effect = RuntimeError("secret")
            with self.assertLogs("ragtime.solidworks.service", level="ERROR") as captured:
                response = await self.execute("pdm.search", {"query": "private-document"})
            self.assertEqual(response["error"]["message"], "The PDM index search failed; check the embedding provider and the index.")
            self.assertNotIn("secret", json.dumps(response))
            self.assertIn("tool tool, operation pdm.search: RuntimeError", captured.output[0])
            self.assertNotIn("secret", captured.output[0])
            self.assertNotIn("private-document", captured.output[0])
            self.assertEqual(self.transport.calls, [])


if __name__ == "__main__":
    unittest.main()
