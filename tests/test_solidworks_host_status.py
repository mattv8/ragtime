import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core.ssh import SSHResult
from ragtime.solidworks.host_status import PDM_VIEWS_WARNING, PROBE_SCRIPT, SolidworksHostStatusService
from ragtime.solidworks.pdm_link import PdmLink, PdmLinkError
from ragtime.solidworks.transport import SolidworksTransportError, TransportErrorKind

LINK = PdmLink(tool_id="pdm-1", tool_name="Render PC", index_name="pdm_render_pc", vault="PDM")
LINKED_CONFIG = {
    "name": "SolidWorks",
    "connection_config": {"pdm_tool_id": "pdm-1", "pdm_approved_folders": ["\\13_Ragtime"]},
}
HOST_VIEWS = [{"vault": "PDM", "root": "C:\\PDM", "logged_in": False, "user": None}]
HOST_PDM = {"client_installed": True, "client_version": "32.3.1", "views": HOST_VIEWS}
LIVE_VIEWS = [{"vault": "PDM", "root": "C:\\PDM", "logged_in": True, "user": "readonly"}]
LIVE_LINKED = {
    "vault": "PDM",
    "root": "C:\\PDM",
    "logged_in": True,
    "user": "readonly",
    "writes_enabled": False,
    "approved_folders": ["\\13_Ragtime"],
    "problem": None,
}


class FakeTransport:
    def __init__(self, probe, response=None):
        self.probe = probe
        self.response = response
        self.calls = []
        self.timeouts = []

    async def run_powershell(self, script, *, timeout_seconds) -> SSHResult:
        self.calls.append(("probe", script))
        self.timeouts.append(timeout_seconds)
        if isinstance(self.probe, Exception):
            raise self.probe
        assert isinstance(self.probe, SSHResult), "the test expected a probe result"
        return self.probe

    async def call(self, request, *, timeout_seconds) -> dict:
        self.calls.append(("call", request))
        self.timeouts.append(timeout_seconds)
        if isinstance(self.response, list):
            item = self.response.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if isinstance(self.response, Exception):
            raise self.response
        assert isinstance(self.response, dict), "the test expected a host response"
        return self.response

    async def upload_file(self, local_path, remote_path, *, timeout_seconds):
        raise AssertionError("upload_file is not used by the status tests")


def probe(present=True, interactive=True, features=None):
    data = {
        "exe_present": present,
        "host_version": "1.0" if present else None,
        "account": {"name": "DOMAIN\\user", "is_administrator": True},
        "interactive": interactive,
        "installed_versions": [2024],
        "host_features": features,
    }
    return SSHResult("noise\n" + json.dumps(data), "", 0, True)


def response():
    return {
        "protocol": 1,
        "id": "1",
        "operation": "status",
        "ok": True,
        "result": {
            "host_version": "1.0",
            "session": {"interactive": True},
            "solidworks": {"installed_versions": [2024], "instances": []},
        },
    }


class HostStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        artifact = mock.patch("ragtime.solidworks.host_status.get_host_artifact", return_value=None)
        artifact.start()
        self.addCleanup(artifact.stop)

    async def test_missing_host_uses_one_probe(self):
        transport = FakeTransport(probe(False))
        status = await SolidworksHostStatusService(lambda _: transport).host_status({})
        self.assertTrue(status["reachable"] and not status["installed"] and not status["running"])
        account = status["account"]
        assert account is not None
        self.assertTrue(account["is_administrator"])
        self.assertIsNone(status["expected_host_version"])
        self.assertIsNone(status["error"])
        self.assertEqual(len(transport.calls), 1)

    async def test_running_status_and_version_check(self):
        transport = FakeTransport(probe(), response())
        service = SolidworksHostStatusService(lambda _: transport)
        ok, _, details = await service.test_connection({"connection_config": {"solidworks_version": 2024}})
        self.assertTrue(ok)
        assert details is not None
        self.assertEqual(details["solidworks_versions"], [2024])
        self.assertEqual(transport.calls[1][1]["operation"], "status")
        self.assertEqual(len(transport.calls), 2)

    async def test_instances_and_updater_hook(self):
        updater = mock.MagicMock()
        updater.state_for.return_value = {"state": "pending"}
        with mock.patch("ragtime.solidworks.host_updater.host_updater", updater):
            config = {"id": "tool"}
            status = await SolidworksHostStatusService(lambda _: FakeTransport(probe(), response())).host_status(config)
            self.assertTrue(status["instances_reported"])
            self.assertEqual(status["auto_update"], {"state": "pending"})
            updater.observe.assert_called_with(config, status["host_version"])
            status = await SolidworksHostStatusService(lambda _: FakeTransport(probe())).host_status(config, probe_only=True)
            self.assertFalse(status["instances_reported"])
            updater.observe.side_effect = ValueError("secret")
            status = await SolidworksHostStatusService(lambda _: FakeTransport(probe(), response())).host_status(config)
            self.assertTrue(status["running"])
            self.assertIsNone(status["error"])

    async def test_no_response_preserves_probe(self):
        cases = [(False, "no_interactive_session"), (True, "host_not_running"), (None, "host_not_running")]
        for interactive, code in cases:
            with self.subTest(interactive=interactive):
                transport = FakeTransport(probe(interactive=interactive), SolidworksTransportError("no_response", "secret"))
                status = await SolidworksHostStatusService(lambda _: transport).host_status({})
                self.assertEqual(status["error"]["code"], code)
                self.assertEqual(status["session"]["interactive"], interactive)
                self.assertEqual(status["solidworks"]["installed_versions"], [2024])
                self.assertNotIn("secret", str(status))

    async def test_unreachable_is_safe(self):
        kinds: list[TransportErrorKind] = ["ssh_unreachable", "ssh_auth_failed"]
        for kind in kinds:
            with self.subTest(kind=kind):
                transport = FakeTransport(SolidworksTransportError(kind, "secret"))
                status = await SolidworksHostStatusService(lambda _: transport).host_status({})
                self.assertFalse(status["reachable"])
                self.assertEqual(status["error"]["code"], kind)
                self.assertIsNone(status["account"])
                self.assertIsNone(status["session"])
                self.assertIsNone(status["solidworks"])
                self.assertNotIn("secret", str(status))

    async def test_update_and_version_mismatch(self):
        with mock.patch("ragtime.solidworks.host_status.get_host_artifact", return_value=SimpleNamespace(host_version="2.0")):
            transport = FakeTransport(probe(), response())
            service = SolidworksHostStatusService(lambda _: transport)
            status = await service.host_status({})
            self.assertTrue(status["update_available"])
            self.assertEqual(status["expected_host_version"], "2.0")
            transport.calls.clear()
            ok, _, details = await service.test_connection({"connection_config": {"solidworks_version": 2023}})
            self.assertFalse(ok)
            assert details is not None
            self.assertEqual(details["error_code"], "version_not_installed")
            self.assertEqual(len(transport.calls), 2)
            ok, _, _ = await service.test_connection({"connection_config": {"solidworks_version": 2023, "instance_mode": "attach"}})
            self.assertTrue(ok)

    async def test_malformed_probe_is_safe(self):
        transport = FakeTransport(SSHResult("not json", "private", 1, False))
        status = await SolidworksHostStatusService(lambda _: transport).host_status({})
        self.assertFalse(status["running"])
        self.assertIsNotNone(status["error"])
        self.assertNotIn("private", str(status))
        self.assertEqual(len(transport.calls), 1)

    async def test_host_error_envelope_is_safe(self):
        envelope = {"operation": "status", "ok": False, "error": {"code": "protocol_mismatch", "message": "secret"}}
        transport = FakeTransport(probe(), envelope)
        status = await SolidworksHostStatusService(lambda _: transport).host_status({})
        self.assertEqual(status["error"]["code"], "protocol_mismatch")
        self.assertNotIn("secret", str(status))

    async def test_new_host_error_codes(self):
        for code in ("host_busy", "host_stopping"):
            with self.subTest(code=code):
                transport = FakeTransport(probe(), {"operation": "status", "ok": False, "error": {"code": code}})
                status = await SolidworksHostStatusService(lambda _: transport).host_status({})
                self.assertEqual(status["error"]["code"], code)

    async def test_features_validation(self):
        self.assertIn("$features = @($v.features | ForEach-Object { [string]$_ })", PROBE_SCRIPT)
        cases = [(["stop-if-idle", "x1"], ["stop-if-idle", "x1"]), ([], []), (None, []), (["good", "BAD"], []), (["a" * 41], []), (["x"] * 21, []), ([1], [])]
        for value, expected in cases:
            with self.subTest(value=value):
                status = await SolidworksHostStatusService(lambda _: FakeTransport(probe(features=value))).host_status({}, probe_only=True)
                self.assertEqual(status["host_features"], expected)

    async def test_activity_validation(self):
        good = {"in_flight": 0, "idle_seconds": 5, "pdm_busy": False}
        cases = [
            (good, good),
            (None, None),
            ({**good, "in_flight": True}, None),
            ({**good, "idle_seconds": -1}, None),
            ({**good, "pdm_busy": 0}, None),
            ({"in_flight": 0}, None),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                host = response()
                host["result"]["activity"] = value
                status = await SolidworksHostStatusService(lambda _: FakeTransport(probe(), host)).host_status({})
                self.assertEqual(status["activity"], expected)

    async def test_unexpected_error_does_not_expose_config(self):
        transport = FakeTransport(RuntimeError("secret"))
        status = await SolidworksHostStatusService(lambda _: transport).host_status({"password": "secret"})
        self.assertEqual(status["error"]["code"], "internal_error")
        self.assertNotIn("secret", str(status))

    async def test_probe_only_never_calls_the_host(self):
        transport = FakeTransport(probe(), response())
        status = await SolidworksHostStatusService(lambda _: transport).host_status({}, probe_only=True)
        self.assertTrue(status["installed"])
        self.assertFalse(status["running"])
        self.assertIsNone(status["error"])
        self.assertEqual([call[0] for call in transport.calls], ["probe"])

    async def test_status_timeouts_are_positive_and_capped_at_sixty(self):
        configs = [{}, {"timeout_max_seconds": 300}, {"timeout_max_seconds": 0}, {"timeout_max_seconds": -5}, {"timeout_max_seconds": 30}]
        expected = [60, 60, 60, 60, 30]
        for config, seconds in zip(configs, expected):
            with self.subTest(config=config):
                transport = FakeTransport(probe(False))
                await SolidworksHostStatusService(lambda _: transport).host_status(config)
                self.assertEqual(transport.timeouts, [seconds])

    async def test_every_status_shape_carries_warning_key(self):
        outcomes = [
            await SolidworksHostStatusService(lambda _: FakeTransport(probe(False))).host_status({}),
            await SolidworksHostStatusService(lambda _: FakeTransport(probe(), response())).host_status({}),
            await SolidworksHostStatusService(lambda _: FakeTransport(SolidworksTransportError("ssh_unreachable", "x"))).host_status({}),
        ]
        for status in outcomes:
            with self.subTest(status=status):
                self.assertIn("warning", status)
                self.assertIsNone(status["warning"])
                self.assertEqual(status["host_features"], [])
                self.assertIsNone(status["activity"])
                self.assertIsNone(status["auto_update"])

    async def test_include_pdm_false_retains_host_pdm_without_extra_call(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        transport = FakeTransport(probe(), host)
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)) as resolve:
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG, include_pdm=False)
        self.assertEqual(status["pdm"], {**HOST_PDM, "linked": None})
        self.assertEqual(len(transport.calls), 2)
        resolve.assert_not_awaited()

    async def test_old_host_reports_no_pdm_block_and_no_warning(self):
        transport = FakeTransport(probe(), response())
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)) as resolve:
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG)
        self.assertIsNone(status["pdm"])
        self.assertIsNone(status["warning"])
        self.assertIsNone(status["error"])
        self.assertEqual(len(transport.calls), 2)
        resolve.assert_not_awaited()

    async def test_pdm_block_without_link_skips_extra_call(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        transport = FakeTransport(probe(), host)
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(side_effect=PdmLinkError("no_link"))):
            status = await SolidworksHostStatusService(lambda _: transport).host_status({"connection_config": {}})
        self.assertEqual(status["pdm"], {"client_installed": True, "client_version": "32.3.1", "views": HOST_VIEWS, "linked": None})
        self.assertIsNone(status["warning"])
        self.assertEqual(len(transport.calls), 2)

    async def test_unexpected_link_failure_only_warns(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        failing_link = mock.AsyncMock(side_effect=RuntimeError("secret database detail"))
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=failing_link):
            transport = FakeTransport(probe(), host)
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG)
            transport = FakeTransport(probe(), dict(host))
            ok, message, details = await SolidworksHostStatusService(lambda _: transport).test_connection(LINKED_CONFIG)
        self.assertTrue(status["reachable"] and status["installed"] and status["running"])
        self.assertEqual(status["warning"], PDM_VIEWS_WARNING)
        self.assertIsNone(status["error"])
        self.assertEqual(status["pdm"], {"client_installed": True, "client_version": "32.3.1", "views": HOST_VIEWS, "linked": None})
        self.assertNotIn("secret", str(status))
        self.assertTrue(ok)
        self.assertIn("is running as", message)
        assert details is not None
        self.assertNotIn("pdm_vault", details)
        self.assertEqual(len(transport.calls), 2)

    async def test_malformed_pdm_block_is_dropped(self):
        host = response()
        host["result"]["pdm"] = {"client_installed": "yes", "views": HOST_VIEWS}
        transport = FakeTransport(probe(), host)
        status = await SolidworksHostStatusService(lambda _: transport).host_status({})
        self.assertIsNone(status["pdm"])
        self.assertIsNone(status["warning"])

    async def test_linked_pdm_status_replaces_views_and_sets_linked(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        pdm_status = {
            "operation": "pdm.status",
            "ok": True,
            "result": {"client_installed": True, "client_version": "32.3.1", "views": LIVE_VIEWS, "linked": LIVE_LINKED},
        }
        transport = FakeTransport(probe(), [host, pdm_status])
        with (
            mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)),
            mock.patch("ragtime.solidworks.host_status.resolve_actor_label", new=mock.AsyncMock(return_value="alice")),
        ):
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG)
        self.assertIsNone(status["warning"])
        self.assertIsNone(status["error"])
        self.assertEqual(status["pdm"]["views"], LIVE_VIEWS)
        self.assertEqual(status["pdm"]["linked"], LIVE_LINKED)
        self.assertEqual(len(transport.calls), 3)
        request = transport.calls[2][1]
        self.assertEqual(request["operation"], "pdm.status")
        self.assertEqual(request["arguments"], {})
        self.assertEqual(
            request["context"]["pdm"],
            {
                "tool_id": "unsaved",
                "vault": "PDM",
                "approved_folders": ["\\13_Ragtime"],
                "comment_prefix": "[Ragtime: SolidWorks; user: alice] ",
            },
        )
        self.assertEqual(request["context"]["solidworks_version"], None)

    async def test_pdm_status_error_keeps_host_views_and_warns_only(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        failure = {"operation": "pdm.status", "ok": False, "error": {"code": "operation_timeout", "message": "secret"}}
        transport = FakeTransport(probe(), [host, failure])
        with (
            mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)),
            mock.patch("ragtime.solidworks.host_status.resolve_actor_label", new=mock.AsyncMock(return_value="alice")),
        ):
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG)
        self.assertEqual(status["warning"], PDM_VIEWS_WARNING)
        self.assertIsNone(status["error"])
        self.assertTrue(status["running"])
        self.assertEqual(status["pdm"], {"client_installed": True, "client_version": "32.3.1", "views": HOST_VIEWS, "linked": None})
        self.assertNotIn("secret", str(status))

    async def test_pdm_status_transport_failure_warns_and_keeps_existing_warning(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        transport = FakeTransport(probe(), [host, SolidworksTransportError("ssh_unreachable", "secret")])
        with (
            mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)),
            mock.patch("ragtime.solidworks.host_status.resolve_actor_label", new=mock.AsyncMock(return_value="alice")),
        ):
            service = SolidworksHostStatusService(lambda _: transport)
            status = await service.host_status(LINKED_CONFIG)
        self.assertEqual(status["warning"], PDM_VIEWS_WARNING)
        self.assertIsNone(status["error"])
        self.assertEqual(status["pdm"]["views"], HOST_VIEWS)

    async def test_probe_only_never_calls_pdm_status(self):
        transport = FakeTransport(probe(), response())
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)) as resolve:
            status = await SolidworksHostStatusService(lambda _: transport).host_status(LINKED_CONFIG, probe_only=True)
        self.assertEqual([call[0] for call in transport.calls], ["probe"])
        self.assertIsNone(status["pdm"])
        resolve.assert_not_awaited()

    async def test_test_connection_details_include_linked_pdm_view(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        pdm_status = {
            "operation": "pdm.status",
            "ok": True,
            "result": {"client_installed": True, "client_version": "32.3.1", "views": LIVE_VIEWS, "linked": LIVE_LINKED},
        }
        transport = FakeTransport(probe(), [host, pdm_status])
        with (
            mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(return_value=LINK)),
            mock.patch("ragtime.solidworks.host_status.resolve_actor_label", new=mock.AsyncMock(return_value="alice")),
        ):
            ok, _, details = await SolidworksHostStatusService(lambda _: transport).test_connection(LINKED_CONFIG)
        self.assertTrue(ok)
        assert details is not None
        self.assertEqual(details["pdm_client_version"], "32.3.1")
        self.assertEqual(details["pdm_vault"], "PDM")
        self.assertEqual(details["pdm_vault_root"], "C:\\PDM")
        self.assertEqual(details["pdm_user"], "readonly")

    async def test_test_connection_details_omit_pdm_without_logged_in_link(self):
        host = response()
        host["result"]["pdm"] = HOST_PDM
        transport = FakeTransport(probe(), host)
        with mock.patch("ragtime.solidworks.host_status.resolve_pdm_link", new=mock.AsyncMock(side_effect=PdmLinkError("no_link"))):
            ok, _, details = await SolidworksHostStatusService(lambda _: transport).test_connection({"connection_config": {}})
        self.assertTrue(ok)
        assert details is not None
        self.assertNotIn("pdm_vault", details)
        self.assertNotIn("pdm_client_version", details)


if __name__ == "__main__":
    unittest.main()
