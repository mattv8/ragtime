import json
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

from fastapi import HTTPException

from ragtime.indexer import routes
from ragtime.indexer.models import CreateToolConfigRequest, ToolConfig, ToolType, UpdateToolConfigRequest
from ragtime.indexer.tool_health import get_heartbeat_timeout_seconds


class SolidworksIndexerWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_switching_link_clears_folders_unless_supplied(self):
        sw = ToolConfig(
            id="sw",
            name="sw",
            tool_type=ToolType.SOLIDWORKS,
            connection_config={
                "host": "h",
                "user": "u",
                "pdm_tool_id": "old",
                "pdm_approved_folders": [r"\Old"],
            },
        )
        pdm = ToolConfig(id="new", name="new", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        save = AsyncMock(return_value=sw)
        with (
            mock.patch.object(routes.repository, "get_tool_config", AsyncMock(side_effect=lambda tool_id: sw if tool_id == "sw" else pdm)),
            mock.patch.object(routes.repository, "update_tool_config", save),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            for incoming, expected in [
                ({"pdm_tool_id": "new"}, []),
                ({"pdm_tool_id": "new", "pdm_approved_folders": ["/New"]}, [r"\New"]),
                ({"pdm_tool_id": "old"}, [r"\Old"]),
            ]:
                await routes.update_tool_config("sw", UpdateToolConfigRequest(connection_config=incoming))
                assert save.await_args is not None
                self.assertEqual(save.await_args.args[1]["connection_config"]["pdm_approved_folders"], expected)

    async def test_create_and_update_reject_invalid_links(self):
        wrong = ToolConfig(id="target", name="wrong", tool_type=ToolType.POSTGRES, connection_config={})
        sw = ToolConfig(id="sw", name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"host": "h", "user": "u"})
        cases = [
            ({"pdm_tool_id": "target"}, None, "The linked PDM tool does not exist."),
            ({"pdm_tool_id": "target"}, wrong, "The linked PDM tool does not exist."),
            ({"pdm_approved_folders": [r"\A"]}, None, "Approved folders need a linked PDM tool."),
        ]
        for incoming, target, message in cases:
            for create in (True, False):
                with self.subTest(incoming=incoming, target=target, create=create):
                    lookup = AsyncMock(side_effect=lambda tool_id: sw if tool_id == "sw" else target)
                    save = AsyncMock()
                    with (
                        mock.patch.object(routes.repository, "get_tool_config", lookup),
                        mock.patch.object(routes.repository, "create_tool_config", save),
                        mock.patch.object(routes.repository, "update_tool_config", save),
                    ):
                        with self.assertRaises(HTTPException) as exc:
                            if create:
                                await routes.create_tool_config(
                                    CreateToolConfigRequest(
                                        name="sw",
                                        tool_type=ToolType.SOLIDWORKS,
                                        connection_config={"host": "h", "user": "u", **incoming},
                                    )
                                )
                            else:
                                await routes.update_tool_config("sw", UpdateToolConfigRequest(connection_config=incoming))
                    self.assertEqual((exc.exception.status_code, exc.exception.detail), (400, message))
                    save.assert_not_awaited()

    async def test_update_merges_and_validates_folders_and_strips_transient_fields(self):
        pdm = ToolConfig(id="pdm", name="vault", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        sw = ToolConfig(
            id="sw",
            name="sw",
            tool_type=ToolType.SOLIDWORKS,
            connection_config={
                "host": "h",
                "user": "u",
                "pdm_tool_id": "pdm",
                "pdm_approved_folders": [r"\Old"],
            },
        )
        lookup = AsyncMock(side_effect=lambda tool_id: sw if tool_id == "sw" else pdm)
        save = AsyncMock(return_value=sw)
        with (
            mock.patch.object(routes.repository, "get_tool_config", lookup),
            mock.patch.object(routes.repository, "update_tool_config", save),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.update_tool_config(
                "sw",
                UpdateToolConfigRequest.model_validate(
                    {
                        "pdm_link": {"tool_id": "fake"},
                        "pdm_link_name": "fake",
                        "connection_config": {
                            "pdm_approved_folders": ["/New"],
                            "pdm_link": {"tool_id": "fake"},
                            "pdm_link_name": "fake",
                        },
                    }
                ),
            )
            assert save.await_args is not None
            config = save.await_args.args[1]["connection_config"]
            self.assertNotIn("pdm_link", save.await_args.args[1])
            self.assertNotIn("pdm_link_name", save.await_args.args[1])
            self.assertEqual((config["pdm_tool_id"], config["pdm_approved_folders"]), ("pdm", [r"\New"]))
            self.assertNotIn("pdm_link", config)
            self.assertNotIn("pdm_link_name", config)
            save.reset_mock()
            with self.assertRaises(HTTPException) as exc:
                await routes.update_tool_config("sw", UpdateToolConfigRequest(connection_config={"pdm_tool_id": None}))
            self.assertEqual((exc.exception.status_code, exc.exception.detail), (400, "Approved folders need a linked PDM tool."))
            save.assert_not_awaited()
            await routes.update_tool_config("sw", UpdateToolConfigRequest(connection_config={"pdm_tool_id": None, "pdm_approved_folders": []}))
            assert save.await_args is not None
            config = save.await_args.args[1]["connection_config"]
            self.assertIsNone(config["pdm_tool_id"])
            self.assertEqual(config["pdm_approved_folders"], [])
            save.reset_mock()
            with self.assertRaises(HTTPException):
                await routes.update_tool_config("sw", UpdateToolConfigRequest(connection_config={"pdm_approved_folders": ["\\"]}))
            save.assert_not_awaited()

    async def test_create_strips_transient_link_fields(self):
        save = AsyncMock(side_effect=lambda config: config)
        request = CreateToolConfigRequest.model_validate(
            {
                "name": "sw",
                "tool_type": "solidworks",
                "pdm_link": {"tool_id": "fake"},
                "pdm_link_name": "fake",
                "connection_config": {"host": "h", "user": "u", "pdm_link": {"tool_id": "fake"}, "pdm_link_name": "fake"},
            }
        )
        with (
            mock.patch.object(routes.repository, "create_tool_config", save),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.create_tool_config(request)
        assert save.await_args is not None
        config = save.await_args.args[0].connection_config
        self.assertNotIn("pdm_link", config)
        self.assertNotIn("pdm_link_name", config)
        self.assertIsNone(save.await_args.args[0].pdm_link)

    async def test_disabling_pdm_only_updates_pdm(self):
        pdm = ToolConfig(id="pdm", name="vault", enabled=False, tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        update = AsyncMock(return_value=pdm)
        with (
            mock.patch.object(routes.repository, "update_tool_config", update),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.toggle_tool_config("pdm", False)
        update.assert_awaited_once_with("pdm", {"enabled": False})

    async def test_pdm_link_validation(self):
        pdm = ToolConfig(id="pdm", name="vault", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        with mock.patch.object(routes.repository, "get_tool_config", AsyncMock(return_value=pdm)):
            await routes._validate_solidworks_pdm_link({"pdm_tool_id": "pdm", "pdm_approved_folders": [r"\A"]})
        with mock.patch.object(routes.repository, "get_tool_config", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as exc:
                await routes._validate_solidworks_pdm_link({"pdm_tool_id": "missing"})
            self.assertEqual(exc.exception.detail, "The linked PDM tool does not exist.")
        with self.assertRaises(HTTPException) as exc:
            await routes._validate_solidworks_pdm_link({"pdm_approved_folders": [r"\A"]})
        self.assertEqual(exc.exception.detail, "Approved folders need a linked PDM tool.")

    async def test_import_clears_missing_pdm_link(self):
        tool = ToolConfig(
            name="sw",
            tool_type=ToolType.SOLIDWORKS,
            connection_config={
                "host": "h",
                "user": "u",
                "pdm_tool_id": "missing",
                "pdm_approved_folders": [r"\A"],
            },
        )
        request = routes.ToolImportRequest(password="password", file_content=json.dumps(routes._build_tool_export_envelope(tool, "password")))
        create = AsyncMock(side_effect=lambda config: config)
        with (
            mock.patch.object(routes.repository, "list_tool_configs", AsyncMock(return_value=[])),
            mock.patch.object(routes.repository, "create_tool_config", create),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.import_tool_config(request)
        assert create.await_args is not None
        config = create.await_args.args[0].connection_config
        self.assertIsNone(config["pdm_tool_id"])
        self.assertEqual(config["pdm_approved_folders"], [])

    async def test_deleting_pdm_does_not_update_solidworks(self):
        pdm = ToolConfig(id="pdm", name="vault", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        update = AsyncMock()
        with (
            mock.patch.object(routes.repository, "get_tool_config", AsyncMock(return_value=pdm)),
            mock.patch.object(routes.repository, "update_tool_config", update),
            mock.patch.object(routes.repository, "delete_tool_config", AsyncMock(return_value=True)),
            mock.patch.object(routes.pdm_indexer, "delete_index", AsyncMock(return_value=(True, "ok"))),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.delete_tool_config("pdm", mock.Mock(), ssh_cleanup=None, confirm_machine_wide=None)
        update.assert_not_awaited()

    def test_computed_pdm_link(self):
        sw = ToolConfig(name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"pdm_tool_id": "pdm"})
        pdm = ToolConfig(id="pdm", name="Vault", enabled=False, tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        self.assertEqual(routes._with_pdm_link(sw, [pdm]).pdm_link, {"tool_id": "pdm", "tool_name": "Vault", "enabled": False, "exists": True})
        pdm.enabled = True
        pdm_link = routes._with_pdm_link(sw, [pdm]).pdm_link
        assert pdm_link is not None
        self.assertTrue(pdm_link["enabled"])
        self.assertEqual(routes._with_pdm_link(sw, []).pdm_link, {"tool_id": "pdm", "tool_name": None, "enabled": False, "exists": False})
        sw.connection_config = {}
        self.assertIsNone(routes._with_pdm_link(sw, [pdm]).pdm_link)

    def test_config_validation_and_defaults(self):
        config = routes._parse_solidworks_connection_config({"host": "host", "user": "user", "shell": "powershell", "pairing_id": "pair"})
        self.assertEqual(config["max_instances"], 1)
        self.assertEqual(config["shell"], "powershell")
        self.assertEqual(config["pairing_id"], "pair")
        self.assertFalse(config["allow_scripts"])
        self.assertTrue(routes._parse_solidworks_connection_config({"host": "host", "user": "user", "allow_scripts": True})["allow_scripts"])
        with self.assertRaises(HTTPException):
            routes._parse_solidworks_connection_config({"host": "host", "user": "user", "allow_scripts": "invalid"})
        with self.assertRaises(HTTPException) as exc:
            routes._parse_solidworks_connection_config({"host": "host", "user": "user", "max_instances": 5})
        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("max_instances", exc.exception.detail)

    async def test_create_validates_config_before_persisting(self):
        create = AsyncMock()
        request = CreateToolConfigRequest(name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"host": "h", "user": "u", "idle_shutdown_seconds": 59})
        with mock.patch.object(routes.repository, "create_tool_config", create):
            with self.assertRaises(HTTPException) as exc:
                await routes.create_tool_config(request)
        self.assertEqual(exc.exception.status_code, 400)
        create.assert_not_awaited()

    async def test_import_validates_config_before_persisting(self):
        tool = ToolConfig(name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"host": "h", "user": "u", "max_instances": 5})
        envelope = routes._build_tool_export_envelope(tool, "password")
        create = AsyncMock()
        request = routes.ToolImportRequest(password="password", file_content=json.dumps(envelope))
        with mock.patch.object(routes.repository, "create_tool_config", create):
            with self.assertRaises(HTTPException) as exc:
                await routes.import_tool_config(request)
        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("max_instances", exc.exception.detail)
        create.assert_not_awaited()

    async def test_update_preserves_secrets_and_other_fields(self):
        tool = ToolConfig(
            id="id",
            name="sw",
            tool_type=ToolType.SOLIDWORKS,
            connection_config={"host": "h", "user": "u", "password": "secret", "shell": "powershell", "pairing_id": "pair"},
        )
        update = AsyncMock(return_value=tool)
        with (
            mock.patch.object(routes.repository, "get_tool_config", AsyncMock(return_value=tool)),
            mock.patch.object(routes.repository, "update_tool_config", update),
            mock.patch.object(routes.rag, "initialize", AsyncMock()),
            mock.patch.object(routes, "invalidate_settings_cache", lambda: None),
            mock.patch.object(routes, "notify_tools_changed", lambda: None),
        ):
            await routes.update_tool_config("id", UpdateToolConfigRequest(connection_config={"password": "", "max_instances": 2, "allow_scripts": True}))
        assert update.await_args is not None
        config = update.await_args.args[1]["connection_config"]
        self.assertEqual(config["password"], "secret")
        self.assertEqual(config["shell"], "powershell")
        self.assertEqual(config["pairing_id"], "pair")
        self.assertEqual(config["max_instances"], 2)
        self.assertTrue(config["allow_scripts"])
        self.assertIsNone(config["pdm_tool_id"])
        self.assertEqual(config["pdm_approved_folders"], [])

    async def test_saved_test_passes_full_config(self):
        tool = ToolConfig(id="id", name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"host": "host"}, timeout_max_seconds=42)
        test = AsyncMock(return_value=(False, "failure", {"error_code": "no_interactive_session"}))
        with (
            mock.patch.object(routes.repository, "get_tool_config", AsyncMock(return_value=tool)),
            mock.patch.object(routes.solidworks_service, "test_connection", test),
            mock.patch.object(
                routes.tool_health_monitor,
                "record_tool_test_result",
                AsyncMock(return_value=SimpleNamespace(changed_tool_ids=set())),
            ),
        ):
            result = await routes.test_saved_tool_connection("id")
        self.assertEqual(result.details, {"error_code": "no_interactive_session"})
        test.assert_awaited_once_with(tool.model_dump())

    async def test_heartbeat_uses_ssh_only(self):
        heartbeat = AsyncMock(return_value=routes.ToolTestResponse(success=True, message="OK"))
        host = AsyncMock()
        with (
            mock.patch.object(routes, "_heartbeat_ssh", heartbeat),
            mock.patch.object(routes.solidworks_service, "host_status", host),
        ):
            await routes._heartbeat_check(ToolType.SOLIDWORKS, {"host": "host"})
        heartbeat.assert_awaited_once_with({"host": "host", "shell": "powershell"})
        host.assert_not_awaited()
        self.assertEqual(get_heartbeat_timeout_seconds({}, ToolType.SOLIDWORKS), 15)


if __name__ == "__main__":
    unittest.main()
