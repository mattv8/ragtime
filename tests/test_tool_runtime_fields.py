import unittest
from unittest import mock

from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.indexer.repository import repository
from ragtime.indexer.tool_runtime_fields import with_runtime_fields

HOST_STATE = {"state": "current", "installed_version": "1.2.0"}
TUNNEL_STATE = {"connected": True, "version": 2, "current_version": 2, "outdated": False}
UPDATER_PATH = "ragtime.solidworks.host_updater.host_updater"
SUMMARY_PATH = "ragtime.remote_client.tunnel_versions.tunnel_client_summary"


def _tool(tool_type: ToolType = ToolType.SOLIDWORKS, connection_config: dict | None = None) -> ToolConfig:
    return ToolConfig(id="tool-1", name="Tool", tool_type=tool_type, connection_config=connection_config or {})


class WithRuntimeFieldsTests(unittest.TestCase):
    def test_solidworks_tool_gets_host_updater_state(self):
        updater = mock.Mock()
        updater.state_for.return_value = HOST_STATE
        with mock.patch(UPDATER_PATH, updater):
            result = with_runtime_fields(_tool())

        updater.state_for.assert_called_once_with("tool-1")
        self.assertEqual(result.solidworks_host_update, HOST_STATE)
        self.assertIsNone(result.tunnel_client)

    def test_non_solidworks_tool_gets_no_host_update_state(self):
        updater = mock.Mock()
        with mock.patch(UPDATER_PATH, updater):
            result = with_runtime_fields(_tool(ToolType.POSTGRES))

        updater.state_for.assert_not_called()
        self.assertIsNone(result.solidworks_host_update)

    def test_tunnel_transport_gets_client_summary(self):
        config = {"transport": "tunnel", "pairing_id": "pair-1"}
        with mock.patch(SUMMARY_PATH, return_value=TUNNEL_STATE) as summary:
            result = with_runtime_fields(_tool(ToolType.SOLIDWORKS, config))

        summary.assert_called_once_with(config)
        self.assertEqual(result.tunnel_client, TUNNEL_STATE)

    def test_direct_transport_gets_no_client_summary(self):
        with mock.patch(SUMMARY_PATH) as summary:
            result = with_runtime_fields(_tool(ToolType.SOLIDWORKS, {"transport": "ssh", "host": "h"}))

        summary.assert_not_called()
        self.assertIsNone(result.tunnel_client)

    def test_raising_host_lookup_leaves_field_none(self):
        updater = mock.Mock()
        updater.state_for.side_effect = RuntimeError("boom")
        with mock.patch(UPDATER_PATH, updater):
            result = with_runtime_fields(_tool())

        self.assertIsNone(result.solidworks_host_update)

    def test_raising_tunnel_lookup_leaves_field_none(self):
        with mock.patch(SUMMARY_PATH, side_effect=ValueError("bad")):
            result = with_runtime_fields(_tool(ToolType.SOLIDWORKS, {"transport": "tunnel", "pairing_id": "p"}))

        self.assertIsNone(result.tunnel_client)

    def test_input_config_is_not_mutated(self):
        config = _tool()
        with mock.patch(UPDATER_PATH, mock.Mock(state_for=mock.Mock(return_value=HOST_STATE))):
            with_runtime_fields(config)

        self.assertIsNone(config.solidworks_host_update)


class RuntimeFieldPersistencePinTests(unittest.IsolatedAsyncioTestCase):
    """Runtime fields are computed for responses only and must never reach the database."""

    RUNTIME_KEYS = ("solidworks_host_update", "tunnel_client")

    def _config_with_runtime_fields(self) -> ToolConfig:
        return _tool(ToolType.SOLIDWORKS, {"transport": "tunnel", "pairing_id": "p"}).model_copy(
            update={"solidworks_host_update": HOST_STATE, "tunnel_client": TUNNEL_STATE}
        )

    async def test_create_does_not_store_runtime_fields(self):
        db = mock.MagicMock()
        db.toolconfig.create = mock.AsyncMock(return_value=object())
        model = _tool(ToolType.SOLIDWORKS, {"transport": "tunnel", "pairing_id": "p"})

        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(repository, "_prisma_tool_config_to_model", return_value=model),
            # The local generated Prisma client predates the SOLIDWORKS enum value; the enum mapping is not under test.
            mock.patch("ragtime.indexer.repository._to_prisma_tool_type", return_value="solidworks"),
        ):
            await repository.create_tool_config(self._config_with_runtime_fields())

        data = db.toolconfig.create.call_args.kwargs["data"]
        for key in self.RUNTIME_KEYS:
            self.assertNotIn(key, data)

    async def test_update_does_not_store_runtime_fields(self):
        db = mock.MagicMock()
        db.toolconfig.update = mock.AsyncMock()
        db.toolconfig.find_unique = mock.AsyncMock(return_value=None)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            await repository.update_tool_config(
                "tool-1",
                {"name": "renamed", "solidworks_host_update": HOST_STATE, "tunnel_client": TUNNEL_STATE},
            )

        data = db.toolconfig.update.call_args.kwargs["data"]
        self.assertEqual(data, {"name": "renamed"})


if __name__ == "__main__":
    unittest.main()
