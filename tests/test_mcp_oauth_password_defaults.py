import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from prisma.enums import McpAuthMethod

from ragtime.core.app_setting_defaults import DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD
from ragtime.indexer.models import AppSettings
from ragtime.indexer.repository import IndexerRepository
from ragtime.mcp import config_routes


def _build_route_record(*, auth_method: McpAuthMethod, auth_password: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id="route-1",
        name="Route 1",
        routePath="route_1",
        description="desc",
        enabled=True,
        requireAuth=True,
        authPassword=auth_password,
        authMethod=auth_method,
        authClientId=None,
        allowedLdapGroup=None,
        includeKnowledgeSearch=True,
        includeGitHistory=True,
        selectedDocumentIndexes=[],
        selectedFilesystemIndexes=[],
        selectedSchemaIndexes=[],
        toolSelections=[],
        createdAt=datetime(2026, 1, 1),
        updatedAt=datetime(2026, 1, 1),
    )


def _build_settings_row(**overrides) -> SimpleNamespace:
    values = {
        "id": "default",
        "serverName": "Ragtime",
        "defaultThemePack": "default",
        "enabledTools": [],
        "odooContainer": "",
        "postgresContainer": "",
        "postgresHost": "",
        "postgresPort": 5432,
        "postgresUser": "",
        "postgresPassword": "",
        "postgresDb": "",
        "enableWriteOps": False,
        "openaiApiKey": "",
        "anthropicApiKey": "",
        "embeddingProvider": "ollama",
        "embeddingModel": "nomic-embed-text",
        "embeddingDimensions": None,
        "ollamaProtocol": "http",
        "ollamaHost": "localhost",
        "ollamaPort": 11434,
        "ollamaBaseUrl": "http://localhost:11434",
        "allowedChatModels": [],
        "mcpDefaultRoutePassword": None,
        "mcpDefaultRouteAuthMethod": McpAuthMethod.oauth2,
        "mcpDefaultRouteAllowedGroup": None,
        "updatedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class McpOauthPasswordDefaultsTests(unittest.IsolatedAsyncioTestCase):
    def test_app_settings_default_mcp_auth_method_is_oauth2(self) -> None:
        settings = AppSettings()
        auth_method_description = AppSettings.model_fields["mcp_default_route_auth_method"].description
        password_description = AppSettings.model_fields["mcp_default_route_password"].description

        self.assertEqual(settings.mcp_default_route_auth_method, "oauth2")
        assert auth_method_description is not None
        assert password_description is not None
        self.assertIn("local or LDAP", auth_method_description)
        self.assertIn("MCP-Password", password_description)

    def test_create_route_request_defaults_to_oauth2(self) -> None:
        request = config_routes.CreateMcpRouteRequest(name="Route", route_path="route")
        auth_password_description = config_routes.CreateMcpRouteRequest.model_fields["auth_password"].description
        allowed_group_description = config_routes.CreateMcpRouteRequest.model_fields["allowed_ldap_group"].description

        self.assertEqual(request.auth_method, "oauth2")
        assert auth_password_description is not None
        assert allowed_group_description is not None
        self.assertIn("MCP-Password", auth_password_description)
        self.assertIn("local or LDAP", allowed_group_description)

    async def test_settings_cache_creates_defaults_with_oauth2_auth_method(self) -> None:
        from ragtime.core.app_settings import SettingsCache

        create = mock.AsyncMock(return_value=_build_settings_row())
        fake_db = SimpleNamespace(
            appsettings=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=None),
                create=create,
            )
        )
        cache = SettingsCache()

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            await cache.get_settings()

        create_call = create.await_args
        assert create_call is not None
        self.assertEqual(
            create_call.kwargs["data"],
            {
                "id": "default",
                "mcpDefaultRouteAuthMethod": McpAuthMethod(DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD),
            },
        )

    async def test_repository_creates_defaults_with_oauth2_auth_method(self) -> None:
        create = mock.AsyncMock(return_value=_build_settings_row())
        fake_db = SimpleNamespace(
            appsettings=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=None),
                create=create,
            )
        )
        repository = IndexerRepository()

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.get_settings()

        create_call = create.await_args
        assert create_call is not None
        self.assertEqual(
            create_call.kwargs["data"],
            {
                "id": "default",
                "mcpDefaultRouteAuthMethod": McpAuthMethod(DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD),
            },
        )

    async def test_create_mcp_route_keeps_oauth2_password_fallback(self) -> None:
        request = config_routes.CreateMcpRouteRequest(
            name="Route",
            route_path="route",
            require_auth=True,
            auth_method="oauth2",
            auth_password="super-secret",
        )
        db = SimpleNamespace(
            mcprouteconfig=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=None),
                create=mock.AsyncMock(),
            ),
        )

        async def _create(*, data):
            return _build_route_record(
                auth_method=data["authMethod"],
                auth_password=data.get("authPassword"),
            )

        db.mcprouteconfig.create.side_effect = _create

        with (
            mock.patch("ragtime.mcp.config_routes.get_db", mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.mcp.config_routes.notify_tools_changed"),
        ):
            result = await config_routes.create_mcp_route(request, _user=mock.sentinel.user)

        create_data = db.mcprouteconfig.create.await_args.kwargs["data"]
        self.assertEqual(create_data["authMethod"], McpAuthMethod.oauth2)
        self.assertNotEqual(create_data["authPassword"], "super-secret")
        self.assertEqual(result.auth_method, "oauth2")
        self.assertEqual(result.auth_password, "super-secret")
        self.assertTrue(result.has_password)

    async def test_update_mcp_route_retains_and_clears_oauth2_password_fallback(self) -> None:
        encrypted_password = config_routes.encrypt_secret("existing-secret")
        existing_route = _build_route_record(
            auth_method=McpAuthMethod.password,
            auth_password=encrypted_password,
        )
        updated_route = _build_route_record(
            auth_method=McpAuthMethod.oauth2,
            auth_password=encrypted_password,
        )
        cleared_route = _build_route_record(
            auth_method=McpAuthMethod.oauth2,
            auth_password=None,
        )
        db = SimpleNamespace(
            mcprouteconfig=SimpleNamespace(
                find_unique=mock.AsyncMock(side_effect=[existing_route, updated_route, updated_route, cleared_route]),
                update=mock.AsyncMock(),
            ),
        )

        with (
            mock.patch("ragtime.mcp.config_routes.get_db", mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.mcp.config_routes.notify_tools_changed"),
        ):
            retained = await config_routes.update_mcp_route(
                "route-1",
                config_routes.UpdateMcpRouteRequest(auth_method="oauth2"),
                _user=mock.sentinel.user,
            )
            cleared = await config_routes.update_mcp_route(
                "route-1",
                config_routes.UpdateMcpRouteRequest(auth_method="oauth2", clear_password=True),
                _user=mock.sentinel.user,
            )

        first_update = db.mcprouteconfig.update.await_args_list[0].kwargs["data"]
        second_update = db.mcprouteconfig.update.await_args_list[1].kwargs["data"]
        self.assertEqual(first_update, {"authMethod": McpAuthMethod.oauth2})
        self.assertEqual(second_update["authMethod"], McpAuthMethod.oauth2)
        self.assertIsNone(second_update["authPassword"])
        assert retained is not None
        assert cleared is not None
        self.assertEqual(retained.auth_password, "existing-secret")
        self.assertTrue(retained.has_password)
        self.assertIsNone(cleared.auth_password)
        self.assertFalse(cleared.has_password)


if __name__ == "__main__":
    unittest.main()
