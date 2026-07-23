import unittest
from types import SimpleNamespace
from unittest import mock


class HttpApiAppSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_tool_configs_decrypts_nested_http_api_secrets_before_caching(self) -> None:
        from ragtime.core.app_settings import SettingsCache
        from ragtime.core.encryption import ENCRYPTED_PREFIX, encrypt_secret
        from ragtime.http_api.secrets import encrypt_http_api_nested_secrets

        original_client_id = "ukg-client"
        original_client_secret = "ukg-secret"
        original_password = "top-level-password"
        encrypted_config = encrypt_http_api_nested_secrets(
            {
                "base_url": "https://example.test",
                "auth_mode": "oauth2_client_credentials",
                "password": encrypt_secret(original_password),
                "token_request_fields": [
                    {"name": "client_id", "value": original_client_id, "secret": False},
                    {"name": "client_secret", "value": original_client_secret, "secret": True},
                ],
            }
        )
        prisma_config = SimpleNamespace(
            id="tool-1",
            name="UKG API",
            toolType="http_api",
            description="HTTP API tool",
            connectionConfig=encrypted_config,
            maxResults=5,
            timeoutMaxSeconds=300,
            allowWrite=False,
        )
        fake_db = SimpleNamespace(toolconfig=SimpleNamespace(find_many=mock.AsyncMock(return_value=[prisma_config])))
        cache = SettingsCache()

        with (
            mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch(
                "ragtime.indexer.tool_health.tool_health_monitor.filter_healthy_tool_config_dicts",
                side_effect=lambda configs: configs,
            ),
        ):
            configs = await cache.get_tool_configs()
            cached_configs = await cache.get_tool_configs()

        self.assertIs(configs, cached_configs)
        fake_db.toolconfig.find_many.assert_awaited_once()

        connection_config = configs[0]["connection_config"]
        self.assertEqual(connection_config["password"], original_password)
        self.assertEqual(connection_config["token_request_fields"][0]["value"], original_client_id)
        self.assertEqual(connection_config["token_request_fields"][1]["value"], original_client_secret)
        self.assertFalse(connection_config["token_request_fields"][0]["value"].startswith(ENCRYPTED_PREFIX))
        self.assertFalse(connection_config["token_request_fields"][1]["value"].startswith(ENCRYPTED_PREFIX))
