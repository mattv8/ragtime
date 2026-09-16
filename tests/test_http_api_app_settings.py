import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

from ragtime.indexer import routes as indexer_routes


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

    async def test_get_enabled_tool_configs_does_not_apply_health_filter(self) -> None:
        from ragtime.core.app_settings import SettingsCache

        prisma_config = SimpleNamespace(
            id="tool-1",
            name="Dockerhost 1",
            toolType="ssh_shell",
            description="Production Docker host",
            connectionConfig={},
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
                return_value=[],
            ),
        ):
            configs = await cache.get_enabled_tool_configs()
            cached_configs = await cache.get_enabled_tool_configs()

        self.assertEqual([config["id"] for config in configs], ["tool-1"])
        self.assertIs(configs, cached_configs)
        fake_db.toolconfig.find_many.assert_awaited_once()

        cache._tool_configs = []  # pyright: ignore[reportPrivateUsage]
        cache.invalidate()

        self.assertIsNone(cache._enabled_tool_configs)  # pyright: ignore[reportPrivateUsage]
        self.assertIsNone(cache._tool_configs)  # pyright: ignore[reportPrivateUsage]

    async def test_openrouter_credits_endpoint_is_admin_only_and_returns_the_redacted_snapshot(self) -> None:
        app = FastAPI()
        app.include_router(indexer_routes.router)
        transport = httpx.ASGITransport(app=app)
        status = {
            "enabled": True,
            "state": "low",
            "key_remaining_usd": 1.25,
            "wallet_remaining_usd": None,
            "threshold_usd": 5.0,
            "checked_at": "2026-09-15T00:00:00+00:00",
            "stale": False,
            "warning": "OpenRouter credits are low.",
        }
        credits = mock.AsyncMock(return_value=status)

        try:
            app.dependency_overrides[indexer_routes.require_admin] = lambda: (_ for _ in ()).throw(
                HTTPException(status_code=403, detail="Admin access required")
            )
            async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                forbidden = await client.get("/indexes/settings/openrouter-credits")
            self.assertEqual(forbidden.status_code, 403)

            app.dependency_overrides[indexer_routes.require_admin] = lambda: SimpleNamespace(id="admin-1", role="admin")
            with mock.patch("ragtime.indexer.routes.get_openrouter_credit_status", credits):
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    allowed = await client.get("/indexes/settings/openrouter-credits")

            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(allowed.json(), status)
            credits.assert_awaited_once_with()
        finally:
            app.dependency_overrides.clear()
