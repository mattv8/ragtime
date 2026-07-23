import json
import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma.models import User
from pydantic import ValidationError
from starlette.requests import Request

from ragtime.http_api.models import HttpApiAuthMode, HttpApiConfiguredHeader, HttpApiConnectionConfig, HttpApiValidationResult
from ragtime.http_api.oauth import OAuthCredentialUpdate, OAuthPollResult
from ragtime.indexer import routes
from ragtime.indexer.models import CreateToolConfigRequest, ToolConfig, ToolTestRequest, ToolType, UpdateToolConfigRequest


class HttpApiRouteTests(unittest.IsolatedAsyncioTestCase):
    def _request(self, *, scheme: str = "https", host: str = "admin.example.test") -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/indexes/tools/http-api/oauth/start",
                "scheme": scheme,
                "headers": [(b"host", host.encode())],
                "server": (host, 443 if scheme == "https" else 80),
            }
        )

    async def test_http_api_oauth_start_uses_configured_external_base_url(self) -> None:
        manager = SimpleNamespace(
            start=mock.AsyncMock(return_value=SimpleNamespace(status="pending", session_id="session-1", authorization_url="https://idp.test/auth"))
        )
        user = cast(User, SimpleNamespace(id="admin-1"))
        request = routes.HttpApiOAuthStartRequest(
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "oauth2",
                "oauth_flow": "authorization_code_pkce",
                "oauth_authorization_url": "https://idp.example.test/authorize",
                "oauth_client_id": "client-1",
                "oauth_token_url": "https://idp.example.test/token",
            }
        )

        with (
            mock.patch.object(routes, "http_api_oauth_manager", manager),
            mock.patch.object(routes.settings, "external_base_url", "https://ragtime.example.test/"),
        ):
            result = await routes.start_http_api_oauth(request, self._request(), user)

        manager.start.assert_awaited_once()
        self.assertEqual(manager.start.await_args.args[0], "admin-1")
        self.assertEqual(manager.start.await_args.args[2], "https://ragtime.example.test/indexes/tools/http-api/oauth/callback")
        self.assertEqual(result["callback_url"], "https://ragtime.example.test/indexes/tools/http-api/oauth/callback")

    async def test_http_api_oauth_callback_does_not_reflect_provider_error(self) -> None:
        manager = SimpleNamespace(complete_authorization_code=mock.AsyncMock(side_effect=RuntimeError("provider secret description")))

        with mock.patch.object(routes, "http_api_oauth_manager", manager):
            response = await routes.http_api_oauth_callback(state="state-1", code=None, error="provider secret description")

        self.assertEqual(response.status_code, 200)
        body = bytes(response.body).decode()
        self.assertNotIn("provider secret description", body)
        self.assertIn("window.close()", body)

    async def test_http_api_oauth_session_is_consumed_only_after_create_persists(self) -> None:
        request = CreateToolConfigRequest(
            name="OAuth API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "oauth2",
                "oauth_flow": "authorization_code_pkce",
                "oauth_authorization_url": "https://idp.example.test/authorize",
                "oauth_client_id": "client-1",
                "oauth_token_url": "https://idp.example.test/token",
                "oauth_session_id": "session-1",
            },
        )
        created = ToolConfig(
            id="tool-1",
            name="OAuth API",
            tool_type=ToolType.HTTP_API,
            connection_config={key: value for key, value in request.connection_config.items() if key != "oauth_session_id"},
        )
        manager = SimpleNamespace(
            peek_credentials=mock.AsyncMock(return_value=OAuthCredentialUpdate(access_token="access-1", refresh_token="refresh-1")),
            consume=mock.AsyncMock(),
        )
        repo = mock.AsyncMock()
        repo.create_tool_config = mock.AsyncMock(return_value=created)

        with (
            mock.patch.object(routes, "http_api_oauth_manager", manager),
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(routes, "notify_tools_changed"),
            mock.patch.object(routes, "invalidate_settings_cache"),
        ):
            await routes.create_tool_config(request, cast(User, SimpleNamespace(id="admin-1")))

        persisted = repo.create_tool_config.await_args.args[0].connection_config
        self.assertEqual(persisted["oauth_access_token"], "access-1")
        self.assertEqual(persisted["oauth_refresh_token"], "refresh-1")
        manager.consume.assert_awaited_once_with("admin-1", "session-1")

    def test_http_api_oauth_poll_serializes_real_result_shape(self) -> None:
        payload = routes._oauth_result_payload(OAuthPollResult(status="pending", session_id="session-1", retry_after=7))

        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["retry_after_seconds"], 7)
        self.assertNotIn("retry_after", payload)

    async def test_http_api_oauth_update_persists_credentials_and_consumes_after_success(self) -> None:
        base = {
            "base_url": "https://api.example.test",
            "auth_mode": "oauth2",
            "oauth_flow": "authorization_code_pkce",
            "oauth_authorization_url": "https://idp.example.test/authorize",
            "oauth_client_id": "client-1",
            "oauth_token_url": "https://idp.example.test/token",
        }
        original = ToolConfig(id="tool-1", name="OAuth API", tool_type=ToolType.HTTP_API, connection_config=base)
        updated = original.model_copy(deep=True)
        manager = SimpleNamespace(
            peek_credentials=mock.AsyncMock(return_value=OAuthCredentialUpdate(access_token="access-2", refresh_token="refresh-2")),
            consume=mock.AsyncMock(),
        )
        repo = mock.AsyncMock()
        repo.get_tool_config = mock.AsyncMock(return_value=original)
        repo.update_tool_config = mock.AsyncMock(return_value=updated)

        with (
            mock.patch.object(routes, "http_api_oauth_manager", manager),
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(routes, "notify_tools_changed"),
            mock.patch.object(routes, "invalidate_settings_cache"),
        ):
            await routes.update_tool_config(
                "tool-1",
                UpdateToolConfigRequest(connection_config={**base, "oauth_session_id": "session-2"}),
                cast(User, SimpleNamespace(id="admin-1")),
            )

        persisted = repo.update_tool_config.await_args.args[1]["connection_config"]
        self.assertEqual(persisted["oauth_access_token"], "access-2")
        self.assertEqual(persisted["oauth_refresh_token"], "refresh-2")
        manager.consume.assert_awaited_once_with("admin-1", "session-2")

    async def test_http_api_oauth_update_preserves_omitted_saved_secrets_without_consuming(self) -> None:
        original_config = {
            "base_url": "https://api.example.test",
            "auth_mode": "oauth2",
            "oauth_flow": "authorization_code_pkce",
            "oauth_authorization_url": "https://idp.example.test/authorize",
            "oauth_client_id": "client-1",
            "oauth_client_secret": "client-secret",
            "oauth_token_url": "https://idp.example.test/token",
            "oauth_access_token": "access-saved",
            "oauth_refresh_token": "refresh-saved",
        }
        original = ToolConfig(id="tool-1", name="OAuth API", tool_type=ToolType.HTTP_API, connection_config=original_config)
        updated = original.model_copy(deep=True)
        manager = SimpleNamespace(consume=mock.AsyncMock())
        repo = mock.AsyncMock()
        repo.get_tool_config = mock.AsyncMock(return_value=original)
        repo.update_tool_config = mock.AsyncMock(return_value=updated)

        with (
            mock.patch.object(routes, "http_api_oauth_manager", manager),
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(routes, "notify_tools_changed"),
            mock.patch.object(routes, "invalidate_settings_cache"),
        ):
            incoming_config = {
                key: value for key, value in original_config.items() if key not in {"oauth_client_secret", "oauth_access_token", "oauth_refresh_token"}
            }
            incoming_config["base_url"] = "https://new-api.example.test"
            await routes.update_tool_config(
                "tool-1",
                UpdateToolConfigRequest(connection_config=incoming_config),
                cast(User, SimpleNamespace(id="admin-1")),
            )

        persisted = repo.update_tool_config.await_args.args[1]["connection_config"]
        self.assertEqual(persisted["base_url"], "https://new-api.example.test")
        self.assertEqual(persisted["oauth_client_secret"], "client-secret")
        self.assertEqual(persisted["oauth_access_token"], "access-saved")
        self.assertEqual(persisted["oauth_refresh_token"], "refresh-saved")
        manager.consume.assert_not_awaited()

    async def test_http_api_oauth_update_failure_keeps_session_for_retry(self) -> None:
        config = {
            "base_url": "https://api.example.test",
            "auth_mode": "oauth2",
            "oauth_flow": "authorization_code_pkce",
            "oauth_authorization_url": "https://idp.example.test/authorize",
            "oauth_client_id": "client-1",
            "oauth_token_url": "https://idp.example.test/token",
        }
        original = ToolConfig(id="tool-1", name="OAuth API", tool_type=ToolType.HTTP_API, connection_config=config)
        manager = SimpleNamespace(
            peek_credentials=mock.AsyncMock(return_value=OAuthCredentialUpdate(access_token="access-3")),
            consume=mock.AsyncMock(),
        )
        repo = mock.AsyncMock()
        repo.get_tool_config = mock.AsyncMock(return_value=original)
        repo.update_tool_config = mock.AsyncMock(side_effect=RuntimeError("database unavailable"))

        with (
            mock.patch.object(routes, "http_api_oauth_manager", manager),
            mock.patch.object(routes, "repository", repo),
        ):
            with self.assertRaises(RuntimeError):
                await routes.update_tool_config(
                    "tool-1",
                    UpdateToolConfigRequest(connection_config={**config, "oauth_session_id": "session-3"}),
                    cast(User, SimpleNamespace(id="admin-1")),
                )

        manager.consume.assert_not_awaited()

    async def test_get_http_api_tool_edit_config_returns_user_entered_values_excluding_provider_tokens(self) -> None:
        config = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "api_key": "api-key-1",
                "oauth_client_secret": "client-secret-1",
                "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                "request_body_fields": [{"name": "tenant", "value": "north", "secret": False}],
                "oauth_access_token": "access-token-1",
                "oauth_refresh_token": "refresh-token-1",
            },
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=config)

        with mock.patch("ragtime.indexer.routes.repository", mock_repo):
            response = await routes.get_http_api_tool_edit_config("tool-http-api")

        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        payload = json.loads(bytes(response.body))
        connection_config = payload["connection_config"]
        self.assertEqual(connection_config["base_url"], "https://api.example.test")
        self.assertEqual(connection_config["api_key"], "api-key-1")
        self.assertEqual(connection_config["oauth_client_secret"], "client-secret-1")
        self.assertEqual(connection_config["request_headers"], [{"name": "X-Tenant", "value": "tenant-secret"}])
        self.assertEqual(connection_config["request_body_fields"], [{"name": "tenant", "value": "north", "secret": False}])
        self.assertNotIn("oauth_access_token", connection_config)
        self.assertNotIn("oauth_refresh_token", connection_config)

    async def test_get_http_api_tool_edit_config_returns_404_for_missing_or_non_http_api_tools(self) -> None:
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(
            side_effect=[None, ToolConfig(id="tool-1", name="Postgres", tool_type=ToolType.POSTGRES, connection_config={})]
        )

        with mock.patch("ragtime.indexer.routes.repository", mock_repo):
            with self.assertRaises(HTTPException) as missing_ctx:
                await routes.get_http_api_tool_edit_config("missing-tool")
            with self.assertRaises(HTTPException) as wrong_type_ctx:
                await routes.get_http_api_tool_edit_config("tool-1")

        self.assertEqual(missing_ctx.exception.status_code, 404)
        self.assertEqual(missing_ctx.exception.detail, "HTTP API tool configuration not found")
        self.assertEqual(wrong_type_ctx.exception.status_code, 404)
        self.assertEqual(wrong_type_ctx.exception.detail, "HTTP API tool configuration not found")

    def test_get_http_api_tool_edit_config_route_requires_admin(self) -> None:
        route = next(
            (r for r in routes.router.routes if getattr(r, "path", None) == "/indexes/tools/{tool_id}/http-api-edit-config"),
            None,
        )

        self.assertIsNotNone(route, "GET /indexes/tools/{tool_id}/http-api-edit-config route not found on router")
        dep_calls = [dep.call for dep in getattr(route, "dependant", SimpleNamespace(dependencies=[])).dependencies]
        self.assertIn(routes.require_admin, dep_calls)

    async def test_create_tool_config_accepts_nested_token_exchange_config_and_response_redacts_nested_secrets(self) -> None:
        request = CreateToolConfigRequest(
            name="UKG API",
            tool_type=ToolType.HTTP_API,
            description="",
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_id", "value": "client-id", "secret": False},
                    {"name": "client_secret", "value": "client-secret", "secret": True},
                ],
                "request_body_format": "json",
                "request_body_fields": [
                    {"name": "tenant", "value": "north", "secret": False},
                    {"name": "client_secret", "value": "resource-secret", "secret": True},
                ],
            },
        )
        created = ToolConfig(
            id="tool-http-api",
            name="UKG API",
            tool_type=ToolType.HTTP_API,
            description="",
            connection_config=request.connection_config,
        )
        mock_repo = mock.AsyncMock()
        mock_repo.create_tool_config = mock.AsyncMock(return_value=created)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            result = await routes.create_tool_config(request)

        mock_repo.create_tool_config.assert_awaited_once()
        persisted = mock_repo.create_tool_config.await_args.args[0]
        self.assertEqual(persisted.connection_config["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(persisted.connection_config["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(persisted.connection_config["token_request_fields"][2]["value"], "client-secret")
        self.assertEqual(persisted.connection_config["request_body_fields"][1]["value"], "resource-secret")
        dumped = result.model_dump(mode="json")
        self.assertEqual(dumped["connection_config"]["request_headers"], [{"name": "X-Tenant", "value": ""}])
        self.assertEqual(dumped["connection_config"]["token_request_headers"], [{"name": "X-Token-Key", "value": ""}])
        self.assertEqual(
            dumped["connection_config"]["token_request_fields"],
            [
                {"name": "grant_type", "value": "", "secret": True},
                {"name": "client_id", "value": "", "secret": True},
                {"name": "client_secret", "value": "", "secret": True},
            ],
        )
        self.assertEqual(
            dumped["connection_config"]["request_body_fields"],
            [
                {"name": "tenant", "value": "", "secret": True},
                {"name": "client_secret", "value": "", "secret": True},
            ],
        )

    async def test_create_tool_config_rejects_invalid_http_api_config_with_400(self) -> None:
        request = CreateToolConfigRequest(
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            description="",
            connection_config={"base_url": "https://api.example.com", "unexpected": True},
        )

        with self.assertRaises(HTTPException) as ctx:
            await routes.create_tool_config(request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("unexpected", str(ctx.exception.detail))

    async def test_create_tool_config_rejects_blocked_nested_header_without_secret_echo(self) -> None:
        request = CreateToolConfigRequest(
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            description="",
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "Host", "value": "tenant-secret"}],
            },
        )

        with self.assertRaises(HTTPException) as ctx:
            await routes.create_tool_config(request)

        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail)
        self.assertIn("request_headers", detail)
        self.assertNotIn("tenant-secret", detail)

    async def test_update_tool_config_merges_saved_http_api_secrets_before_validation(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "auth_mode": "bearer",
                "bearer_token": "stored-token",
            },
        )
        updated = original.model_copy(deep=True)

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)
        mock_repo.update_tool_config = mock.AsyncMock(return_value=updated)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            result = await routes.update_tool_config(
                "tool-http-api",
                UpdateToolConfigRequest(connection_config={"auth_mode": "bearer"}),
            )

        self.assertEqual(result.id, "tool-http-api")
        mock_repo.update_tool_config.assert_awaited_once()
        persisted_updates = mock_repo.update_tool_config.await_args.args[1]
        self.assertEqual(persisted_updates["connection_config"]["bearer_token"], "stored-token")

    async def test_update_tool_config_preserves_nested_saved_secret_values_on_unrelated_edits(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "token_prefix": "Bearer",
                "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_secret", "value": "client-secret", "secret": True},
                ],
                "request_body_format": "json",
                "request_body_fields": [
                    {"name": "tenant", "value": "north", "secret": False},
                    {"name": "client_secret", "value": "resource-secret", "secret": True},
                ],
            },
        )
        updated = original.model_copy(deep=True)
        updated.connection_config["token_prefix"] = "Token"

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)
        mock_repo.update_tool_config = mock.AsyncMock(return_value=updated)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            result = await routes.update_tool_config(
                "tool-http-api",
                UpdateToolConfigRequest(
                    connection_config={
                        "auth_mode": "token_exchange",
                        "token_prefix": "Token",
                        "request_headers": [{"name": "X-Tenant", "value": ""}],
                        "token_request_headers": [{"name": "X-Token-Key", "value": ""}],
                        "token_request_fields": [
                            {"name": "grant_type", "value": "client_credentials", "secret": False},
                            {"name": "client_secret", "value": "", "secret": True},
                        ],
                        "request_body_format": "json",
                        "request_body_fields": [
                            {"name": "tenant", "value": "north", "secret": False},
                            {"name": "client_secret", "value": "", "secret": True},
                        ],
                    }
                ),
            )

        self.assertEqual(result.connection_config["token_prefix"], "Token")
        persisted_updates = mock_repo.update_tool_config.await_args.args[1]
        self.assertEqual(persisted_updates["connection_config"]["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(persisted_updates["connection_config"]["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(persisted_updates["connection_config"]["token_request_fields"][1]["value"], "client-secret")
        self.assertEqual(persisted_updates["connection_config"]["request_body_fields"][1]["value"], "resource-secret")

    async def test_update_tool_config_rejects_duplicate_nested_header_without_secret_echo(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
            },
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)

        with self.assertRaises(HTTPException) as ctx:
            with mock.patch("ragtime.indexer.routes.repository", mock_repo):
                await routes.update_tool_config(
                    "tool-http-api",
                    UpdateToolConfigRequest(
                        connection_config={
                            "auth_mode": "token_exchange",
                            "request_headers": [
                                {"name": "X-Tenant", "value": "tenant-secret"},
                                {"name": "x-tenant", "value": "other-secret"},
                            ],
                        }
                    ),
                )

        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail)
        self.assertIn("duplicate header name", detail.lower())
        self.assertNotIn("tenant-secret", detail)
        self.assertNotIn("other-secret", detail)
        mock_repo.update_tool_config.assert_not_called()

    async def test_update_tool_config_normalizes_legacy_non_secret_blank_secret_row(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "token_request_fields": [{"name": "client_secret", "value": "client-secret", "secret": True}],
            },
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)
        mock_repo.update_tool_config = mock.AsyncMock(return_value=original)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            await routes.update_tool_config(
                "tool-http-api",
                UpdateToolConfigRequest(
                    connection_config={
                        "auth_mode": "token_exchange",
                        "token_request_fields": [{"name": "client_secret", "value": "", "secret": False}],
                    }
                ),
            )

        persisted = mock_repo.update_tool_config.await_args.args[1]["connection_config"]
        self.assertEqual(persisted["token_request_fields"][0]["value"], "client-secret")
        self.assertTrue(persisted["token_request_fields"][0]["secret"])

    async def test_update_tool_config_allows_explicit_empty_http_api_secret_clear(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "auth_mode": "bearer",
                "bearer_token": "stored-token",
            },
        )
        updated = original.model_copy(update={"connection_config": {"base_url": "https://api.example.com", "auth_mode": "bearer", "bearer_token": ""}})

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)
        mock_repo.update_tool_config = mock.AsyncMock(return_value=updated)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            await routes.update_tool_config(
                "tool-http-api",
                UpdateToolConfigRequest(connection_config={"auth_mode": "bearer", "bearer_token": ""}),
            )

        persisted_updates = mock_repo.update_tool_config.await_args.args[1]
        self.assertEqual(persisted_updates["connection_config"]["bearer_token"], "")

    async def test_update_tool_config_returns_400_for_invalid_http_api_update(self) -> None:
        original = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "auth_mode": "bearer",
                "bearer_token": "stored-token",
            },
        )

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)

        with self.assertRaises(HTTPException) as ctx:
            with mock.patch("ragtime.indexer.routes.repository", mock_repo):
                await routes.update_tool_config(
                    "tool-http-api",
                    UpdateToolConfigRequest(connection_config={"auth_mode": "invalid-mode"}),
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("invalid http api configuration", str(ctx.exception.detail).lower())
        mock_repo.update_tool_config.assert_not_called()

    async def test_unsaved_http_api_test_connection_uses_config_only_validation_for_static_modes(self) -> None:
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(
                return_value=HttpApiValidationResult(
                    success=True,
                    message="Configuration is valid - no live request was sent.",
                    details={"auth_mode": "bearer"},
                )
            )
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "bearer",
                        "bearer_token": "secret-token",
                    },
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Configuration is valid - no live request was sent.")
        broker.validate_configuration.assert_awaited_once_with(
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.BEARER,
                bearer_token="secret-token",
            ),
            perform_login=False,
        )

    async def test_unsaved_http_api_test_connection_uses_connectivity_for_headers_mode(self) -> None:
        broker = SimpleNamespace(
            validate_connectivity=mock.AsyncMock(
                return_value=HttpApiValidationResult(
                    success=True,
                    message="Connectivity succeeded with HTTP 401.",
                    details={"status": 401, "auth_tested": False},
                )
            ),
            validate_configuration=mock.AsyncMock(),
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "headers",
                        "request_headers": [{"name": "X-Api-Key", "value": "secret-key"}],
                    },
                )
            )

        self.assertTrue(result.success)
        broker.validate_connectivity.assert_awaited_once_with(
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.HEADERS,
                request_headers=[HttpApiConfiguredHeader(name="X-Api-Key", value="secret-key")],
            )
        )
        broker.validate_configuration.assert_not_awaited()

    async def test_http_api_test_connection_returns_fixed_configuration_error_safely(self) -> None:
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(
                side_effect=routes.HttpApiConfigurationError("Configured request body format does not match the operation body format")
            )
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "bearer",
                        "bearer_token": "planted-secret-token",
                    },
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Configured request body format does not match the operation body format")
        self.assertIsNone(result.details)
        self.assertNotIn("planted-secret-token", json.dumps(result.model_dump()))

    async def test_headers_connectivity_failure_propagates_safe_failure_result(self) -> None:
        broker = SimpleNamespace(
            validate_connectivity=mock.AsyncMock(
                return_value=HttpApiValidationResult(
                    success=False,
                    message="Connectivity failed: ConnectError",
                    details=None,
                )
            ),
            validate_configuration=mock.AsyncMock(),
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "headers",
                        "request_headers": [{"name": "X-Api-Key", "value": "network-secret"}],
                    },
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Connectivity failed: ConnectError")
        self.assertIsNone(result.details)
        self.assertNotIn("network-secret", json.dumps(result.model_dump()))
        broker.validate_configuration.assert_not_awaited()

    def test_http_api_configuration_errors_preserve_fixed_structural_messages(self) -> None:
        error = routes.HttpApiConfigurationError("Configured request body format does not match the operation body format")

        self.assertEqual(
            routes._sanitize_http_api_error_message(error, fallback="fallback"),
            "Configured request body format does not match the operation body format",
        )

    def test_http_api_error_sanitizer_redacts_secret_bearing_messages(self) -> None:
        error = ValueError("Configured request body format does not match secret-token")

        self.assertEqual(routes._sanitize_http_api_error_message(error, fallback="fallback"), "fallback")

    async def test_unsaved_http_api_test_connection_performs_login_only_for_login_exchange(self) -> None:
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(return_value=HttpApiValidationResult(success=True, message="Login exchange succeeded - token received."))
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "login_exchange",
                        "login_path": "/auth/token",
                        "login_username": "demo",
                        "login_password": "secret",
                    },
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(broker.validate_configuration.await_args.kwargs["perform_login"])

    async def test_unsaved_http_api_test_connection_performs_login_for_token_exchange_without_leaking_secrets(self) -> None:
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(
                return_value=HttpApiValidationResult(
                    success=True,
                    message="Token exchange succeeded - token received.",
                    details={"auth_mode": "token_exchange", "token_response_path": "access_token"},
                )
            )
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.test_tool_connection(
                ToolTestRequest(
                    tool_type=ToolType.HTTP_API,
                    connection_config={
                        "base_url": "https://api.example.com",
                        "auth_mode": "token_exchange",
                        "login_path": "/oauth/token",
                        "token_request_fields": [
                            {"name": "client_secret", "value": "dummy-client-secret", "secret": True},
                        ],
                        "token_request_headers": [{"name": "X-Key", "value": "dummy-header-secret"}],
                    },
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Token exchange succeeded - token received.")
        self.assertEqual(result.details, {"auth_mode": "token_exchange", "token_response_path": "access_token"})
        self.assertTrue(broker.validate_configuration.await_args.kwargs["perform_login"])
        dumped = json.dumps(result.model_dump())
        self.assertNotIn("dummy-client-secret", dumped)
        self.assertNotIn("dummy-header-secret", dumped)

    async def test_saved_http_api_test_connection_uses_persisted_secret_values(self) -> None:
        config = ToolConfig(
            id="tool-http-api",
            name="Demo API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "auth_mode": "login_exchange",
                "login_path": "/auth/token",
                "login_username": "demo",
                "login_password": "stored-secret",
            },
        )
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(return_value=HttpApiValidationResult(success=True, message="Login exchange succeeded - token received."))
        )

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=config)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch.object(routes, "http_api_broker", broker),
            mock.patch(
                "ragtime.indexer.routes.tool_health_monitor.record_tool_test_result", mock.AsyncMock(return_value=SimpleNamespace(changed_tool_ids=set()))
            ),
        ):
            result = await routes.test_saved_tool_connection("tool-http-api")

        self.assertTrue(result.success)
        validated_config = broker.validate_configuration.await_args.args[0]
        self.assertEqual(validated_config.login_password, "stored-secret")

    async def test_http_api_heartbeat_uses_non_login_validation(self) -> None:
        broker = SimpleNamespace(
            validate_configuration=mock.AsyncMock(
                return_value=HttpApiValidationResult(
                    success=True,
                    message="Configuration is valid - no live request was sent.",
                )
            )
        )

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes._heartbeat_check(
                ToolType.HTTP_API,
                {
                    "base_url": "https://api.example.com",
                    "auth_mode": "login_exchange",
                    "login_path": "/auth/token",
                    "login_username": "demo",
                    "login_password": "secret",
                },
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Configuration is valid - no live request was sent.")
        self.assertFalse(broker.validate_configuration.await_args.kwargs["perform_login"])

    async def test_normalize_http_api_openapi_document_from_spec_url_fetches_then_normalizes(self) -> None:
        spec = json.dumps(
            {
                "openapi": "3.1.0",
                "info": {"title": "Demo", "version": "1.0"},
                "paths": {"/items": {"get": {"operationId": "listItems"}}},
            }
        )
        request = routes.HttpApiOpenApiNormalizeRequest.model_validate({"spec_url": "https://api.example.com/openapi.json"})
        broker = SimpleNamespace(fetch_openapi_document=mock.AsyncMock(return_value=spec))

        with mock.patch.object(routes, "http_api_broker", broker):
            result = await routes.normalize_http_api_openapi(request)

        broker.fetch_openapi_document.assert_awaited_once_with("https://api.example.com/openapi.json")
        self.assertEqual(result.openapi_source_url, "https://api.example.com/openapi.json")
        self.assertEqual(result.openapi_source_name, "")
        self.assertTrue(result.openapi_source_hash)
        self.assertEqual(result.openapi_catalog.title, "Demo")
        self.assertEqual(result.openapi_catalog.operations[0].path, "/items")

    def test_normalize_http_api_openapi_request_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            routes.HttpApiOpenApiNormalizeRequest.model_validate(
                {
                    "spec_url": "https://api.example.com/openapi.json",
                    "url": "https://api.example.com/openapi.json",
                }
            )

    async def test_normalize_http_api_openapi_document_accepts_inline_document(self) -> None:
        request = routes.HttpApiOpenApiNormalizeRequest(
            document='{"openapi":"3.1.0","info":{"title":"Inline","version":"1.0"},"paths":{}}',
            document_name="inline.json",
        )

        result = await routes.normalize_http_api_openapi(request)

        self.assertEqual(result.openapi_source_url, "")
        self.assertEqual(result.openapi_source_name, "inline.json")
        self.assertTrue(result.openapi_source_hash)
        self.assertEqual(result.openapi_catalog.title, "Inline")

    async def test_normalize_http_api_openapi_document_requires_exactly_one_source(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await routes.normalize_http_api_openapi(
                routes.HttpApiOpenApiNormalizeRequest(
                    spec_url="https://api.example.com/openapi.json",
                    document="{}",
                )
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("exactly one", str(ctx.exception.detail).lower())

    async def test_normalize_http_api_openapi_document_sanitizes_fetch_errors(self) -> None:
        request = routes.HttpApiOpenApiNormalizeRequest(spec_url="https://api.example.com/openapi.json")
        broker = SimpleNamespace(fetch_openapi_document=mock.AsyncMock(side_effect=ValueError("token=secret raw_openapi_document=boom")))

        with mock.patch.object(routes, "http_api_broker", broker):
            with self.assertRaises(HTTPException) as ctx:
                await routes.normalize_http_api_openapi(request)

        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail).lower()
        self.assertNotIn("secret", detail)
        self.assertNotIn("raw_openapi_document", detail)


if __name__ == "__main__":
    unittest.main()
