import json
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from pydantic import ValidationError

from ragtime.http_api.models import HttpApiAuthMode, HttpApiConnectionConfig, HttpApiValidationResult
from ragtime.indexer import routes
from ragtime.indexer.models import CreateToolConfigRequest, ToolConfig, ToolTestRequest, ToolType, UpdateToolConfigRequest


class HttpApiRouteTests(unittest.IsolatedAsyncioTestCase):
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
        dumped = result.model_dump(mode="json")
        self.assertEqual(dumped["connection_config"]["request_headers"], [{"name": "X-Tenant", "value": ""}])
        self.assertEqual(dumped["connection_config"]["token_request_headers"], [{"name": "X-Token-Key", "value": ""}])
        self.assertEqual(
            dumped["connection_config"]["token_request_fields"],
            [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_id", "value": "client-id", "secret": False},
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
                    }
                ),
            )

        self.assertEqual(result.connection_config["token_prefix"], "Token")
        persisted_updates = mock_repo.update_tool_config.await_args.args[1]
        self.assertEqual(persisted_updates["connection_config"]["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(persisted_updates["connection_config"]["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(persisted_updates["connection_config"]["token_request_fields"][1]["value"], "client-secret")

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

    async def test_update_tool_config_rejects_secret_row_toggled_non_secret_blank(self) -> None:
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

        with self.assertRaises(HTTPException) as ctx:
            with mock.patch("ragtime.indexer.routes.repository", mock_repo):
                await routes.update_tool_config(
                    "tool-http-api",
                    UpdateToolConfigRequest(
                        connection_config={
                            "auth_mode": "token_exchange",
                            "token_request_fields": [{"name": "client_secret", "value": "", "secret": False}],
                        }
                    ),
                )

        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail)
        self.assertIn("token_request_fields", detail)
        self.assertNotIn("client-secret", detail)
        mock_repo.update_tool_config.assert_not_called()

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
