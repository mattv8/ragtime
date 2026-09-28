import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException
from prisma.models import User

from ragtime.indexer import routes
from ragtime.indexer.models import AppSettings


def _create_mock_user():
    """Create a mock User object for testing."""
    return SimpleNamespace(id="test-user", email="test@example.com", workspace_id="test-workspace")


class OpenAICompatibleRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_uses_saved_key_only_for_same_normalized_url(self) -> None:
        settings = SimpleNamespace(
            openai_compatible_base_url="https://proxy.example/v1",
            openai_compatible_api_key="saved-key",
        )
        fetch = mock.AsyncMock(return_value=[])

        with (
            mock.patch("ragtime.indexer.routes.repository.get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch("ragtime.indexer.routes.list_compatible_models", fetch),
        ):
            await routes.fetch_llm_models(
                routes.LLMModelsRequest(provider="openai_compatible", base_url="https://proxy.example/v1/"),
                cast(User, _create_mock_user()),
            )
            await routes.fetch_llm_models(
                routes.LLMModelsRequest(provider="openai_compatible", base_url="https://other.example/v1"),
                cast(User, _create_mock_user()),
            )
            await routes.fetch_llm_models(
                routes.LLMModelsRequest(provider="openai_compatible", base_url="https://proxy.example/v1", api_key=""),
                cast(User, _create_mock_user()),
            )

        self.assertEqual(fetch.await_args_list[0].kwargs["api_key"], "saved-key")
        self.assertEqual(fetch.await_args_list[1].kwargs["api_key"], "")
        self.assertEqual(fetch.await_args_list[2].kwargs["api_key"], "")

    async def test_aggregate_and_validation_use_saved_key_but_explicit_empty_does_not(self) -> None:
        settings = SimpleNamespace(
            openai_compatible_base_url="https://proxy.example/v1",
            openai_compatible_api_key="saved-key",
            openai_compatible_catalog_provider="",
            openai_compatible_model_limits={},
        )
        authorization_headers: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            authorization_headers.append(request.headers.get("Authorization"))
            if request.headers.get("Authorization") != "Bearer saved-key":
                return httpx.Response(401, request=request)
            return httpx.Response(200, json={"data": [{"id": "listed"}]}, request=request)

        original_client = httpx.AsyncClient

        def client_factory(**kwargs: Any) -> httpx.AsyncClient:
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with mock.patch("ragtime.core.openai_compatible.httpx.AsyncClient", client_factory):
            result = await routes._fetch_llm_models_for_provider("openai_compatible", settings=cast(AppSettings, settings))
            await routes._fetch_llm_models_for_provider("openai_compatible", settings=cast(AppSettings, settings), api_key="")
            await routes._validate_conversation_model_selection(
                provider="openai_compatible", model_id="listed", settings=cast(AppSettings, settings), force_refresh=True
            )

        assert result is not None
        self.assertTrue(result.success)
        self.assertEqual(authorization_headers, ["Bearer saved-key", None, "Bearer saved-key"])

    async def test_unconfigured_compatible_provider_has_400_validation_boundary(self) -> None:
        settings = SimpleNamespace(openai_compatible_base_url="")
        with self.assertRaises(HTTPException) as error:
            await routes._fetch_llm_models_for_provider("openai_compatible", settings=cast(AppSettings, settings), raise_on_unconfigured=True)
        self.assertEqual(error.exception.status_code, 400)

    def test_generic_catalog_model_preserves_unknown_limits_without_legacy_inference(self) -> None:
        generic = routes.LLMModel(
            id="gpt-4o",
            name="Generic GPT-4o",
            context_limit=None,
            max_output_tokens=None,
            context_limit_source=None,
            output_limit_source=None,
            tool_call_supported=False,
        )

        with mock.patch("ragtime.indexer.routes.get_context_limit", new=mock.AsyncMock()) as legacy_context:
            available = routes._generic_available_model(generic)
            routes._assign_model_groups([available])

        self.assertEqual(available.provider, "openai_compatible")
        self.assertIsNone(available.context_limit)
        self.assertIsNone(available.context_limit_source)
        self.assertFalse(available.tool_call_supported)
        self.assertEqual(available.supported_endpoints, ["/chat/completions"])
        legacy_context.assert_not_awaited()

    async def test_catalog_provider_route_is_admin_only_and_returns_service_records(self) -> None:
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin")
        try:
            with mock.patch(
                "ragtime.indexer.routes.list_catalog_providers",
                mock.AsyncMock(return_value=[{"id": "openrouter", "name": "OpenRouter", "api": "https://openrouter.ai"}]),
            ):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://ragtime.example") as client:
                    response = await client.get("/indexes/llm/model-catalog-providers")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["id"], "openrouter")
        finally:
            app.dependency_overrides.clear()

    async def test_catalog_provider_route_rejects_non_admin(self) -> None:
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.require_admin] = lambda: (_ for _ in ()).throw(HTTPException(status_code=403, detail="Admin access required"))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://ragtime.example") as client:
                response = await client.get("/indexes/llm/model-catalog-providers")
            self.assertEqual(response.status_code, 403)
        finally:
            app.dependency_overrides.clear()
