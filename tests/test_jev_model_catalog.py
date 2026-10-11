import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

from ragtime.content_protection import model_catalog, routes


def _mock_client(handler: httpx.AsyncByteStream | object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10.0)  # type: ignore[arg-type]


class JevModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    def test_detect_key_transport(self) -> None:
        self.assertEqual(model_catalog.detect_key_transport(" sk-or-key "), "openrouter")
        self.assertEqual(model_catalog.detect_key_transport("ts_key"), "typesafe")
        self.assertEqual(model_catalog.detect_key_transport("ts-key"), "typesafe")
        self.assertIsNone(model_catalog.detect_key_transport("other-key"))

    async def test_auto_uses_draft_key_and_rejects_unrecognized_key(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"data": []})

        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(handler)):
            result = await model_catalog.list_jev_models("auto", "sk-or-draft", {})
        self.assertTrue(result["success"])
        self.assertEqual(result["transport"], "openrouter")
        self.assertEqual(seen[0].url, httpx.URL("https://openrouter.ai/api/v1/models"))
        self.assertEqual(seen[0].headers["Authorization"], "Bearer sk-or-draft")

        result = await model_catalog.list_jev_models("auto", "unknown", {})
        self.assertEqual(result["message"], "Choose TypeSafe or OpenRouter for this key.")

    async def test_explicit_transport_rejects_contradictory_key_and_missing_key(self) -> None:
        result = await model_catalog.list_jev_models("typesafe", "sk-or-draft", {})
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "The API key does not match the selected provider.")

        result = await model_catalog.list_jev_models("auto", None, {})
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Enter or save a TypeSafe or OpenRouter API key first.")

    async def test_auto_prefers_saved_typesafe_key_and_explicit_transport_does_not_fallback(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"models": []})

        settings = {"typesafe_api_key": "typesafe-saved", "openrouter_api_key": "openrouter-saved"}
        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(handler)):
            result = await model_catalog.list_jev_models("auto", None, settings)
            no_fallback = await model_catalog.list_jev_models("openrouter", None, {"typesafe_api_key": "typesafe-saved"})

        self.assertTrue(result["success"])
        self.assertEqual(result["transport"], "typesafe")
        self.assertEqual(seen[0].headers["Authorization"], "Bearer typesafe-saved")
        self.assertFalse(no_fallback["success"])
        self.assertEqual(no_fallback["message"], "Enter or save a TypeSafe or OpenRouter API key first.")

    async def test_parses_vendor_catalogs_filters_models_and_deduplicates(self) -> None:
        responses = [
            {
                "models": [
                    {"name": "jev-latest", "description": "Current", "release_date": "2026-01-01"},
                    {"name": "jev-1.2", "description": "Version"},
                    {"name": "jev-1.2"},
                    {"name": "other"},
                ]
            },
            {
                "data": [
                    {"id": "typesafe/jev-latest"},
                    {"id": "typesafe/jev-2.0", "name": "Jev 2"},
                    {"id": "typesafe/jev-router"},
                    {"id": "typesafe/jev-2.0"},
                ]
            },
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=responses.pop(0))

        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(handler)):
            typesafe = await model_catalog.list_jev_models("typesafe", None, {"typesafe_api_key": "saved"})
            openrouter = await model_catalog.list_jev_models("openrouter", None, {"openrouter_api_key": "saved"})

        self.assertEqual([row["id"] for row in typesafe["models"]], ["jev-latest", "jev-1.2"])
        self.assertEqual([row["id"] for row in openrouter["models"]], ["jev-latest", "typesafe/jev-2.0"])
        self.assertEqual(typesafe["models"][0]["description"], "Current")
        self.assertEqual(typesafe["models"][0]["release_date"], "2026-01-01")

    async def test_provider_errors_do_not_leak_keys_or_provider_bodies(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="provider body sk-or-secret")

        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(handler)):
            result = await model_catalog.list_jev_models("openrouter", "sk-or-secret", {})

        self.assertEqual(result["message"], "The OpenRouter API key was rejected.")
        self.assertNotIn("sk-or-secret", str(result))
        self.assertNotIn("provider body", str(result))

    async def test_network_and_invalid_json_fail_safely(self) -> None:
        def network_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("network sk-or-secret", request=request)

        def invalid_json(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not JSON")

        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(network_error)):
            network_result = await model_catalog.list_jev_models("openrouter", "sk-or-secret", {})
        with mock.patch.object(model_catalog, "_make_http_client", side_effect=lambda timeout: _mock_client(invalid_json)):
            json_result = await model_catalog.list_jev_models("openrouter", "sk-or-secret", {})

        self.assertEqual(network_result["message"], "Could not load models from OpenRouter.")
        self.assertEqual(json_result["message"], "Could not load models from OpenRouter.")
        self.assertNotIn("sk-or-secret", str(network_result))


class JevModelCatalogRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_models_requires_admin_and_loads_settings(self) -> None:
        def reject() -> None:
            raise HTTPException(status_code=403, detail="Admin access required")

        self.app.dependency_overrides[routes.require_admin] = reject
        response = await self.client.post("/indexes/content-protection/models", json={"transport": "auto"})
        self.assertEqual(response.status_code, 403)

        self.app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin")
        with (
            mock.patch.object(routes.app_settings, "get_app_settings", new=mock.AsyncMock(return_value={})) as get_settings,
            mock.patch.object(routes, "list_jev_models", new=mock.AsyncMock(return_value={"success": True})) as list_models,
        ):
            response = await self.client.post("/indexes/content-protection/models", json={"transport": "auto", "api_key": "ts_draft"})

        self.assertEqual(response.status_code, 200)
        get_settings.assert_awaited_once()
        list_models.assert_awaited_once_with("auto", "ts_draft", {})


def test_models_rejects_a_too_long_key_without_logging_or_echoing_it(caplog: Any) -> None:
    async def request_models() -> httpx.Response:
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with mock.patch.object(routes.app_settings, "get_app_settings", new=mock.AsyncMock(return_value={})):
                return await client.post("/indexes/content-protection/models", json={"transport": "auto", "api_key": api_key})

    api_key = f"ts_{'x' * 598}"
    response = asyncio.run(request_models())

    assert response.status_code == 200
    assert response.json()["message"] == "API key is too long."
    assert api_key not in response.text
    assert all(api_key not in record.getMessage() for record in caplog.records)
