import asyncio
import unittest
from unittest import mock

import httpx

import ragtime.core.omlx as omlx
from ragtime.rag.components import RAGComponents


class OmlxContextResolutionTests(unittest.TestCase):
    def test_resolves_authenticated_omlx_context_limit(self) -> None:
        expected_key = "test-omlx-key"
        base_url = "https://omlx.example.test"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, httpx.URL(f"{base_url}/v1/models/status"))
            self.assertEqual(request.headers.get("Authorization"), f"Bearer {expected_key}")
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "id": "qwen3",
                            "max_context_window": 131072,
                        }
                    ]
                },
            )

        original_async_client = httpx.AsyncClient

        def mock_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        rag = RAGComponents()
        rag._app_settings = {
            "llm_omlx_base_url": base_url,
            "omlx_api_key": expected_key,
            "llm_max_tokens": 400000,
        }

        with mock.patch.object(omlx.httpx, "AsyncClient", side_effect=mock_async_client):
            context_limit = asyncio.run(rag._resolve_local_context_limit("omlx", "qwen3"))
            resolved_limit = asyncio.run(rag._resolve_llm_max_tokens("omlx", "qwen3"))

        self.assertEqual(context_limit, 131072)
        self.assertEqual(resolved_limit, 131072)
