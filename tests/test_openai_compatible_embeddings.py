import asyncio
import unittest
from unittest import mock

import httpx

from ragtime.core import openai_compatible


class OpenAICompatibleEmbeddingDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_returns_empty_when_no_embedding_candidates_are_declared(self) -> None:
        with mock.patch(
            "ragtime.core.openai_compatible._fetch_live_models",
            new=mock.AsyncMock(return_value=[{"id": "chat-only", "type": "chat"}]),
        ):
            models = await openai_compatible.list_embedding_models("https://compatible.test/root")

        self.assertEqual(models, [])

    async def test_discovery_only_probes_explicit_embedding_metadata(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.url.path, request.headers.get("Authorization"), request.content))
            if request.url.path == "/api/root/models":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "chat-only", "type": "chat"},
                            {"id": "embed-only", "supported_endpoints": ["/embeddings"]},
                        ]
                    },
                    request=request,
                )
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}, request=request)

        original = httpx.AsyncClient
        with mock.patch("ragtime.core.openai_compatible.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
            models = await openai_compatible.list_embedding_models("https://compatible.test/api/root", "key")

        self.assertEqual([(model.id, model.dimensions) for model in models], [("embed-only", 3)])
        self.assertEqual([path for path, *_ in requests], ["/api/root/models", "/api/root/embeddings"])
        self.assertEqual(requests[-1][1], "Bearer key")

    async def test_explicit_model_is_probed_when_models_is_unsupported_and_keyless_has_no_auth_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/root/models":
                return httpx.Response(404, request=request)
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}, request=request)

        original = httpx.AsyncClient
        with mock.patch("ragtime.core.openai_compatible.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
            models = await openai_compatible.list_embedding_models("https://compatible.test/root", selected_model="manually-typed")

        self.assertEqual([(model.id, model.dimensions) for model in models], [("manually-typed", 2)])

    async def test_probe_rejects_malformed_vectors_and_redacts_upstream_auth_error(self) -> None:
        original = httpx.AsyncClient
        for vector in ([True, 0.2], ["0.1"], [[0.1]], [float("inf")], [None]):
            with (
                self.subTest(vector=vector),
                mock.patch(
                    "ragtime.core.openai_compatible.httpx.AsyncClient",
                    lambda **kwargs: original(
                        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"embedding": vector}]}, request=request)),
                        **kwargs,
                    ),
                ),
            ):
                with self.assertRaisesRegex(openai_compatible.CompatibleProviderError, "invalid embedding response"):
                    await openai_compatible.probe_embedding_dimension("https://compatible.test/root", "embed")

        with mock.patch(
            "ragtime.core.openai_compatible.httpx.AsyncClient",
            lambda **kwargs: original(
                transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret upstream detail", request=request)), **kwargs
            ),
        ):
            with self.assertRaises(openai_compatible.CompatibleProviderError) as error:
                await openai_compatible.probe_embedding_dimension("https://compatible.test/root", "embed", "secret-key")
        self.assertEqual(error.exception.code, "authentication")
        self.assertNotIn("secret", str(error.exception))

    async def test_discovery_deduplicates_and_surfaces_all_probe_failures(self) -> None:
        original = httpx.AsyncClient
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/root/models":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "embed", "type": "embedding"},
                            {"id": "embed", "supported_endpoints": ["/embeddings"]},
                        ]
                    },
                    request=request,
                )
            return httpx.Response(429, text="secret rate detail", request=request)

        with mock.patch("ragtime.core.openai_compatible.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)):
            with self.assertRaises(openai_compatible.CompatibleProviderError) as error:
                await openai_compatible.list_embedding_models("https://compatible.test/root")
        self.assertEqual(error.exception.code, "rate_limited")
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual([request.url.path for request in requests], ["/root/models", "/root/embeddings"])

    async def test_discovery_caps_concurrent_probe_candidates(self) -> None:
        rows = [{"id": f"embed-{index}", "type": "embedding"} for index in range(25)]
        probes = []

        async def probe(_base_url: str, model_id: str, _api_key: str = "") -> int:
            probes.append(model_id)
            return 3

        with (
            mock.patch("ragtime.core.openai_compatible._fetch_live_models", new=mock.AsyncMock(return_value=rows)),
            mock.patch("ragtime.core.openai_compatible.probe_embedding_dimension", new=probe),
        ):
            models = await openai_compatible.list_embedding_models("https://compatible.test/root")

        self.assertEqual(len(models), 20)
        self.assertEqual(probes, [f"embed-{index}" for index in range(20)])

    async def test_discovery_returns_completed_alternatives_when_selected_and_pending_probes_fail(self) -> None:
        never = asyncio.Event()
        cancelled = asyncio.Event()

        async def probe(_base_url: str, model_id: str, _api_key: str = "") -> int:
            if model_id == "selected":
                raise openai_compatible.CompatibleProviderError("unavailable", "safe")
            if model_id == "alternative":
                return 3
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("pending probe unexpectedly completed")

        rows = [
            {"id": "selected", "type": "embedding"},
            {"id": "alternative", "type": "embedding"},
            {"id": "pending", "type": "embedding"},
        ]
        with (
            mock.patch("ragtime.core.openai_compatible._fetch_live_models", new=mock.AsyncMock(return_value=rows)),
            mock.patch("ragtime.core.openai_compatible.probe_embedding_dimension", new=probe),
            mock.patch("ragtime.core.openai_compatible._EMBEDDING_PROBE_CONCURRENCY", 3),
            mock.patch("ragtime.core.openai_compatible._EMBEDDING_DISCOVERY_TIMEOUT_SECONDS", 0.01),
        ):
            models = await openai_compatible.list_embedding_models("https://compatible.test/root", selected_model="selected")

        self.assertEqual([(model.id, model.dimensions) for model in models], [("alternative", 3)])
        self.assertTrue(cancelled.is_set())

    async def test_discovery_cancellation_cancels_and_awaits_probe_tasks(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def probe(_base_url: str, _model_id: str, _api_key: str = "") -> int:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("probe unexpectedly completed")

        with (
            mock.patch(
                "ragtime.core.openai_compatible._fetch_live_models",
                new=mock.AsyncMock(return_value=[{"id": "embed", "type": "embedding"}]),
            ),
            mock.patch("ragtime.core.openai_compatible.probe_embedding_dimension", new=probe),
        ):
            task = asyncio.create_task(openai_compatible.list_embedding_models("https://compatible.test/root"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(cancelled.is_set())
