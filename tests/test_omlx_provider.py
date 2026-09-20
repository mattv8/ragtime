import unittest
from contextlib import contextmanager
from typing import Iterator, Optional
from unittest.mock import patch

import httpx

from ragtime.core import omlx
from ragtime.core.model_providers import (
    EMBEDDING_PROVIDER_NAMES,
    LLM_PROVIDER_NAMES,
    LOCAL_EMBEDDING_PROVIDER_NAMES,
    LOCAL_LLM_PROVIDER_NAMES,
    get_provider,
)


@contextmanager
def patched_successful_omlx_discovery() -> Iterator[None]:
    async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
        _ = (base_url, api_key)
        return [
            omlx.OmlxModelInfo(id="chat-a", name="chat-a", model_type="llm"),
            omlx.OmlxModelInfo(id="embed-a", name="embed-a", model_type="embedding"),
        ]

    async def fake_probe_embedding_dimension(
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
    ) -> Optional[int]:
        _ = (base_url, api_key)
        return 1024 if model == "embed-a" else None

    async def fake_probe_chat_capability(
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
    ) -> bool:
        _ = (base_url, model, api_key)
        return False

    with (
        patch.object(omlx, "list_status_models", fake_list_status_models),
        patch.object(omlx, "probe_embedding_dimension", fake_probe_embedding_dimension),
        patch.object(omlx, "probe_chat_capability", fake_probe_chat_capability),
    ):
        yield


class OmlxProviderTests(unittest.TestCase):
    def test_parse_openai_model_row(self) -> None:
        parsed = omlx._parse_model_row({"id": "qwen3-coder-next-8bit"})

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.id, "qwen3-coder-next-8bit")
        self.assertEqual(parsed.name, "qwen3-coder-next-8bit")
        self.assertEqual(parsed.supported_endpoints, ["/chat/completions"])

    def test_extract_embedding_dimension_from_openai_payload(self) -> None:
        payload = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}

        self.assertEqual(omlx.extract_embedding_dimension(payload), 4)

    def test_provider_registry_marks_omlx_openai_compatible(self) -> None:
        provider = get_provider("omlx")

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.label, "oMLX")
        self.assertTrue(provider.supports_llm)
        self.assertTrue(provider.supports_embeddings)
        self.assertTrue(provider.openai_compatible_chat)
        self.assertTrue(provider.openai_compatible_embeddings)
        self.assertIsNotNone(provider.llm_connection)
        self.assertIsNotNone(provider.embedding_connection)
        assert provider.llm_connection is not None
        assert provider.embedding_connection is not None
        self.assertEqual(provider.llm_connection.default_port, 8000)
        self.assertEqual(provider.embedding_connection.default_port, 8000)

    def test_provider_registry_includes_omlx_in_derived_provider_sets(self) -> None:
        self.assertIn("omlx", LLM_PROVIDER_NAMES)
        self.assertIn("omlx", EMBEDDING_PROVIDER_NAMES)
        self.assertIn("omlx", LOCAL_LLM_PROVIDER_NAMES)
        self.assertIn("omlx", LOCAL_EMBEDDING_PROVIDER_NAMES)


class OmlxContextMetadataTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_model_row_uses_positive_max_model_len(self) -> None:
        parsed = omlx._parse_model_row({"id": "gpt-oss", "max_model_len": 131072, "max_tokens": 400000})

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.context_limit, 131072)

    def test_parse_status_model_row_prefers_effective_context_override(self) -> None:
        parsed = omlx._parse_status_model_row(
            {
                "id": "qwen",
                "max_context_window": 262144,
                "max_model_len": 131072,
                "model_context_length": 1048576,
                "max_tokens": 400000,
            }
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.context_limit, 262144)

    def test_parse_status_model_row_uses_model_and_native_context_fallbacks(self) -> None:
        from_models = omlx._parse_status_model_row({"id": "from-models", "max_model_len": 131072})
        from_native = omlx._parse_status_model_row({"id": "from-native", "model_context_length": 1048576})

        self.assertIsNotNone(from_models)
        self.assertIsNotNone(from_native)
        assert from_models is not None
        assert from_native is not None
        self.assertEqual(from_models.context_limit, 131072)
        self.assertEqual(from_native.context_limit, 1048576)

    def test_parsers_ignore_max_tokens_and_invalid_context_values(self) -> None:
        only_generation_default = omlx._parse_status_model_row({"id": "default", "max_tokens": 400000})
        invalid_model = omlx._parse_model_row({"id": "invalid", "max_model_len": "zero"})
        invalid_status = omlx._parse_status_model_row(
            {
                "id": "invalid-status",
                "max_context_window": 0,
                "max_model_len": -1,
                "model_context_length": "not-a-number",
            }
        )

        self.assertIsNotNone(only_generation_default)
        self.assertIsNotNone(invalid_model)
        self.assertIsNotNone(invalid_status)
        assert only_generation_default is not None
        assert invalid_model is not None
        assert invalid_status is not None
        self.assertIsNone(only_generation_default.context_limit)
        self.assertIsNone(invalid_model.context_limit)
        self.assertIsNone(invalid_status.context_limit)

    async def test_get_model_context_length_prefers_matching_status_model(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="target", name="target", context_limit=262144)]

        async def fail_list_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            self.fail("models fallback should not run when status metadata is usable")

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fail_list_models),
        ):
            result = await omlx.get_model_context_length("target", "http://example.test")

        self.assertEqual(result, 262144)

    async def test_get_model_context_length_falls_back_to_matching_models_row(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="target", name="target", context_limit=None)]

        async def fake_list_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="target", name="target", context_limit=131072)]

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fake_list_models),
        ):
            result = await omlx.get_model_context_length("target", "http://example.test")

        self.assertEqual(result, 131072)

    async def test_get_model_context_length_returns_none_for_unknown_target(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="other", name="other", context_limit=262144)]

        async def fake_list_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="another", name="another", context_limit=131072)]

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fake_list_models),
        ):
            result = await omlx.get_model_context_length("target", "http://example.test")

        self.assertIsNone(result)


class OmlxEmbeddingDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        omlx._CAPABILITY_CACHE.clear()

    def tearDown(self) -> None:
        omlx._CAPABILITY_CACHE.clear()

    async def test_list_embedding_models_discovers_successfully_probed_models(
        self,
    ) -> None:
        with patched_successful_omlx_discovery():
            models = await omlx.list_embedding_models("http://example.test")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "embed-a")
        self.assertEqual(models[0].dimensions, 1024)
        self.assertEqual(models[0].supported_endpoints, ["/embeddings"])

    async def test_list_chat_models_excludes_embedding_only_models(self) -> None:
        with patched_successful_omlx_discovery():
            models = await omlx.list_chat_models("http://example.test")

        self.assertEqual([model.id for model in models], ["chat-a"])

    async def test_list_embedding_models_returns_empty_when_probes_fail(self) -> None:
        async def fake_list_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [
                omlx.OmlxModelInfo(id="chat-a", name="chat-a"),
                omlx.OmlxModelInfo(id="chat-b", name="chat-b"),
            ]

        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return []

        async def fake_probe_embedding_dimension(
            base_url: str,
            model: str,
            api_key: Optional[str] = None,
        ) -> Optional[int]:
            _ = (base_url, model, api_key)
            raise RuntimeError("model does not support embeddings")

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fake_list_models),
            patch.object(omlx, "probe_embedding_dimension", fake_probe_embedding_dimension),
        ):
            models = await omlx.list_embedding_models("http://example.test", selected_model="chat-a")

        self.assertEqual(models, [])

    async def test_list_embedding_models_keeps_successfully_probed_models(self) -> None:
        async def fake_list_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="embed-a", name="embed-a")]

        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return []

        async def fake_probe_embedding_dimension(
            base_url: str,
            model: str,
            api_key: Optional[str] = None,
        ) -> Optional[int]:
            _ = (base_url, model, api_key)
            return 1024

        async def fake_probe_chat_capability(
            base_url: str,
            model: str,
            api_key: Optional[str] = None,
        ) -> bool:
            _ = (base_url, model, api_key)
            return False

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fake_list_models),
            patch.object(omlx, "probe_embedding_dimension", fake_probe_embedding_dimension),
            patch.object(omlx, "probe_chat_capability", fake_probe_chat_capability),
        ):
            models = await omlx.list_embedding_models("http://example.test", selected_model="embed-a")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "embed-a")
        self.assertEqual(models[0].dimensions, 1024)


class OmlxContextMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_metadata_forwards_auth_and_only_accepts_context_window(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200, json={"models": [{"id": "chat-a", "max_context_window": 8192, "max_tokens": 400000}, {"id": "output-only", "max_tokens": 400000}]}
            )

        real_async_client = httpx.AsyncClient
        with patch(
            "ragtime.core.omlx.httpx.AsyncClient",
            new=lambda *args, **kwargs: real_async_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout")),
        ):
            models = await omlx.list_status_models("http://omlx.test/", api_key="test-key")

        self.assertEqual(str(captured[0].url), "http://omlx.test/v1/models/status")
        self.assertEqual(captured[0].headers.get("Authorization"), "Bearer test-key")
        self.assertEqual([(model.id, model.context_limit) for model in models], [("chat-a", 8192), ("output-only", None)])

    async def test_get_context_length_selects_only_the_requested_model(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None) -> list[omlx.OmlxModelInfo]:
            _ = (base_url, api_key)
            return [
                omlx.OmlxModelInfo(id="other", name="other", context_limit=4096),
                omlx.OmlxModelInfo(id="selected", name="selected", context_limit=8192),
            ]

        with patch.object(omlx, "list_status_models", fake_list_status_models):
            context_limit = await omlx.get_model_context_length("selected", "http://omlx.test", api_key="test-key")

        self.assertEqual(context_limit, 8192)

    async def test_get_context_length_returns_none_for_unknown_or_invalid_metadata(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None) -> list[omlx.OmlxModelInfo]:
            _ = (base_url, api_key)
            return [omlx.OmlxModelInfo(id="invalid", name="invalid", context_limit=None)]

        with patch.object(omlx, "list_status_models", fake_list_status_models):
            unknown = await omlx.get_model_context_length("missing", "http://omlx.test")
            invalid = await omlx.get_model_context_length("invalid", "http://omlx.test")

        self.assertIsNone(unknown)
        self.assertIsNone(invalid)


if __name__ == "__main__":
    unittest.main()
