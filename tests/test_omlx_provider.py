import unittest
from typing import Optional
from unittest.mock import patch

from ragtime.core import omlx
from ragtime.core.model_providers import get_provider


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


class OmlxEmbeddingDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        omlx._CAPABILITY_CACHE.clear()

    def tearDown(self) -> None:
        omlx._CAPABILITY_CACHE.clear()

    async def test_list_embedding_models_discovers_successfully_probed_models(
        self,
    ) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [
                omlx.OmlxModelInfo(id="chat-a", name="chat-a", model_type="llm"),
                omlx.OmlxModelInfo(
                    id="embed-a", name="embed-a", model_type="embedding"
                ),
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
            patch.object(
                omlx, "probe_embedding_dimension", fake_probe_embedding_dimension
            ),
            patch.object(omlx, "probe_chat_capability", fake_probe_chat_capability),
        ):
            models = await omlx.list_embedding_models("http://example.test")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "embed-a")
        self.assertEqual(models[0].dimensions, 1024)
        self.assertEqual(models[0].supported_endpoints, ["/embeddings"])

    async def test_list_chat_models_excludes_embedding_only_models(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: Optional[str] = None):
            _ = (base_url, api_key)
            return [
                omlx.OmlxModelInfo(id="chat-a", name="chat-a", model_type="llm"),
                omlx.OmlxModelInfo(
                    id="embed-a", name="embed-a", model_type="embedding"
                ),
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
            patch.object(
                omlx, "probe_embedding_dimension", fake_probe_embedding_dimension
            ),
            patch.object(omlx, "probe_chat_capability", fake_probe_chat_capability),
        ):
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
            patch.object(
                omlx, "probe_embedding_dimension", fake_probe_embedding_dimension
            ),
        ):
            models = await omlx.list_embedding_models(
                "http://example.test", selected_model="chat-a"
            )

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
            patch.object(
                omlx, "probe_embedding_dimension", fake_probe_embedding_dimension
            ),
            patch.object(omlx, "probe_chat_capability", fake_probe_chat_capability),
        ):
            models = await omlx.list_embedding_models(
                "http://example.test", selected_model="embed-a"
            )

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "embed-a")
        self.assertEqual(models[0].dimensions, 1024)


if __name__ == "__main__":
    unittest.main()
