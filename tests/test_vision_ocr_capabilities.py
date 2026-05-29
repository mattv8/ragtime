import unittest
from unittest.mock import patch

from ragtime.core import llama_cpp, lmstudio, model_limits, omlx, openrouter, vision_models
from ragtime.core.embedding_models import EmbeddingModelInfo
from ragtime.indexer import routes as indexer_routes


class VisionOcrCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        model_limits._provider_supports_image_input.clear()

    def tearDown(self) -> None:
        model_limits._provider_supports_image_input.clear()

    async def test_supports_image_input_uses_registered_metadata_only(self) -> None:
        with patch.object(model_limits, "_ensure_cache_loaded"):
            self.assertFalse(await model_limits.supports_image_input("gpt-4o"))

            model_limits.register_model_input_modalities("gpt-4o", ["text", "image"])

            self.assertTrue(await model_limits.supports_image_input("gpt-4o"))

    async def test_provider_capability_metadata_registers_modalities(self) -> None:
        row = {
            "id": "provider-vlm",
            "modalities": {"input": ["text", "image"]},
            "capabilities": ["tool_use"],
        }

        capabilities, *_ = indexer_routes._extract_provider_capability_metadata(row)

        self.assertIn("image_input", capabilities)
        with patch.object(model_limits, "_ensure_cache_loaded"):
            self.assertTrue(await model_limits.supports_image_input("provider-vlm"))

    async def test_lmstudio_vision_discovery_trusts_provider_metadata(self) -> None:
        async def fake_list_chat_models(base_url: str, api_key: str | None = None):
            _ = (base_url, api_key)
            return [
                lmstudio.LmStudioModelInfo(
                    id="native-vlm",
                    name="native-vlm",
                    model_type="vlm",
                    capabilities=["vision"],
                    context_limit=8192,
                    loaded=True,
                ),
                lmstudio.LmStudioModelInfo(
                    id="native-llm",
                    name="native-llm",
                    model_type="llm",
                ),
            ]

        async def fail_probe(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("metadata-confirmed VLM should not require probing")

        with (
            patch.object(lmstudio, "list_chat_models", fake_list_chat_models),
            patch.object(
                vision_models,
                "probe_openai_compatible_vision_capability",
                fail_probe,
            ),
        ):
            models = await vision_models.list_provider_vision_models("lmstudio", base_url="http://example.test")

        self.assertEqual([model.name for model in models], ["native-vlm"])
        self.assertEqual(models[0].provider, "lmstudio")
        self.assertTrue(models[0].loaded)

    async def test_llama_cpp_vision_discovery_does_not_live_probe_unknown_models(self) -> None:
        async def fake_list_chat_models(base_url: str):
            _ = base_url
            return [
                llama_cpp.LlamaCppModelInfo(
                    id="metadata-vlm",
                    name="metadata-vlm",
                    capabilities=["chat", "image_input"],
                    context_limit=4096,
                ),
                llama_cpp.LlamaCppModelInfo(
                    id="chat-only",
                    name="chat-only",
                    capabilities=["chat"],
                ),
            ]

        async def fail_probe(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("llama.cpp vision discovery should not warm models with live probes")

        with (
            patch.object(llama_cpp, "list_chat_models", fake_list_chat_models),
            patch.object(
                vision_models,
                "probe_openai_compatible_vision_capability",
                fail_probe,
            ),
        ):
            models = await vision_models.list_provider_vision_models("llama_cpp", base_url="http://example.test")

        self.assertEqual([model.name for model in models], ["metadata-vlm"])
        self.assertEqual(models[0].provider, "llama_cpp")
        self.assertEqual(models[0].context_limit, 4096)

    async def test_omlx_vision_discovery_does_not_live_probe_unknown_models(self) -> None:
        async def fake_list_status_models(base_url: str, api_key: str | None = None):
            _ = (base_url, api_key)
            return [
                omlx.OmlxModelInfo(
                    id="metadata-vlm",
                    name="metadata-vlm",
                    model_type="vlm",
                    context_limit=32768,
                    loaded=True,
                ),
                omlx.OmlxModelInfo(
                    id="unknown-model",
                    name="unknown-model",
                ),
            ]

        async def fail_list_models(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("status metadata should be enough for oMLX")

        async def fail_probe(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("oMLX vision discovery should not warm models with live probes")

        with (
            patch.object(omlx, "list_status_models", fake_list_status_models),
            patch.object(omlx, "list_models", fail_list_models),
            patch.object(
                vision_models,
                "probe_openai_compatible_vision_capability",
                fail_probe,
            ),
        ):
            models = await vision_models.list_provider_vision_models("omlx", base_url="http://example.test")

        self.assertEqual([model.name for model in models], ["metadata-vlm"])
        self.assertEqual(models[0].provider, "omlx")
        self.assertEqual(models[0].context_limit, 32768)
        self.assertTrue(models[0].loaded)

    async def test_openai_vision_discovery_uses_metadata_catalog_without_live_probe(self) -> None:
        async def fake_list_models_dev_vision_models(provider: str, available_model_ids: set[str] | None = None):
            self.assertEqual(provider, "openai")
            self.assertEqual(available_model_ids, {"gpt-4o", "gpt-4.1"})
            return [
                vision_models.VisionModelInfo(
                    name="gpt-4.1",
                    provider="openai",
                    capabilities=["image_input"],
                ),
                vision_models.VisionModelInfo(
                    name="gpt-4o",
                    provider="openai",
                    capabilities=["image_input"],
                ),
            ]

        async def fake_fetch_openai_available_model_ids(api_key: str | None):
            self.assertEqual(api_key, "sk-test")
            return {"gpt-4o", "gpt-4.1"}

        async def fail_probe(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("OpenAI vision discovery should not issue image requests")

        with (
            patch.object(
                vision_models,
                "_list_models_dev_vision_models",
                fake_list_models_dev_vision_models,
            ),
            patch.object(
                vision_models,
                "_fetch_openai_available_model_ids",
                fake_fetch_openai_available_model_ids,
            ),
            patch.object(
                vision_models,
                "probe_openai_compatible_vision_capability",
                fail_probe,
            ),
        ):
            models = await vision_models.list_provider_vision_models("openai", api_key="sk-test")

        self.assertEqual([model.name for model in models], ["gpt-4.1", "gpt-4o"])

    async def test_openrouter_vision_discovery_uses_catalog_capabilities_without_live_probe(self) -> None:
        async def fake_list_models(api_key: str, timeout: float = 20.0):
            self.assertEqual(api_key, "sk-or-test")
            self.assertEqual(timeout, 20.0)
            return [
                {
                    "id": "provider/vision-model",
                    "name": "Vision Model",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "limit": {"context": 128000},
                },
                {
                    "id": "provider/text-model",
                    "name": "Text Model",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                },
                {
                    "id": "provider/image-generator",
                    "name": "Image Generator",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["image"],
                    },
                },
            ]

        async def fail_probe(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("OpenRouter vision discovery should not issue image requests")

        with (
            patch.object(openrouter, "list_models", fake_list_models),
            patch.object(
                vision_models,
                "probe_openai_compatible_vision_capability",
                fail_probe,
            ),
        ):
            models = await vision_models.list_provider_vision_models("openrouter", api_key="sk-or-test")

        self.assertEqual([model.name for model in models], ["provider/vision-model"])
        self.assertEqual(models[0].provider, "openrouter")
        self.assertEqual(models[0].context_limit, 128000)
        self.assertIn("image_input", models[0].capabilities or [])

    async def test_openrouter_embedding_discovery_uses_catalog_capabilities(self) -> None:
        async def fake_list_embedding_models(api_key: str, timeout: float = 20.0):
            self.assertEqual(api_key, "sk-or-test")
            self.assertEqual(timeout, 15.0)
            return [
                {
                    "id": "openai/text-embedding-3-small",
                    "name": "Text Embedding 3 Small",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["embeddings"],
                    },
                },
                {
                    "id": "provider/embedding-endpoint-model",
                    "name": "Endpoint Embedding Model",
                    "supported_endpoints": ["/embeddings"],
                    "limit": {"context": 4096, "output": 1024},
                },
                {
                    "id": "provider/chat-model",
                    "name": "Chat Model",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                },
            ]

        async def fake_get_embedding_models():
            return {
                "text-embedding-3-small": EmbeddingModelInfo(
                    id="text-embedding-3-small",
                    provider="openai",
                    max_input_tokens=8191,
                    output_vector_size=1536,
                )
            }

        with (
            patch.object(openrouter, "list_embedding_models", fake_list_embedding_models),
            patch.object(indexer_routes, "get_embedding_models", fake_get_embedding_models),
        ):
            response = await indexer_routes._fetch_openrouter_embedding_models("sk-or-test")

        self.assertTrue(response.success)
        self.assertEqual(
            [model.id for model in response.models],
            ["provider/embedding-endpoint-model", "openai/text-embedding-3-small"],
        )
        self.assertEqual(response.models[0].dimensions, 1024)
        self.assertEqual(response.models[1].context_limit, 8191)

    def test_models_dev_vision_catalog_uses_modalities_metadata(self) -> None:
        payload = {
            "openai": {
                "models": {
                    "text-only": {
                        "id": "text-only",
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                    "image-model": {
                        "id": "image-model",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "limit": {"context": 128000},
                    },
                    "image-output-only": {
                        "id": "image-output-only",
                        "modalities": {"input": ["text"], "output": ["image"]},
                    },
                }
            }
        }

        models = vision_models._models_dev_vision_models_from_payload(
            payload,
            "openai",
            available_model_ids={"image-model", "text-only", "image-output-only"},
        )

        self.assertEqual([model.name for model in models], ["image-model"])
        self.assertEqual(models[0].capabilities, ["image_input"])
        self.assertEqual(models[0].context_limit, 128000)


if __name__ == "__main__":
    unittest.main()
