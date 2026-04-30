import unittest

from ragtime.core import lmstudio


class LmStudioProviderTests(unittest.TestCase):
    def test_parse_api_v1_llm_row_captures_metadata(self) -> None:
        row = {
            "type": "vlm",
            "key": "gemma-4-31b-it-mlx",
            "display_name": "Gemma 4 31B",
            "architecture": "gemma4",
            "quantization": {"name": "8bit"},
            "format": "mlx",
            "loaded_instances": [{"identifier": "gemma-loaded"}],
            "max_context_length": 262144,
            "capabilities": {"trained_for_tool_use": True, "vision": True},
        }

        parsed = lmstudio.parse_model_row(row)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.id, "gemma-4-31b-it-mlx")
        self.assertEqual(parsed.name, "Gemma 4 31B")
        self.assertEqual(parsed.model_type, "vlm")
        self.assertEqual(parsed.context_limit, 262144)
        self.assertEqual(parsed.architecture, "gemma4")
        self.assertEqual(parsed.quantization, "8bit")
        self.assertEqual(parsed.format, "mlx")
        self.assertTrue(parsed.loaded)
        self.assertIn("trained_for_tool_use", parsed.capabilities or [])

    def test_parse_api_v0_embedding_row_captures_state(self) -> None:
        row = {
            "id": "text-embedding-nomic-embed-text-v1.5",
            "type": "embeddings",
            "arch": "nomic-bert",
            "compatibility_type": "gguf",
            "quantization": "Q4_K_M",
            "state": "not-loaded",
            "max_context_length": 2048,
            "capabilities": [],
        }

        parsed = lmstudio.parse_model_row(row)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.id, "text-embedding-nomic-embed-text-v1.5")
        self.assertEqual(parsed.model_type, "embeddings")
        self.assertEqual(parsed.context_limit, 2048)
        self.assertEqual(parsed.architecture, "nomic-bert")
        self.assertEqual(parsed.quantization, "Q4_K_M")
        self.assertEqual(parsed.format, "gguf")
        self.assertFalse(parsed.loaded)

    def test_model_type_filters(self) -> None:
        chat = lmstudio.parse_model_row({"key": "chat", "type": "llm"})
        vision = lmstudio.parse_model_row({"key": "vision", "type": "vlm"})
        embedding = lmstudio.parse_model_row({"key": "embed", "type": "embeddings"})

        assert chat is not None
        assert vision is not None
        assert embedding is not None
        self.assertIn(chat.model_type, lmstudio.CHAT_MODEL_TYPES)
        self.assertIn(vision.model_type, lmstudio.CHAT_MODEL_TYPES)
        self.assertIn(embedding.model_type, lmstudio.EMBEDDING_MODEL_TYPES)

    def test_extract_embedding_dimension_from_openai_payload(self) -> None:
        payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        self.assertEqual(lmstudio.extract_embedding_dimension(payload), 3)


if __name__ == "__main__":
    unittest.main()
