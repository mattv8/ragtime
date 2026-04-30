import unittest

from ragtime.core import llama_cpp


class LlamaCppProviderTests(unittest.TestCase):
    def test_extract_props_context_prefers_default_generation_settings(self) -> None:
        payload = {
            "n_ctx": 2048,
            "default_generation_settings": {"n_ctx": 16384},
        }

        self.assertEqual(llama_cpp.extract_props_context(payload), 16384)

    def test_extract_model_context_uses_meta_n_ctx_train(self) -> None:
        row = {"id": "chat-model", "meta": {"n_ctx_train": 32768}}

        self.assertEqual(llama_cpp.extract_model_context(row), 32768)

    def test_extract_model_context_falls_back_to_top_level_context(self) -> None:
        row = {"id": "chat-model", "context_length": "8192"}

        self.assertEqual(llama_cpp.extract_model_context(row), 8192)

    def test_extract_embedding_dimension_from_openai_payload(self) -> None:
        payload = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}

        self.assertEqual(llama_cpp.extract_embedding_dimension(payload), 4)

    def test_normalize_base_url_removes_trailing_slash(self) -> None:
        self.assertEqual(
            llama_cpp.normalize_base_url("http://localhost:8080/"),
            "http://localhost:8080",
        )


if __name__ == "__main__":
    unittest.main()
