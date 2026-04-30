import unittest

from ragtime.indexer.routes import (
    AvailableModel,
    LLMModel,
    LLMModelsResponse,
    _assign_model_groups,
    _extract_version_parts,
    _group_models,
    _merge_llm_model_results,
    _parse_model_identifier,
)
from ragtime.api.routes import (
    _normalize_openapi_model_id,
    _normalize_runtime_model,
    _owned_by_from_openapi_model_id,
)


class ModelResolutionTests(unittest.TestCase):
    def test_parse_model_identifier_accepts_llama_cpp(self) -> None:
        self.assertEqual(
            _parse_model_identifier("llama_cpp::my-chat-model"),
            ("llama_cpp", "my-chat-model"),
        )

    def test_parse_model_identifier_accepts_lmstudio(self) -> None:
        self.assertEqual(
            _parse_model_identifier("lmstudio::gemma-4-31b-it-mlx"),
            ("lmstudio", "gemma-4-31b-it-mlx"),
        )

    def test_openapi_normalization_accepts_llama_cpp_token(self) -> None:
        self.assertEqual(
            _normalize_openapi_model_id("llama_cpp", "llama_cpp::my-chat-model"),
            "lc::my-chat-model",
        )
        self.assertEqual(
            _normalize_runtime_model("openai", "lc::my-chat-model"),
            "llama_cpp::my-chat-model",
        )
        self.assertEqual(
            _owned_by_from_openapi_model_id("openai", "lc::my-chat-model"),
            "llama_cpp",
        )

    def test_openapi_normalization_accepts_lmstudio_token(self) -> None:
        self.assertEqual(
            _normalize_openapi_model_id("lmstudio", "lmstudio::gemma-4-31b-it-mlx"),
            "ls::gemma-4-31b-it-mlx",
        )
        self.assertEqual(
            _normalize_runtime_model("openai", "ls::gemma-4-31b-it-mlx"),
            "lmstudio::gemma-4-31b-it-mlx",
        )
        self.assertEqual(
            _owned_by_from_openapi_model_id("openai", "ls::gemma-4-31b-it-mlx"),
            "lmstudio",
        )

    def test_extract_version_parts_prefers_model_id_detail(self) -> None:
        self.assertEqual(
            _extract_version_parts("claude-opus-4.7", "Claude Opus 4"),
            (4, 7),
        )

    def test_group_models_marks_opus_47_as_latest(self) -> None:
        models = [
            LLMModel(id="claude-opus-4", name="Claude Opus 4", created=100),
            LLMModel(id="claude-opus-4.7", name="Claude Opus 4", created=200),
        ]

        grouped = _group_models(models, "github_copilot")
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "claude-opus-4.7")
        self.assertTrue(all(model.group == "Claude Opus 4" for model in grouped))

    def test_group_models_marks_sonnet_46_as_latest(self) -> None:
        models = [
            LLMModel(
                id="anthropic/claude-sonnet-4",
                name="Claude Sonnet 4",
                created=100,
            ),
            LLMModel(
                id="anthropic/claude-sonnet-4.6",
                name="Claude Sonnet 4",
                created=200,
            ),
        ]

        grouped = _group_models(models, "github_copilot")
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "anthropic/claude-sonnet-4.6")
        self.assertTrue(all(model.group == "Claude Sonnet 4" for model in grouped))

    def test_group_models_dynamic_claude_family_is_grouped_by_major(self) -> None:
        models = [
            LLMModel(id="claude-mythos-5", name="Claude Mythos 5", created=100),
            LLMModel(id="claude-mythos-5.1", name="Claude Mythos 5.1", created=200),
        ]

        grouped = _group_models(models, "anthropic")
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "claude-mythos-5.1")
        self.assertTrue(all(model.group == "Claude Mythos 5" for model in grouped))

    def test_group_models_keeps_legacy_claude_groups(self) -> None:
        models = [
            LLMModel(id="claude-3-opus", name="Claude 3 Opus", created=100),
            LLMModel(id="claude-3.5-sonnet", name="Claude 3.5 Sonnet", created=200),
        ]

        grouped = _group_models(models, "anthropic")
        grouped_by_id = {model.id: model.group for model in grouped}

        self.assertEqual(grouped_by_id["claude-3-opus"], "Claude 3 Opus")
        self.assertEqual(grouped_by_id["claude-3.5-sonnet"], "Claude 3.5 Sonnet")

    def test_assign_model_groups_marks_newest_available_model(self) -> None:
        models = [
            AvailableModel(
                id="claude-opus-4",
                name="Claude Opus 4",
                provider="github_copilot",
                created=100,
            ),
            AvailableModel(
                id="claude-opus-4.7",
                name="Claude Opus 4",
                provider="github_copilot",
                created=200,
            ),
        ]

        grouped = _assign_model_groups(models)
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "claude-opus-4.7")

    def test_assign_model_groups_handles_dynamic_claude_families_for_copilot(
        self,
    ) -> None:
        models = [
            AvailableModel(
                id="anthropic/claude-mythos-5",
                name="Claude Mythos 5",
                provider="github_copilot",
                created=100,
            ),
            AvailableModel(
                id="anthropic/claude-mythos-5.1",
                name="Claude Mythos 5.1",
                provider="github_copilot",
                created=200,
            ),
        ]

        grouped = _assign_model_groups(models)
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "anthropic/claude-mythos-5.1")
        self.assertTrue(all(model.group == "Claude Mythos 5" for model in grouped))

    def test_group_models_dynamic_gpt_minor_family(self) -> None:
        models = [
            LLMModel(id="gpt-5.5", name="GPT-5.5", created=100),
            LLMModel(id="gpt-5.5-mini", name="GPT-5.5 Mini", created=200),
        ]

        grouped = _group_models(models, "openai")
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "gpt-5.5-mini")
        self.assertTrue(all(model.group == "GPT-5.5" for model in grouped))

    def test_group_models_dynamic_gpt_major_family(self) -> None:
        models = [
            LLMModel(id="gpt-6", name="GPT-6", created=100),
            LLMModel(id="gpt-6.0-mini", name="GPT-6.0 Mini", created=200),
        ]

        grouped = _group_models(models, "openai")
        groups = {model.id: model.group for model in grouped}

        self.assertEqual(groups["gpt-6"], "GPT-6")
        self.assertEqual(groups["gpt-6.0-mini"], "GPT-6.0")

    def test_group_models_dynamic_gemini_family(self) -> None:
        models = [
            LLMModel(id="google/gemini-3", name="Gemini 3", created=100),
            LLMModel(
                id="google/gemini-3.1-flash", name="Gemini 3.1 Flash", created=200
            ),
        ]

        grouped = _group_models(models, "github_copilot")
        latest = [model for model in grouped if model.is_latest]
        groups = {model.id: model.group for model in grouped}

        self.assertEqual(groups["google/gemini-3"], "Gemini 3")
        self.assertEqual(groups["google/gemini-3.1-flash"], "Gemini 3")
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "google/gemini-3.1-flash")

    def test_group_models_keeps_legacy_gemini_25_group(self) -> None:
        models = [
            LLMModel(id="google/gemini-2.5-pro", name="Gemini 2.5 Pro", created=100),
        ]

        grouped = _group_models(models, "github_copilot")

        self.assertEqual(grouped[0].group, "Gemini 2")

    def test_merge_prefers_primary_metadata_for_duplicates(self) -> None:
        primary = LLMModelsResponse(
            success=True,
            message="primary",
            models=[
                LLMModel(
                    id="claude-opus-4.7",
                    name="Claude Opus 4.7",
                    created=200,
                    max_output_tokens=32000,
                    capabilities=["reasoning"],
                )
            ],
            default_model="claude-opus-4.7",
        )
        extra = LLMModelsResponse(
            success=True,
            message="extra",
            models=[
                LLMModel(
                    id="claude-opus-4.7",
                    name="Claude Opus 4",
                    created=100,
                    max_output_tokens=16000,
                    capabilities=["text"],
                    supported_endpoints=["/chat/completions"],
                )
            ],
        )

        merged = _merge_llm_model_results(primary, extra)

        self.assertEqual(len(merged.models), 1)
        model = merged.models[0]
        self.assertEqual(model.name, "Claude Opus 4.7")
        self.assertEqual(model.created, 200)
        self.assertEqual(model.max_output_tokens, 32000)
        self.assertEqual(model.capabilities, ["reasoning"])
        self.assertEqual(model.supported_endpoints, ["/chat/completions"])
        self.assertEqual(merged.default_model, "claude-opus-4.7")


if __name__ == "__main__":
    unittest.main()
