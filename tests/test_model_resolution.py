import unittest
from types import SimpleNamespace
from unittest import mock

import ragtime.indexer.routes as indexer_routes

from ragtime.core.model_limits import extract_openrouter_model_limits
from ragtime.core.model_providers import resolve_model_family_from_metadata
from ragtime.indexer.routes import (
    AvailableModel,
    LLMModel,
    LLMModelsResponse,
    _assign_model_groups,
    _extract_version_parts,
    _group_models,
    _identifier_in_allowed_models,
    _merge_llm_model_results,
    _normalize_conversation_model_provider,
    _parse_model_identifier,
)
from ragtime.api.routes import (
    _normalize_openapi_model_id,
    _normalize_runtime_model,
    _owned_by_from_openapi_model_id,
)
from ragtime.rag.components import RAGComponents


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

    def test_parse_model_identifier_accepts_omlx_from_registry(self) -> None:
        self.assertEqual(
            _parse_model_identifier("omlx::qwen3-coder-next-8bit"),
            ("omlx", "qwen3-coder-next-8bit"),
        )

    def test_parse_model_identifier_accepts_openrouter_from_registry(self) -> None:
        self.assertEqual(
            _parse_model_identifier("openrouter::anthropic/claude-sonnet-4.5"),
            ("openrouter", "anthropic/claude-sonnet-4.5"),
        )

    def test_parse_model_identifier_accepts_openrouter_short_alias(self) -> None:
        self.assertEqual(
            _parse_model_identifier("or::openai/gpt-4o"),
            ("openrouter", "openai/gpt-4o"),
        )

    def test_conversation_model_provider_accepts_registered_omlx(self) -> None:
        self.assertEqual(_normalize_conversation_model_provider("omlx"), "omlx")

    def test_conversation_model_provider_accepts_registered_openrouter(self) -> None:
        self.assertEqual(
            _normalize_conversation_model_provider("openrouter"), "openrouter"
        )

    def test_conversation_model_provider_rejects_unknown_provider(self) -> None:
        with self.assertRaises(indexer_routes.HTTPException) as raised:
            _normalize_conversation_model_provider("not_a_provider")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("omlx", str(raised.exception.detail))

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

    def test_openapi_normalization_accepts_omlx_token(self) -> None:
        self.assertEqual(
            _normalize_openapi_model_id("omlx", "omlx::qwen3-coder-next-8bit"),
            "om::qwen3-coder-next-8bit",
        )
        self.assertEqual(
            _normalize_runtime_model("openai", "om::qwen3-coder-next-8bit"),
            "omlx::qwen3-coder-next-8bit",
        )
        self.assertEqual(
            _owned_by_from_openapi_model_id("openai", "om::qwen3-coder-next-8bit"),
            "omlx",
        )

    def test_openapi_normalization_accepts_openrouter_token(self) -> None:
        self.assertEqual(
            _normalize_openapi_model_id(
                "openrouter", "openrouter::anthropic/claude-sonnet-4.5"
            ),
            "or::anthropic/claude-sonnet-4.5",
        )
        self.assertEqual(
            _normalize_runtime_model("openai", "or::openai/gpt-4o"),
            "openrouter::openai/gpt-4o",
        )
        self.assertEqual(
            _owned_by_from_openapi_model_id("openai", "or::openai/gpt-4o"),
            "openrouter",
        )

    def test_allowed_model_matching_accepts_provider_alias_and_publisher_prefix(
        self,
    ) -> None:
        self.assertTrue(
            _identifier_in_allowed_models(
                "github_copilot::openai/gpt-5.3-codex",
                ["openai::gpt-5.3-codex"],
            )
        )

    def test_allowed_model_matching_rejects_filtered_model(self) -> None:
        self.assertFalse(
            _identifier_in_allowed_models(
                "github_copilot::openai/gpt-5.3-codex",
                ["github_copilot::claude-sonnet-4.6"],
            )
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

    def test_group_models_openrouter_without_metadata_uses_default_group(self) -> None:
        models = [
            LLMModel(id="openai/gpt-5.5", name="GPT-5.5", created=100),
            LLMModel(id="openai/gpt-5.5-mini", name="GPT-5.5 Mini", created=200),
        ]

        grouped = _group_models(models, "openrouter")
        latest = [model for model in grouped if model.is_latest]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].id, "openai/gpt-5.5-mini")
        self.assertTrue(all(model.group == "Other Openrouter" for model in grouped))

    def test_openrouter_family_uses_structured_tokenizer_metadata(self) -> None:
        self.assertEqual(
            resolve_model_family_from_metadata(
                "openrouter",
                {"architecture": {"tokenizer": "Mistral"}},
            ),
            "Mistral",
        )
        self.assertEqual(
            resolve_model_family_from_metadata(
                "openrouter",
                {"architecture": {"tokenizer": "Qwen3"}},
            ),
            "Qwen",
        )

    def test_openrouter_group_models_preserves_structured_family(self) -> None:
        models = [
            LLMModel(
                id="mistralai/mistral-medium-3-5",
                name="Mistral Medium 3.5",
                group="Mistral",
                created=100,
            )
        ]

        grouped = _group_models(models, "openrouter")

        self.assertEqual(grouped[0].group, "Mistral")
        self.assertTrue(grouped[0].is_latest)

    def test_openrouter_limits_use_structured_api_fields(self) -> None:
        context_limit, output_limit = extract_openrouter_model_limits(
            {
                "context_length": 1_000_000,
                "top_provider": {
                    "context_length": 999_000,
                    "max_completion_tokens": 128_000,
                },
            }
        )

        self.assertEqual(context_limit, 1_000_000)
        self.assertEqual(output_limit, 128_000)

    def test_openrouter_limits_fall_back_to_top_provider_context(self) -> None:
        context_limit, output_limit = extract_openrouter_model_limits(
            {
                "top_provider": {
                    "context_length": 262_144,
                    "max_completion_tokens": 65_536,
                },
            }
        )

        self.assertEqual(context_limit, 262_144)
        self.assertEqual(output_limit, 65_536)

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


class ModelSendEligibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_validation_rejects_model_removed_from_chat_models(self) -> None:
        settings = SimpleNamespace(
            allowed_chat_models=["github_copilot::claude-sonnet-4.6"],
            llm_provider="github_copilot",
        )

        with (
            mock.patch.object(
                indexer_routes.repository,
                "get_settings",
                mock.AsyncMock(return_value=settings),
            ),
            mock.patch.object(
                indexer_routes,
                "_validate_conversation_model_selection",
                mock.AsyncMock(),
            ) as validate_live_model,
        ):
            with self.assertRaises(indexer_routes.HTTPException) as raised:
                await indexer_routes._validate_conversation_model_before_send(
                    "github_copilot::openai/gpt-5.3-codex"
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("Select another model", str(raised.exception.detail))
        validate_live_model.assert_not_awaited()

    async def test_send_validation_allows_allowed_provider_alias_variant(self) -> None:
        settings = SimpleNamespace(
            allowed_chat_models=["openai::gpt-5.3-codex"],
            llm_provider="github_copilot",
        )

        with (
            mock.patch.object(
                indexer_routes.repository,
                "get_settings",
                mock.AsyncMock(return_value=settings),
            ),
            mock.patch.object(
                indexer_routes,
                "_validate_conversation_model_selection",
                mock.AsyncMock(),
            ) as validate_live_model,
        ):
            await indexer_routes._validate_conversation_model_before_send(
                "github_copilot::openai/gpt-5.3-codex"
            )

        validate_live_model.assert_awaited_once()


class RequestScopedLLMResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unscoped_allowed_copilot_model_uses_copilot_despite_openai_first(
        self,
    ) -> None:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_provider": "openai",
            "llm_model": "gpt-4.1",
            "openai_api_key": "openai-key",
            "github_copilot_access_token": "copilot-token",
            "allowed_chat_models": ["github_copilot::claude-haiku-4.5"],
            "model_provider_precedence": {
                "providers": ["openai", "github_copilot"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }
        copilot_llm = object()

        async def build_llm(provider: str, model: str, max_tokens: int):
            self.assertEqual(provider, "github_copilot")
            self.assertEqual(model, "claude-haiku-4.5")
            return copilot_llm

        with (
            mock.patch.object(rag, "_build_llm", side_effect=build_llm),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm("claude-haiku-4.5")

        self.assertIs(resolution.llm, copilot_llm)
        self.assertEqual(resolution.provider, "github_copilot")
        self.assertEqual(resolution.attempted_providers, ("github_copilot",))

    async def test_provider_precedence_falls_back_when_first_provider_unavailable(
        self,
    ) -> None:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_provider": "openai",
            "llm_model": "gpt-5.1-codex-mini",
            "openai_api_key": "openai-key",
            "github_copilot_access_token": "copilot-token",
            "allowed_chat_models": [],
            "model_provider_precedence": {
                "providers": ["openai", "github_copilot"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }
        copilot_llm = object()
        attempted: list[str] = []

        async def build_llm(provider: str, model: str, max_tokens: int):
            attempted.append(provider)
            return copilot_llm if provider == "github_copilot" else None

        with (
            mock.patch.object(rag, "_build_llm", side_effect=build_llm),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm("gpt-5.1-codex-mini")

        self.assertEqual(attempted, ["openai", "github_copilot"])
        self.assertIs(resolution.llm, copilot_llm)
        self.assertEqual(resolution.provider, "github_copilot")
        self.assertEqual(resolution.attempted_providers, ("openai", "github_copilot"))

    async def test_openai_scoped_copilot_model_routes_through_copilot(self) -> None:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_provider": "github_copilot",
            "llm_model": "claude-haiku-4.5",
            "openai_api_key": "openai-key",
            "github_copilot_access_token": "copilot-token",
            "allowed_chat_models": ["github_copilot::gpt-5.4"],
            "model_provider_precedence": {
                "providers": ["github_copilot", "openai", "anthropic"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }
        copilot_llm = object()
        attempted: list[str] = []

        async def build_llm(provider: str, model: str, max_tokens: int):
            attempted.append(provider)
            return copilot_llm

        with (
            mock.patch.object(rag, "_build_llm", side_effect=build_llm),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm("openai::gpt-5.4")

        self.assertEqual(attempted, ["github_copilot"])
        self.assertIs(resolution.llm, copilot_llm)
        self.assertEqual(resolution.provider, "github_copilot")
        self.assertEqual(resolution.model, "gpt-5.4")
        self.assertEqual(resolution.attempted_providers, ("github_copilot",))

    async def test_allowed_openai_alias_uses_configured_copilot_equivalent(
        self,
    ) -> None:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_provider": "github_copilot",
            "llm_model": "claude-haiku-4.5",
            "openai_api_key": "",
            "github_copilot_access_token": "copilot-token",
            "allowed_chat_models": ["openai::gpt-5.4"],
            "model_provider_precedence": {
                "providers": ["github_copilot", "openai", "anthropic"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }
        copilot_llm = object()
        attempted: list[str] = []

        async def build_llm(provider: str, model: str, max_tokens: int):
            attempted.append(provider)
            return copilot_llm

        with (
            mock.patch.object(rag, "_build_llm", side_effect=build_llm),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm(
                "github_copilot::openai/gpt-5.4"
            )

        self.assertEqual(attempted, ["github_copilot"])
        self.assertIs(resolution.llm, copilot_llm)
        self.assertEqual(resolution.provider, "github_copilot")
        self.assertEqual(resolution.model, "openai/gpt-5.4")
        self.assertEqual(resolution.attempted_providers, ("github_copilot",))

    async def test_no_llm_error_names_attempted_providers(self) -> None:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_provider": "openai",
            "llm_model": "gpt-4.1",
            "openai_api_key": "",
            "allowed_chat_models": [],
            "model_provider_precedence": {
                "providers": ["openai", "github_copilot"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }

        with (
            mock.patch.object(rag, "_build_llm", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm("gpt-4.1")

        message = rag._no_llm_configured_message(resolution)
        self.assertIn("Tried providers for model 'gpt-4.1'", message)
        self.assertIn("OpenAI", message)
        self.assertIn("API key is missing", message)

    async def test_missing_cached_llm_refreshes_settings_before_resolution(self) -> None:
        rag = RAGComponents()
        rag._core_ready = True
        rag._app_settings = {
            "llm_provider": "openai",
            "llm_model": "gpt-4.1",
            "openai_api_key": "",
        }
        fresh_settings = {
            "llm_provider": "github_copilot",
            "llm_model": "claude-haiku-4.5",
            "github_copilot_access_token": "copilot-token",
            "allowed_chat_models": ["github_copilot::claude-haiku-4.5"],
            "model_provider_precedence": {
                "providers": ["openai", "github_copilot"],
                "model_overrides": {},
                "family_overrides": {},
            },
        }
        copilot_llm = object()

        async def build_llm(provider: str, model: str, max_tokens: int):
            self.assertEqual(provider, "github_copilot")
            self.assertEqual(model, "claude-haiku-4.5")
            return copilot_llm

        with (
            mock.patch(
                "ragtime.rag.components.get_app_settings",
                new=mock.AsyncMock(return_value=fresh_settings),
            ),
            mock.patch.object(rag, "_build_llm", side_effect=build_llm),
            mock.patch.object(
                rag,
                "_resolve_chat_request_max_tokens",
                new=mock.AsyncMock(return_value=4096),
            ),
        ):
            resolution = await rag._get_request_scoped_llm(None)

        self.assertIs(resolution.llm, copilot_llm)
        self.assertEqual(resolution.provider, "github_copilot")
        self.assertEqual(rag._app_settings, fresh_settings)


if __name__ == "__main__":
    unittest.main()
