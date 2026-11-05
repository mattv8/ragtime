import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core.model_providers import get_provider
from ragtime.core.ocr_target import OcrTargetError, resolve_ocr_target
from ragtime.indexer import routes
from ragtime.indexer.models import AppSettings, IndexConfig, OcrMode, OcrProvider, UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository


class OcrTargetTests(unittest.TestCase):
    def test_inherits_global_model_when_provider_matches(self) -> None:
        target = resolve_ocr_target(
            {
                "default_ocr_mode": "vision",
                "default_ocr_provider": "openai",
                "default_ocr_vision_model": "gpt-4.1-mini",
            },
            {"ocr_provider": "openai"},
        )

        assert target is not None
        self.assertEqual(target.provider, "openai")
        self.assertEqual(target.model, "gpt-4.1-mini")
        self.assertEqual(target.base_url, "https://api.openai.com/v1")

    def test_rejects_overridden_provider_without_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "ocr_vision_model"):
            resolve_ocr_target(
                {
                    "default_ocr_mode": "vision",
                    "default_ocr_provider": "openai",
                    "default_ocr_vision_model": "gpt-4.1-mini",
                },
                {"ocr_provider": "anthropic"},
            )

    def test_explicitly_cleared_per_index_values_inherit_global_target(self) -> None:
        target = resolve_ocr_target(
            {
                "default_ocr_mode": "vision",
                "default_ocr_provider": "openai",
                "default_ocr_vision_model": "gpt-4.1-mini",
            },
            {"ocr_provider": None, "ocr_vision_model": None},
        )

        assert target is not None
        self.assertEqual(target.provider, "openai")
        self.assertEqual(target.model, "gpt-4.1-mini")

    def test_same_provider_blank_model_inherits_global_model(self) -> None:
        target = resolve_ocr_target(
            {
                "default_ocr_mode": "vision",
                "default_ocr_provider": "openai",
                "default_ocr_vision_model": "gpt-4.1-mini",
            },
            {"ocr_provider": "openai", "ocr_vision_model": None},
        )

        assert target is not None
        self.assertEqual(target.model, "gpt-4.1-mini")

    def test_rejects_blank_model_for_a_different_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "ocr_vision_model"):
            resolve_ocr_target(
                {
                    "default_ocr_mode": "vision",
                    "default_ocr_provider": "openai",
                    "default_ocr_vision_model": "gpt-4.1-mini",
                },
                {"ocr_provider": "anthropic", "ocr_vision_model": None},
            )

    def test_missing_global_provider_is_invalid_for_vision(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider"):
            resolve_ocr_target(
                {"default_ocr_mode": "vision", "default_ocr_provider": None, "default_ocr_vision_model": None},
                {"ocr_mode": "vision"},
            )

    def test_ollama_prefers_the_llm_connection_for_ocr(self) -> None:
        target = resolve_ocr_target(
            {
                "default_ocr_mode": "vision",
                "default_ocr_provider": "ollama",
                "default_ocr_vision_model": "llama3.2-vision",
                "ollama_base_url": "http://ocr.example.test:11434",
                "llm_ollama_base_url": "http://chat.example.test:11434",
            }
        )

        assert target is not None
        self.assertEqual(target.base_url, "http://chat.example.test:11434")

    def test_ollama_uses_embedding_connection_when_llm_uses_its_default(self) -> None:
        target = resolve_ocr_target(
            {
                "default_ocr_mode": "vision",
                "default_ocr_provider": "ollama",
                "default_ocr_vision_model": "llama3.2-vision",
                "ollama_base_url": "http://ocr.example.test:11434",
                "llm_ollama_base_url": "http://localhost:11434",
            }
        )

        assert target is not None
        self.assertEqual(target.base_url, "http://ocr.example.test:11434")

    def test_resolves_native_and_cli_endpoints_without_credentials(self) -> None:
        anthropic = resolve_ocr_target({"default_ocr_mode": "vision", "default_ocr_provider": "anthropic", "default_ocr_vision_model": "claude-sonnet-4-5"})
        claude_code = resolve_ocr_target({"default_ocr_mode": "vision", "default_ocr_provider": "claude_code", "default_ocr_vision_model": "claude-sonnet-4-5"})

        assert anthropic is not None
        assert claude_code is not None
        self.assertEqual(anthropic.base_url, "https://api.anthropic.com")
        self.assertIsNone(claude_code.base_url)
        self.assertNotIn("api_key", anthropic.to_snapshot())

    def test_subscription_providers_are_vision_only(self) -> None:
        claude_code_provider = get_provider("claude_code")
        openai_codex_provider = get_provider("openai_codex")
        assert claude_code_provider is not None
        assert openai_codex_provider is not None
        self.assertTrue(claude_code_provider.supports_vision_ocr)
        self.assertTrue(openai_codex_provider.supports_vision_ocr)
        self.assertTrue(openai_codex_provider.supports_embeddings)

    def test_resolves_real_enum_models(self) -> None:
        settings = AppSettings(
            default_ocr_mode=OcrMode.VISION,
            default_ocr_provider=OcrProvider.OLLAMA,
            default_ocr_vision_model="llama3.2-vision",
        )
        config = IndexConfig(name="manual", ocr_mode=OcrMode.VISION)

        target = resolve_ocr_target(settings, config)

        assert target is not None
        self.assertEqual(target.provider, "ollama")
        self.assertEqual(target.model, "llama3.2-vision")

    def test_non_vision_modes_return_no_target(self) -> None:
        target = resolve_ocr_target(
            AppSettings(default_ocr_mode=OcrMode.TESSERACT, default_ocr_provider=OcrProvider.OPENAI),
            IndexConfig(name="manual", ocr_mode=OcrMode.TESSERACT),
        )

        self.assertIsNone(target)

    def test_pydantic_config_uses_its_default_disabled_mode(self) -> None:
        target = resolve_ocr_target(
            AppSettings(default_ocr_provider=OcrProvider.OPENAI),
            IndexConfig(name="manual"),
        )

        self.assertIsNone(target)

    def test_dict_without_mode_requires_explicit_vision_mode(self) -> None:
        self.assertIsNone(
            resolve_ocr_target(
                {"default_ocr_provider": "openai", "default_ocr_vision_model": "gpt-4.1-mini"},
                {"ocr_provider": "openai"},
            )
        )

    def test_rejects_credential_bearing_snapshot_endpoint(self) -> None:
        with self.assertRaises(OcrTargetError):
            resolve_ocr_target(
                {
                    "default_ocr_mode": "vision",
                    "default_ocr_provider": "openai_codex",
                    "default_ocr_vision_model": "gpt-4.1",
                    "openai_codex_base_url": "https://token@example.test/codex",
                }
            )


if __name__ == "__main__":
    unittest.main()


class OcrRouteValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_vision_catalog_forwards_only_server_image_models(self) -> None:
        settings = AppSettings()
        catalog = routes.LLMModelsResponse(
            success=True,
            message="ok",
            models=[
                routes.LLMModel(id="codex-image", name="Codex Image", capabilities=["image"]),
                routes.LLMModel(id="codex-text", name="Codex Text", capabilities=["reasoning"]),
            ],
        )
        infos = [SimpleNamespace(name="codex-image", provider="openai_codex", capabilities=["image_input"])]
        with (
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(routes, "_fetch_subscription_vision_candidates", mock.AsyncMock(return_value=catalog)),
            mock.patch.object(routes, "list_provider_vision_models", mock.AsyncMock(return_value=infos)) as list_models,
        ):
            response = await routes.get_provider_vision_models(
                routes.VisionModelsRequest(provider="openai_codex", candidate_models=["forged-client-id"]),
                _user=mock.sentinel.user,
            )

        self.assertTrue(response.success)
        self.assertEqual([model.name for model in response.models], ["codex-image"])
        assert list_models.await_args is not None
        self.assertEqual(list_models.await_args.kwargs["available_image_model_ids"], {"codex-image"})
        self.assertEqual(list_models.await_args.kwargs["candidate_models"], [])

    async def test_settings_reject_invalid_provider_while_ocr_is_disabled(self) -> None:
        current_settings = AppSettings(default_ocr_mode=OcrMode.DISABLED)
        with (
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=current_settings)),
            mock.patch.object(routes.repository, "update_settings", mock.AsyncMock()) as save,
        ):
            with self.assertRaisesRegex(Exception, "does not support vision OCR"):
                await routes.update_settings(
                    UpdateSettingsRequest(default_ocr_provider="unknown-provider"),
                    _user=mock.sentinel.user,
                )

        save.assert_not_awaited()

    async def test_unrelated_settings_save_does_not_validate_legacy_ocr_target(self) -> None:
        current_settings_data = AppSettings().model_dump()
        current_settings_data.update(default_ocr_provider="unknown-provider", default_ocr_mode=OcrMode.DISABLED)
        current_settings = AppSettings.model_construct(**current_settings_data)
        with (
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=current_settings)),
            mock.patch.object(routes.repository, "update_settings", mock.AsyncMock(return_value=current_settings)) as save,
            mock.patch.object(routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(routes, "notify_tools_changed"),
        ):
            await routes.update_settings(UpdateSettingsRequest(default_theme_pack="default"), _user=mock.sentinel.user)

        save.assert_awaited_once_with({"default_theme_pack": "default"})

    async def test_filesystem_trigger_rejects_unresolved_vision_target_before_job_start(self) -> None:
        tool = SimpleNamespace(
            tool_type=routes.ToolType.FILESYSTEM_INDEXER,
            connection_config={"base_path": "/data", "index_name": "docs", "ocr_mode": "vision"},
        )
        settings = AppSettings(default_ocr_provider=None, default_ocr_vision_model=None)
        with (
            mock.patch.object(routes.repository, "get_tool_config", mock.AsyncMock(return_value=tool)),
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(routes.filesystem_indexer, "ensure_pgvector_extension", mock.AsyncMock()) as ensure_pgvector,
        ):
            with self.assertRaisesRegex(Exception, "OCR provider is required"):
                await routes.trigger_filesystem_index(
                    "tool-1",
                    routes.TriggerFilesystemIndexRequest(),
                    _user=mock.sentinel.user,
                )

        ensure_pgvector.assert_not_awaited()

    async def test_index_config_normalizes_legacy_ollama_before_validating_a_clear(self) -> None:
        metadata = SimpleNamespace(
            sourceType="git",
            configSnapshot={"ocr_mode": "ollama"},
            source="https://example.test/repo.git",
            webhookId=None,
            webhookSecret=None,
        )
        settings = AppSettings(
            default_ocr_mode=OcrMode.VISION,
            default_ocr_provider=OcrProvider.OLLAMA,
            default_ocr_vision_model="llama3.2-vision",
        )
        with (
            mock.patch.object(routes.repository, "get_index_metadata", mock.AsyncMock(return_value=metadata)),
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(routes.repository, "update_index_config", mock.AsyncMock(return_value=True)) as save,
        ):
            response = await routes.update_index_config(
                "docs",
                routes.UpdateIndexConfigRequest(ocr_provider=None),
                _user=mock.sentinel.user,
            )

        self.assertEqual(response["config_snapshot"]["ocr_mode"], "vision")
        self.assertIsNone(response["config_snapshot"]["ocr_provider"])
        save.assert_awaited_once()


class _TokenSettingsStore:
    def __init__(self, refresh_token: str) -> None:
        self.openaiCodexRefreshToken = refresh_token


class _TokenSettingsModel:
    def __init__(self, settings: _TokenSettingsStore) -> None:
        self.settings = settings
        self.update = mock.AsyncMock()

    async def find_unique(self, **_kwargs):
        return self.settings


class _TokenTransaction:
    def __init__(self, settings: _TokenSettingsStore) -> None:
        self.appsettings = _TokenSettingsModel(settings)
        self.query_raw = mock.AsyncMock(return_value=[{"id": "default"}])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _TokenDatabase:
    def __init__(self, transaction: _TokenTransaction) -> None:
        self.transaction = transaction

    def tx(self):
        return self.transaction


class CodexTokenRotationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rotates_tokens_in_the_locked_transaction_when_refresh_token_matches(self) -> None:
        transaction = _TokenTransaction(_TokenSettingsStore("stored-refresh"))
        repository = IndexerRepository()
        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=_TokenDatabase(transaction))),
            mock.patch("ragtime.indexer.repository.decrypt_secret", return_value="expected-refresh"),
            mock.patch("ragtime.indexer.repository.encrypt_secret", side_effect=lambda value: f"encrypted:{value}"),
        ):
            rotated = await repository.rotate_openai_codex_tokens_if_refresh_token_matches(
                "expected-refresh",
                {"openai_codex_access_token": "new-access", "openai_codex_refresh_token": "new-refresh"},
            )

        self.assertTrue(rotated)
        transaction.query_raw.assert_awaited_once_with("SELECT id FROM app_settings WHERE id = 'default' FOR UPDATE")
        transaction.appsettings.update.assert_awaited_once_with(
            where={"id": "default"},
            data={"openaiCodexAccessToken": "encrypted:new-access", "openaiCodexRefreshToken": "encrypted:new-refresh"},
        )

    async def test_refuses_stale_refresh_token_without_writing(self) -> None:
        transaction = _TokenTransaction(_TokenSettingsStore("stored-refresh"))
        repository = IndexerRepository()
        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=_TokenDatabase(transaction))),
            mock.patch("ragtime.indexer.repository.decrypt_secret", return_value="different-refresh"),
        ):
            rotated = await repository.rotate_openai_codex_tokens_if_refresh_token_matches(
                "expected-refresh",
                {"openai_codex_access_token": "new-access"},
            )

        self.assertFalse(rotated)
        transaction.appsettings.update.assert_not_awaited()
