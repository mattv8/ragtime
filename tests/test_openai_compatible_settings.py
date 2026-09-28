import unittest
from unittest import mock

from ragtime.core.encryption import ENCRYPTED_PREFIX
from ragtime.indexer.models import AppSettings, UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository
from tests.test_db_fixtures import FakeDb, make_settings_row


class OpenAICompatibleSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_cache_loads_decrypted_key_and_json_overrides(self) -> None:
        from ragtime.core.app_settings import SettingsCache
        from ragtime.core.encryption import encrypt_secret

        fake_db = FakeDb(
            make_settings_row(
                openaiCompatibleBaseUrl="https://proxy.example/v1",
                openaiCompatibleApiKey=encrypt_secret("saved-compatible-key"),
                openaiCompatibleCatalogProvider="openrouter",
                openaiCompatibleModelLimits={"gpt-4o": {"context_limit": 128000, "max_output_tokens": 4096}},
            )
        )
        cache = SettingsCache()

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            loaded = await cache.get_settings()

        self.assertEqual(loaded["openai_compatible_base_url"], "https://proxy.example/v1")
        self.assertEqual(loaded["openai_compatible_api_key"], "saved-compatible-key")
        self.assertEqual(loaded["openai_compatible_catalog_provider"], "openrouter")
        self.assertEqual(loaded["openai_compatible_model_limits"], {"gpt-4o": {"context_limit": 128000, "max_output_tokens": 4096}})

    async def test_settings_persist_encrypted_key_and_json_limit_overrides(self) -> None:
        request = UpdateSettingsRequest.model_validate(
            {
                "openai_compatible_base_url": "https://proxy.example/v1/",
                "openai_compatible_api_key": "compatible-secret",
                "openai_compatible_catalog_provider": "openrouter",
                "openai_compatible_model_limits": {"model-A": {"context_limit": 32768, "max_output_tokens": 4096}},
            }
        )
        fake_db = FakeDb(make_settings_row())
        repository = IndexerRepository()

        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(repository, "get_settings", mock.AsyncMock(return_value=AppSettings())),
        ):
            await repository.update_settings(request.model_dump(exclude_unset=True))

        update = fake_db.appsettings.last_update_data
        assert update is not None
        self.assertEqual(update["openaiCompatibleBaseUrl"], "https://proxy.example/v1")
        self.assertTrue(update["openaiCompatibleApiKey"].startswith(ENCRYPTED_PREFIX))
        self.assertEqual(update["openaiCompatibleCatalogProvider"], "openrouter")
        self.assertEqual(
            update["openaiCompatibleModelLimits"].data,
            {"model-A": {"context_limit": 32768, "max_output_tokens": 4096}},
        )

    def test_settings_override_json_round_trip_preserves_nullable_limits(self) -> None:
        request = UpdateSettingsRequest.model_validate({"openai_compatible_model_limits": {"gpt-4o": {"context_limit": 128000}, "unknown": {}}})
        settings = AppSettings(**request.model_dump(exclude_unset=True))

        self.assertEqual(settings.openai_compatible_model_limits["gpt-4o"].context_limit, 128000)
        self.assertIsNone(settings.openai_compatible_model_limits["unknown"].max_output_tokens)

    async def test_settings_can_clear_compatible_key(self) -> None:
        fake_db = FakeDb(make_settings_row(openaiCompatibleApiKey="enc::saved"))
        repository = IndexerRepository()

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings({"openai_compatible_api_key": ""})

        self.assertEqual(fake_db.appsettings.last_update_data, {"openaiCompatibleApiKey": ""})

    async def test_settings_root_change_clears_saved_key_when_no_key_is_supplied(self) -> None:
        fake_db = FakeDb(make_settings_row(openaiCompatibleApiKey="enc::saved"))
        repository = IndexerRepository()
        existing = AppSettings(
            openai_compatible_base_url="https://old.example/v1",
            openai_compatible_api_key="saved-key",
        )
        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(repository, "get_settings", mock.AsyncMock(return_value=existing)),
        ):
            await repository.update_settings({"openai_compatible_base_url": "https://new.example/v1"})

        assert fake_db.appsettings.last_update_data is not None
        self.assertEqual(fake_db.appsettings.last_update_data["openaiCompatibleApiKey"], "")

    def test_compatible_settings_validate_url_and_reject_embedding_provider(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings(openai_compatible_base_url="not-a-url")
        with self.assertRaises(ValueError):
            UpdateSettingsRequest(openai_compatible_base_url="https://proxy.example/?query=yes")
        with self.assertRaises(ValueError):
            UpdateSettingsRequest(embedding_provider="openai_compatible")
