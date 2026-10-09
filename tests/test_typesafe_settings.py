import unittest
from unittest import mock

from ragtime.core.encryption import ENCRYPTED_PREFIX, decrypt_secret, encrypt_secret
from ragtime.indexer.models import AppSettings, UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository
from tests.test_db_fixtures import FakeDb, make_settings_row


class TypeSafeSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_cache_returns_decrypted_typesafe_key(self) -> None:
        from ragtime.core.app_settings import SettingsCache

        cache = SettingsCache()
        fake_db = FakeDb(make_settings_row(typesafeApiKey=encrypt_secret("typesafe-secret")))

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await cache.get_settings()

        self.assertEqual(settings["typesafe_api_key"], "typesafe-secret")

    async def test_settings_encrypts_returns_and_clears_typesafe_key(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            stored = await repository.update_settings({"typesafe_api_key": "typesafe-secret"})
            update = fake_db.appsettings.last_update_data
            assert update is not None
            encrypted = update["typesafeApiKey"]
            self.assertTrue(encrypted.startswith(ENCRYPTED_PREFIX))
            self.assertNotEqual(encrypted, "typesafe-secret")
            self.assertEqual(decrypt_secret(encrypted), "typesafe-secret")
            self.assertEqual(stored.typesafe_api_key, "typesafe-secret")

            cleared = await repository.update_settings({"typesafe_api_key": ""})

        self.assertEqual(fake_db.appsettings.last_update_data, {"typesafeApiKey": ""})
        self.assertEqual(cleared.typesafe_api_key, "")

    def test_settings_request_and_response_accept_typesafe_key(self) -> None:
        request = UpdateSettingsRequest.model_validate({"typesafe_api_key": "typesafe-secret"})
        response = AppSettings(typesafe_api_key="typesafe-secret")

        self.assertEqual(request.typesafe_api_key, "typesafe-secret")
        self.assertEqual(response.typesafe_api_key, "typesafe-secret")
