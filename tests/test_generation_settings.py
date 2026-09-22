import unittest
from unittest import mock

from ragtime.indexer.models import UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository
from tests.test_db_fixtures import FakeDb, make_settings_row


class GenerationSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_request_persists_each_generation_surface_independently(self) -> None:
        for field, db_field in (("chat_enabled", "chatEnabled"), ("userspace_generation_enabled", "userspaceGenerationEnabled")):
            for enabled in (True, False):
                with self.subTest(field=field, enabled=enabled):
                    request = UpdateSettingsRequest.model_validate({field: enabled})
                    updates = request.model_dump(exclude_unset=True)
                    fake_db = FakeDb(make_settings_row(chatEnabled=not enabled, userspaceGenerationEnabled=not enabled))
                    repository = IndexerRepository()

                    with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
                        settings = await repository.update_settings(updates)

                    self.assertEqual(fake_db.appsettings.last_update_data, {db_field: enabled})
                    self.assertEqual(getattr(settings, field), enabled)

    async def test_omitted_generation_fields_preserve_existing_policy(self) -> None:
        request = UpdateSettingsRequest.model_validate({})
        fake_db = FakeDb(make_settings_row(chatEnabled=False, userspaceGenerationEnabled=True))
        repository = IndexerRepository()

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.update_settings(request.model_dump(exclude_unset=True))

        self.assertIsNone(fake_db.appsettings.last_update_data)
        self.assertFalse(settings.chat_enabled)
        self.assertTrue(settings.userspace_generation_enabled)
