import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.repository import IndexerRepository


class _FakeAppSettingsClient:
    def __init__(self, row: SimpleNamespace) -> None:
        self._row = row

    async def find_unique(self, _where: dict[str, str]) -> SimpleNamespace:
        return self._row


class _FakeDb:
    def __init__(self, row: SimpleNamespace) -> None:
        self.appsettings = _FakeAppSettingsClient(row)


def _settings_row(**overrides: str) -> SimpleNamespace:
    values = {
        "id": "default",
        "serverName": "HammRAG",
        "defaultThemePack": "serif",
        "openaiApiKey": "",
        "anthropicApiKey": "",
        "ollamaProtocol": "http",
        "ollamaHost": "localhost",
        "ollamaPort": 11434,
        "ollamaBaseUrl": "http://localhost:11434",
        "allowedChatModels": [],
        "enabledTools": [],
        "postgresHost": "",
        "postgresUser": "",
        "postgresPassword": "",
        "postgresDb": "",
        "enableWriteOps": False,
        "updatedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AppSettingsThemeTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_settings_preserves_default_theme_pack(self) -> None:
        # Given: app settings stored with a non-default global theme pack.
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(defaultThemePack="serif"))

        # When: settings are read through the repository mapping.
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        # Then: the theme pack is returned instead of falling back to the model default.
        self.assertEqual(settings.default_theme_pack, "serif")
