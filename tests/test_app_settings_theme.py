import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.repository import IndexerRepository


class _FakeAppSettingsClient:
    def __init__(self, row: SimpleNamespace) -> None:
        self._row = row

    async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace:
        return self._row

    async def update(self, *, where: dict[str, str], data: dict) -> SimpleNamespace:
        self._last_update = data
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

    async def test_get_settings_canonicalizes_legacy_vscode_id(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(defaultThemePack="vscode"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertEqual(settings.default_theme_pack, "modern")

    async def test_update_settings_canonicalizes_legacy_vscode_default_theme_pack(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(defaultThemePack="default"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings({"default_theme_pack": "vscode"})

        client = fake_db.appsettings
        self.assertEqual(getattr(client, "_last_update", {}).get("defaultThemePack"), "modern")

    def test_theme_migration_updates_both_persisted_columns(self) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "prisma" / "migrations"
        migration_dirs = sorted(p for p in migrations_dir.iterdir() if p.is_dir() and "rename_vscode" in p.name)
        self.assertGreaterEqual(len(migration_dirs), 1, "Expected a vscode→modern migration folder")
        sql = (migration_dirs[0] / "migration.sql").read_text(encoding="utf-8").lower()
        self.assertIn("update", sql)
        self.assertIn("theme_pack", sql)
        self.assertIn("default_theme_pack", sql)
        self.assertIn("'modern'", sql)
        # migration.sql should only contain vscode as the legacy value to backfill, not as a target
        self.assertIn("= 'vscode'", sql)
