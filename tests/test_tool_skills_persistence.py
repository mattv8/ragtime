import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from pydantic import ValidationError

from ragtime.core.app_settings import SettingsCache
from ragtime.indexer.models import FaissSearchConcurrencyMode, UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository, _sql_quote_literal
from tests.test_db_fixtures import FakeDb, FakeTx, make_settings_row


class ToolSkillsPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_cache_fallback_enables_tool_skills_by_default(self) -> None:
        cache = SettingsCache()

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(side_effect=RuntimeError("db down"))):
            settings = await cache.get_settings()

        self.assertTrue(settings["tool_skills_enabled"])

    async def test_repository_get_settings_maps_tool_skills_enabled(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row(toolSkillsEnabled=False))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertFalse(settings.tool_skills_enabled)

    async def test_repository_update_settings_maps_tool_skills_enabled(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings({"tool_skills_enabled": False})

        self.assertEqual(fake_db.appsettings.last_update_data, {"toolSkillsEnabled": False})

    async def test_settings_cache_defaults_faiss_search_concurrency_mode_to_per_index(self) -> None:
        cache = SettingsCache()

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(side_effect=RuntimeError("db down"))):
            settings = await cache.get_settings()

        self.assertEqual(settings["faiss_search_concurrency_mode"], "per_index")

    async def test_settings_cache_reads_faiss_search_concurrency_mode(self) -> None:
        cache = SettingsCache()
        fake_db = FakeDb(make_settings_row(faissSearchConcurrencyMode="global_mode"))

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await cache.get_settings()

        self.assertEqual(settings["faiss_search_concurrency_mode"], "global")

    async def test_repository_get_settings_maps_faiss_search_concurrency_mode(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row(faissSearchConcurrencyMode="global_mode"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertEqual(settings.faiss_search_concurrency_mode, "global")

    async def test_repository_update_settings_maps_faiss_search_concurrency_mode(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings({"faiss_search_concurrency_mode": "global"})

        self.assertEqual(fake_db.appsettings.last_update_data, {"faissSearchConcurrencyMode": "global_mode"})

    async def test_repository_get_settings_normalizes_raw_global_label(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row(faissSearchConcurrencyMode="global"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertEqual(settings.faiss_search_concurrency_mode, "global")

    def test_update_settings_request_rejects_invalid_faiss_search_concurrency_mode(self) -> None:
        with self.assertRaises(ValidationError):
            UpdateSettingsRequest.model_validate({"faiss_search_concurrency_mode": "parallel"})

    def test_update_settings_request_preserves_faiss_search_concurrency_mode(self) -> None:
        request = UpdateSettingsRequest(faiss_search_concurrency_mode=FaissSearchConcurrencyMode.GLOBAL)

        self.assertEqual(
            request.model_dump(mode="json", exclude_unset=True),
            {"faiss_search_concurrency_mode": "global"},
        )

    def test_update_settings_request_preserves_tool_skills_enabled_toggle(self) -> None:
        request = UpdateSettingsRequest(tool_skills_enabled=False)

        self.assertEqual(request.model_dump(exclude_unset=True), {"tool_skills_enabled": False})

    async def test_create_conversation_persists_normalized_loaded_tool_skill_ids(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            conversation = await repository.create_conversation(
                title="Tool skills",
                model="gpt-4.1",
                loaded_tool_skill_ids=[" skill.alpha ", "", "skill.alpha", "skill.beta "],
            )

        self.assertEqual(
            fake_db.conversation.create_calls[0]["loadedToolSkillIds"].data,
            ["skill.alpha", "skill.beta"],
        )
        self.assertEqual(conversation.loaded_tool_skill_ids, ["skill.alpha", "skill.beta"])

    async def test_get_conversation_loaded_tool_skill_ids_normalizes_values(self) -> None:
        repository = IndexerRepository()
        row = SimpleNamespace(loadedToolSkillIds=[" skill.alpha ", "", "skill.alpha", None, "skill.beta"])
        fake_db = FakeDb(make_settings_row(), conversation_row=row)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("conversation-1")

        self.assertEqual(loaded_ids, ["skill.alpha", "skill.beta"])

    async def test_get_conversation_loaded_tool_skill_ids_ignores_corrupt_scalar_json_values(self) -> None:
        repository = IndexerRepository()
        row = SimpleNamespace(loadedToolSkillIds=123)
        fake_db = FakeDb(make_settings_row(), conversation_row=row)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("conversation-1")

        self.assertEqual(loaded_ids, [])

    async def test_get_conversation_loaded_tool_skill_ids_returns_empty_list_when_missing(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row(), conversation_row=None)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("missing-conversation")

        self.assertEqual(loaded_ids, [])

    async def test_mutate_conversation_loaded_tool_skill_ids_removes_before_adding(self) -> None:
        repository = IndexerRepository()
        tx = FakeTx(["skill.alpha", "skill.beta", "skill.gamma"])
        fake_db = FakeDb(make_settings_row(), tx=tx)
        quoted_conversation_id = _sql_quote_literal("conversation-1")

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.mutate_conversation_loaded_tool_skill_ids(
                "conversation-1",
                add_ids=[" skill.beta ", "skill.delta", "skill.delta"],
                remove_ids=["skill.beta", " skill.delta "],
            )

        self.assertEqual(loaded_ids, ["skill.alpha", "skill.gamma"])
        self.assertTrue(any("FOR UPDATE" in query for query in tx.queries))
        self.assertTrue(any(quoted_conversation_id in query for query in tx.queries))
        self.assertTrue(any(quoted_conversation_id in query for query in tx.updates))
        self.assertTrue(any('["skill.alpha","skill.gamma"]' in query for query in tx.updates))

    async def test_mutate_conversation_loaded_tool_skill_ids_returns_empty_list_when_missing(self) -> None:
        repository = IndexerRepository()
        tx = FakeTx(["skill.alpha"], row_exists=False)
        fake_db = FakeDb(make_settings_row(), tx=tx)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.mutate_conversation_loaded_tool_skill_ids(
                "missing-conversation",
                add_ids=["skill.beta"],
            )

        self.assertEqual(loaded_ids, [])
        self.assertEqual(tx.updates, [])


if __name__ == "__main__":
    unittest.main()
