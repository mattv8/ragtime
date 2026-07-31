import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from pydantic import ValidationError

from ragtime.core.app_settings import SettingsCache
from ragtime.indexer.models import UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository, _sql_quote_literal


def _settings_row(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": "default",
        "serverName": "Ragtime",
        "defaultThemePack": "default",
        "faissSearchConcurrencyMode": "per_index",
        "openaiApiKey": "",
        "anthropicApiKey": "",
        "mcpDefaultRoutePassword": None,
        "mcpDefaultRouteAllowedGroup": None,
        "odooContainer": "odoo-server",
        "postgresContainer": "odoo-postgres",
        "ollamaProtocol": "http",
        "ollamaHost": "localhost",
        "ollamaPort": 11434,
        "ollamaBaseUrl": "http://localhost:11434",
        "embeddingProvider": "ollama",
        "embeddingModel": "nomic-embed-text",
        "embeddingDimensions": None,
        "allowedChatModels": [],
        "enabledTools": [],
        "postgresHost": "",
        "postgresPort": 5432,
        "postgresUser": "",
        "postgresPassword": "",
        "postgresDb": "",
        "enableWriteOps": False,
        "updatedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeAppSettingsClient:
    def __init__(self, row: SimpleNamespace) -> None:
        self._row = row
        self.last_update_data: dict[str, Any] | None = None

    async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace:
        return self._row

    async def update(self, *, where: dict[str, str], data: dict[str, Any]) -> SimpleNamespace:
        self.last_update_data = data
        for key, value in data.items():
            setattr(self._row, key, value)
        return self._row


class _FakeConversationClient:
    def __init__(self, row: SimpleNamespace | None = None) -> None:
        self.row = row
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, *, data: dict[str, Any], include: dict[str, bool] | None = None) -> SimpleNamespace:
        self.create_calls.append(data)
        created = {
            "id": data["id"],
            "title": data["title"],
            "model": data["model"],
            "messages": [],
            "totalTokens": data["totalTokens"],
            "userId": data.get("userId"),
            "workspaceId": data.get("workspaceId"),
            "toolOutputMode": "default",
            "toolSelectionMode": data.get("toolSelectionMode", "default_all"),
            "disabledBuiltinToolIds": [],
            "loadedToolSkillIds": data.get("loadedToolSkillIds", []),
            "subagentsEnabled": data.get("subagentsEnabled", True),
            "parentConversationId": data.get("parentConversationId"),
            "subagentRole": data.get("subagentRole"),
            "subagentIndex": data.get("subagentIndex"),
            "activeTaskId": None,
            "activeBranchId": None,
            "createdAt": data.get("createdAt"),
            "updatedAt": data.get("updatedAt"),
            "user": None,
        }
        return SimpleNamespace(**created)

    async def find_unique(self, *, where: dict[str, str], include: dict[str, bool] | None = None) -> SimpleNamespace | None:
        return self.row


class _FakeTx:
    def __init__(self, loaded_ids: Any, *, row_exists: bool = True) -> None:
        self.loaded_ids = loaded_ids
        self.row_exists = row_exists
        self.queries: list[str] = []
        self.updates: list[str] = []

    async def query_raw(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        if not self.row_exists:
            return []
        return [{"loaded_tool_skill_ids": self.loaded_ids}]

    async def execute_raw(self, query: str) -> int:
        self.updates.append(query)
        return 1

    async def __aenter__(self) -> "_FakeTx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeDb:
    def __init__(self, settings_row: SimpleNamespace, *, conversation_row: SimpleNamespace | None = None, tx: _FakeTx | None = None) -> None:
        self.appsettings = _FakeAppSettingsClient(settings_row)
        self.conversation = _FakeConversationClient(conversation_row)
        self._tx = tx

    def tx(self) -> _FakeTx:
        assert self._tx is not None
        return self._tx


class ToolSkillsPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_cache_fallback_enables_tool_skills_by_default(self) -> None:
        cache = SettingsCache()

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(side_effect=RuntimeError("db down"))):
            settings = await cache.get_settings()

        self.assertTrue(settings["tool_skills_enabled"])

    async def test_repository_get_settings_maps_tool_skills_enabled(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(toolSkillsEnabled=False))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertFalse(settings.tool_skills_enabled)

    async def test_repository_update_settings_maps_tool_skills_enabled(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row())

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
        fake_db = _FakeDb(_settings_row(faissSearchConcurrencyMode="global_mode"))

        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await cache.get_settings()

        self.assertEqual(settings["faiss_search_concurrency_mode"], "global")

    async def test_repository_get_settings_maps_faiss_search_concurrency_mode(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(faissSearchConcurrencyMode="global_mode"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertEqual(settings.faiss_search_concurrency_mode, "global")

    async def test_repository_update_settings_maps_faiss_search_concurrency_mode(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings({"faiss_search_concurrency_mode": "global"})

        self.assertEqual(fake_db.appsettings.last_update_data, {"faissSearchConcurrencyMode": "global_mode"})

    async def test_repository_get_settings_normalizes_raw_global_label(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(faissSearchConcurrencyMode="global"))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertEqual(settings.faiss_search_concurrency_mode, "global")

    def test_update_settings_request_rejects_invalid_faiss_search_concurrency_mode(self) -> None:
        with self.assertRaises(ValidationError):
            UpdateSettingsRequest(faiss_search_concurrency_mode="parallel")

    def test_update_settings_request_preserves_faiss_search_concurrency_mode(self) -> None:
        request = UpdateSettingsRequest(faiss_search_concurrency_mode="global")

        self.assertEqual(
            request.model_dump(mode="json", exclude_unset=True),
            {"faiss_search_concurrency_mode": "global"},
        )

    def test_update_settings_request_preserves_tool_skills_enabled_toggle(self) -> None:
        request = UpdateSettingsRequest(tool_skills_enabled=False)

        self.assertEqual(request.model_dump(exclude_unset=True), {"tool_skills_enabled": False})

    async def test_create_conversation_persists_normalized_loaded_tool_skill_ids(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row())

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
        fake_db = _FakeDb(_settings_row(), conversation_row=row)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("conversation-1")

        self.assertEqual(loaded_ids, ["skill.alpha", "skill.beta"])

    async def test_get_conversation_loaded_tool_skill_ids_ignores_corrupt_scalar_json_values(self) -> None:
        repository = IndexerRepository()
        row = SimpleNamespace(loadedToolSkillIds=123)
        fake_db = _FakeDb(_settings_row(), conversation_row=row)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("conversation-1")

        self.assertEqual(loaded_ids, [])

    async def test_get_conversation_loaded_tool_skill_ids_returns_empty_list_when_missing(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row(), conversation_row=None)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.get_conversation_loaded_tool_skill_ids("missing-conversation")

        self.assertEqual(loaded_ids, [])

    async def test_mutate_conversation_loaded_tool_skill_ids_removes_before_adding(self) -> None:
        repository = IndexerRepository()
        tx = _FakeTx(["skill.alpha", "skill.beta", "skill.gamma"])
        fake_db = _FakeDb(_settings_row(), tx=tx)
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
        tx = _FakeTx(["skill.alpha"], row_exists=False)
        fake_db = _FakeDb(_settings_row(), tx=tx)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            loaded_ids = await repository.mutate_conversation_loaded_tool_skill_ids(
                "missing-conversation",
                add_ids=["skill.beta"],
            )

        self.assertEqual(loaded_ids, [])
        self.assertEqual(tx.updates, [])


if __name__ == "__main__":
    unittest.main()
