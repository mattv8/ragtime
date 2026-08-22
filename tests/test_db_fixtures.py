"""Shared database mock fixtures for test suites."""

from types import SimpleNamespace
from typing import Any


def make_settings_row(**overrides: Any) -> SimpleNamespace:
    """Create a fake app settings row with standard defaults.

    Args:
        **overrides: Override specific settings values

    Returns:
        SimpleNamespace representing a settings database row
    """
    values: dict[str, Any] = {
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


class FakeAppSettingsClient:
    """Mock app settings client for database testing."""

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


class FakeConversationClient:
    """Mock conversation client for database testing."""

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


class FakeTx:
    """Mock transaction object for database testing."""

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

    async def __aenter__(self) -> "FakeTx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeDb:
    """Mock database client for testing."""

    def __init__(self, settings_row: SimpleNamespace, *, conversation_row: SimpleNamespace | None = None, tx: FakeTx | None = None) -> None:
        self.appsettings = FakeAppSettingsClient(settings_row)
        self.conversation = FakeConversationClient(conversation_row)
        self._tx = tx

    def tx(self) -> FakeTx:
        assert self._tx is not None
        return self._tx
