"""Shared test helpers for userspace SQLite tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, TypedDict


class _CapturedAuditEvent(TypedDict):
    """Audit event captured during testing."""

    workspace_id: str
    user_id: str | None
    event_type: str
    payload: dict[str, object]
    session_id: str | None
    created_at: object  # datetime | None


@dataclass
class _WorkspaceRowBase:
    """Base workspace row for testing."""

    owner_user_id: str
    name: str


class _FakeWorkspaceMemberTable:
    """Fake workspace member table for testing."""

    def __init__(self, roles: dict[tuple[str, str], str]) -> None:
        self.roles = roles
        self.find_first_calls = 0

    async def find_first(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        self.find_first_calls += 1
        key = (str(where.get("workspaceId") or ""), str(where.get("userId") or ""))
        role = self.roles.get(key)
        if role is None:
            return None
        return SimpleNamespace(workspaceId=key[0], userId=key[1], role=role)

    async def find_many(self, *, where: dict[str, Any] | None = None) -> list[SimpleNamespace]:
        workspace_ids = {str(value) for value in (where or {}).get("workspaceId", {}).get("in", [])}
        user_id = str((where or {}).get("userId") or "")
        return [
            SimpleNamespace(workspaceId=workspace_id, userId=member_user_id, role=role)
            for (workspace_id, member_user_id), role in self.roles.items()
            if (not workspace_ids or workspace_id in workspace_ids) and (not user_id or member_user_id == user_id)
        ]


class _FakeGrantTable:
    """Fake grant table for testing."""

    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = rows or []
        self.find_first_calls = 0

    async def find_first(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        self.find_first_calls += 1
        for row in self.rows:
            if str(getattr(row, "sourceWorkspaceId", "") or "") == str(where.get("sourceWorkspaceId") or "") and str(
                getattr(row, "targetWorkspaceId", "") or ""
            ) == str(where.get("targetWorkspaceId") or ""):
                return row
        return None

    async def find_many(self, *, where: dict[str, object] | None = None, order: dict[str, str] | None = None) -> list[SimpleNamespace]:
        source_workspace_id = str((where or {}).get("sourceWorkspaceId") or "")
        rows = [row for row in self.rows if not source_workspace_id or str(getattr(row, "sourceWorkspaceId", "") or "") == source_workspace_id]
        if order == {"targetWorkspaceId": "asc"}:
            rows.sort(key=lambda row: str(getattr(row, "targetWorkspaceId", "") or ""))
        return rows


class _FakeUserTable:
    """Fake user table for testing."""

    async def find_unique(self, **_: object) -> SimpleNamespace | None:
        return None
