from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock


class _FakeToolAccessPolicyDelegate:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    async def upsert(self, **kwargs: Any) -> SimpleNamespace:
        self.upserts.append(kwargs)
        create = kwargs["data"]["create"]
        return SimpleNamespace(
            id="policy-1",
            toolConfigId=create["toolConfig"]["connect"]["id"],
            defaultChatAccess=create["defaultChatAccess"],
            defaultWorkspaceAccess=create["defaultWorkspaceAccess"],
        )


class _FakeDb:
    def __init__(
        self,
        *,
        policy_rows: list[dict[str, Any]] | None = None,
        user_rows: list[dict[str, Any]] | None = None,
        group_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.policy_rows = list(policy_rows or [])
        self.user_rows = list(user_rows or [])
        self.group_rows = list(group_rows or [])
        self.query_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.toolaccesspolicy = _FakeToolAccessPolicyDelegate()

    async def query_raw(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.query_calls.append((query, params))
        normalized = " ".join(query.split())
        if 'FROM "tool_access_policies"' in normalized:
            return list(self.policy_rows)
        if 'FROM "tool_user_access"' in normalized:
            return list(self.user_rows)
        if 'FROM "tool_auth_group_access"' in normalized:
            return list(self.group_rows)
        raise AssertionError(f"Unexpected query: {query}")


class ToolAccessResolverTests(unittest.IsolatedAsyncioTestCase):
    def _load_module(self) -> Any:
        try:
            return importlib.import_module("ragtime.core.tool_access")
        except ModuleNotFoundError:
            self.fail("ragtime.core.tool_access module is missing")

    async def test_direct_and_group_deny_beat_matching_grants(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[
                {"tool_config_id": "tool-1", "default_access": "read"},
                {"tool_config_id": "tool-2", "default_access": "read"},
            ],
            user_rows=[
                {"tool_config_id": "tool-1", "access_level": "read_write"},
                {"tool_config_id": "tool-2", "access_level": "deny"},
            ],
            group_rows=[
                {"tool_config_id": "tool-1", "access_level": "deny"},
                {"tool_config_id": "tool-2", "access_level": "read_write"},
            ],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=["tool-1", "tool-2"],
            )

        self.assertEqual(resolved, {"tool-1": "deny", "tool-2": "deny"})

    async def test_highest_matching_grant_wins_when_no_deny_matches(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[
                {"tool_config_id": "tool-1", "default_access": "deny"},
                {"tool_config_id": "tool-2", "default_access": "deny"},
            ],
            user_rows=[{"tool_config_id": "tool-1", "access_level": "read"}],
            group_rows=[
                {"tool_config_id": "tool-1", "access_level": "read_write"},
                {"tool_config_id": "tool-2", "access_level": "read"},
            ],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=["tool-1", "tool-2"],
            )

        self.assertEqual(resolved, {"tool-1": "read_write", "tool-2": "read"})

    async def test_inherit_rows_fall_through_to_surface_policy(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[{"tool_config_id": "tool-1", "default_access": "read_write", "allow_write": True}],
            user_rows=[{"tool_config_id": "tool-1", "access_level": None}],
            group_rows=[{"tool_config_id": "tool-1", "access_level": None}],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="workspace",
                tool_config_ids=["tool-1"],
            )

        self.assertEqual(resolved, {"tool-1": "read_write"})

    async def test_read_only_tool_caps_fallback_read_write_to_read(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[{"tool_config_id": "tool-1", "default_access": "read_write", "allow_write": False}],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="workspace",
                tool_config_ids=["tool-1"],
            )

        self.assertEqual(resolved, {"tool-1": "read"})

    async def test_read_only_tool_keeps_explicit_read_write_grants_uncapped(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[{"tool_config_id": "tool-1", "default_access": "deny", "allow_write": False}],
            user_rows=[{"tool_config_id": "tool-1", "access_level": "read_write"}],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=["tool-1"],
            )

        self.assertEqual(resolved, {"tool-1": "read_write"})

    async def test_missing_policy_fails_closed_to_deny(self) -> None:
        tool_access = self._load_module()

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=_FakeDb())):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=["tool-1"],
            )

        self.assertEqual(resolved, {"tool-1": "deny"})

    async def test_admin_bypass_returns_read_write_without_db_lookup(self) -> None:
        tool_access = self._load_module()
        get_db = mock.AsyncMock(side_effect=AssertionError("admin bypass should not hit db"))

        with mock.patch("ragtime.core.tool_access.get_db", get_db):
            resolved = await tool_access.resolve_tool_access(
                user_id="admin-1",
                is_admin=True,
                surface="chat",
                tool_config_ids=["tool-1", "tool-2"],
            )

        self.assertEqual(resolved, {"tool-1": "read_write", "tool-2": "read_write"})
        get_db.assert_not_awaited()

    async def test_filter_tool_ids_by_access_preserves_order(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[
                {"tool_config_id": "tool-1", "default_access": "deny"},
                {"tool_config_id": "tool-2", "default_access": "read"},
                {"tool_config_id": "tool-3", "default_access": "read_write"},
            ]
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            allowed = await tool_access.filter_tool_ids_by_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=["tool-3", "tool-1", "tool-2", "tool-3"],
            )

        self.assertEqual(allowed, ["tool-3", "tool-2", "tool-3"])

    async def test_group_query_uses_current_timestamp_without_extra_now_param(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb(
            policy_rows=[
                {"tool_config_id": "tool-1", "default_access": "deny"},
                {"tool_config_id": "tool-2", "default_access": "deny"},
            ],
            group_rows=[
                {"tool_config_id": "tool-1", "access_level": "read"},
                {"tool_config_id": "tool-2", "access_level": "read_write"},
            ],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="workspace",
                tool_config_ids=["tool-1", "tool-2"],
            )

        group_query, group_params = next((query, params) for query, params in fake_db.query_calls if 'FROM "tool_auth_group_access"' in query)
        self.assertIn('JOIN "auth_group_memberships"', group_query)
        self.assertNotIn('JOIN "auth_groups"', group_query)
        self.assertIn('membership."expires_at" IS NULL OR membership."expires_at" > CURRENT_TIMESTAMP', group_query)
        self.assertEqual(group_params, (["tool-1", "tool-2"], "user-1"))
        self.assertEqual(resolved, {"tool-1": "read", "tool-2": "read_write"})

    async def test_resolver_batches_policy_user_and_group_queries_across_many_tool_ids(self) -> None:
        tool_access = self._load_module()
        tool_ids = [f"tool-{index}" for index in range(1, 6)]
        fake_db = _FakeDb(
            policy_rows=[{"tool_config_id": tool_id, "default_access": "read" if tool_id != "tool-5" else "deny", "allow_write": True} for tool_id in tool_ids],
            user_rows=[{"tool_config_id": "tool-5", "access_level": "read_write"}],
        )

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            resolved = await tool_access.resolve_tool_access(
                user_id="user-1",
                is_admin=False,
                surface="chat",
                tool_config_ids=tool_ids,
            )

        self.assertEqual(len(fake_db.query_calls), 3)
        policy_query, _policy_params = fake_db.query_calls[0]
        self.assertIn('JOIN "tool_configs" tool ON tool."id" = policy."tool_config_id"', policy_query)
        self.assertIn('tool."allow_write" AS allow_write', policy_query)
        for _query, params in fake_db.query_calls:
            self.assertEqual(params[0], tool_ids)
        self.assertEqual(resolved["tool-5"], "read_write")
        self.assertTrue(all(tool_id in resolved for tool_id in tool_ids))


class EnsureToolAccessPolicyTests(unittest.IsolatedAsyncioTestCase):
    def _load_module(self) -> Any:
        try:
            return importlib.import_module("ragtime.core.tool_access")
        except ModuleNotFoundError:
            self.fail("ragtime.core.tool_access module is missing")

    async def test_ensure_tool_access_policy_upserts_deny_defaults(self) -> None:
        tool_access = self._load_module()
        fake_db = _FakeDb()

        with mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=fake_db)):
            await tool_access.ensure_tool_access_policy("tool-123")
            await tool_access.ensure_tool_access_policy("tool-123")

        self.assertEqual(len(fake_db.toolaccesspolicy.upserts), 2)
        first = fake_db.toolaccesspolicy.upserts[0]
        self.assertEqual(first["where"], {"toolConfigId": "tool-123"})
        self.assertEqual(first["data"]["update"], {})
        self.assertEqual(first["data"]["create"]["defaultChatAccess"], "deny")
        self.assertEqual(first["data"]["create"]["defaultWorkspaceAccess"], "deny")
        self.assertEqual(first["data"]["create"]["toolConfig"], {"connect": {"id": "tool-123"}})


if __name__ == "__main__":
    unittest.main()
