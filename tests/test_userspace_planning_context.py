"""Tests for the curated external-agent planning context."""

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from ragtime.rag.prompts import _WORKSPACE_CONTINUITY_EXISTING_RULES


def _fake_workspace() -> SimpleNamespace:
    return SimpleNamespace(
        id="ws-1",
        name="Sales Dashboard",
        description="Revenue tracking",
        sqlite_persistence_mode="include",
        tool_selection_mode="custom",
        selected_tool_ids=["tool-1"],
        selected_tool_group_ids=[],
        tool_options={},
        owner_user_id="user-1",
        members=[],
        updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )


def _fake_files():
    return [
        SimpleNamespace(path="index.html", size_bytes=100, updated_at=None, entry_type="file"),
        SimpleNamespace(path="dashboard/main.ts", size_bytes=2000, updated_at=None, entry_type="file"),
        SimpleNamespace(path=".ragtime/runtime-entrypoint.json", size_bytes=50, updated_at=None, entry_type="file"),
    ]


def _fake_entrypoint():
    return SimpleNamespace(
        state="valid",
        framework="node",
        command="npx esbuild dashboard/main.ts --serve=0.0.0.0:$PORT",
        cwd=".",
        error=None,
    )


def _fake_tool_config():
    return SimpleNamespace(
        id="tool-1",
        name="Production DB",
        tool_type=SimpleNamespace(value="postgres"),
        description="Main production database",
        allow_write=False,
        group_id=None,
        group_name=None,
        connection_config={"password": "SECRET-VALUE"},
    )


class PlanningContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from ragtime.userspace import planning_service as module

        self.module = module
        self.service = module.planning_service

    def _patches(self):
        from ragtime.userspace.service import userspace_service

        return [
            mock.patch.object(
                userspace_service,
                "enforce_workspace_role",
                mock.AsyncMock(return_value=_fake_workspace()),
            ),
            mock.patch.object(
                userspace_service,
                "list_workspace_files",
                mock.AsyncMock(return_value=_fake_files()),
            ),
            mock.patch.object(
                userspace_service,
                "get_workspace_entrypoint_status",
                mock.Mock(return_value=_fake_entrypoint()),
            ),
            mock.patch.object(
                userspace_service,
                "is_default_static_entrypoint",
                mock.Mock(return_value=False),
            ),
            mock.patch.object(
                userspace_service,
                "list_snapshots",
                mock.AsyncMock(return_value=[SimpleNamespace(message="init", created_at=None)]),
            ),
            mock.patch.object(
                self.module,
                "resolve_effective_tool_ids",
                mock.AsyncMock(return_value=["tool-1"]),
            ),
            mock.patch.object(
                self.module.repository,
                "list_tool_configs",
                mock.AsyncMock(return_value=[_fake_tool_config()]),
            ),
            mock.patch.object(
                self.module,
                "get_db",
                mock.AsyncMock(
                    return_value=SimpleNamespace(
                        user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user"))),
                    )
                ),
            ),
        ]

    async def test_context_shape(self) -> None:
        patches = self._patches()
        for p in patches:
            p.start()
        try:
            context = await self.service.get_workspace_context("ws-1", "user-1")
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(context["workspace"]["name"], "Sales Dashboard")
        self.assertEqual(context["architecture"]["framework"], "node")
        self.assertEqual(context["architecture"]["file_count"], 2)
        self.assertNotIn(".ragtime/runtime-entrypoint.json", context["architecture"]["key_files"])
        self.assertEqual(context["selected_tools"][0]["component_id"], "tool-1")
        self.assertTrue(context["context_revision"])

    async def test_context_never_leaks_secrets(self) -> None:
        patches = self._patches()
        for p in patches:
            p.start()
        try:
            context = await self.service.get_workspace_context("ws-1", "user-1")
        finally:
            for p in patches:
                p.stop()

        dumped = json.dumps(context, default=str)
        self.assertNotIn("SECRET-VALUE", dumped)
        self.assertNotIn("connection_config", dumped)
        self.assertNotIn("enc::", dumped)

    async def test_builder_contract_matches_internal_prompts(self) -> None:
        from ragtime.userspace.planning_contract import build_builder_contract, continuity_rules

        contract = build_builder_contract(sqlite_persistence_mode="include", has_live_data_tools=True)
        joined = "\n".join(contract["rules"])
        self.assertIn("$PORT", joined)
        self.assertIn("0.0.0.0", joined)
        for rule in _WORKSPACE_CONTINUITY_EXISTING_RULES:
            self.assertIn(rule.lstrip("- ").strip(), contract["rules"])
        self.assertEqual(continuity_rules(), [r.lstrip("- ").strip() for r in _WORKSPACE_CONTINUITY_EXISTING_RULES])


class PlanningFileReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_file_bounds_and_pagination(self) -> None:
        from ragtime.userspace import planning_service as module
        from ragtime.userspace.service import userspace_service

        content = "\n".join(f"line {i}" for i in range(1, 101))
        fake_response = SimpleNamespace(path="app.py", content=content)
        with (
            mock.patch.object(type(module.planning_service), "_is_admin", mock.AsyncMock(return_value=False)),
            mock.patch.object(
                userspace_service,
                "get_workspace_file",
                mock.AsyncMock(return_value=fake_response),
            ),
        ):
            result = await module.planning_service.read_file("ws-1", "user-1", "app.py", start_line=1, max_lines=10)
        self.assertEqual(result["total_lines"], 100)
        self.assertEqual(result["end_line"], 10)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["next_start_line"], 11)
        self.assertIn("line 10", result["content"])
        self.assertNotIn("line 11", result["content"])

    async def test_list_files_filters_prefix_and_reserved(self) -> None:
        from ragtime.userspace import planning_service as module
        from ragtime.userspace.service import userspace_service

        with (
            mock.patch.object(type(module.planning_service), "_is_admin", mock.AsyncMock(return_value=False)),
            mock.patch.object(
                userspace_service,
                "list_workspace_files",
                mock.AsyncMock(return_value=_fake_files()),
            ),
        ):
            result = await module.planning_service.list_files("ws-1", "user-1", prefix="dashboard/")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["files"][0]["path"], "dashboard/main.ts")

    async def test_character_limit_keeps_pagination_moving(self) -> None:
        from ragtime.userspace import planning_service as module
        from ragtime.userspace.service import userspace_service

        fake_response = SimpleNamespace(path="data.txt", content=("x" * 300) + "\nsecond")
        with (
            mock.patch.object(type(module.planning_service), "_is_admin", mock.AsyncMock(return_value=False)),
            mock.patch.object(userspace_service, "get_workspace_file", mock.AsyncMock(return_value=fake_response)),
        ):
            result = await module.planning_service.read_file("ws-1", "user-1", "data.txt", max_chars=200)

        self.assertEqual(len(result["content"]), 200)
        self.assertTrue(result["line_clipped"])
        self.assertEqual(result["end_line"], 1)
        self.assertEqual(result["next_start_line"], 2)

    async def test_admin_identity_is_forwarded_to_file_service(self) -> None:
        from ragtime.userspace import planning_service as module
        from ragtime.userspace.service import userspace_service

        list_mock = mock.AsyncMock(return_value=[])
        read_mock = mock.AsyncMock(return_value=SimpleNamespace(path="app.py", content="ok"))
        with (
            mock.patch.object(type(module.planning_service), "_is_admin", mock.AsyncMock(return_value=True)),
            mock.patch.object(userspace_service, "list_workspace_files", list_mock),
            mock.patch.object(userspace_service, "get_workspace_file", read_mock),
        ):
            await module.planning_service.list_files("ws-1", "admin-1")
            await module.planning_service.read_file("ws-1", "admin-1", "app.py")

        list_mock.assert_awaited_once_with("ws-1", "admin-1", is_admin=True)
        read_mock.assert_awaited_once_with("ws-1", "app.py", "admin-1", is_admin=True)


if __name__ == "__main__":
    unittest.main()
