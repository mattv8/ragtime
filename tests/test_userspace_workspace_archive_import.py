import sys
import types
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    fake_prompts_module.build_workspace_scm_setup_prompt = lambda *args, **kwargs: ""
    fake_rag_package.prompts = fake_prompts_module
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.service import UserSpaceService


class _FakeToolConfigTable:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._by_id = {str(record.id): record for record in records}

    async def find_unique(self, *, where: dict[str, str]) -> Optional[SimpleNamespace]:
        return self._by_id.get(str(where.get("id") or ""))


class _FakeToolGroupTable:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._by_id = {str(record.id): record for record in records}

    async def find_unique(self, *, where: dict[str, str]) -> Optional[SimpleNamespace]:
        return self._by_id.get(str(where.get("id") or ""))


class _CaptureTable:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.updated: list[dict[str, object]] = []

    async def create(self, *, data: dict[str, object]) -> SimpleNamespace:
        self.created.append(data)
        return SimpleNamespace(**data)

    async def update(
        self,
        *,
        where: dict[str, object],
        data: dict[str, object],
    ) -> SimpleNamespace:
        self.updated.append({"where": where, "data": data})
        return SimpleNamespace(**data)


class WorkspaceArchiveImportTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_workspace_archive_selection_id_sets_keeps_exact_ids_only(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            toolconfig=_FakeToolConfigTable(
                [
                    SimpleNamespace(id="tool-a", enabled=True),
                    SimpleNamespace(id="tool-disabled", enabled=False),
                ]
            ),
            toolgroup=_FakeToolGroupTable([SimpleNamespace(id="group-a")]),
        )
        manifest = {
            "workspace": {
                "selected_tool_ids": ["tool-a", "tool-disabled", "tool-missing"],
                "selected_tool_group_ids": ["group-a", "group-missing"],
            },
            "chats": [
                {
                    "tool_config_ids": ["tool-a", "tool-disabled", "tool-missing"],
                    "tool_group_ids": ["group-a", "group-missing"],
                }
            ],
        }

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            (
                allowed_tool_ids,
                allowed_tool_group_ids,
                warnings,
            ) = await service._resolve_workspace_archive_selection_id_sets(manifest)

        self.assertEqual(allowed_tool_ids, {"tool-a"})
        self.assertEqual(allowed_tool_group_ids, {"group-a"})
        self.assertTrue(
            any(
                "Skipped 2 archived tool selection references" in warning
                for warning in warnings
            )
        )
        self.assertTrue(
            any(
                "Skipped 1 archived tool group reference" in warning
                for warning in warnings
            )
        )

    async def test_import_workspace_chat_payloads_filters_to_allowed_exact_ids(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            conversation=_CaptureTable(),
            conversationtoolselection=_CaptureTable(),
            conversationtoolgroupselection=_CaptureTable(),
            conversationbranch=_CaptureTable(),
        )
        chat_payloads = [
            {
                "title": "Imported chat",
                "messages": [],
                "tool_config_ids": ["tool-a", "tool-missing", "tool-a"],
                "tool_group_ids": ["group-a", "group-missing", "group-a"],
                "branches": [],
            }
        ]

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            imported_count = await service._import_workspace_chat_payloads(
                "workspace-1",
                "user-1",
                chat_payloads,
                allowed_tool_config_ids={"tool-a"},
                allowed_tool_group_ids={"group-a"},
            )

        self.assertEqual(imported_count, 1)
        self.assertEqual(len(fake_db.conversation.created), 1)
        self.assertEqual(
            [row["toolConfigId"] for row in fake_db.conversationtoolselection.created],
            ["tool-a"],
        )
        self.assertEqual(
            [
                row["toolGroupId"]
                for row in fake_db.conversationtoolgroupselection.created
            ],
            ["group-a"],
        )


if __name__ == "__main__":
    unittest.main()
