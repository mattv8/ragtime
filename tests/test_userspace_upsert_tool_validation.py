import json
import sys
import types
import unittest
from unittest import mock

fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _fake_ensure_copilot_token_fresh(*_args, **_kwargs):
    return None


setattr(fake_copilot_auth, "ensure_copilot_token_fresh", _fake_ensure_copilot_token_fresh)
sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

from ragtime.rag.components import RAGComponents


class UserSpaceUpsertToolValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _upsert_tool(self):
        tools = await RAGComponents()._create_userspace_file_tools("workspace-1", "user-1")
        for tool in tools:
            if tool.name == "upsert_userspace_file":
                return tool
        raise AssertionError("upsert_userspace_file tool not found")

    async def test_single_file_upsert_rejects_missing_content_without_writing(self) -> None:
        tool = await self._upsert_tool()
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.userspace_service.upsert_workspace_file",
            new_callable=mock.AsyncMock,
        ) as upsert_workspace_file:
            raw = await coroutine(path="dashboard/main.ts")

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "rejected_not_persisted")
        self.assertEqual(payload["failure_class"], "content_missing")
        self.assertEqual(payload["next_best_tool"], "upsert_userspace_file")
        self.assertTrue(payload["rejected"])
        self.assertFalse(payload["persisted"])
        upsert_workspace_file.assert_not_called()

    async def test_batched_upsert_rejects_entry_missing_content_without_writing(self) -> None:
        tool = await self._upsert_tool()
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.userspace_service.upsert_workspace_file",
            new_callable=mock.AsyncMock,
        ) as upsert_workspace_file:
            raw = await coroutine(files=[{"path": "dashboard/main.ts"}])

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "rejected_not_persisted")
        self.assertEqual(payload["failure_class"], "content_missing")
        self.assertEqual(payload["next_best_tool"], "upsert_userspace_file")
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertEqual(payload["summary"]["rejected"], 1)
        self.assertEqual(payload["files"][0]["failure_class"], "content_missing")
        upsert_workspace_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
