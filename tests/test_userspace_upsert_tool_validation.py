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
    async def _tool(self, name: str):
        tools = await RAGComponents()._create_userspace_file_tools("workspace-1", "user-1")
        for tool in tools:
            if tool.name == name:
                return tool
        raise AssertionError(f"{name} tool not found")

    async def _upsert_tool(self):
        return await self._tool("upsert_userspace_file")

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

    async def test_discover_userspace_primitives_reuses_primitive_payload_helpers(self) -> None:
        tool = await self._tool("discover_userspace_primitives")
        coroutine = tool.coroutine
        assert coroutine is not None

        capabilities_payload = {
            "workspace_id": "workspace-1",
            "endpoints": {
                "capabilities": "/__ragtime/capabilities",
                "session": "/__ragtime/session",
            },
        }
        session_payload = {
            "workspace_id": "workspace-1",
            "user_fingerprint": "fp_test",
            "auth": {
                "methods": [{"key": "ldap", "available": True}],
                "browser_auth_endpoint": "/__ragtime/browser-auth",
            },
        }
        with (
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_capabilities",
                new_callable=mock.AsyncMock,
                return_value=capabilities_payload,
            ) as primitive_capabilities,
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_session_payload",
                new_callable=mock.AsyncMock,
                return_value=session_payload,
            ) as primitive_session,
        ):
            raw = await coroutine(include_session=True)

        payload = json.loads(raw)
        self.assertLess(len(raw), 6000)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool"], "discover_userspace_primitives")
        self.assertEqual(payload["workspace_id"], "workspace-1")
        self.assertEqual(payload["next_best_tool"], "patch_userspace_file")
        self.assertIn("/__ragtime/session", raw)
        self.assertIn("/__ragtime/browser-auth", raw)
        self.assertIn("user_fingerprint", raw)
        self.assertEqual(payload["capabilities"], capabilities_payload)
        self.assertEqual(payload["session"], session_payload)
        primitive_capabilities.assert_awaited_once_with("workspace-1", "user-1", preview_mode="workspace")
        primitive_session.assert_awaited_once_with("workspace-1", "user-1", mode="workspace")


if __name__ == "__main__":
    unittest.main()
