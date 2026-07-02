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

from ragtime.rag.components import (
    FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
    RAGComponents,
    should_truncate_stream_display_output,
    wrap_tool_with_truncation,
)
from ragtime.rag.prompts import _USERSPACE_MODE_PROMPT_TEMPLATE, _USERSPACE_TURN_REMINDER_BASE, build_workspace_scm_setup_prompt


class UserSpaceUpsertToolValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _tool(self, name: str):
        tools = await RAGComponents()._create_userspace_file_tools("workspace-1", "user-1")
        for tool in tools:
            if tool.name == name:
                return tool
        raise AssertionError(f"{name} tool not found")

    async def _upsert_tool(self):
        return await self._tool("upsert_userspace_file")

    async def test_assay_userspace_code_defaults_are_compact_but_preserve_diagnostics(self) -> None:
        tool = await self._tool("assay_userspace_code")
        coroutine = tool.coroutine
        assert coroutine is not None

        schema = tool.args_schema
        assert schema is not None
        self.assertEqual(schema.model_fields["max_files"].default, 6)
        self.assertEqual(schema.model_fields["max_chars_per_file"].default, 800)
        self.assertIn("structure", tool.description)
        self.assertIn("contract", tool.description)
        self.assertIn("search_userspace_code", tool.description)

        files = [
            types.SimpleNamespace(path="dashboard/main.ts"),
            *[types.SimpleNamespace(path=f"dashboard/file{i}.ts") for i in range(1, 8)],
            types.SimpleNamespace(path="index.html"),
        ]

        async def get_workspace_file(_workspace_id: str, path: str, _user_id: str, **_kwargs):
            return types.SimpleNamespace(
                path=path,
                artifact_type="module_ts" if path.endswith(".ts") else "html",
                content=(f"// {path}\n" + "x" * 1200),
                live_data_connections=[],
                live_data_checks=[],
            )

        with (
            mock.patch(
                "ragtime.rag.components.userspace_service.enforce_workspace_role",
                new_callable=mock.AsyncMock,
            ) as enforce_workspace_role,
            mock.patch(
                "ragtime.rag.components.userspace_service.list_workspace_files",
                new_callable=mock.AsyncMock,
                return_value=files,
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_file",
                new_callable=mock.AsyncMock,
                side_effect=get_workspace_file,
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace",
                new_callable=mock.AsyncMock,
                return_value=types.SimpleNamespace(selected_tool_ids=["tool-1"]),
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_entrypoint_status",
                return_value=types.SimpleNamespace(state="valid", framework="node", error=None),
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.is_default_static_entrypoint",
                return_value=False,
            ),
        ):
            raw = await coroutine()

        payload = json.loads(raw)
        workspace = payload["workspace"]
        self.assertEqual(payload["tool"], "assay_userspace_code")
        self.assertEqual(workspace["summary"]["inspected_file_count"], 6)
        self.assertEqual(len(workspace["inspected_files"]), 6)
        self.assertLessEqual(len(workspace["inspected_files"][0]["preview"]), 800)
        self.assertTrue(workspace["structure"]["has_dashboard_entry"])
        self.assertTrue(workspace["structure"]["has_index_html"])
        self.assertEqual(workspace["structure"]["authoritative_entrypoint"], ".ragtime/runtime-entrypoint.json")
        self.assertTrue(workspace["live_data_contract"]["workspace_has_selected_tools"])
        self.assertEqual(workspace["live_data_contract"]["selected_tool_ids"], ["tool-1"])
        enforce_workspace_role.assert_awaited_once_with("workspace-1", "user-1", "editor")

    def test_userspace_prompt_guidance_splits_assay_from_indexed_search_with_fallback(self) -> None:
        scm_prompt = build_workspace_scm_setup_prompt(git_url="https://example.test/repo.git", git_branch="main")

        self.assertIn("assay_userspace_code to assess the current structure", scm_prompt)
        self.assertIn("search_userspace_code", scm_prompt)
        self.assertIn("when the code index is unavailable", scm_prompt)
        self.assertIn("list_userspace_files plus targeted reads", scm_prompt)
        self.assertIn("assay (structure/contract) -> search/read", _USERSPACE_TURN_REMINDER_BASE)
        self.assertIn("Use `assay_userspace_code` for structure and contract context", _USERSPACE_MODE_PROMPT_TEMPLATE)

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
        primitive_session.assert_awaited_once_with("workspace-1", "user-1", mode="workspace", same_origin_auth_endpoints=True)

    async def test_frontend_json_display_tools_bypass_global_output_truncation(self) -> None:
        tool = await self._tool("discover_userspace_primitives")

        wrapped = wrap_tool_with_truncation(
            tool,
            32,
            preserve_output_tool_names=FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
        )

        self.assertIs(wrapped, tool)

    def test_frontend_json_display_tools_bypass_stream_display_truncation(self) -> None:
        output = json.dumps({"tool": "discover_userspace_primitives", "payload": "x" * 3000}, indent=2)

        for tool_name in FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES:
            self.assertFalse(
                should_truncate_stream_display_output(tool_name, output),
                tool_name,
            )

        self.assertTrue(should_truncate_stream_display_output("generic_tool", output))

    async def test_search_userspace_code_tool_delegates_to_workspace_index_service(self) -> None:
        tool = await self._tool("search_userspace_code")
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.workspace_code_index_service.search_workspace_code",
            new_callable=mock.AsyncMock,
            return_value={"status": "ready", "results": [{"path": "src/app.py"}]},
        ) as search_workspace_code:
            raw = await coroutine(query="where is startup configured?", max_results=3)

        payload = json.loads(raw)
        self.assertEqual(payload["tool"], "search_userspace_code")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["results"], [{"path": "src/app.py"}])
        search_workspace_code.assert_awaited_once_with(
            workspace_id="workspace-1",
            query="where is startup configured?",
            mode="hybrid",
            max_results=3,
            max_chars_per_result=1200,
        )


if __name__ == "__main__":
    unittest.main()
