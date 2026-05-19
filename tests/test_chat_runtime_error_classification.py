import sys
import types
import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from ragtime.rag.components import (
    RAGComponents,
    RequestLLMResolution,
    _is_userspace_workspace_not_found,
)
from ragtime.userspace.service import UserSpaceService

if "ragtime.indexer.service" not in sys.modules:
    fake_indexer_service = types.ModuleType("ragtime.indexer.service")
    setattr(fake_indexer_service, "IndexerService", object)
    sys.modules["ragtime.indexer.service"] = fake_indexer_service


class ChatRuntimeErrorClassificationTests(unittest.TestCase):
    def test_platform_workspace_404_is_not_labeled_as_provider_failure(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )
        exc = HTTPException(status_code=404, detail="Workspace not found")

        self.assertTrue(_is_userspace_workspace_not_found(exc))

        message = RAGComponents()._chat_runtime_error_message(exc, resolution)

        self.assertIn("workspace this chat tried to access", message)
        self.assertNotIn("GitHub Copilot", message)
        self.assertNotIn("gpt-5.4", message)

    def test_non_userspace_exception_preserves_existing_contextual_message(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )
        exc = RuntimeError("404: Workspace not found")

        self.assertFalse(_is_userspace_workspace_not_found(exc))

        message = RAGComponents()._chat_runtime_error_message(exc, resolution)

        self.assertIn("I encountered an error processing your request", message)
        self.assertIn("while using GitHub Copilot for model 'gpt-5.4'", message)
        self.assertIn("404: Workspace not found", message)

    def test_unknown_error_preserves_existing_contextual_message(self) -> None:
        resolution = RequestLLMResolution(
            llm=object(),
            provider="github_copilot",
            model="gpt-5.4",
            attempted_providers=("github_copilot",),
        )

        message = RAGComponents()._chat_runtime_error_message(
            RuntimeError("boom"),
            resolution,
        )

        self.assertIn("I encountered an error processing your request", message)
        self.assertIn("while using GitHub Copilot for model 'gpt-5.4'", message)
        self.assertIn("boom", message)

    def test_userspace_runtime_hint_includes_live_data_warning(self) -> None:
        status = SimpleNamespace(
            session_state="running",
            runtime_operation_phase=None,
            devserver_running=True,
            last_error=None,
            live_data_warning="Error: column invoice_created_date does not exist",
        )

        hint = RAGComponents._build_userspace_runtime_status_turn_hint(status)

        self.assertIn("WARNING Possible live data query issue", hint)
        self.assertIn("column invoice_created_date does not exist", hint)


class CrossWorkspaceResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_granted_target_without_user_access_becomes_tool_denial(self) -> None:
        service = UserSpaceService()

        async def deny_workspace_access(*_args, **_kwargs):
            raise HTTPException(status_code=404, detail="Workspace not found")

        service._enforce_workspace_access = deny_workspace_access  # type: ignore[method-assign]

        with self.assertRaisesRegex(ValueError, "current user"):
            await service.resolve_cross_workspace_target(
                source_workspace_id="source-workspace",
                target_workspace_id="target-workspace",
                user_id="user-id",
                accessible_modes={"target-workspace": "read"},
                action="read",
            )


if __name__ == "__main__":
    unittest.main()
