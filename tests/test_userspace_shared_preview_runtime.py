import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class SharedPreviewRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_preview_auto_starts_runtime_for_workspace_owner(self) -> None:
        session = SimpleNamespace(workspace_id="workspace-1")
        workspace_table = SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(ownerUserId="owner-1")))
        service = UserSpaceRuntimeService()
        service._ensure_session_row = mock.AsyncMock(return_value=session)  # type: ignore[method-assign]
        service._cache_preview_upstream_session = mock.AsyncMock()  # type: ignore[method-assign]

        with mock.patch(
            "ragtime.userspace.runtime_service.get_db",
            new=mock.AsyncMock(return_value=SimpleNamespace(workspace=workspace_table)),
        ):
            result = await service.ensure_shared_preview_session("workspace-1")

        self.assertIs(result, session)
        service._ensure_session_row.assert_awaited_once_with(
            "workspace-1",
            "owner-1",
            auto_start=True,
        )
        service._cache_preview_upstream_session.assert_awaited_once_with(session)
