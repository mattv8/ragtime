import asyncio
import unittest
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

from ragtime.userspace.models import (
    UserSpaceWorkspaceScmImportRequest,
    UserSpaceWorkspaceScmPreviewResponse,
    UserSpaceWorkspaceScmStatus,
    UserSpaceWorkspaceScmSyncResponse,
    WorkspaceScmPreviewState,
)
from ragtime.userspace.service import UserSpaceService


def _workspace(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": "workspace-1",
        "scmGitUrl": "https://git.example/owner/repo.git",
        "scmGitBranch": "main",
        "scmProvider": "github",
        "scmRepoVisibility": "private",
        "scmRemoteRole": "upstream",
        "scmAutoPullEnabled": True,
        "scmSyncPaused": False,
        "scmWebhookPaused": False,
        "ownerUserId": "owner-1",
        "scmLastRemoteCommitHash": "remote-1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _workspace_db(workspace: SimpleNamespace | None):
    class _WorkspaceClient:
        def __init__(self, row: SimpleNamespace | None) -> None:
            self._row = row

        async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace | None:
            if where != {"id": "workspace-1"}:
                raise AssertionError(f"unexpected where clause: {where}")
            return self._row

    return SimpleNamespace(workspace=_WorkspaceClient(workspace))


def _preview(state: WorkspaceScmPreviewState = "safe") -> UserSpaceWorkspaceScmPreviewResponse:
    return UserSpaceWorkspaceScmPreviewResponse(
        workspace_id="workspace-1",
        direction="import",
        state=state,
        summary=f"preview:{state}",
        git_url="https://git.example/owner/repo.git",
        git_branch="main",
        provider="github",
        repo_visibility="private",
        remote_commit_hash="remote-2",
    )


def _sync_response(summary: str = "Imported") -> UserSpaceWorkspaceScmSyncResponse:
    return UserSpaceWorkspaceScmSyncResponse(
        workspace_id="workspace-1",
        direction="import",
        state="success",
        summary=summary,
        scm=UserSpaceWorkspaceScmStatus(),
        snapshot=None,
        remote_commit_hash="remote-2",
        suggested_setup_prompt=None,
    )


@contextmanager
def _webhook_pull_mocks(
    service: UserSpaceService,
    *,
    preview_state: WorkspaceScmPreviewState = "safe",
    sync_response: UserSpaceWorkspaceScmSyncResponse | None = None,
):
    with (
        mock.patch.object(
            service,
            "_build_workspace_scm_preview",
            new=mock.AsyncMock(return_value=(_preview(preview_state), "fingerprint")),
        ) as preview_mock,
        mock.patch.object(
            service,
            "import_workspace_from_scm",
            new=mock.AsyncMock(return_value=sync_response or _sync_response()),
        ) as import_mock,
    ):
        yield preview_mock, import_mock


class WorkspaceScmWebhookPullTests(unittest.IsolatedAsyncioTestCase):
    async def test_webhook_pull_does_not_require_interval_auto_pull(self) -> None:
        workspace = _workspace(scmAutoPullEnabled=False)
        service = UserSpaceService()

        with (
            _webhook_pull_mocks(service) as (preview_mock, import_mock),
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(workspace)),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "imported")
        preview_mock.assert_awaited_once()
        import_mock.assert_awaited_once()
        await_args = import_mock.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        import_request = await_args.args[2]
        self.assertIsInstance(import_request, UserSpaceWorkspaceScmImportRequest)
        self.assertEqual(import_request.git_url, "https://git.example/owner/repo.git")
        self.assertEqual(import_request.git_branch, "main")

    async def test_webhook_pull_returns_paused_without_preview(self) -> None:
        workspace = _workspace(scmSyncPaused=True)
        service = UserSpaceService()

        with (
            mock.patch.object(service, "_build_workspace_scm_preview", new=mock.AsyncMock()) as preview_mock,
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(workspace)),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "paused")
        preview_mock.assert_not_called()

    async def test_webhook_pull_ignores_webhook_pause_state_and_still_imports(self) -> None:
        workspace = _workspace(scmWebhookPaused=True, scmAutoPullEnabled=False)
        service = UserSpaceService()

        with (
            _webhook_pull_mocks(service) as (preview_mock, import_mock),
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(workspace)),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "imported")
        preview_mock.assert_awaited_once()
        import_mock.assert_awaited_once()

    async def test_webhook_pull_returns_not_upstream_for_publish_remote(self) -> None:
        service = UserSpaceService()

        with mock.patch(
            "ragtime.userspace.service.get_db",
            return_value=_workspace_db(_workspace(scmRemoteRole="publish")),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "not_upstream")

    async def test_webhook_pull_returns_disconnected_when_remote_missing(self) -> None:
        service = UserSpaceService()

        with mock.patch(
            "ragtime.userspace.service.get_db",
            return_value=_workspace_db(_workspace(scmGitUrl=None)),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "disconnected")

    async def test_webhook_pull_returns_up_to_date_preview_state(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("up_to_date"), "fingerprint")),
            ),
            mock.patch.object(service, "import_workspace_from_scm", new=mock.AsyncMock()) as import_mock,
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(_workspace())),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "up_to_date")
        import_mock.assert_not_called()

    async def test_webhook_pull_returns_missing_branch_preview_state(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("missing_branch"), "fingerprint")),
            ),
            mock.patch.object(service, "import_workspace_from_scm", new=mock.AsyncMock()) as import_mock,
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(_workspace())),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "missing_branch")
        import_mock.assert_not_called()

    async def test_webhook_pull_returns_conflict_for_non_safe_preview(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("destructive"), "fingerprint")),
            ),
            mock.patch.object(service, "import_workspace_from_scm", new=mock.AsyncMock()) as import_mock,
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(_workspace())),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "conflict")
        import_mock.assert_not_called()

    async def test_scm_operation_serializes_two_callers(self) -> None:
        service = UserSpaceService()
        entered: list[str] = []
        release = asyncio.Event()
        first_entered = asyncio.Event()

        async def run(label: str) -> None:
            async with service.workspace_scm_operation("workspace-1"):
                entered.append(label)
                if label == "first":
                    first_entered.set()
                    await release.wait()

        first = asyncio.create_task(run("first"))
        await first_entered.wait()
        second = asyncio.create_task(run("second"))
        await asyncio.sleep(0)
        self.assertEqual(entered, ["first"])
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(entered, ["first", "second"])

    async def test_webhook_pull_locked_does_not_reenter_operation_context(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(
                service,
                "workspace_scm_operation",
                side_effect=AssertionError("locked pull must not reacquire the SCM operation context"),
            ),
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("safe"), "fingerprint")),
            ),
            mock.patch.object(
                service,
                "import_workspace_from_scm",
                new=mock.AsyncMock(return_value=_sync_response()),
            ),
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(_workspace())),
        ):
            outcome = await service.run_workspace_scm_webhook_pull_locked("workspace-1")

        self.assertEqual(outcome.state, "imported")

    async def test_import_workspace_from_scm_marks_code_index_after_import_finishes(self) -> None:
        service = UserSpaceService()
        call_order: list[str] = []

        async def execute_import(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            call_order.append("execute")
            return SimpleNamespace(
                task_summary="Imported",
                scm=UserSpaceWorkspaceScmStatus(),
                imported_snapshot=None,
                remote_commit_hash="remote-2",
                suggested_prompt=None,
            )

        async def mark_dirty(*_args: Any, **_kwargs: Any) -> None:
            call_order.append("mark_dirty")

        with (
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("safe"), "fingerprint")),
            ),
            mock.patch.object(service, "_maybe_store_workspace_scm_token", new=mock.AsyncMock()),
            mock.patch.object(
                service,
                "_execute_workspace_scm_import",
                new=mock.AsyncMock(side_effect=execute_import),
            ),
            mock.patch.object(
                service,
                "_mark_workspace_code_index_dirty",
                new=mock.AsyncMock(side_effect=mark_dirty),
            ),
        ):
            response = await service.import_workspace_from_scm(
                "workspace-1",
                "owner-1",
                UserSpaceWorkspaceScmImportRequest(),
            )

        self.assertEqual(response.state, "success")
        self.assertEqual(call_order, ["execute", "mark_dirty"])

    async def test_run_workspace_auto_pull_respects_interval_gating(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(service, "_run_workspace_safe_pull_locked", new=mock.AsyncMock()) as locked_pull_mock,
            mock.patch(
                "ragtime.userspace.service.get_db",
                return_value=_workspace_db(_workspace(scmAutoPullEnabled=False)),
            ),
        ):
            await service._run_workspace_auto_pull("workspace-1")

        locked_pull_mock.assert_not_called()

    async def test_run_workspace_auto_pull_delegates_to_locked_pull_when_enabled(self) -> None:
        service = UserSpaceService()

        with (
            mock.patch.object(service, "_run_workspace_safe_pull_locked", new=mock.AsyncMock()) as locked_pull_mock,
            mock.patch("ragtime.userspace.service.get_db", return_value=_workspace_db(_workspace())),
        ):
            await service._run_workspace_auto_pull("workspace-1")

        locked_pull_mock.assert_awaited_once_with("workspace-1")

    async def test_auto_sync_uses_workspace_scm_operation_once_for_import(self) -> None:
        service = UserSpaceService()
        entered: list[str] = []

        @asynccontextmanager
        async def fake_operation(workspace_id: str):
            entered.append(workspace_id)
            yield

        with (
            mock.patch.object(service, "workspace_scm_operation", new=fake_operation),
            mock.patch.object(service, "_run_workspace_auto_pull", new=mock.AsyncMock()) as auto_pull_mock,
        ):
            await service._run_workspace_scm_auto_sync("workspace-1", "import", 300)

        self.assertEqual(entered, ["workspace-1"])
        auto_pull_mock.assert_awaited_once_with("workspace-1")

    async def test_import_task_runner_uses_workspace_scm_operation_once(self) -> None:
        service = UserSpaceService()
        entered: list[str] = []

        @asynccontextmanager
        async def fake_operation(workspace_id: str):
            entered.append(workspace_id)
            yield

        with (
            mock.patch.object(service, "workspace_scm_operation", new=fake_operation),
            mock.patch.object(
                service,
                "_run_workspace_scm_import_task_body",
                new=mock.AsyncMock(),
            ) as task_body_mock,
        ):
            await service._run_workspace_scm_import_task(
                "task-1",
                "workspace-1",
                "owner-1",
                UserSpaceWorkspaceScmImportRequest(),
            )

        self.assertEqual(entered, ["workspace-1"])
        task_body_mock.assert_awaited_once()

    async def test_preview_import_task_runner_uses_workspace_scm_operation_once(self) -> None:
        service = UserSpaceService()
        entered: list[str] = []

        @asynccontextmanager
        async def fake_operation(workspace_id: str):
            entered.append(workspace_id)
            yield

        with (
            mock.patch.object(service, "workspace_scm_operation", new=fake_operation),
            mock.patch.object(service, "_update_workspace_scm_import_task_phase", new=mock.AsyncMock()) as update_phase_mock,
            mock.patch.object(
                service,
                "_build_workspace_scm_preview",
                new=mock.AsyncMock(return_value=(_preview("safe"), "fingerprint")),
            ),
            mock.patch.object(service, "_maybe_store_workspace_scm_token", new=mock.AsyncMock()),
        ):
            await service._run_workspace_scm_preview_import_task(
                "task-1",
                "workspace-1",
                "owner-1",
                UserSpaceWorkspaceScmImportRequest(),
            )

        self.assertEqual(entered, ["workspace-1"])
        update_phase_mock.assert_any_await(
            "workspace-1",
            "task-1",
            "preview_ready",
            summary="preview:safe",
            remote_commit_hash="remote-2",
            preview=_preview("safe"),
        )
