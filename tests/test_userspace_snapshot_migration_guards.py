import sys
import types
import unittest
from pathlib import Path
from unittest import mock

fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _fake_ensure_copilot_token_fresh(*_args, **_kwargs):
    return None


setattr(fake_copilot_auth, "ensure_copilot_token_fresh", _fake_ensure_copilot_token_fresh)
sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

from fastapi import HTTPException

from ragtime.userspace.models import UserSpaceSnapshotDiffSummaryResponse
from ragtime.userspace.service import UserSpaceService


class _FakeDb:
    def __init__(self, responses):
        self._responses = list(responses)

    async def query_raw(self, _query: str):
        if not self._responses:
            return []
        return self._responses.pop(0)

    async def execute_raw(self, _query: str):
        return None


class UserSpaceSnapshotMigrationGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_workspace_git_repo_reinitializes_when_snapshot_metadata_exists(self) -> None:
        service = UserSpaceService()

        with self.subTest("snapshot rows present"):
            with (
                mock.patch.object(service, "_workspace_files_dir", return_value=Path("/tmp/workspace-files")),
                mock.patch.object(service, "_workspace_git_dir", return_value=Path("/tmp/workspace-files/.git")),
                mock.patch("pathlib.Path.exists", return_value=False),
                mock.patch.object(service, "_run_git", new=mock.AsyncMock()) as run_git,
                mock.patch.object(service, "_ensure_workspace_git_identity", new=mock.AsyncMock()),
                mock.patch.object(service, "_ensure_workspace_gitignore"),
            ):
                await service._ensure_workspace_git_repo("workspace-1")

            run_git.assert_awaited_once_with("workspace-1", ["init"])

    async def test_snapshot_diff_summary_returns_unavailable_warning_for_missing_commit(self) -> None:
        service = UserSpaceService()
        snapshot_row = {"git_commit_hash": "718b30af6313323f1e88e41e6e5d230565830c02"}

        with (
            mock.patch.object(service, "_get_snapshot_record", new=mock.AsyncMock(return_value=snapshot_row)),
            mock.patch.object(
                service,
                "_prepare_snapshot_diff_context",
                new=mock.AsyncMock(
                    side_effect=HTTPException(
                        status_code=409,
                        detail="Snapshot commit is no longer available in the current workspace repository.",
                    )
                ),
            ),
        ):
            result = await service.get_snapshot_diff_summary("workspace-1", "snapshot-1", "user-1")

        self.assertIsInstance(result, UserSpaceSnapshotDiffSummaryResponse)
        self.assertFalse(result.available)
        self.assertEqual(result.files, [])
        self.assertEqual(result.snapshot_commit_hash, snapshot_row["git_commit_hash"])
        self.assertIn("no longer available", result.warning or "")
