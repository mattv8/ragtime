from __future__ import annotations

import unittest
from unittest import mock

from ragtime.userspace.service import UserSpaceService


class RuntimeGuardedCodeRestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_guarded_restore_holds_parent_lease_for_git_and_finishes(self) -> None:
        service = UserSpaceService.__new__(UserSpaceService)
        history = mock.Mock(runtime_history_active=mock.AsyncMock(return_value=True))
        history._runtime_history_request = mock.AsyncMock(return_value={"status": "completed"})
        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(service, "_workspace_files_dir", return_value=mock.Mock()),
            mock.patch(
                "ragtime.userspace.service.runtime_manager_request", new=mock.AsyncMock(return_value={"returncode": 0, "stdout_b64": "", "stderr_b64": ""})
            ) as git,
        ):
            async with service._guarded_code_restore("workspace-1"):
                await service._run_git("workspace-1", ["status"])

        self.assertEqual("POST", history._runtime_history_request.await_args_list[0].args[0])
        self.assertIn("/begin", history._runtime_history_request.await_args_list[0].args[1])
        self.assertIn("/finish", history._runtime_history_request.await_args_list[1].args[1])
        self.assertIsNotNone(git.await_args)
        await_args = git.await_args
        assert await_args is not None
        self.assertTrue(await_args.kwargs["json_payload"]["sqlite_history_operation_id"])
