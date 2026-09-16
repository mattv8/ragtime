import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class UserSpaceExecTimeoutEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = UserSpaceRuntimeService()
        self.enforce = mock.patch(
            "ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role",
            new=mock.AsyncMock(),
        )
        self.enforce.start()

    async def asyncTearDown(self) -> None:
        self.enforce.stop()

    async def test_omitted_timeout_uses_current_admin_default_before_runtime_start(self) -> None:
        ensure = mock.AsyncMock(return_value=SimpleNamespace(provider_session_id="session-1"))
        refresh = mock.AsyncMock()
        provider = mock.AsyncMock(return_value={"exit_code": 0})
        with (
            mock.patch(
                "ragtime.userspace.runtime_service.get_app_settings",
                new=mock.AsyncMock(return_value={"userspace_exec_timeout_default_seconds": 180, "userspace_exec_timeout_max_seconds": 1800}),
            ),
            mock.patch.object(self.service, "ensure_workspace_preview_session", ensure),
            mock.patch.object(self.service, "_refresh_runtime_mounts_if_specs_changed", refresh),
            mock.patch.object(self.service, "_runtime_provider_exec_command", provider),
        ):
            await self.service.exec_workspace_command("workspace", "user", "npm run build")

        provider.assert_awaited_once_with("session-1", "npm run build", timeout_seconds=180, cwd=None)

    async def test_explicit_timeout_receives_transport_grace_without_retry(self) -> None:
        with mock.patch.object(self.service, "_runtime_manager_request", new=mock.AsyncMock(return_value={})) as request:
            await self.service._runtime_provider_exec_command("session-1", "npm run build", timeout_seconds=1800)

        request_args = request.await_args
        assert request_args is not None
        self.assertEqual(request_args.kwargs["timeout_override_seconds"], 1830.0)
        self.assertFalse(request_args.kwargs["retry_safe"])

    async def test_invalid_timeout_is_rejected_before_runtime_start(self) -> None:
        ensure = mock.AsyncMock()
        with (
            mock.patch(
                "ragtime.userspace.runtime_service.get_app_settings",
                new=mock.AsyncMock(return_value={"userspace_exec_timeout_default_seconds": 120, "userspace_exec_timeout_max_seconds": 600}),
            ),
            mock.patch.object(self.service, "ensure_workspace_preview_session", ensure),
            self.assertRaisesRegex(HTTPException, "between 1 and 600"),
        ):
            await self.service.exec_workspace_command("workspace", "user", "npm run build", timeout_seconds=601)

        ensure.assert_not_awaited()

    async def test_boolean_and_fractional_timeouts_are_rejected_before_runtime_start(self) -> None:
        ensure = mock.AsyncMock()
        with (
            mock.patch("ragtime.userspace.runtime_service.get_app_settings", new=mock.AsyncMock(return_value={})),
            mock.patch.object(self.service, "ensure_workspace_preview_session", ensure),
        ):
            for timeout in (True, 12.5, "NaN"):
                with self.assertRaisesRegex(HTTPException, "must be an integer"):
                    await self.service.exec_workspace_command("workspace", "user", "pwd", timeout_seconds=timeout)

        ensure.assert_not_awaited()
