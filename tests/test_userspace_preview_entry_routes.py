from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from starlette.requests import Request

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.userspace.models import UserSpacePreviewWarning

_workspace_preview_entry_url = getattr(_RUNTIME_ROUTES, "_workspace_preview_entry_url")
issue_workspace_preview_launch = _RUNTIME_ROUTES.issue_workspace_preview_launch
workspace_preview_entry = _RUNTIME_ROUTES.workspace_preview_entry
UserSpacePreviewLaunchResponse = _RUNTIME_ROUTES.UserSpacePreviewLaunchResponse


def _build_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [(b"host", b"ragtime.dev.visnovsky.us")],
            "scheme": "https",
            "server": ("ragtime.dev.visnovsky.us", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


class _FakeRuntimeService:
    def __init__(self) -> None:
        self.describe_calls: list[tuple[str, str, str]] = []
        self.issue_calls: list[tuple[str, str, str, str, str | None]] = []

    async def describe_workspace_preview_launch(
        self,
        workspace_id: str,
        user_id: str,
        *,
        control_plane_origin: str,
        session_expires_at=None,
    ):
        self.describe_calls.append((workspace_id, user_id, control_plane_origin))
        return (
            "https://workspace-id.ragtime.dev.visnovsky.us",
            datetime(2026, 4, 23, tzinfo=timezone.utc),
            UserSpacePreviewWarning(
                issue_code="preview_host_unreachable",
                title="probe failed",
                dismiss_key="preview-warning",
            ),
        )

    async def issue_workspace_preview_launch(
        self,
        workspace_id: str,
        user_id: str,
        *,
        control_plane_origin: str,
        path: str = "/",
        parent_origin: str | None = None,
    ):
        self.issue_calls.append(
            (workspace_id, user_id, control_plane_origin, path, parent_origin)
        )
        return UserSpacePreviewLaunchResponse(
            workspace_id=workspace_id,
            preview_origin="https://workspace-id.ragtime.dev.visnovsky.us",
            preview_url="https://workspace-id.ragtime.dev.visnovsky.us/__ragtime/bootstrap?grant=test",
            expires_at=datetime(2026, 4, 23, tzinfo=timezone.utc),
        )


class PreviewEntryRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_preview_entry_url_omits_blank_parent_origin(self) -> None:
        self.assertEqual(
            _workspace_preview_entry_url(
                "https://ragtime.dev.visnovsky.us",
                "workspace-id",
                "/preview",
                "   ",
            ),
            "https://ragtime.dev.visnovsky.us/indexes/userspace/runtime/workspaces/workspace-id/preview-entry?path=%2Fpreview",
        )

    async def test_workspace_preview_launch_uses_entry_url_without_minting_grant(self) -> None:
        fake_runtime_service = _FakeRuntimeService()
        request = _build_request("/indexes/userspace/runtime/workspaces/workspace-id/preview-launch")

        with mock.patch.object(
            _RUNTIME_ROUTES,
            "_runtime_service",
            return_value=fake_runtime_service,
        ), mock.patch.object(
            _RUNTIME_ROUTES,
            "get_browser_matched_origin",
            return_value="https://ragtime.dev.visnovsky.us",
        ):
            response = await issue_workspace_preview_launch(
                "workspace-id",
                SimpleNamespace(path="/preview", parent_origin=None),
                request,
                SimpleNamespace(id="user-123"),
                None,
            )

        self.assertEqual(fake_runtime_service.describe_calls, [("workspace-id", "user-123", "https://ragtime.dev.visnovsky.us")])
        self.assertEqual(fake_runtime_service.issue_calls, [])
        self.assertEqual(
            response.preview_url,
            "https://ragtime.dev.visnovsky.us/indexes/userspace/runtime/workspaces/workspace-id/preview-entry?path=%2Fpreview",
        )
        self.assertEqual(
            response.preview_origin,
            "https://workspace-id.ragtime.dev.visnovsky.us",
        )
        self.assertIsNone(response.preview_warning)

    async def test_workspace_preview_entry_redirects_to_bootstrap_url(self) -> None:
        fake_runtime_service = _FakeRuntimeService()
        request = _build_request("/indexes/userspace/runtime/workspaces/workspace-id/preview-entry")

        with mock.patch.object(
            _RUNTIME_ROUTES,
            "_runtime_service",
            return_value=fake_runtime_service,
        ), mock.patch.object(
            _RUNTIME_ROUTES,
            "get_browser_matched_origin",
            return_value="https://ragtime.dev.visnovsky.us",
        ):
            response = await workspace_preview_entry(
                "workspace-id",
                request,
                path="/preview",
                parent_origin=None,
                user=SimpleNamespace(id="user-123"),
            )

        self.assertEqual(response.status_code, 307)
        self.assertEqual(
            response.headers["location"],
            "https://workspace-id.ragtime.dev.visnovsky.us/__ragtime/bootstrap?grant=test",
        )
        self.assertEqual(
            fake_runtime_service.issue_calls,
            [
                (
                    "workspace-id",
                    "user-123",
                    "https://ragtime.dev.visnovsky.us",
                    "/preview",
                    None,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()