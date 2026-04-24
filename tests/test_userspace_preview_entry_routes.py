from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from pydantic import BaseModel, Field
from starlette.requests import Request


def _install_runtime_route_test_stubs() -> None:
    ragtime_pkg: Any = sys.modules.setdefault("ragtime", types.ModuleType("ragtime"))
    ragtime_pkg.__path__ = getattr(ragtime_pkg, "__path__", [])

    config_pkg: Any = sys.modules.setdefault(
        "ragtime.config", types.ModuleType("ragtime.config")
    )
    config_pkg.__path__ = getattr(config_pkg, "__path__", [])
    config_settings: Any = types.ModuleType("ragtime.config.settings")
    config_settings.settings = SimpleNamespace()
    sys.modules["ragtime.config.settings"] = config_settings

    core_pkg: Any = sys.modules.setdefault("ragtime.core", types.ModuleType("ragtime.core"))
    core_pkg.__path__ = getattr(core_pkg, "__path__", [])

    core_app_settings: Any = types.ModuleType("ragtime.core.app_settings")
    core_app_settings.get_app_settings = lambda: None
    sys.modules["ragtime.core.app_settings"] = core_app_settings

    core_auth: Any = types.ModuleType("ragtime.core.auth")
    core_auth.decode_access_token = lambda *args, **kwargs: None
    core_auth.get_browser_matched_origin = (
        lambda request, browser_origin=None: "https://ragtime.dev.visnovsky.us"
    )
    sys.modules["ragtime.core.auth"] = core_auth

    class _NoopLimiter:
        def limit(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    core_rate_limit: Any = types.ModuleType("ragtime.core.rate_limit")
    core_rate_limit.SHARE_AUTH_RATE_LIMIT = "60/minute"
    core_rate_limit.limiter = _NoopLimiter()
    sys.modules["ragtime.core.rate_limit"] = core_rate_limit

    core_security: Any = types.ModuleType("ragtime.core.security")

    async def _unused_current_user(*_args, **_kwargs):
        raise AssertionError("dependency injection should not run in this test")

    async def _unused_current_user_optional(*_args, **_kwargs):
        return None

    core_security.get_current_user = _unused_current_user
    core_security.get_current_user_optional = _unused_current_user_optional
    core_security.get_session_token = lambda *args, **kwargs: None
    sys.modules["ragtime.core.security"] = core_security

    userspace_pkg: Any = sys.modules.setdefault(
        "ragtime.userspace", types.ModuleType("ragtime.userspace")
    )
    userspace_pkg.__path__ = getattr(userspace_pkg, "__path__", [])

    userspace_models: Any = types.ModuleType("ragtime.userspace.models")

    class _TestPreviewLaunchRequest(BaseModel):
        path: str = "/"
        parent_origin: str | None = None
        prefer_root_proxy: bool = False

    class _TestPreviewWarning(BaseModel):
        issue_code: str
        title: str
        warnings: list[str] = Field(default_factory=list)
        dismiss_key: str
        preview_routing_mode: str = "subdomain"
        resolved_base_domain: str | None = None
        preview_host: str | None = None
        source: str = "derived"

    class _TestPreviewLaunchResponse(BaseModel):
        workspace_id: str
        preview_url: str
        preview_origin: str
        expires_at: datetime
        preview_warning: _TestPreviewWarning | None = None

    class _PlaceholderModel(BaseModel):
        pass

    userspace_models.UserSpaceBrowserAuthorization = _PlaceholderModel
    userspace_models.UserSpaceBrowserAuthRequest = _PlaceholderModel
    userspace_models.UserSpaceBrowserAuthResponse = _PlaceholderModel
    userspace_models.UserSpaceBrowserSurface = str
    userspace_models.UserSpaceCapabilityTokenResponse = _PlaceholderModel
    userspace_models.UserSpaceFileResponse = _PlaceholderModel
    userspace_models.UserSpacePreviewLaunchRequest = _TestPreviewLaunchRequest
    userspace_models.UserSpacePreviewLaunchResponse = _TestPreviewLaunchResponse
    userspace_models.UserSpacePreviewWarning = _TestPreviewWarning
    userspace_models.UserSpaceRuntimeActionResponse = _PlaceholderModel
    userspace_models.UserSpaceRuntimeSessionResponse = _PlaceholderModel
    userspace_models.UserSpaceRuntimeStatusResponse = _PlaceholderModel
    userspace_models.UserSpaceWorkspaceTabStateResponse = _PlaceholderModel
    sys.modules["ragtime.userspace.models"] = userspace_models

    runtime_errors: Any = types.ModuleType("ragtime.userspace.runtime_errors")

    class RuntimeVersionConflictError(Exception):
        pass

    runtime_errors.RuntimeVersionConflictError = RuntimeVersionConflictError
    sys.modules["ragtime.userspace.runtime_errors"] = runtime_errors

    share_auth: Any = types.ModuleType("ragtime.userspace.share_auth")
    share_auth.set_share_auth_cookie = lambda *args, **kwargs: None
    share_auth.share_auth_token_from_request = lambda *args, **kwargs: None
    sys.modules["ragtime.userspace.share_auth"] = share_auth


def _load_runtime_routes_module():
    _install_runtime_route_test_stubs()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "ragtime"
        / "userspace"
        / "runtime_routes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ragtime_userspace_runtime_routes_under_test",
        module_path,
    )
    assert spec and spec.loader, f"failed to build spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RUNTIME_ROUTES = _load_runtime_routes_module()
_workspace_preview_entry_url = getattr(_RUNTIME_ROUTES, "_workspace_preview_entry_url")
issue_workspace_preview_launch = _RUNTIME_ROUTES.issue_workspace_preview_launch
workspace_preview_entry = _RUNTIME_ROUTES.workspace_preview_entry
UserSpacePreviewLaunchResponse = _RUNTIME_ROUTES.UserSpacePreviewLaunchResponse
UserSpacePreviewWarning = sys.modules[
    "ragtime.userspace.models"
].UserSpacePreviewWarning


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