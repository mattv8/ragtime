import unittest
from types import SimpleNamespace
from unittest import mock

from starlette.requests import Request

import ragtime.indexer.routes as indexer_routes
import ragtime.pdm_automation.routes as pdm_routes
import ragtime.userspace.routes as userspace_routes
from ragtime.core import auth


def _request(
    *,
    scheme: str = "http",
    host: str = "internal.example:8000",
    forwarded_host: str | None = None,
    forwarded_proto: str | None = None,
) -> Request:
    headers = [(b"host", host.encode())]
    if forwarded_host:
        headers.append((b"x-forwarded-host", forwarded_host.encode()))
    if forwarded_proto:
        headers.append((b"x-forwarded-proto", forwarded_proto.encode()))

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "scheme": scheme,
            "server": ("internal.example", 8000),
        },
        receive,
    )


def _admin() -> SimpleNamespace:
    return SimpleNamespace(id="admin-1", role="admin")


class WebhookUrlOriginRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_webhook_management_uses_configured_external_origin(self) -> None:
        request = _request()
        expected_origin = "https://public.example/prefix"
        repository = indexer_routes.git_webhook_repository
        metadata = SimpleNamespace(sourceType="git", source="https://git.example/repo.git")
        with (
            mock.patch.object(auth.settings, "external_base_url", expected_origin + "/"),
            mock.patch.object(indexer_routes.repository, "get_index_metadata", mock.AsyncMock(return_value=metadata)),
            mock.patch.object(repository, "get_index_config", mock.AsyncMock()) as get_config,
            mock.patch.object(repository, "enable_index", mock.AsyncMock()) as enable,
            mock.patch.object(repository, "rotate_index_secret", mock.AsyncMock()) as rotate,
            mock.patch.object(repository, "pause_index", mock.AsyncMock()) as pause,
            mock.patch.object(repository, "resume_index", mock.AsyncMock()) as resume,
            mock.patch.object(repository, "resolve_index_target", mock.AsyncMock(return_value=None)),
        ):
            await indexer_routes.get_index_webhook("git-index", request, _admin())
            await indexer_routes.enable_index_webhook("git-index", request, _admin())
            await indexer_routes.rotate_index_webhook_secret("git-index", request, _admin())
            await indexer_routes.pause_index_webhook("git-index", request, _admin())
            await indexer_routes.resume_index_webhook("git-index", request, _admin())

        get_config.assert_awaited_once_with("git-index", expected_origin)
        enable.assert_awaited_once_with("git-index", expected_origin)
        rotate.assert_awaited_once_with("git-index", expected_origin)
        pause.assert_awaited_once_with("git-index", expected_origin)
        resume.assert_awaited_once_with("git-index", expected_origin)

    async def test_workspace_webhook_management_uses_configured_external_origin(self) -> None:
        request = _request()
        expected_origin = "https://public.example/prefix"
        repository = userspace_routes.git_webhook_repository
        workspace = SimpleNamespace(scmGitUrl="https://git.example/repo.git", scmRemoteRole="upstream")
        db = SimpleNamespace(workspace=SimpleNamespace(find_unique=mock.AsyncMock(return_value=workspace)))
        with (
            mock.patch.object(auth.settings, "external_base_url", expected_origin + "/"),
            mock.patch.object(userspace_routes.userspace_service, "enforce_workspace_role", mock.AsyncMock()),
            mock.patch.object(userspace_routes, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(repository, "get_workspace_config", mock.AsyncMock()) as get_config,
            mock.patch.object(repository, "enable_workspace", mock.AsyncMock()) as enable,
            mock.patch.object(repository, "rotate_workspace_secret", mock.AsyncMock()) as rotate,
            mock.patch.object(repository, "pause_workspace", mock.AsyncMock()) as pause,
            mock.patch.object(repository, "resume_workspace", mock.AsyncMock()) as resume,
            mock.patch.object(repository, "resolve_workspace_target", mock.AsyncMock(return_value=None)),
        ):
            await userspace_routes.get_workspace_scm_webhook("workspace-1", request, _admin())
            await userspace_routes.enable_workspace_scm_webhook("workspace-1", request, _admin())
            await userspace_routes.rotate_workspace_scm_webhook_secret("workspace-1", request, _admin())
            await userspace_routes.pause_workspace_scm_webhook("workspace-1", request, _admin())
            await userspace_routes.resume_workspace_scm_webhook("workspace-1", request, _admin())

        get_config.assert_awaited_once_with("workspace-1", expected_origin)
        enable.assert_awaited_once_with("workspace-1", expected_origin)
        rotate.assert_awaited_once_with("workspace-1", expected_origin)
        pause.assert_awaited_once_with("workspace-1", expected_origin)
        resume.assert_awaited_once_with("workspace-1", expected_origin)

    async def test_pdm_webhook_management_uses_forwarded_external_origin(self) -> None:
        request = _request(forwarded_host="internal.example:8443", forwarded_proto="https")
        expected_origin = "https://internal.example:8443"
        repository = pdm_routes.pdm_automation_repository
        with (
            mock.patch.object(auth.settings, "external_base_url", ""),
            mock.patch.object(pdm_routes, "_pdm_tool", mock.AsyncMock()),
            mock.patch.object(repository, "get_status", mock.AsyncMock()) as get_status,
            mock.patch.object(repository, "enable_webhook", mock.AsyncMock()) as enable,
            mock.patch.object(repository, "rotate_webhook", mock.AsyncMock()) as rotate,
            mock.patch.object(repository, "set_paused", mock.AsyncMock()) as set_paused,
        ):
            await pdm_routes.get_pdm_webhook("tool-1", request, _admin())
            await pdm_routes.enable_pdm_webhook("tool-1", request, _admin())
            await pdm_routes.rotate_pdm_webhook("tool-1", request, _admin())
            await pdm_routes.pause_pdm_webhook("tool-1", request, _admin())
            await pdm_routes.resume_pdm_webhook("tool-1", request, _admin())

        get_status.assert_awaited_once_with("tool-1", expected_origin)
        enable.assert_awaited_once_with("tool-1", expected_origin)
        rotate.assert_awaited_once_with("tool-1", expected_origin)
        set_paused.assert_has_awaits(
            [
                mock.call("tool-1", True, expected_origin),
                mock.call("tool-1", False, expected_origin),
            ]
        )

    async def test_index_webhook_origin_uses_safe_request_resolution_when_unconfigured(self) -> None:
        repository = indexer_routes.git_webhook_repository
        cases = (
            (_request(forwarded_host="internal.example:8443", forwarded_proto="https"), "https://internal.example:8443"),
            (_request(scheme="http"), "http://internal.example:8000"),
            (_request(scheme="https", host="internal.example"), "https://internal.example"),
            (_request(forwarded_host="attacker.example", forwarded_proto="https"), "http://internal.example:8000"),
        )
        with (
            mock.patch.object(auth.settings, "external_base_url", ""),
            mock.patch.object(repository, "get_index_config", mock.AsyncMock()) as get_config,
        ):
            for request, expected_origin in cases:
                get_config.reset_mock()
                await indexer_routes.get_index_webhook("git-index", request, _admin())
                get_config.assert_awaited_once_with("git-index", expected_origin)
