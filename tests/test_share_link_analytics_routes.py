from __future__ import annotations

import json
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

from ragtime.config.settings import settings
from tests.rag_prompts_stub import install_fake_rag_prompts, remove_fake_rag_prompts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

inserted_fake_rag_prompts = install_fake_rag_prompts()

import ragtime.indexer.routes as indexer_routes_module
import ragtime.main as main_module
import ragtime.userspace.routes as userspace_routes_module
from ragtime.indexer.models import ConversationShareLinkListResponse, ConversationShareLinkStatus
from ragtime.userspace.models import UserSpaceWorkspaceShareLinkListResponse, UserSpaceWorkspaceShareLinkStatus

remove_fake_rag_prompts(inserted_fake_rag_prompts)


def _build_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [(b"host", b"ragtime.dev")],
            "scheme": "https",
            "server": ("ragtime.dev", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


class _DisconnectingRequest:
    def __init__(self, states: list[bool], path: str = "/") -> None:
        self._states = list(states)
        self._request = _build_request(path)

    @property
    def url(self) -> Any:
        return self._request.url

    @property
    def headers(self) -> Any:
        return self._request.headers

    @property
    def base_url(self) -> Any:
        return self._request.base_url

    async def is_disconnected(self) -> bool:
        if self._states:
            return self._states.pop(0)
        return True


async def _read_sse_chunks(response: Any, count: int) -> list[str]:
    chunks: list[str] = []
    iterator = response.body_iterator
    for _ in range(count):
        chunk = await anext(iterator)
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    return chunks


@contextmanager
def _token_share_launch_patches(*, share_record: SimpleNamespace, analytics_mock: mock.AsyncMock):
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(main_module, "_share_current_user_from_request", mock.AsyncMock(return_value=None)))
        stack.enter_context(mock.patch.object(main_module, "share_auth_token_from_request", return_value=None))
        stack.enter_context(mock.patch.object(main_module, "get_external_origin", return_value="https://ragtime.dev"))
        stack.enter_context(
            mock.patch.object(
                main_module.userspace_service,
                "get_share_prompt_metadata_by_token",
                mock.AsyncMock(return_value=("Shared workspace", None)),
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module.userspace_service,
                "_resolve_public_share_record_by_token",
                mock.AsyncMock(return_value=("workspace", share_record)),
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module.userspace_service,
                "resolve_shared_workspace_id",
                mock.AsyncMock(return_value="workspace-1"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module.userspace_service,
                "get_share_access_mode",
                mock.AsyncMock(return_value="token"),
            )
        )
        stack.enter_context(mock.patch.object(main_module.userspace_service, "record_public_share_hit", analytics_mock))
        stack.enter_context(
            mock.patch.object(
                main_module.userspace_runtime_service,
                "issue_shared_preview_launch",
                mock.AsyncMock(return_value=SimpleNamespace(preview_url="https://preview.example/bootstrap")),
            )
        )
        yield


class PublicShareAnalyticsRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_token_root_redirect_records_single_workspace_hit(self) -> None:
        request = _build_request("/shared/token-123")
        share_record = SimpleNamespace(id="ws-share-1", workspaceId="workspace-1")
        record_hit = mock.AsyncMock()

        with _token_share_launch_patches(share_record=share_record, analytics_mock=record_hit):
            response = await main_module._shared_launch_redirect_by_token("token-123", request, "")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "https://preview.example/bootstrap")
        record_hit.assert_awaited_once_with(
            request,
            "workspace",
            "ws-share-1",
            "redirect",
            event_name="public_share_token_hit",
            metadata={"route_kind": "token"},
        )

    async def test_shared_token_root_redirect_swallow_analytics_failure_and_returns_redirect(self) -> None:
        request = _build_request("/shared/token-123")
        share_record = SimpleNamespace(id="ws-share-1", workspaceId="workspace-1")

        with _token_share_launch_patches(
            share_record=share_record,
            analytics_mock=mock.AsyncMock(side_effect=RuntimeError("analytics write failed")),
        ):
            response = await main_module._shared_launch_redirect_by_token("token-123", request, "")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "https://preview.example/bootstrap")

    async def test_shared_token_path_proxy_does_not_record_asset_refresh(self) -> None:
        request = _build_request("/shared/token-123/assets/app.js")
        share_record = SimpleNamespace(id="ws-share-1", workspaceId="workspace-1")
        record_hit = mock.AsyncMock()

        with _token_share_launch_patches(share_record=share_record, analytics_mock=record_hit):
            response = await main_module._shared_launch_redirect_by_token("token-123", request, "assets/app.js")

        self.assertEqual(response.status_code, 302)
        record_hit.assert_not_awaited()

    async def test_shared_slug_password_prompt_records_single_conversation_hit(self) -> None:
        request = _build_request("/alice/shared-chat")
        share_record = SimpleNamespace(id="conv-share-1", conversationId="conversation-1")
        record_hit = mock.AsyncMock()

        with (
            mock.patch.object(main_module, "_share_current_user_from_request", mock.AsyncMock(return_value=None)),
            mock.patch.object(main_module, "share_auth_token_from_request", return_value=None),
            mock.patch.object(
                main_module.userspace_service,
                "get_share_prompt_metadata_by_slug",
                mock.AsyncMock(return_value=("Shared chat", "Alice")),
            ),
            mock.patch.object(
                main_module.userspace_service,
                "_resolve_public_share_record_by_slug",
                mock.AsyncMock(return_value=("conversation", share_record)),
            ),
            mock.patch.object(
                main_module.userspace_service,
                "authorize_shared_conversation_access_by_slug",
                mock.AsyncMock(side_effect=HTTPException(status_code=401, detail="Password required")),
            ),
            mock.patch.object(main_module.userspace_service, "record_public_share_hit", record_hit),
        ):
            response = await main_module._shared_launch_redirect_by_slug("alice", "shared-chat", request, "")

        self.assertEqual(response.status_code, 200)
        record_hit.assert_awaited_once_with(
            request,
            "conversation",
            "conv-share-1",
            "password_prompt",
            event_name="public_share_slug_hit",
            metadata={"route_kind": "slug"},
        )


class ShareLinkAnalyticsSseRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_share_link_events_use_external_base_url(self) -> None:
        payload = UserSpaceWorkspaceShareLinkListResponse(
            workspace_id="workspace-1",
            owner_username="alice",
            links=[
                UserSpaceWorkspaceShareLinkStatus(
                    id="ws-share-1",
                    workspace_id="workspace-1",
                    has_share_link=True,
                    owner_username="alice",
                    share_token="token-123",
                    share_url="/alice/shared-workspace",
                )
            ],
        )
        list_links = mock.AsyncMock(return_value=payload)

        with (
            mock.patch.object(settings, "external_base_url", "https://ragtime.hammerton.com"),
            mock.patch.object(userspace_routes_module.userspace_service, "list_workspace_share_links", list_links),
        ):
            response = await userspace_routes_module.stream_workspace_share_link_events(
                "workspace-1",
                _DisconnectingRequest(
                    [False],
                    "/indexes/userspace/workspaces/workspace-1/share-links/events",
                ),
                user=SimpleNamespace(id="user-1", role="editor"),
            )
            chunk = (await _read_sse_chunks(response, 1))[0]

        event_payload = json.loads(chunk.split("data: ", 1)[1])
        self.assertEqual(
            event_payload["links"][0]["anonymous_share_url"],
            "https://ragtime.hammerton.com/shared/token-123",
        )
        list_links.assert_awaited_once_with(
            "workspace-1",
            "user-1",
            base_url="https://ragtime.hammerton.com",
        )

    async def test_share_link_events_emit_changed_payload_then_keepalive(self) -> None:
        cases = [
            (
                "workspace",
                getattr(userspace_routes_module, "stream_workspace_share_link_events"),
                mock.patch.object,
                userspace_routes_module,
                "list_workspace_share_links",
                UserSpaceWorkspaceShareLinkListResponse(
                    workspace_id="workspace-1",
                    owner_username="alice",
                    links=[
                        UserSpaceWorkspaceShareLinkStatus(
                            id="ws-share-1",
                            workspace_id="workspace-1",
                            has_share_link=True,
                            owner_username="alice",
                            public_hit_count=3,
                        )
                    ],
                ),
                "workspace-1",
            ),
            (
                "conversation",
                getattr(indexer_routes_module, "stream_conversation_share_link_events"),
                mock.patch.object,
                indexer_routes_module,
                "list_conversation_share_links",
                ConversationShareLinkListResponse(
                    conversation_id="conversation-1",
                    owner_username="alice",
                    links=[
                        ConversationShareLinkStatus(
                            id="conv-share-1",
                            conversation_id="conversation-1",
                            has_share_link=True,
                            owner_username="alice",
                            public_hit_count=4,
                        )
                    ],
                ),
                "conversation-1",
            ),
        ]

        for label, stream_route, _patch_object, module, list_method, payload, resource_id in cases:
            with self.subTest(label=label):
                with (
                    mock.patch.object(
                        module.userspace_service,
                        list_method,
                        mock.AsyncMock(side_effect=[payload, payload, payload]),
                    ),
                    mock.patch.object(module.asyncio, "sleep", mock.AsyncMock(return_value=None)),
                ):
                    response = await stream_route(
                        resource_id,
                        _DisconnectingRequest([False, False, True]),
                        user=SimpleNamespace(id="user-1", role="editor"),
                    )
                    chunks = await _read_sse_chunks(response, 2)

                event_payload = payload
                if label == "workspace":
                    event_payload = payload.model_copy(
                        update={
                            "links": [
                                userspace_routes_module._apply_share_link_variants(
                                    link,
                                    base_url="https://ragtime.dev",
                                )
                                for link in payload.links
                            ],
                        }
                    )
                payload_json = json.dumps(event_payload.model_dump(mode="json"))
                self.assertEqual(chunks[0], f"event: share_links\ndata: {payload_json}\n\n")
                self.assertEqual(chunks[1], ": keepalive\n\n")

    async def test_workspace_share_link_events_reject_unauthorized_management_user(self) -> None:
        stream_route = getattr(userspace_routes_module, "stream_workspace_share_link_events")

        with mock.patch.object(
            userspace_routes_module.userspace_service,
            "list_workspace_share_links",
            mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Editor access required")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await stream_route(
                    "workspace-1",
                    _DisconnectingRequest([True]),
                    user=SimpleNamespace(id="user-2", role="viewer"),
                )

        self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
