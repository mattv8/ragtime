from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rag_prompts_stub import install_fake_rag_prompts, remove_fake_rag_prompts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

inserted_fake_rag_prompts = install_fake_rag_prompts()

from ragtime.userspace import service as userspace_service_module
from ragtime.userspace.service import UserSpaceService

remove_fake_rag_prompts(inserted_fake_rag_prompts)


def _build_request(
    path: str,
    *,
    method: str = "GET",
    client_host: str = "203.0.113.10",
    referrer: str | None = "https://example.com/ref",
    user_agent: str | None = "ShareAnalyticsTest/1.0",
) -> SimpleNamespace:
    headers: dict[str, str] = {"host": "ragtime.dev"}
    if referrer is not None:
        headers["referer"] = referrer
    if user_agent is not None:
        headers["user-agent"] = user_agent
    return SimpleNamespace(
        method=method,
        headers=headers,
        url=SimpleNamespace(path=path),
        scope={"path": path},
        client=SimpleNamespace(host=client_host),
        state=SimpleNamespace(),
    )


class _FakeShareModel:
    def __init__(self, rows: dict[str, SimpleNamespace]) -> None:
        self.rows = rows
        self.update_calls: list[dict[str, object]] = []

    async def find_unique(self, *, where):  # type: ignore[no-untyped-def]
        return self.rows.get(where["id"])

    async def update(self, *, where, data):  # type: ignore[no-untyped-def]
        self.update_calls.append({"where": where, "data": data})
        row = self.rows[where["id"]]
        for field, value in data.items():
            if isinstance(value, dict) and "increment" in value:
                current = int(getattr(row, field, 0) or 0)
                setattr(row, field, current + int(value["increment"]))
            else:
                setattr(row, field, value)
        return row


class _FakeAnalyticsEventModel:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create(self, *, data):  # type: ignore[no-untyped-def]
        self.created.append(data)
        return SimpleNamespace(**data)


class _FakeTx:
    def __init__(self, db: SimpleNamespace) -> None:
        self._db = db

    async def __aenter__(self) -> SimpleNamespace:
        return self._db

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False


class _FakeDb(SimpleNamespace):
    def tx(self) -> _FakeTx:
        return _FakeTx(self)


class ShareLinkAnalyticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_workspace_token_hit_and_exposes_status_counts(self) -> None:
        service = UserSpaceService()
        workspace_model = _FakeShareModel({})
        workspace_share = SimpleNamespace(
            id="ws-share-1",
            workspaceId="workspace-1",
            shareToken="token-123",
            shareSlug="alice-workspace",
            shareTokenCreatedAt=datetime.now(timezone.utc),
            shareAccessMode="token",
            shareSelectedUserIds=[],
            shareSelectedLdapGroups=[],
            activeShareStyle="anonymous",
            publicHitCount=0,
            lastPublicHitAt=None,
        )
        workspace_model.rows[workspace_share.id] = workspace_share
        analytics = _FakeAnalyticsEventModel()
        fake_db = _FakeDb(
            workspaceshare=workspace_model,
            conversationshare=_FakeShareModel({}),
            sharelinkrequestlog=analytics,
        )

        request = _build_request("/shared/token-123")

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_public_share_hit(
                request,
                "workspace",
                workspace_share.id,
                "redirect",
                event_name="public_share_token_hit",
                metadata={"route_kind": "token"},
            )

        self.assertEqual(workspace_share.publicHitCount, 1)
        self.assertIsNotNone(workspace_share.lastPublicHitAt)
        self.assertEqual(
            workspace_model.update_calls[0]["data"]["publicHitCount"],
            {"increment": 1},
        )
        self.assertEqual(len(analytics.created), 1)
        created = analytics.created[0]
        self.assertEqual(created["shareTargetType"], "workspace")
        self.assertEqual(created["shareId"], workspace_share.id)
        self.assertEqual(created["eventName"], "public_share_token_hit")
        self.assertEqual(created["outcome"], "redirect")
        self.assertEqual(created["requestPath"], "/shared/token-123")
        self.assertEqual(created["requestMethod"], "GET")
        self.assertEqual(created["referrer"], "https://example.com/ref")
        self.assertEqual(created["userAgent"], "ShareAnalyticsTest/1.0")
        self.assertIsNone(created["authenticatedUserId"])
        self.assertEqual(created["metadata"], {"route_kind": "token"})
        self.assertTrue(created["clientFingerprint"])
        persisted_payload = json.dumps(created, default=str)
        self.assertNotIn("203.0.113.10", persisted_payload)

        status = service._workspace_share_status_from_record("workspace-1", "alice", workspace_share)
        self.assertEqual(status.public_hit_count, 1)
        self.assertEqual(status.last_public_hit_at, workspace_share.lastPublicHitAt)

    async def test_records_conversation_slug_hit_and_exposes_status_counts(self) -> None:
        service = UserSpaceService()
        conversation_model = _FakeShareModel({})
        conversation_share = SimpleNamespace(
            id="conv-share-1",
            conversationId="conversation-1",
            shareToken="token-456",
            shareSlug="alice-chat",
            shareTokenCreatedAt=datetime.now(timezone.utc),
            shareAccessMode="token",
            shareSelectedUserIds=[],
            shareSelectedLdapGroups=[],
            activeShareStyle="named",
            grantedRole="viewer",
            scopeAnchorMessageIdx=None,
            scopeDirection=None,
            publicHitCount=0,
            lastPublicHitAt=None,
        )
        conversation_model.rows[conversation_share.id] = conversation_share
        analytics = _FakeAnalyticsEventModel()
        fake_db = _FakeDb(
            workspaceshare=_FakeShareModel({}),
            conversationshare=conversation_model,
            sharelinkrequestlog=analytics,
        )

        request = _build_request("/alice/alice-chat")

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_public_share_hit(
                request,
                "conversation",
                conversation_share.id,
                "password_prompt",
                event_name="public_share_slug_hit",
                authenticated_user_id="user-42",
            )

        self.assertEqual(conversation_share.publicHitCount, 1)
        self.assertIsNotNone(conversation_share.lastPublicHitAt)
        self.assertEqual(
            conversation_model.update_calls[0]["data"]["publicHitCount"],
            {"increment": 1},
        )
        self.assertEqual(len(analytics.created), 1)
        created = analytics.created[0]
        self.assertEqual(created["shareTargetType"], "conversation")
        self.assertEqual(created["shareId"], conversation_share.id)
        self.assertEqual(created["eventName"], "public_share_slug_hit")
        self.assertEqual(created["outcome"], "password_prompt")
        self.assertEqual(created["requestPath"], "/alice/alice-chat")
        self.assertEqual(created["authenticatedUserId"], "user-42")
        self.assertTrue(created["clientFingerprint"])
        self.assertNotIn("203.0.113.10", json.dumps(created, default=str))

        status = service._conversation_share_status_from_record("conversation-1", "alice", conversation_share)
        self.assertEqual(status.public_hit_count, 1)
        self.assertEqual(status.last_public_hit_at, conversation_share.lastPublicHitAt)
