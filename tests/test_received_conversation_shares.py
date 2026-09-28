import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

import httpx
from fastapi import FastAPI, Response
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.core.security import get_current_user
from ragtime.indexer.models import ReceivedConversationShare
from ragtime.indexer.repository import repository
from ragtime.indexer.routes import list_received_conversation_shares, router
from ragtime.userspace.service import userspace_service
from tests.content_protection_support import use_disabled_content_protection

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def _user(*, user_id: str = "recipient-1", admin: bool = False) -> User:
    return User(
        id=user_id,
        username="recipient",
        authProvider=AuthProvider.local,
        cachedGroups=cast(Json, "[]"),
        role=UserRole.admin if admin else UserRole.user,
        roleManuallySet=False,
        createdAt=NOW,
        updatedAt=NOW,
        securityGeneration=0,
    )


def _share() -> ReceivedConversationShare:
    return ReceivedConversationShare(
        id="share-1",
        conversation_id="conversation-1",
        title="Shared chat",
        owner_username="owner",
        owner_display_name="Owner",
        share_token="token-1",
        label="Scoped link",
        granted_role="editor",
        scope_anchor_message_idx=3,
        scope_direction="forward",
        created_at=NOW,
    )


def _app_for(user: User) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


class ReceivedConversationShareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        use_disabled_content_protection(self)

    async def test_repository_uses_recipient_only_metadata_query_and_created_cursor(self) -> None:
        query_raw = mock.AsyncMock(return_value=[])
        db = SimpleNamespace(query_raw=query_raw)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            shares = await repository.list_received_conversation_shares(
                "recipient-'quoted",
                limit=1,
                cursor_created_at=NOW,
                cursor_id="share-a",
            )

        self.assertEqual(shares, [])
        assert query_raw.await_args is not None
        sql = query_raw.await_args.args[0]
        self.assertIn("s.share_access_mode = 'selected_users'", sql)
        self.assertIn("s.share_selected_user_ids @> '[\"recipient-''quoted\"]'::jsonb", sql)
        self.assertIn("c.workspace_id IS NULL", sql)
        self.assertIn("c.parent_conversation_id IS NULL", sql)
        self.assertIn("s.owner_user_id IS DISTINCT FROM 'recipient-''quoted'", sql)
        self.assertIn("NULLIF(BTRIM(s.share_token), '') IS NOT NULL", sql)
        self.assertIn("s.created_at DESC, s.id DESC", sql)
        self.assertIn("s.created_at <", sql)
        self.assertIn("s.id < 'share-a'", sql)
        self.assertIn("LIMIT 1", sql)
        self.assertNotIn("updated_at", sql)
        self.assertNotIn("c.messages", sql)
        self.assertNotIn("conversation_members", sql)

    async def test_repository_maps_minimal_metadata_and_keeps_multiple_scopes(self) -> None:
        rows = [
            {
                "id": "share-2",
                "conversation_id": "conversation-1",
                "title": "Old shared chat",
                "owner_username": "owner",
                "owner_display_name": None,
                "share_token": "token-2",
                "label": "Earlier messages",
                "granted_role": "viewer",
                "scope_anchor_message_idx": 1,
                "scope_direction": "backward",
                "created_at": NOW,
            },
            {
                "id": "share-1",
                "conversation_id": "conversation-1",
                "title": "Old shared chat",
                "owner_username": "owner",
                "owner_display_name": None,
                "share_token": "token-1",
                "label": "Later messages",
                "granted_role": "editor",
                "scope_anchor_message_idx": 4,
                "scope_direction": "forward",
                "created_at": NOW,
            },
        ]
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=rows))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            shares = await repository.list_received_conversation_shares("recipient-1")

        self.assertEqual([share.id for share in shares], ["share-2", "share-1"])
        self.assertEqual(shares[0].scope_direction, "backward")
        self.assertEqual(shares[1].granted_role, "editor")

    async def test_repository_reflects_revocation_without_caching_or_member_writes(self) -> None:
        row = {
            "id": "share-1",
            "conversation_id": "conversation-1",
            "title": "Old shared chat",
            "owner_username": "owner",
            "owner_display_name": None,
            "share_token": "token-1",
            "label": None,
            "granted_role": "viewer",
            "scope_anchor_message_idx": None,
            "scope_direction": None,
            "created_at": NOW,
        }
        query_raw = mock.AsyncMock(side_effect=[[row], []])
        db = SimpleNamespace(query_raw=query_raw)

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            before_revocation = await repository.list_received_conversation_shares("recipient-1")
            after_revocation = await repository.list_received_conversation_shares("recipient-1")

        self.assertEqual([share.id for share in before_revocation], ["share-1"])
        self.assertEqual(after_revocation, [])
        self.assertEqual(query_raw.await_count, 2)
        self.assertTrue(all("conversation_members" not in call.args[0] for call in query_raw.await_args_list))

    async def test_selected_user_access_parity_rejects_unlisted_admin(self) -> None:
        share_record = SimpleNamespace(
            ownerUserId="owner-1",
            shareAccessMode="selected_users",
            shareSelectedUserIds=["recipient-1"],
            shareSelectedLdapGroups=[],
            sharePassword=None,
        )
        await userspace_service._enforce_share_access(share_record, _user(), None)
        with self.assertRaisesRegex(Exception, "User not allowed for this share"):
            await userspace_service._enforce_share_access(share_record, _user(user_id="admin-1", admin=True), None)

    async def test_http_route_uses_default_limit_static_path_and_no_store(self) -> None:
        share = _share()
        app = _app_for(_user(admin=True))
        with mock.patch.object(repository, "list_received_conversation_shares", mock.AsyncMock(return_value=[share])) as list_mock:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                response = await client.get("/indexes/conversations/shared-with-me")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.json(), [_share().model_dump(mode="json")])
        list_mock.assert_awaited_once_with("recipient-1", limit=50, cursor_created_at=None, cursor_id=None)

    async def test_http_route_validates_cursor_and_forwards_created_at(self) -> None:
        app = _app_for(_user())
        with mock.patch.object(repository, "list_received_conversation_shares", mock.AsyncMock(return_value=[])) as list_mock:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                missing_created_at = await client.get("/indexes/conversations/shared-with-me?cursor_id=share-1")
                missing_id = await client.get("/indexes/conversations/shared-with-me?cursor_created_at=2026-09-27T12:00:00Z")
                malformed = await client.get("/indexes/conversations/shared-with-me?cursor_created_at=not-a-date&cursor_id=share-1")
                oversized = await client.get("/indexes/conversations/shared-with-me?limit=201")
                valid = await client.get("/indexes/conversations/shared-with-me?cursor_created_at=2026-09-27T12:00:00Z&cursor_id=share-1")

        self.assertEqual(missing_created_at.status_code, 400)
        self.assertIn("cursor_id requires cursor_created_at", missing_created_at.json()["detail"])
        self.assertEqual(missing_id.status_code, 400)
        self.assertIn("cursor_created_at requires cursor_id", missing_id.json()["detail"])
        self.assertEqual(malformed.status_code, 400)
        self.assertIn("cursor_created_at", malformed.json()["detail"])
        self.assertEqual(oversized.status_code, 422)
        self.assertEqual(valid.status_code, 200)
        list_mock.assert_awaited_once_with("recipient-1", limit=50, cursor_created_at=NOW, cursor_id="share-1")

    async def test_route_protects_only_displayed_metadata_and_rejects_blocked_content(self) -> None:
        share = _share()
        with (
            mock.patch.object(repository, "list_received_conversation_shares", mock.AsyncMock(return_value=[share])),
            mock.patch("ragtime.indexer.routes._authorize_conversation_release", mock.AsyncMock(side_effect=RuntimeError("blocked"))) as authorize,
        ):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                await list_received_conversation_shares(response=Response(), limit=50, user=_user())
        authorize.assert_awaited_once_with(
            [
                {
                    "title": "Shared chat",
                    "label": "Scoped link",
                    "owner_username": "owner",
                    "owner_display_name": "Owner",
                }
            ],
            user=mock.ANY,
            owner_user_id=None,
        )


if __name__ == "__main__":
    unittest.main()
