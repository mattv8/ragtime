import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma.models import User

import ragtime.indexer.conversation_reads as conversation_reads
import ragtime.indexer.routes as routes


class ConversationMetadataReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_metadata_read_uses_parameterized_projection_from_shared_client(self) -> None:
        query_raw = mock.AsyncMock(
            return_value=[
                {
                    "userId": None,
                    "workspaceId": None,
                    "toolSelectionMode": "custom",
                    "disabledBuiltinToolIds": '["tool-1"]',
                    "subagentsEnabled": True,
                    "memberUserId": "member-1",
                    "memberRole": "viewer",
                }
            ]
        )
        db = SimpleNamespace(query_raw=query_raw)

        with mock.patch.object(conversation_reads, "get_db", mock.AsyncMock(return_value=db)) as get_db:
            metadata = await conversation_reads.get_conversation_read_metadata("conversation-1")

        get_db.assert_awaited_once()
        query_call = query_raw.await_args
        assert query_call is not None
        query = query_call.args[0]
        self.assertNotIn("messages", query)
        self.assertIn('cm."user_id" AS "memberUserId"', query)
        self.assertEqual(query_call.args[1], "conversation-1")
        assert metadata is not None
        self.assertEqual(metadata.disabledBuiltinToolIds, ["tool-1"])
        self.assertEqual(metadata.members[0].role, "viewer")

    async def test_legacy_null_owner_is_denied_to_nonmember_but_admin_and_member_can_read(self) -> None:
        conversation = SimpleNamespace(
            userId=None,
            workspaceId=None,
            members=[SimpleNamespace(userId="member-1", role="viewer")],
        )

        with mock.patch.object(routes, "get_conversation_read_metadata", mock.AsyncMock(return_value=conversation)):
            with self.assertRaises(HTTPException) as denied:
                await routes.get_conversation_members("conversation-1", cast(User, SimpleNamespace(id="other-1", role="user")))
            self.assertEqual(denied.exception.status_code, 403)

            member_result = await routes.get_conversation_members("conversation-1", cast(User, SimpleNamespace(id="member-1", role="user")))
            admin_result = await routes.get_conversation_members("conversation-1", cast(User, SimpleNamespace(id="admin-1", role="admin")))

        self.assertEqual(member_result, [{"user_id": "member-1", "role": "viewer"}])
        self.assertEqual(admin_result, member_result)

    async def test_workspace_member_route_uses_workspace_acl(self) -> None:
        conversation = SimpleNamespace(userId=None, workspaceId="workspace-1", members=[])
        assert_workspace_access = mock.AsyncMock()

        with (
            mock.patch.object(routes, "get_conversation_read_metadata", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(routes, "_assert_workspace_access", assert_workspace_access),
        ):
            result = await routes.get_conversation_members("conversation-1", cast(User, SimpleNamespace(id="user-1", role="user")))

        self.assertEqual(result, [])
        assert_workspace_access.assert_awaited_once_with("workspace-1", mock.ANY, "viewer")
