"""Opt-in API/Postgres regression checks using disposable local conversations.

Run inside ragtime-dev with RAGTIME_BRANCH_INTEGRATION=1 after migrations.
"""

import asyncio
import os
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
from prisma import Json, Prisma
from prisma.enums import ChatTaskStatus, ConversationBranchKind, UserRole
from prisma.types import ChatTaskCreateInput, ConversationBranchCreateInput, ConversationCreateInput, SessionCreateInput, UserCreateInput

from ragtime.core.auth import create_access_token, hash_token


def messages(*labels):
    return [{"role": "user" if index % 2 == 0 else "assistant", "content": label, "message_id": str(uuid.uuid4())} for index, label in enumerate(labels)]


@unittest.skipUnless(os.environ.get("RAGTIME_BRANCH_INTEGRATION") == "1", "local API/Postgres opt-in")
class ConversationBranchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = Prisma()
        await self.db.connect()
        self.addAsyncCleanup(self.db.disconnect)
        self.client = httpx.AsyncClient(base_url="http://localhost:8000", timeout=30)
        self.addAsyncCleanup(self.client.aclose)
        # Use a disposable test identity/session; do not depend on the local
        # administrator's MFA enrollment or alter an existing user's account.
        user_data = cast(UserCreateInput, {"username": f"branch-test-{uuid.uuid4()}", "role": UserRole.admin})
        user = await self.db.user.create(data=user_data)
        self.addAsyncCleanup(self.db.user.delete, where={"id": user.id})
        token = create_access_token(user.id, user.username, "admin", mfa_verified=True)
        session_data = cast(
            SessionCreateInput,
            {
                "userId": user.id,
                "tokenHash": hash_token(token),
                "expiresAt": datetime.now(timezone.utc) + timedelta(minutes=5),
                "mfaVerifiedAt": datetime.now(timezone.utc),
            },
        )
        await self.db.session.create(data=session_data)
        self.client.headers["Authorization"] = f"Bearer {token}"
        self.conversation_id = str(uuid.uuid4())
        conversation_data = cast(
            ConversationCreateInput,
            {
                "id": self.conversation_id,
                "title": "Branch integration disposable",
                "messages": Json([]),
                "toolSelectionMode": "custom",
            },
        )
        await self.db.conversation.create(data=conversation_data)
        self.addAsyncCleanup(self.db.conversation.delete, where={"id": self.conversation_id})
        self.path = f"/indexes/conversations/{self.conversation_id}"

    async def seed(self, payload, active=None):
        await self.db.conversation.update(
            where={"id": self.conversation_id},
            data={
                "messages": Json(payload),
                "activeBranchId": active,
            },
        )

    async def branch(self, point, payload, *, parent=None, kind: str | None = "edit", legacy=False):
        data = cast(
            ConversationBranchCreateInput,
            {
                "conversationId": self.conversation_id,
                "branchPointIndex": point,
                "parentBranchId": parent,
                "branchKind": cast(ConversationBranchKind | None, kind),
                "preservedMessages": Json(payload[point:]),
            },
        )
        if not legacy:
            data["baseMessages"] = Json(payload[:point])
        return await self.db.conversationbranch.create(data=data)

    async def switch(self, branch, expected):
        started = time.perf_counter()
        response = await self.client.post(self.path + "/branches/switch", json={"branch_id": branch.id})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["active_branch_id"], branch.id)
        self.assertEqual([message["content"] for message in body["messages"]], [message["content"] for message in expected])
        return time.perf_counter() - started

    async def test_same_group_repeated_round_trips_keep_distinct_suffixes(self):
        root = messages("root", "answer")
        old = root + messages("old prompt", "old answer")
        live = root + messages("new prompt", "new answer")
        original = await self.branch(2, old)
        await self.seed(live)
        await self.switch(original, old)
        rows = await self.db.conversationbranch.find_many(where={"conversationId": self.conversation_id})
        current = next(row for row in rows if row.branchKind is None)
        durations = []
        for _ in range(5):
            durations.append(await self.switch(current, live))
            durations.append(await self.switch(original, old))
        self.assertEqual(await self.db.conversationbranch.count(where={"conversationId": self.conversation_id}), 2)
        print(f"branch API round trips: n={len(durations)}, max={max(durations):.3f}s")

    async def test_legacy_production_shape_and_nested_edit_preserve_both_paths(self):
        root = messages(*[f"message {index}" for index in range(25)])
        parent = await self.branch(23, root, kind=None, legacy=True)
        old = root + messages("old nested prompt", "old nested answer")
        await self.seed(old, parent.id)
        response = await self.client.post(
            self.path + "/branches",
            json={
                "from_message_index": 25,
                "branch_kind": "edit",
                "auto_snapshot": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        original = await self.db.conversationbranch.find_unique(where={"id": response.json()["id"]})
        live = root + messages("new nested prompt", "new nested answer")
        await self.seed(live)
        await self.switch(original, old)
        rows = await self.db.conversationbranch.find_many(where={"conversationId": self.conversation_id})
        current = next(row for row in rows if row.branchKind is None and row.branchPointIndex == 25)
        await self.switch(current, live)
        await self.switch(parent, old)
        await self.switch(original, old)
        await self.switch(current, live)

    async def test_cross_prefix_short_path_and_empty_delete_restore(self):
        short = messages("short root", "short answer")
        long = messages("different root", "different answer", "long prompt", "long answer")
        source = await self.branch(0, short)
        target = await self.branch(3, long)
        await self.seed(short, source.id)
        await self.switch(target, long)
        await self.switch(source, short)
        response = await self.client.post(
            self.path + "/branches",
            json={
                "from_message_index": 0,
                "branch_kind": "delete",
                "auto_snapshot": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        deleted = await self.db.conversationbranch.find_unique(where={"id": response.json()["id"]})
        await self.switch(deleted, short)
        response = await self.client.post(self.path + "/branches/release")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["messages"], [])

    async def test_active_task_and_concurrent_switches_are_serialized(self):
        old, live = messages("old"), messages("live")
        first = await self.branch(0, old)
        second = await self.branch(0, live, kind=None)
        await self.seed(live, second.id)
        task_data = cast(
            ChatTaskCreateInput,
            {
                "conversationId": self.conversation_id,
                "userMessage": "fixture",
                "status": ChatTaskStatus.pending,
            },
        )
        task = await self.db.chattask.create(data=task_data)
        await self.db.conversation.update(where={"id": self.conversation_id}, data={"activeTaskId": task.id})
        for endpoint, body in [("switch", {"branch_id": first.id}), ("release", None)]:
            response = await self.client.post(self.path + "/branches/" + endpoint, json=body)
            self.assertEqual(response.status_code, 409, response.text)
        await self.db.conversation.update(where={"id": self.conversation_id}, data={"activeTaskId": None})
        await self.db.chattask.delete(where={"id": task.id})
        responses = await asyncio.gather(
            *[self.client.post(self.path + "/branches/switch", json={"branch_id": branch.id}) for branch in [first, second, first, second]]
        )
        self.assertTrue(all(response.status_code == 200 for response in responses))
        await self.switch(first, old)
        await self.switch(second, live)
