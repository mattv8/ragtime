"""Tests for workspace external-agent access tokens."""

import importlib
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import agent_access

UniqueViolationError: Any = getattr(importlib.import_module("prisma.errors"), "UniqueViolationError")


def _await_kwargs(async_mock: mock.AsyncMock) -> dict[str, Any]:
    assert async_mock.await_args is not None
    return dict(async_mock.await_args.kwargs)


def _fake_record(**overrides):
    base = dict(
        id="acc-1",
        workspaceId="ws-1",
        createdByUserId="user-1",
        token="tok-abc",
        enabled=True,
        allowTaskSubmission=True,
        lastUsedAt=None,
        hitCount=0,
        createdAt=None,
        updatedAt=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_db(record=None, user_role="user"):
    db = SimpleNamespace()
    db.workspaceagentaccess = SimpleNamespace(
        find_first=mock.AsyncMock(return_value=record),
        find_unique=mock.AsyncMock(return_value=record),
        create=mock.AsyncMock(return_value=record or _fake_record()),
        update=mock.AsyncMock(return_value=record or _fake_record()),
    )
    db.externalbuildrequest = SimpleNamespace(
        find_unique=mock.AsyncMock(return_value=None),
        find_first=mock.AsyncMock(return_value=None),
        create=mock.AsyncMock(return_value=_fake_record(id="ebr-1")),
        update=mock.AsyncMock(return_value=None),
        delete=mock.AsyncMock(return_value=None),
    )
    db.user = SimpleNamespace(
        find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role=user_role)),
    )
    return db


class AgentAccessTokenTests(unittest.IsolatedAsyncioTestCase):
    async def test_enable_mints_token_and_returns_status(self) -> None:
        db = _fake_db(record=None)
        created = _fake_record(token="tok-new")
        db.workspaceagentaccess.create = mock.AsyncMock(return_value=created)

        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.enable_agent_access("ws-1", "user-1")

        self.assertTrue(status["enabled"])
        self.assertEqual(status["token"], "tok-new")
        create_kwargs = _await_kwargs(db.workspaceagentaccess.create)["data"]
        self.assertGreaterEqual(len(create_kwargs["token"]), 32)

    async def test_status_hides_token_when_disabled(self) -> None:
        record = _fake_record(enabled=False)
        db = _fake_db(record=record)
        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.get_agent_access_status("ws-1", "user-1")
        self.assertFalse(status["enabled"])
        self.assertIsNone(status["token"])

    async def test_reenable_disabled_access_preserves_existing_token_and_creator(self) -> None:
        record = _fake_record(enabled=False, token="old-token", createdByUserId="old-owner")
        db = _fake_db(record=record)
        db.workspaceagentaccess.update = mock.AsyncMock(return_value=_fake_record(enabled=True, token="old-token", createdByUserId="old-owner"))
        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.enable_agent_access("ws-1", "new-owner")

        update_data = _await_kwargs(db.workspaceagentaccess.update)["data"]
        self.assertNotIn("token", update_data)
        self.assertNotIn("createdByUserId", update_data)
        self.assertEqual(status["token"], "old-token")

    async def test_enable_by_different_owner_preserves_active_url_and_creator(self) -> None:
        record = _fake_record(enabled=True, token="old-token", createdByUserId="old-owner")
        db = _fake_db(record=record)
        db.workspaceagentaccess.update = mock.AsyncMock(return_value=_fake_record(token="old-token", createdByUserId="old-owner", allowTaskSubmission=False))
        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.enable_agent_access("ws-1", "new-owner", allow_task_submission=False)

        update_data = _await_kwargs(db.workspaceagentaccess.update)["data"]
        self.assertNotIn("token", update_data)
        self.assertNotIn("createdByUserId", update_data)
        self.assertEqual(status["token"], "old-token")

    async def test_status_shows_token_to_different_current_owner(self) -> None:
        record = _fake_record(enabled=True, token="tok-secret", createdByUserId="owner-1")
        db = _fake_db(record=record)

        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.get_agent_access_status("ws-1", "owner-2")

        self.assertTrue(status["enabled"])
        self.assertEqual(status["token"], "tok-secret")

    async def test_concurrent_first_enable_rereads_and_updates_winning_row(self) -> None:
        winning = _fake_record(id="acc-winning", token="tok-winning", createdByUserId="owner-1", enabled=True)
        db = _fake_db(record=None)
        db.workspaceagentaccess.find_unique = mock.AsyncMock(side_effect=[None, winning])
        db.workspaceagentaccess.create = mock.AsyncMock(side_effect=UniqueViolationError({}))
        db.workspaceagentaccess.update = mock.AsyncMock(
            return_value=_fake_record(
                id="acc-winning",
                token="tok-winning",
                createdByUserId="owner-1",
                enabled=True,
                allowTaskSubmission=False,
            )
        )

        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            status = await agent_access.enable_agent_access("ws-1", "owner-2", allow_task_submission=False)

        self.assertEqual(status["token"], "tok-winning")
        update_data = _await_kwargs(db.workspaceagentaccess.update)["data"]
        self.assertEqual(update_data["enabled"], True)
        self.assertEqual(update_data["allowTaskSubmission"], False)
        self.assertNotIn("token", update_data)
        self.assertNotIn("createdByUserId", update_data)

    async def test_resolve_unknown_token_raises_404(self) -> None:
        db = _fake_db(record=None)
        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as ctx:
                await agent_access.resolve_agent_access_token("nope")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_resolve_disabled_token_raises_404(self) -> None:
        db = _fake_db(record=_fake_record(enabled=False))
        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException):
                await agent_access.resolve_agent_access_token("tok-abc")

    async def test_resolve_valid_token_returns_context(self) -> None:
        db = _fake_db(record=_fake_record())
        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            ctx = await agent_access.resolve_agent_access_token("tok-abc")
        self.assertEqual(ctx.workspace_id, "ws-1")
        self.assertEqual(ctx.acting_user_id, "user-1")
        self.assertFalse(ctx.acting_user_is_admin)
        self.assertTrue(ctx.allow_task_submission)
        db.workspaceagentaccess.update.assert_awaited()

    async def test_rotate_requires_existing_record(self) -> None:
        db = _fake_db(record=None)
        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await agent_access.rotate_agent_access_token("ws-1", "user-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_rotate_rebinds_token_to_current_owner(self) -> None:
        db = _fake_db(record=_fake_record(createdByUserId="old-owner"))
        db.workspaceagentaccess.update = mock.AsyncMock(return_value=_fake_record(token="rotated", createdByUserId="new-owner"))
        with (
            mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(agent_access, "_mint_token", return_value="rotated"),
            mock.patch(
                "ragtime.userspace.service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
        ):
            await agent_access.rotate_agent_access_token("ws-1", "new-owner")

        update_data = _await_kwargs(db.workspaceagentaccess.update)["data"]
        self.assertEqual(update_data["createdByUserId"], "new-owner")


class ExternalBuildRequestLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def test_find_external_build_request_uses_compound_lookup(self) -> None:
        record = _fake_record(id="ebr-1")
        db = _fake_db()
        db.externalbuildrequest.find_unique = mock.AsyncMock(return_value=record)

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            found = await agent_access.find_external_build_request("user-1", agent_access.AGENT_ACCESS_SOURCE, "req-1")

        assert found is not None
        self.assertEqual(found.id, "ebr-1")
        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.find_unique)["where"],
            {
                "userId_source_requestId": {
                    "userId": "user-1",
                    "source": agent_access.AGENT_ACCESS_SOURCE,
                    "requestId": "req-1",
                }
            },
        )

    async def test_find_external_build_request_by_task_scopes_owner(self) -> None:
        record = _fake_record(id="ebr-1")
        db = _fake_db()
        db.externalbuildrequest.find_first = mock.AsyncMock(return_value=record)

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            found = await agent_access.find_external_build_request_by_task("user-1", agent_access.AGENT_ACCESS_SOURCE, "ws-1", "task-1")

        assert found is not None
        self.assertEqual(found.id, "ebr-1")
        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.find_first)["where"],
            {
                "userId": "user-1",
                "source": agent_access.AGENT_ACCESS_SOURCE,
                "workspaceId": "ws-1",
                "taskId": "task-1",
            },
        )

    async def test_find_external_build_request_by_conversation_scopes_owner(self) -> None:
        record = _fake_record(id="ebr-1")
        db = _fake_db()
        db.externalbuildrequest.find_first = mock.AsyncMock(return_value=record)

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            found = await agent_access.find_external_build_request_by_conversation("user-1", agent_access.AGENT_ACCESS_SOURCE, "ws-1", "conv-1")

        assert found is not None
        self.assertEqual(found.id, "ebr-1")
        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.find_first)["where"],
            {
                "userId": "user-1",
                "source": agent_access.AGENT_ACCESS_SOURCE,
                "workspaceId": "ws-1",
                "conversationId": "conv-1",
            },
        )

    async def test_create_external_build_request_persists_claim(self) -> None:
        record = _fake_record(id="ebr-1")
        db = _fake_db()
        db.externalbuildrequest.create = mock.AsyncMock(return_value=record)

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            created = await agent_access.create_external_build_request(
                user_id="user-1",
                source=agent_access.AGENT_ACCESS_SOURCE,
                request_id="req-1",
                payload_hash="hash-1",
                workspace_id="ws-1",
            )

        self.assertEqual(created.id, "ebr-1")
        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.create)["data"],
            {
                "userId": "user-1",
                "source": agent_access.AGENT_ACCESS_SOURCE,
                "requestId": "req-1",
                "payloadHash": "hash-1",
                "workspaceId": "ws-1",
            },
        )

    async def test_finalize_external_build_request_records_ownership(self) -> None:
        db = _fake_db()

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            await agent_access.finalize_external_build_request("ebr-1", conversation_id="conv-1", task_id="task-1")

        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.update),
            {
                "where": {"id": "ebr-1"},
                "data": {"conversationId": "conv-1", "taskId": "task-1"},
            },
        )

    async def test_delete_external_build_request_removes_unfinished_claim(self) -> None:
        db = _fake_db()

        with mock.patch.object(agent_access, "get_db", mock.AsyncMock(return_value=db)):
            await agent_access.delete_external_build_request("ebr-1")

        self.assertEqual(
            _await_kwargs(db.externalbuildrequest.delete),
            {"where": {"id": "ebr-1"}},
        )


if __name__ == "__main__":
    unittest.main()
