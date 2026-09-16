"""Database integration tests for user space runtime restart behavior.

This test suite requires a live Postgres/Prisma database and is opt-in via
RAGTIME_RUNTIME_RESTART_DATABASE_INTEGRATION=1 environment variable.
"""

import asyncio
import contextvars
import os
import unittest
from datetime import timedelta
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException
from prisma import Prisma
from prisma.enums import AuthProvider, RuntimeSessionState

from ragtime.core.datetimes import utc_now
from ragtime.userspace.runtime_service import UserSpaceRuntimeService

_task_db: contextvars.ContextVar[Prisma] = contextvars.ContextVar("runtime_restart_test_db")


async def _get_task_db() -> Prisma:
    return _task_db.get()


@unittest.skipUnless(os.environ.get("RAGTIME_RUNTIME_RESTART_DATABASE_INTEGRATION") == "1", "local Prisma/Postgres opt-in")
class RuntimeRestartDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.first = Prisma()
        self.second = Prisma()
        await self.first.connect()
        await self.second.connect()
        self.user_ids = [str(uuid4()), str(uuid4())]
        self.workspace_id = str(uuid4())
        for index, user_id in enumerate(self.user_ids):
            await self.first.user.create(data={"id": user_id, "username": f"restart-db-{user_id}", "authProvider": AuthProvider.local})
        await self.first.workspace.create(data={"id": self.workspace_id, "name": f"restart-db-{self.workspace_id}", "ownerUserId": self.user_ids[0]})
        await self.first.userspaceruntimesession.create(
            data={
                "id": str(uuid4()),
                "workspaceId": self.workspace_id,
                "leasedByUserId": self.user_ids[0],
                "state": RuntimeSessionState.running,
                "providerSessionId": "provider-session",
            }
        )

    async def asyncTearDown(self) -> None:
        await getattr(self.first, "workspaceruntimeoperation").delete_many(where={"workspaceId": self.workspace_id})
        await self.first.userspaceruntimesession.delete_many(where={"workspaceId": self.workspace_id})
        await self.first.workspace.delete(where={"id": self.workspace_id})
        await self.first.user.delete_many(where={"id": {"in": self.user_ids}})
        await self.first.disconnect()
        await self.second.disconnect()

    async def test_advisory_lock_serializes_idempotency_and_workspace_throttle(self) -> None:
        first_service = UserSpaceRuntimeService()
        second_service = UserSpaceRuntimeService()
        first_dispatch = mock.AsyncMock(return_value={"runtime_operation_id": "provider-operation-1"})
        second_dispatch = mock.AsyncMock(return_value={"runtime_operation_id": "provider-operation-2"})

        async def request(service: UserSpaceRuntimeService, db: Prisma, user_id: str, key: str, reason: str = "") -> dict:
            token = _task_db.set(db)
            try:
                return await service.request_app_restart(self.workspace_id, user_id, key, reason=reason)
            finally:
                _task_db.reset(token)

        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", new=_get_task_db),
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", new=mock.AsyncMock()),
            mock.patch.object(first_service, "_runtime_provider_restart_app", new=first_dispatch),
            mock.patch.object(second_service, "_runtime_provider_restart_app", new=second_dispatch),
        ):
            same_key = await asyncio.gather(
                request(first_service, self.first, self.user_ids[0], "same-key-123"),
                request(second_service, self.second, self.user_ids[0], "same-key-123"),
            )
            self.assertEqual(same_key[0]["id"], same_key[1]["id"])
            self.assertEqual(first_dispatch.await_count + second_dispatch.await_count, 1)
            dispatched_args = first_dispatch.await_args or second_dispatch.await_args
            assert dispatched_args is not None
            dispatched_request_id = dispatched_args.args[1]
            self.assertEqual(dispatched_request_id, same_key[0]["id"])

            with self.assertRaises(HTTPException) as throttled:
                await request(second_service, self.second, self.user_ids[1], "other-key-123")
            self.assertEqual(throttled.exception.status_code, 429)

            await getattr(self.first, "workspaceruntimeoperation").update(
                where={"id": same_key[0]["id"]},
                data={"createdAt": utc_now() - timedelta(seconds=61)},
            )
            other_user_same_key = await request(second_service, self.second, self.user_ids[1], "same-key-123")
            self.assertNotEqual(other_user_same_key["id"], same_key[0]["id"])
            self.assertEqual(first_dispatch.await_count + second_dispatch.await_count, 2)
            request_ids = {args.args[1] for args in (first_dispatch.await_args_list + second_dispatch.await_args_list)}
            self.assertEqual(request_ids, {same_key[0]["id"], other_user_same_key["id"]})

            with self.assertRaises(HTTPException) as conflict:
                await request(second_service, self.second, self.user_ids[0], "same-key-123", reason="different")
            self.assertEqual(conflict.exception.status_code, 409)
