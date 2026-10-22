"""Opt-in PostgreSQL integration coverage for development account create claims."""

from __future__ import annotations

import asyncio
import contextvars
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException
from prisma import Prisma
from prisma.enums import AuthProvider

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_accounts import AccountDevelopmentService, AccountWorkspaceCreateRequest

_task_db: contextvars.ContextVar[Prisma] = contextvars.ContextVar("development_account_test_db")


async def _get_task_db() -> Prisma:
    return _task_db.get()


@unittest.skipUnless(
    os.environ.get("RAGTIME_BYO_INTEGRATION") == "1" and bool(os.environ.get("RAGTIME_BYO_DATABASE_URL") or os.environ.get("DATABASE_URL")),
    "requires RAGTIME_BYO_INTEGRATION=1 and a task-owned RAGTIME_BYO_DATABASE_URL or DATABASE_URL",
)
class DevelopmentAccountsDatabaseTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the ledger with independent Prisma clients and real transactions."""

    async def asyncSetUp(self) -> None:
        database_url = os.environ.get("RAGTIME_BYO_DATABASE_URL") or os.environ["DATABASE_URL"]
        self.first = Prisma(datasource={"url": database_url})
        self.second = Prisma(datasource={"url": database_url})
        await self.first.connect()
        await self.second.connect()
        self.user_ids = [str(uuid4()), str(uuid4())]
        self.created_request_ids: list[str] = []
        for user_id in self.user_ids:
            await self.first.user.create(
                data={"id": user_id, "username": f"byoh-account-db-{user_id}", "authProvider": AuthProvider.local}
            )
        self._database_patch = mock.patch("ragtime.userspace.development_accounts.get_db", new=_get_task_db)
        self._database_patch.start()

    async def asyncTearDown(self) -> None:
        ledger = getattr(self.first, "workspacedevelopmentcreaterequest")
        rows = await ledger.find_many(where={"userId": {"in": self.user_ids}})
        self.created_request_ids.extend(str(row.id) for row in rows)
        if self.created_request_ids:
            await ledger.delete_many(where={"id": {"in": list(set(self.created_request_ids))}})
        await self.first.user.delete_many(where={"id": {"in": self.user_ids}})
        self._database_patch.stop()
        await self.first.disconnect()
        await self.second.disconnect()

    @staticmethod
    def _principal(user_id: str) -> DevelopmentPrincipal:
        return DevelopmentPrincipal(user_id=user_id, is_admin=False, auth_source="user")

    @staticmethod
    def _request(key: str, *, name: str = "Synthetic account workspace") -> AccountWorkspaceCreateRequest:
        return AccountWorkspaceCreateRequest(name=name, idempotency_key=key, tool_selection_mode="custom", selected_tool_ids=[])

    async def _create(self, service: AccountDevelopmentService, db: Prisma, user_id: str, request: AccountWorkspaceCreateRequest) -> dict[str, object]:
        token = _task_db.set(db)
        try:
            return await service.create_workspace(self._principal(user_id), request)
        finally:
            _task_db.reset(token)

    async def test_two_services_single_dispatch_conflict_and_completed_replay(self) -> None:
        first_service, second_service = AccountDevelopmentService(), AccountDevelopmentService()
        dispatch = mock.AsyncMock(side_effect=lambda request, user_id, **kwargs: SimpleNamespace(id=kwargs["creation_workspace_id"], name=request.name))
        replay_workspace = mock.AsyncMock(side_effect=lambda db, workspace_id, user_id: SimpleNamespace(id=workspace_id, name="Synthetic account workspace"))
        request = self._request("same-key-0001")

        with (
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=dispatch),
            mock.patch.object(AccountDevelopmentService, "_authorized_workspace", new=replay_workspace),
        ):
            first, second = await asyncio.gather(
                self._create(first_service, self.first, self.user_ids[0], request),
                self._create(second_service, self.second, self.user_ids[0], request),
            )
            self.assertEqual(first["workspace_id"], second["workspace_id"])
            self.assertEqual(dispatch.await_count, 1)
            self.assertIn(first["status"], {"completed", "pending", "interrupted"})
            self.assertIn(second["status"], {"completed", "pending", "interrupted"})

            replay = await self._create(second_service, self.second, self.user_ids[0], request)
            self.assertEqual(replay["status"], "completed")
            self.assertTrue(replay["replayed"])
            self.assertEqual(dispatch.await_count, 1)

            with self.assertRaises(HTTPException) as conflict:
                await self._create(second_service, self.second, self.user_ids[0], self._request("same-key-0001", name="different payload"))
            self.assertEqual(conflict.exception.status_code, 409)

    async def test_atomic_five_claim_limit_replays_are_free_and_users_are_isolated(self) -> None:
        first_service, second_service = AccountDevelopmentService(), AccountDevelopmentService()
        dispatch = mock.AsyncMock(side_effect=lambda request, user_id, **kwargs: SimpleNamespace(id=kwargs["creation_workspace_id"], name=request.name))
        replay_workspace = mock.AsyncMock(side_effect=lambda db, workspace_id, user_id: SimpleNamespace(id=workspace_id, name="Synthetic account workspace"))
        with (
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=dispatch),
            mock.patch.object(AccountDevelopmentService, "_authorized_workspace", new=replay_workspace),
        ):
            requests = [self._request(f"first-user-key-{number:02d}") for number in range(6)]
            results = await asyncio.gather(
                *(self._create(first_service if number % 2 == 0 else second_service, self.first if number % 2 == 0 else self.second, self.user_ids[0], request) for number, request in enumerate(requests)),
                return_exceptions=True,
            )
            admitted = [(request, result) for request, result in zip(requests, results) if isinstance(result, dict)]
            limited = [result for result in results if isinstance(result, HTTPException)]
            self.assertEqual(len(admitted), 5)
            self.assertEqual(len(limited), 1)
            self.assertEqual(limited[0].status_code, 429)
            self.assertEqual(dispatch.await_count, 5)

            replay = await self._create(second_service, self.second, self.user_ids[0], admitted[0][0])
            self.assertTrue(replay["replayed"])
            self.assertEqual(dispatch.await_count, 5)

            with self.assertRaises(HTTPException) as limited:
                await self._create(second_service, self.second, self.user_ids[0], self._request("first-user-key-06"))
            self.assertEqual(limited.exception.status_code, 429)

            isolated = await self._create(second_service, self.second, self.user_ids[1], self._request("first-user-key-00"))
            self.assertEqual(isolated["status"], "completed")
            self.assertEqual(dispatch.await_count, 6)
