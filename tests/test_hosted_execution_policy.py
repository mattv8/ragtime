import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.core import hosted_execution_policy


class HostedExecutionPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_disable_wins_over_user_enable(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=False))),
            user=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=True)),
                find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="user-1", hostedChatEnabled=True)]),
            ),
        )
        with mock.patch.object(hosted_execution_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertFalse(await hosted_execution_policy.hosted_execution_enabled("user-1"))

    async def test_context_principals_are_checked(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=True))),
            user=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=False)),
                find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="user-1", hostedChatEnabled=False)]),
            ),
        )
        with mock.patch.object(hosted_execution_policy, "get_db", mock.AsyncMock(return_value=db)):
            with hosted_execution_policy.hosted_execution_context("user-1"):
                with self.assertRaises(HTTPException) as raised:
                    await hosted_execution_policy.require_hosted_execution()
        self.assertEqual(raised.exception.status_code, 403)
        detail = raised.exception.detail
        assert isinstance(detail, dict)
        self.assertEqual(detail.get("code"), "hosted_execution_disabled")

    async def test_explicit_principal_cannot_bypass_disabled_context_owner(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=True))),
            user=SimpleNamespace(
                find_many=mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(id="caller", hostedChatEnabled=True),
                        SimpleNamespace(id="owner", hostedChatEnabled=False),
                    ]
                )
            ),
        )
        with mock.patch.object(hosted_execution_policy, "get_db", mock.AsyncMock(return_value=db)):
            with hosted_execution_policy.hosted_execution_context("owner"):
                with self.assertRaises(HTTPException):
                    await hosted_execution_policy.require_hosted_execution("caller")
        db.user.find_many.assert_awaited_once()

    async def test_database_failure_fails_closed(self) -> None:
        with mock.patch.object(hosted_execution_policy, "get_db", mock.AsyncMock(side_effect=RuntimeError("offline"))):
            self.assertFalse(await hosted_execution_policy.hosted_execution_enabled())
