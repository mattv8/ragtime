import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.core import generation_policy


class ChatGenerationPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_disable_wins_over_user_enable(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=False))),
            user=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=True)),
                find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="user-1", chatEnabled=True)]),
            ),
        )
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertFalse(await generation_policy.chat_generation_enabled("user-1"))

    async def test_context_principals_are_checked(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=True))),
            user=SimpleNamespace(
                find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=False)),
                find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="user-1", chatEnabled=False)]),
            ),
        )
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "user-1"):
                with self.assertRaises(HTTPException) as raised:
                    await generation_policy.require_generation()
        self.assertEqual(raised.exception.status_code, 403)
        detail = raised.exception.detail
        assert isinstance(detail, dict)
        self.assertEqual(detail.get("code"), "chat_generation_disabled")

    async def test_explicit_principal_cannot_bypass_disabled_context_owner(self) -> None:
        db = SimpleNamespace(
            appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=True))),
            user=SimpleNamespace(
                find_many=mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(id="caller", chatEnabled=True),
                        SimpleNamespace(id="owner", chatEnabled=False),
                    ]
                )
            ),
        )
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "owner"):
                with self.assertRaises(HTTPException):
                    await generation_policy.require_generation("caller")
        db.user.find_many.assert_awaited_once()

    async def test_database_failure_fails_closed(self) -> None:
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(side_effect=RuntimeError("offline"))):
            self.assertFalse(await generation_policy.chat_generation_enabled())
