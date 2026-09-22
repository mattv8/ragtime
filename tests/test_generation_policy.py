import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException

from ragtime.core import generation_policy


def _db(*, chat: bool = True, userspace: bool = True, user_chat: bool | None = None, user_userspace: bool | None = None) -> SimpleNamespace:
    user = SimpleNamespace(id="user", chatEnabled=user_chat, userspaceGenerationEnabled=user_userspace)
    return SimpleNamespace(
        appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=chat, userspaceGenerationEnabled=userspace))),
        user=SimpleNamespace(find_many=mock.AsyncMock(return_value=[user])),
    )


class GenerationPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_and_user_overrides_are_independent(self) -> None:
        db = _db(chat=False, userspace=True, user_chat=True, user_userspace=None)
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertFalse(await generation_policy.chat_generation_enabled("user"))
            self.assertTrue(await generation_policy.userspace_generation_enabled("user"))

    async def test_userspace_override_does_not_change_chat(self) -> None:
        db = _db(chat=True, userspace=True, user_chat=True, user_userspace=False)
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertTrue(await generation_policy.chat_generation_enabled("user"))
            self.assertFalse(await generation_policy.userspace_generation_enabled("user"))

    async def test_context_checks_caller_and_owner(self) -> None:
        db = _db()
        db.user.find_many.return_value = [
            SimpleNamespace(id="caller", chatEnabled=True),
            SimpleNamespace(id="owner", chatEnabled=False),
        ]
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "owner"):
                with self.assertRaises(HTTPException) as raised:
                    await generation_policy.require_generation("caller")
        detail = cast(dict[str, str], raised.exception.detail)
        self.assertEqual(detail["code"], "chat_generation_disabled")

    async def test_v1_is_trusted_but_unknown_scope_denies(self) -> None:
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(side_effect=RuntimeError("offline"))):
            with generation_policy.generation_context("v1"):
                await generation_policy.require_generation()
            with self.assertRaises(HTTPException) as raised:
                await generation_policy.require_generation()
        detail = cast(dict[str, str], raised.exception.detail)
        self.assertEqual(detail["code"], "generation_scope_unknown")

    async def test_nested_context_preserves_root_owner_principal(self) -> None:
        db = _db()
        db.user.find_many.return_value = [
            SimpleNamespace(id="caller", chatEnabled=True),
            SimpleNamespace(id="owner", chatEnabled=False),
        ]
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "owner"):
                with generation_policy.generation_context("chat", "caller"):
                    with self.assertRaises(HTTPException):
                        await generation_policy.require_generation()
