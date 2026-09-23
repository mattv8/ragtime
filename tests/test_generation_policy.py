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
    async def test_nullable_override_precedence_matrix_is_independent_per_surface(self) -> None:
        for global_enabled, override, expected in (
            (False, None, False),
            (False, True, True),
            (False, False, False),
            (True, None, True),
            (True, True, True),
            (True, False, False),
        ):
            for surface in ("chat", "userspace"):
                with self.subTest(surface=surface, global_enabled=global_enabled, override=override):
                    db = _db(
                        chat=global_enabled,
                        userspace=global_enabled,
                        user_chat=override,
                        user_userspace=override,
                    )
                    check = generation_policy.chat_generation_enabled if surface == "chat" else generation_policy.userspace_generation_enabled
                    with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
                        self.assertEqual(await check("user"), expected)

    async def test_userspace_override_does_not_change_chat(self) -> None:
        db = _db(chat=True, userspace=True, user_chat=True, user_userspace=False)
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertTrue(await generation_policy.chat_generation_enabled("user"))
            self.assertFalse(await generation_policy.userspace_generation_enabled("user"))

    async def test_explicit_enable_overrides_disabled_global_for_each_surface(self) -> None:
        db = _db(chat=False, userspace=False, user_chat=True, user_userspace=True)
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertTrue(await generation_policy.chat_generation_enabled("user"))
            self.assertTrue(await generation_policy.userspace_generation_enabled("user"))

    async def test_no_principal_uses_global_default_instead_of_unconditional_allow(self) -> None:
        for global_enabled in (False, True):
            with self.subTest(global_enabled=global_enabled):
                db = _db(chat=global_enabled)
                with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
                    self.assertEqual(await generation_policy.chat_generation_enabled(), global_enabled)

    async def test_missing_settings_or_required_setting_fails_closed(self) -> None:
        for settings_row in (None, SimpleNamespace(userspaceGenerationEnabled=True)):
            with self.subTest(settings_row=settings_row):
                db = _db()
                db.appsettings.find_unique.return_value = settings_row
                with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
                    self.assertFalse(await generation_policy.chat_generation_enabled("user"))

    async def test_missing_concrete_principal_fails_closed(self) -> None:
        db = _db()
        db.user.find_many.return_value = []
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            self.assertFalse(await generation_policy.chat_generation_enabled("missing-user"))

    async def test_mixed_principals_each_require_an_explicit_enable_when_global_is_disabled(self) -> None:
        db = _db(chat=False)
        db.user.find_many.return_value = [
            SimpleNamespace(id="caller", chatEnabled=True),
            SimpleNamespace(id="owner", chatEnabled=None),
        ]
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "owner"):
                with self.assertRaises(HTTPException):
                    await generation_policy.require_generation("caller")

        db.user.find_many.return_value[1].chatEnabled = True
        with mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)):
            with generation_policy.generation_context("chat", "owner"):
                await generation_policy.require_generation("caller")

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
