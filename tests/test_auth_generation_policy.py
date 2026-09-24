import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

import httpx
from fastapi import FastAPI
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User
from starlette.requests import Request

from ragtime.api import auth as api_auth
from ragtime.core import generation_policy

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _user(
    *,
    chat: bool | None,
    userspace: bool | None,
    user_id: str = "user-1",
    username: str = "alice",
    role: UserRole = UserRole.user,
) -> User:
    return User(
        id=user_id,
        username=username,
        displayName="Alice",
        email="alice@example.com",
        role=role,
        authProvider=AuthProvider.local_managed,
        themePack=None,
        roleManuallySet=False,
        sourceProvider=None,
        sourceSyncedAt=None,
        sourceExpiresAt=None,
        cachedGroups=cast(Json, "[]"),
        chatEnabled=chat,
        userspaceGenerationEnabled=userspace,
        createdAt=NOW,
        updatedAt=NOW,
        securityGeneration=0,
    )


def _db(settings_row: object | None) -> SimpleNamespace:
    empty_delegate = SimpleNamespace(find_many=mock.AsyncMock(return_value=[]))
    return SimpleNamespace(
        appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=settings_row)),
        authgroupmembership=empty_delegate,
        authgroup=empty_delegate,
        usermfafactor=empty_delegate,
        usermfarecoverycode=SimpleNamespace(find_many=mock.AsyncMock(return_value=[]), count=mock.AsyncMock(return_value=0)),
        userwebauthncredential=empty_delegate,
    )


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/auth/status",
            "headers": [(b"host", b"ragtime.test")],
            "scheme": "https",
        }
    )


class BulkUserGenerationPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_bulk_responses_allow_explicit_overrides_when_global_defaults_are_disabled(self) -> None:
        db = _db(SimpleNamespace(chatEnabled=False, userspaceGenerationEnabled=False))
        with (
            mock.patch.object(api_auth, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                api_auth,
                "get_auth_provider_config",
                mock.AsyncMock(return_value=SimpleNamespace(totp_policy="optional", totp_required_group_ids=[])),
            ),
        ):
            response = await api_auth._bulk_user_responses([_user(chat=True, userspace=True)])

        self.assertTrue(response[0].chat_enabled_effective)
        self.assertTrue(response[0].userspace_generation_enabled_effective)

    async def test_bulk_responses_fail_closed_when_settings_row_or_required_field_is_missing(self) -> None:
        for settings_row in (None, SimpleNamespace()):
            with self.subTest(settings_row=settings_row):
                db = _db(settings_row)
                with (
                    mock.patch.object(api_auth, "get_db", mock.AsyncMock(return_value=db)),
                    mock.patch.object(
                        api_auth,
                        "get_auth_provider_config",
                        mock.AsyncMock(return_value=SimpleNamespace(totp_policy="optional", totp_required_group_ids=[])),
                    ),
                ):
                    response = await api_auth._bulk_user_responses([_user(chat=True, userspace=True)])

                self.assertFalse(response[0].chat_enabled_effective)
                self.assertFalse(response[0].userspace_generation_enabled_effective)


class UserGenerationPolicyRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_patch_serializes_explicit_enable_and_null_reset_with_global_default_off(self) -> None:
        user = _user(chat=None, userspace=None)
        db = _db(SimpleNamespace(chatEnabled=False, userspaceGenerationEnabled=False))

        async def update_user(*, where: dict, data: dict) -> User:
            self.assertEqual(where, {"id": "user-1"})
            for field, value in data.items():
                setattr(user, field, value)
            return user

        db.user = SimpleNamespace(
            find_many=mock.AsyncMock(return_value=[user]),
            update=mock.AsyncMock(side_effect=update_user),
        )
        admin = _user(chat=None, userspace=None, user_id="admin-1", username="admin", role=UserRole.admin)
        patches = (
            mock.patch.object(api_auth, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "user_has_enabled_totp", mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "mfa_needed_for_user", mock.AsyncMock(return_value=False)),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            enabled = await api_auth.update_user_generation_policy(
                "user-1",
                api_auth.UpdateUserGenerationPolicyRequest.model_validate({"chat_enabled": True}),
                current_user=admin,
            )
            reset = await api_auth.update_user_generation_policy(
                "user-1",
                api_auth.UpdateUserGenerationPolicyRequest.model_validate({"chat_enabled": None}),
                current_user=admin,
            )

        self.assertTrue(enabled.chat_enabled)
        self.assertTrue(enabled.chat_enabled_effective)
        self.assertIsNone(enabled.userspace_generation_enabled)
        self.assertFalse(enabled.userspace_generation_enabled_effective)
        self.assertIsNone(reset.chat_enabled)
        self.assertFalse(reset.chat_enabled_effective)

    async def test_auth_status_agrees_with_explicit_user_enable_under_global_default_off(self) -> None:
        user = _user(chat=True, userspace=True)
        db = _db(SimpleNamespace(chatEnabled=False, userspaceGenerationEnabled=False))
        db.user = SimpleNamespace(find_many=mock.AsyncMock(return_value=[user]))
        with (
            mock.patch.object(generation_policy, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                api_auth,
                "get_app_settings",
                mock.AsyncMock(return_value={"chat_enabled": False, "userspace_generation_enabled": False}),
            ),
            mock.patch.object(api_auth, "build_auth_method_statuses", mock.AsyncMock(return_value=[])),
            mock.patch.object(api_auth, "_get_debug_totp_code", mock.AsyncMock(return_value=None)),
        ):
            response = await api_auth.get_auth_status(_request(), current_user=user)

        self.assertTrue(response.chat_enabled)
        self.assertTrue(response.userspace_generation_enabled)

    async def test_route_rejects_non_admin_before_generation_policy_update(self) -> None:
        app = FastAPI()
        app.include_router(api_auth.router)
        app.dependency_overrides[api_auth.get_current_user] = lambda: SimpleNamespace(id="user-1", role="user")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.patch("/auth/users/user-1/generation-policy", json={"chat_enabled": True})
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"detail": "Admin access required"})
