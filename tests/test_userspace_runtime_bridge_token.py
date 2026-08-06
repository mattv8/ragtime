from __future__ import annotations

import time
import unittest
from unittest import mock

from fastapi import HTTPException
from jose import jwt  # type: ignore[import-untyped]

from ragtime.config.settings import settings
from ragtime.userspace.runtime_service import (
    _RUNTIME_BRIDGE_TOKEN_KIND,
    UserSpaceRuntimeService,
)


def _decode(token: str) -> dict:
    return jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])


class RuntimeBridgeTokenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceRuntimeService()

    def test_build_token_embeds_kind_workspace_session_and_expiry(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")
        claims = _decode(token)
        self.assertEqual(claims["kind"], _RUNTIME_BRIDGE_TOKEN_KIND)
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")
        self.assertGreater(claims["exp"], int(time.time()))

    async def test_verify_rejects_missing_and_malformed_tokens(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await self.service.verify_runtime_bridge_token(None)
        self.assertEqual(ctx.exception.status_code, 401)
        with self.assertRaises(HTTPException):
            await self.service.verify_runtime_bridge_token("not-a-jwt")

    async def test_verify_rejects_wrong_kind(self) -> None:
        wrong = jwt.encode(
            {
                "kind": "userspace_preview_session",
                "workspace_id": "ws-1",
                "session_id": "s",
                "exp": int(time.time()) + 60,
            },
            settings.encryption_key,
            algorithm=settings.jwt_algorithm,
        )
        with self.assertRaises(HTTPException):
            await self.service.verify_runtime_bridge_token(wrong)

    async def test_verify_rejects_stopped_or_mismatched_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")

        stopped = mock.Mock()
        stopped.state = "stopped"
        stopped.workspaceId = "ws-1"
        stopped.leasedByUserId = "user-1"
        with mock.patch.object(
            self.service,
            "_get_runtime_session_record",
            mock.AsyncMock(return_value=stopped),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

        other = mock.Mock()
        other.state = "running"
        other.workspaceId = "ws-2"
        other.leasedByUserId = "user-1"
        with mock.patch.object(
            self.service,
            "_get_runtime_session_record",
            mock.AsyncMock(return_value=other),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

        with mock.patch.object(
            self.service,
            "_get_runtime_session_record",
            mock.AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

    async def test_verify_rejects_empty_lease_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")
        active = mock.Mock()
        active.state = "running"
        active.workspaceId = "ws-1"
        active.leasedByUserId = "   "
        with mock.patch.object(
            self.service,
            "_get_runtime_session_record",
            mock.AsyncMock(return_value=active),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service.verify_runtime_bridge_token(token)
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_verify_accepts_active_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")
        active = mock.Mock()
        active.state = "running"
        active.workspaceId = "ws-1"
        active.leasedByUserId = "user-7"
        with mock.patch.object(
            self.service,
            "_get_runtime_session_record",
            mock.AsyncMock(return_value=active),
        ):
            claims = await self.service.verify_runtime_bridge_token(token)
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")
        self.assertEqual(claims["leased_by_user_id"], "user-7")
