from __future__ import annotations

import asyncio
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
        with (
            mock.patch.object(self.service, "_audit", mock.AsyncMock()),
            self.assertRaises(HTTPException),
        ):
            await self.service.verify_runtime_bridge_token(wrong)

    async def test_verify_rejects_stopped_or_mismatched_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")

        stopped = mock.Mock()
        stopped.state = "stopped"
        stopped.workspaceId = "ws-1"
        stopped.leasedByUserId = "user-1"
        current = mock.Mock()
        current.id = "sess-1"
        current.state = "running"
        current.workspaceId = "ws-1"
        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=stopped),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current),
            ),
            mock.patch.object(self.service, "_audit", mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

        other = mock.Mock()
        other.state = "running"
        other.workspaceId = "ws-2"
        other.leasedByUserId = "user-1"
        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=other),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current),
            ),
            mock.patch.object(self.service, "_audit", mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=None),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current),
            ),
            mock.patch.object(self.service, "_audit", mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException):
                await self.service.verify_runtime_bridge_token(token)

    async def test_verify_rejects_empty_lease_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")
        active = mock.Mock()
        active.state = "running"
        active.workspaceId = "ws-1"
        active.leasedByUserId = "   "
        active.id = "sess-1"
        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=active),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=active),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service.verify_runtime_bridge_token(token)
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_verify_accepts_active_session(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-1")
        active = mock.Mock()
        active.id = "sess-1"
        active.state = "running"
        active.workspaceId = "ws-1"
        active.leasedByUserId = "user-7"
        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=active),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=active),
            ),
        ):
            claims = await self.service.verify_runtime_bridge_token(token)
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")
        self.assertEqual(claims["leased_by_user_id"], "user-7")

    async def test_verify_rejects_mismatched_current_session_and_audits_sanitized_failure(self) -> None:
        token = self.service.build_runtime_bridge_token("ws-1", "sess-stale")
        stale = mock.Mock()
        stale.id = "sess-stale"
        stale.state = "running"
        stale.workspaceId = "ws-1"
        stale.leasedByUserId = "user-1"
        current = mock.Mock()
        current.id = "sess-current"
        current.state = "running"
        current.workspaceId = "ws-1"

        with (
            mock.patch.object(
                self.service,
                "_get_runtime_session_record",
                mock.AsyncMock(return_value=stale),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current),
            ),
            mock.patch.object(self.service, "_audit", mock.AsyncMock()) as audit,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service.verify_runtime_bridge_token(token)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Runtime bridge session mismatch")
        audit.assert_awaited_once()
        payload = audit.await_args.kwargs["payload"]
        self.assertEqual(payload["reason"], "session_mismatch")
        self.assertEqual(payload["workspace_id"], "ws-1")
        self.assertEqual(payload["token_session_id"], "sess-stale")
        self.assertEqual(payload["current_session_id"], "sess-current")
        self.assertNotIn(token, str(payload))

    async def test_verify_logs_but_does_not_audit_expired_token(self) -> None:
        expired = jwt.encode(
            {
                "kind": _RUNTIME_BRIDGE_TOKEN_KIND,
                "workspace_id": "ws-1",
                "session_id": "sess-1",
                "iat": int(time.time()) - 120,
                "exp": int(time.time()) - 60,
            },
            settings.encryption_key,
            algorithm=settings.jwt_algorithm,
        )
        with (
            mock.patch.object(self.service, "_audit", mock.AsyncMock()) as audit,
            mock.patch("ragtime.userspace.runtime_service.logger.warning") as log_warning,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await self.service.verify_runtime_bridge_token(expired)

        self.assertEqual(ctx.exception.status_code, 401)
        audit.assert_not_awaited()
        log_warning.assert_called_once()
        self.assertNotIn(expired, str(log_warning.call_args))

    async def test_runtime_bridge_auth_failure_audit_deduplicates_repeated_failures(self) -> None:
        claims = {"workspace_id": "ws-1", "session_id": "sess-1"}
        with (
            mock.patch.object(self.service, "_audit", mock.AsyncMock()) as audit,
            mock.patch("ragtime.userspace.runtime_service.logger.warning") as log_warning,
        ):
            await self.service._log_runtime_bridge_auth_failure(
                reason="session_mismatch",
                claims=claims,
                current_session_id="sess-current",
            )
            await self.service._log_runtime_bridge_auth_failure(
                reason="session_mismatch",
                claims=claims,
                current_session_id="sess-current",
            )

        audit.assert_awaited_once()
        self.assertEqual(log_warning.call_count, 2)
        self.assertNotIn("Bearer", str(log_warning.call_args_list))
        self.assertNotIn(".", next(iter(self.service._runtime_bridge_auth_failure_audit_dedupe.keys())))

    async def test_runtime_bridge_auth_failure_audit_deduplicates_concurrent_failures(self) -> None:
        claims = {"workspace_id": "ws-1", "session_id": "sess-1"}
        audit = mock.AsyncMock()

        with (
            mock.patch.object(self.service, "_audit", audit),
            mock.patch("ragtime.userspace.runtime_service.logger.warning"),
        ):
            await asyncio.gather(
                *[
                    self.service._log_runtime_bridge_auth_failure(
                        reason="session_mismatch",
                        claims=claims,
                        current_session_id="sess-current",
                    )
                    for _ in range(5)
                ]
            )

        audit.assert_awaited_once()
