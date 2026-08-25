from __future__ import annotations

import sys
import types
import unittest
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest import mock

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.core.datetimes import utc_now
from ragtime.userspace.models import UserSpaceRuntimeSession
from ragtime.userspace.runtime_service import (
    _RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
    UserSpaceRuntimeService,
)


def _session() -> UserSpaceRuntimeSession:
    return cast(
        UserSpaceRuntimeSession,
        SimpleNamespace(
            id="sess-1",
            workspace_id="workspace-1",
            provider_session_id="mgr-1",
            state="running",
        ),
    )


def _healthy_provider_status() -> dict[str, object]:
    now = utc_now()
    return {
        "bridge_credential": {
            "bridge_url": "http://ragtime:8000/indexes/userspace/runtime-bridge",
            "token_kind": "userspace_runtime_bridge",
            "workspace_id": "workspace-1",
            "session_id": "sess-1",
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=2)).isoformat(),
        }
    }


class BridgeReadinessFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthy_precheck_uses_status_cache_and_skips_audit_query(self) -> None:
        service = UserSpaceRuntimeService()
        get_status = mock.AsyncMock(return_value=_healthy_provider_status())
        audit_lookup = mock.AsyncMock()

        with (
            mock.patch.object(service, "_runtime_provider_get_status", new=get_status),
            mock.patch.object(service, "_get_latest_runtime_bridge_success_at", new=audit_lookup),
        ):
            await service._ensure_workspace_preview_bridge_ready(_session())

        get_status.assert_awaited_once_with(
            "mgr-1",
            max_age_seconds=_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
            allow_stale_on_error=False,
        )
        audit_lookup.assert_not_awaited()

    async def test_recovery_recheck_forces_fresh_status(self) -> None:
        service = UserSpaceRuntimeService()
        unhealthy = {"bridge_credential": None}
        get_status = mock.AsyncMock(side_effect=[unhealthy, unhealthy, _healthy_provider_status()])
        get_bridge_status = mock.AsyncMock(
            side_effect=[
                SimpleNamespace(state="missing", expires_at=None, detail=None),
                SimpleNamespace(state="missing", expires_at=None, detail=None),
                SimpleNamespace(state="healthy", expires_at=utc_now() + timedelta(hours=2), detail=None),
            ]
        )

        with (
            mock.patch.object(service, "_runtime_provider_get_status", new=get_status),
            mock.patch.object(service, "_get_runtime_bridge_status_for_session", new=get_bridge_status),
            mock.patch.object(
                service,
                "restart_runtime_env_vars_and_wait",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                service,
                "_get_active_session_row",
                new=mock.AsyncMock(return_value=SimpleNamespace(id="sess-1")),
            ),
            mock.patch.object(service, "_to_runtime_session", new=mock.Mock(return_value=_session())),
        ):
            await service._ensure_workspace_preview_bridge_ready(_session())

        self.assertEqual(get_status.await_count, 3)
        second_call = get_status.await_args_list[1]
        third_call = get_status.await_args_list[2]
        self.assertEqual(second_call.kwargs["max_age_seconds"], 0)
        self.assertEqual(third_call.kwargs["max_age_seconds"], 0)
        self.assertEqual(get_bridge_status.await_count, 3)
        for call in get_bridge_status.await_args_list:
            self.assertFalse(call.kwargs["include_last_success"])

    async def test_user_facing_status_still_includes_last_success(self) -> None:
        service = UserSpaceRuntimeService()
        audit_lookup = mock.AsyncMock(return_value=None)

        with mock.patch.object(service, "_get_latest_runtime_bridge_success_at", new=audit_lookup):
            status = await service._get_runtime_bridge_status_for_session(
                _session(),
                _healthy_provider_status(),
            )

        audit_lookup.assert_awaited_once_with("workspace-1", "sess-1")
        self.assertEqual(status.state, "healthy")

    async def test_workspace_preview_status_excludes_last_success_when_requested(self) -> None:
        service = UserSpaceRuntimeService()
        provider_status = _healthy_provider_status()

        with (
            mock.patch.object(service, "_runtime_provider_get_status", new=mock.AsyncMock(return_value=provider_status)) as get_status,
            mock.patch.object(
                service,
                "_get_runtime_bridge_status_for_session",
                new=mock.AsyncMock(return_value=SimpleNamespace(state="healthy")),
            ) as get_bridge_status,
        ):
            await service._get_workspace_preview_bridge_status(
                _session(),
                max_age_seconds=0,
                include_last_success=False,
            )

        get_status.assert_awaited_once()
        get_bridge_status.assert_awaited_once_with(
            _session(),
            provider_status,
            include_last_success=False,
        )


if __name__ == "__main__":
    unittest.main()
