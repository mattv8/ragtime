import sys
import time as _time
import types
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    fake_prompts_module.build_workspace_scm_setup_prompt = lambda *args, **kwargs: ""
    fake_rag_package.prompts = fake_prompts_module
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import UserSpaceRuntimeService
from ragtime.userspace.service import UserSpaceService

_NOW = datetime(2026, 5, 8, tzinfo=timezone.utc)


class _FakeRuntimeSessionTable:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def update(
        self,
        *,
        where: dict[str, str],
        data: dict[str, Any],
    ) -> SimpleNamespace:
        self.updates.append({"where": where, "data": data})
        return SimpleNamespace(id=where["id"], **data)


class _FakeRuntimeDb:
    def __init__(self, table: _FakeRuntimeSessionTable) -> None:
        self.userspaceruntimesession = table


class _RuntimeRefreshService(UserSpaceRuntimeService):
    def __init__(self, active_row: SimpleNamespace) -> None:
        super().__init__()
        self.active_row = active_row
        self.invalidated: list[tuple[str, bool]] = []

    async def _get_active_session_row(self, workspace_id: str) -> SimpleNamespace | None:
        return self.active_row

    async def _runtime_provider_refresh_mounts(
        self,
        provider_session_id: str | None,
        workspace_mounts: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        raise HTTPException(
            status_code=502,
            detail='Runtime manager request failed (404): {"detail":"Runtime session not found"}',
        )

    async def _invalidate_workspace_runtime_caches(
        self,
        workspace_id: str,
        *,
        invalidate_preview_host: bool = False,
    ) -> None:
        self.invalidated.append((workspace_id, invalidate_preview_host))


class _FakeWorkspaceMountTable:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows

    async def find_many(self, **_: Any) -> list[SimpleNamespace]:
        return self.rows


class _FakeMountWatchDb:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.workspacemount = _FakeWorkspaceMountTable(rows)


class _MountWatchService(UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.signature_checks = 0

    async def _global_workspace_mount_sync_interval_seconds(self) -> int:
        return 30

    async def _workspace_mount_target_signature(
        self,
        workspace_id: str,
        target_path: str,
    ) -> tuple[Any, ...]:
        self.signature_checks += 1
        return ((target_path, True, 1, 1, 1, "digest"),)


class UserSpaceRuntimeMountRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_refresh_marks_missing_provider_session_stopped(self) -> None:
        workspace_id = "workspace-1"
        mount_id = "mount-1"
        active_row = SimpleNamespace(
            id="session-1",
            workspaceId=workspace_id,
            leasedByUserId="user-1",
            state="running",
            runtimeProvider="microvm_pool_v1",
            providerSessionId="mgr-missing",
            previewInternalUrl=None,
            launchFramework=None,
            launchCommand=None,
            launchCwd=None,
            launchPort=None,
            createdAt=_NOW,
            updatedAt=_NOW,
            lastHeartbeatAt=_NOW,
            idleExpiresAt=None,
            ttlExpiresAt=None,
            lastError=None,
        )
        table = _FakeRuntimeSessionTable()
        service = _RuntimeRefreshService(active_row)

        with (
            patch(
                "ragtime.userspace.runtime_service.get_db",
                AsyncMock(return_value=_FakeRuntimeDb(table)),
            ),
            patch(
                "ragtime.userspace.runtime_service.userspace_service.resolve_workspace_mounts_for_runtime",
                AsyncMock(
                    return_value=[
                        {"target_path": "mnt", "source_local_path": "/tmp/mnt"}
                    ]
                ),
            ),
        ):
            notice = await service.refresh_workspace_mount_after_sync(
                workspace_id,
                mount_id,
            )

        self.assertIn("previous runtime session is no longer active", notice or "")
        self.assertEqual(table.updates[0]["where"], {"id": "session-1"})
        self.assertEqual(table.updates[0]["data"]["state"], "stopped")
        self.assertIn(
            "Runtime manager request failed (404)",
            table.updates[0]["data"]["lastError"],
        )
        self.assertEqual(service.invalidated, [(workspace_id, True)])

    async def test_mount_watch_skips_full_signature_scan_until_throttle_expires(self) -> None:
        mount_id = "mount-1"
        mount = SimpleNamespace(
            id=mount_id,
            workspaceId="workspace-1",
            enabled=True,
            autoSyncEnabled=True,
            targetPath="mnt",
            syncIntervalSeconds=30,
            mountSource=SimpleNamespace(
                enabled=True,
                sourceType="ssh",
                syncIntervalSeconds=None,
            ),
            userMountSource=None,
        )
        service = _MountWatchService()
        now = _time.monotonic()
        service._workspace_mount_watch_next_due_monotonic[mount_id] = now + 30.0
        service._workspace_mount_watch_next_signature_monotonic[mount_id] = now + 30.0

        with patch(
            "ragtime.userspace.service.get_db",
            AsyncMock(return_value=_FakeMountWatchDb([mount])),
        ):
            await service._workspace_mount_watch_tick()

        self.assertEqual(service.signature_checks, 0)
        self.assertNotIn(mount_id, service._workspace_mount_watch_inflight)


if __name__ == "__main__":
    unittest.main()
