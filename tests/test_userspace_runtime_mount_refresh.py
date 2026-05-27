import sys
import tempfile
import time as _time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.models import UpdateUserspaceMountSourceRequest, UserSpaceWorkspace
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
        self.update_many_calls: list[dict[str, Any]] = []

    async def find_many(self, **_: Any) -> list[SimpleNamespace]:
        return self.rows

    async def update_many(
        self,
        *,
        where: dict[str, Any],
        data: dict[str, Any],
    ) -> None:
        self.update_many_calls.append({"where": where, "data": data})


class _FakeMountWatchDb:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.workspacemount = _FakeWorkspaceMountTable(rows)


class _FakeMountSourceTable:
    def __init__(self) -> None:
        self.update_calls: list[dict[str, Any]] = []

    async def update(
        self,
        *,
        where: dict[str, Any],
        data: dict[str, Any],
        include: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        self.update_calls.append({"where": where, "data": data, "include": include})
        return SimpleNamespace(id=str(where.get("id") or ""), enabled=bool(data.get("enabled", True)))


class _FakeWorkspaceMountCascadeTable:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows
        self.update_calls: list[dict[str, Any]] = []
        self.update_many_calls: list[dict[str, Any]] = []

    async def find_many(self, *, where: dict[str, Any]) -> list[SimpleNamespace]:
        mount_source_id = where.get("mountSourceId")
        require_enabled = where.get("enabled")
        results: list[SimpleNamespace] = []
        for row in self.rows:
            if str(getattr(row, "mountSourceId", "")) != str(mount_source_id):
                continue
            if require_enabled is not None and bool(getattr(row, "enabled", False)) != bool(require_enabled):
                continue
            results.append(row)
        return results

    async def update(self, *, where: dict[str, Any], data: dict[str, Any]) -> SimpleNamespace:
        self.update_calls.append({"where": where, "data": data})
        return SimpleNamespace(id=str(where.get("id") or ""), **data)

    async def update_many(self, *, where: dict[str, Any], data: dict[str, Any]) -> None:
        self.update_many_calls.append({"where": where, "data": data})


class _FakeWorkspaceTable:
    def __init__(self) -> None:
        self.update_calls: list[dict[str, Any]] = []

    async def update(self, *, where: dict[str, Any], data: dict[str, Any]) -> SimpleNamespace:
        self.update_calls.append({"where": where, "data": data})
        return SimpleNamespace(id=str(where.get("id") or ""), **data)


class _FakeMountSourceDisableDb:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.userspacemountsource = _FakeMountSourceTable()
        self.workspacemount = _FakeWorkspaceMountCascadeTable(rows)
        self.workspace = _FakeWorkspaceTable()

    async def query_raw(self, query: str, mount_source_id: str) -> list[dict[str, Any]]:
        if "SELECT id FROM workspace_mounts" in query:
            return [{"id": str(getattr(row, "id", ""))} for row in self.workspacemount.rows if str(getattr(row, "mountSourceId", "")) == str(mount_source_id)]
        return []


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


class _MountListService(UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.availability_checks: list[str] = []

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
        is_admin: bool = False,
    ) -> UserSpaceWorkspace:
        return UserSpaceWorkspace(
            id=workspace_id,
            name="Workspace",
            owner_user_id=user_id,
            members=[],
            created_at=_NOW,
            updated_at=_NOW,
        )

    async def _workspace_mount_editable_by_user(self, row: Any, user_id: str) -> bool:
        return True

    async def _check_mount_source_available(self, row: Any, mount_source: Any) -> bool:
        self.availability_checks.append(str(getattr(row, "id", "")))
        return False


class UserSpaceRuntimeMountRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabling_mount_source_force_unmounts_enabled_workspace_mounts(self) -> None:
        mount_source_id = "source-1"
        rows = [
            SimpleNamespace(
                id="mount-enabled-1",
                workspaceId="workspace-1",
                mountSourceId=mount_source_id,
                enabled=True,
                targetPath="/workspace/mount-a",
            ),
            SimpleNamespace(
                id="mount-enabled-2",
                workspaceId="workspace-2",
                mountSourceId=mount_source_id,
                enabled=True,
                targetPath="/workspace/mount-b",
            ),
            SimpleNamespace(
                id="mount-disabled",
                workspaceId="workspace-3",
                mountSourceId=mount_source_id,
                enabled=False,
                targetPath="/workspace/mount-c",
            ),
        ]
        db = _FakeMountSourceDisableDb(rows)
        service = UserSpaceService()

        def invalidate_file_list_cache(workspace_id: str) -> None:
            return None

        setattr(service, "invalidate_file_list_cache", invalidate_file_list_cache)

        async def _fake_get_mount_source_record(_db: Any, _mount_source_id: str) -> SimpleNamespace:
            return SimpleNamespace(id=mount_source_id, enabled=True)

        def _fake_source_from_record(record: Any, **_: Any) -> SimpleNamespace:
            enabled = bool(getattr(record, "enabled", False))
            return SimpleNamespace(
                id=mount_source_id,
                name="Global Source",
                description=None,
                enabled=enabled,
                source_type="filesystem",
                connection_config={},
                approved_paths=[],
                sync_interval_seconds=30,
                access_user_ids=[],
                access_group_identifiers=[],
            )

        with (
            patch("ragtime.userspace.service.get_db", AsyncMock(return_value=db)),
            patch.object(service, "_get_mount_source_record", AsyncMock(side_effect=_fake_get_mount_source_record)),
            patch.object(service, "_userspace_mount_source_from_record", side_effect=_fake_source_from_record),
            patch.object(service, "_normalize_mount_source_payload", return_value=({}, [], "docker_volume")),
            patch.object(service, "_invalidate_workspace_mount_sync_preview", AsyncMock()),
            patch.object(service, "_resolve_workspace_mount_runtime_target_dir", return_value=None),
        ):
            await service.update_userspace_mount_source(
                mount_source_id,
                UpdateUserspaceMountSourceRequest(enabled=False),
                user_id="admin-1",
            )

        updated_mount_ids = {str(call["where"].get("id") or "") for call in db.workspacemount.update_calls}
        self.assertEqual(updated_mount_ids, {"mount-enabled-1", "mount-enabled-2"})
        self.assertEqual(len(db.workspace.update_calls), 2)
        self.assertEqual(
            db.workspacemount.update_many_calls[0]["where"],
            {"mountSourceId": mount_source_id, "autoSyncEnabled": True},
        )

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
                AsyncMock(return_value=[{"target_path": "mnt", "source_local_path": "/tmp/mnt"}]),
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

    async def test_mount_list_skips_source_probe_for_syncing_mounts(self) -> None:
        syncing_mount = SimpleNamespace(
            id="mount-syncing",
            workspaceId="workspace-1",
            mountSourceId=None,
            userMountSourceId=None,
            mountSource=None,
            userMountSource=None,
            sourcePath=".",
            targetPath="/cloud",
            description=None,
            enabled=True,
            syncMode="merge",
            syncDeletes=False,
            syncStatus="syncing",
            syncBackend="google_drive",
            syncNotice=None,
            syncProgressFilesDone=2,
            syncProgressFilesTotal=5,
            syncProgressMessage="Uploading files",
            syncStartedAt=_NOW,
            lastSyncAt=None,
            lastSyncError=None,
            autoSyncEnabled=False,
            syncIntervalSeconds=None,
            createdAt=_NOW,
            updatedAt=_NOW,
        )
        pending_mount = SimpleNamespace(
            id="mount-pending",
            workspaceId="workspace-1",
            mountSourceId=None,
            userMountSourceId=None,
            mountSource=None,
            userMountSource=None,
            sourcePath=".",
            targetPath="/pending",
            description=None,
            enabled=True,
            syncMode="merge",
            syncDeletes=False,
            syncStatus="pending",
            syncBackend=None,
            syncNotice=None,
            syncProgressFilesDone=0,
            syncProgressFilesTotal=None,
            syncProgressMessage=None,
            syncStartedAt=None,
            lastSyncAt=None,
            lastSyncError=None,
            autoSyncEnabled=False,
            syncIntervalSeconds=None,
            createdAt=_NOW,
            updatedAt=_NOW,
        )
        service = _MountListService()

        with patch(
            "ragtime.userspace.service.get_db",
            AsyncMock(return_value=_FakeMountWatchDb([syncing_mount, pending_mount])),
        ):
            mounts = await service.list_workspace_mounts("workspace-1", "user-1")

        self.assertTrue(mounts[0].source_available)
        self.assertFalse(mounts[1].source_available)
        self.assertEqual(service.availability_checks, ["mount-pending"])

    async def test_workspace_tree_file_path_prefers_mount_source_over_stale_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_id = "workspace-1"
            source_dir = tmp / "source"
            source_dir.mkdir()
            source_file = source_dir / "note.txt"
            source_file.write_text("source\n", encoding="utf-8")
            rootfs_file = tmp / "workspaces" / workspace_id / "rootfs" / "workspace" / "mounted" / "note.txt"
            rootfs_file.parent.mkdir(parents=True)
            rootfs_file.write_text("stale-rootfs\n", encoding="utf-8")

            service = UserSpaceService()
            service._workspaces_dir = tmp / "workspaces"
            service._base_dir = tmp

            with patch.object(
                service,
                "resolve_workspace_mounts_for_runtime",
                AsyncMock(
                    return_value=[
                        {
                            "target_path": "/workspace/mounted",
                            "source_local_path": str(source_dir),
                        }
                    ]
                ),
            ):
                resolved = await service._resolve_workspace_tree_file_path(
                    workspace_id,
                    "mounted/note.txt",
                )

            self.assertEqual(resolved.resolve(), source_file.resolve())

    async def test_cleanup_interrupted_workspace_mount_syncs_marks_syncing_rows_error(self) -> None:
        table = _FakeWorkspaceMountTable([])
        service = UserSpaceService()

        with patch(
            "ragtime.userspace.service.get_db",
            AsyncMock(return_value=_FakeMountWatchDb([])),
        ) as patched_get_db:
            patched_get_db.return_value.workspacemount = table
            await service.cleanup_interrupted_workspace_mount_syncs()

        self.assertEqual(table.update_many_calls[0]["where"], {"syncStatus": "syncing"})
        data = table.update_many_calls[0]["data"]
        self.assertEqual(data["syncStatus"], "error")
        self.assertIn("server restart", data["lastSyncError"])
        self.assertIsNone(data["syncStartedAt"])


if __name__ == "__main__":
    unittest.main()
