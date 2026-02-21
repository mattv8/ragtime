from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.userspace.models import (ArtifactType, CreateWorkspaceRequest,
                                      PaginatedWorkspacesResponse,
                                      UpdateWorkspaceMembersRequest,
                                      UpdateWorkspaceRequest,
                                      UpsertWorkspaceFileRequest,
                                      UserSpaceFileInfo, UserSpaceFileResponse,
                                      UserSpaceLiveDataConnection,
                                      UserSpaceSnapshot, UserSpaceWorkspace,
                                      WorkspaceMember)

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserSpaceService:
    def __init__(self) -> None:
        self._base_dir = Path(settings.index_data_path) / "_userspace"
        self._workspaces_dir = self._base_dir / "workspaces"
        self._workspaces_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_synced = False
        self._metadata_sync_lock = asyncio.Lock()

    @property
    def root_path(self) -> Path:
        return self._base_dir

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self._workspaces_dir / workspace_id

    def _workspace_meta_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "workspace.json"

    def _workspace_files_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "files"

    def _workspace_git_dir(self, workspace_id: str) -> Path:
        return self._workspace_files_dir(workspace_id) / ".git"

    def _run_git(
        self,
        workspace_id: str,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        files_dir = self._workspace_files_dir(workspace_id)
        try:
            return subprocess.run(
                ["git", *args],
                cwd=files_dir,
                capture_output=True,
                text=True,
                check=check,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="Git binary not available for User Space snapshots",
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Git snapshot operation failed: {stderr or 'unknown error'}",
            ) from exc

    def _ensure_workspace_git_repo(self, workspace_id: str) -> None:
        files_dir = self._workspace_files_dir(workspace_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        git_dir = self._workspace_git_dir(workspace_id)
        if git_dir.exists() and git_dir.is_dir():
            return

        self._run_git(workspace_id, ["init"])
        self._run_git(workspace_id, ["config", "user.name", "Ragtime User Space"])
        self._run_git(
            workspace_id,
            ["config", "user.email", "userspace@ragtime.local"],
        )

    def _resolve_workspace_file_path(
        self, workspace_id: str, relative_path: str
    ) -> Path:
        if not relative_path or relative_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")

        files_dir = self._workspace_files_dir(workspace_id)
        target = (files_dir / relative_path).resolve()
        if files_dir.resolve() not in target.parents and target != files_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid file path")
        return target

    def _is_reserved_internal_path(self, relative_path: str) -> bool:
        normalized = relative_path.strip("/")
        if normalized.endswith(".artifact.json"):
            return True
        parts = Path(normalized).parts
        return ".git" in parts

    async def _touch_workspace(self, workspace_id: str, ts: datetime | None = None) -> None:
        db = await get_db()
        try:
            await db.workspace.update(
                where={"id": workspace_id},
                data={"updatedAt": ts or _utc_now()},
            )
        except Exception:
            logger.debug("Failed to update workspace timestamp for %s", workspace_id)

    def _workspace_from_record(self, record: Any) -> UserSpaceWorkspace:
        member_rows = list(getattr(record, "members", []) or [])
        members: list[WorkspaceMember] = []
        owner_present = False

        for member in member_rows:
            member_user_id = getattr(member, "userId", "")
            role_value = getattr(member, "role", "viewer")
            role = (
                role_value
                if isinstance(role_value, str)
                else str(getattr(role_value, "value", role_value))
            )
            if member_user_id == getattr(record, "ownerUserId", "") and role == "owner":
                owner_present = True
            members.append(
                WorkspaceMember(
                    user_id=member_user_id,
                    role=cast(Any, role),
                )
            )

        if not owner_present and getattr(record, "ownerUserId", None):
            members.insert(
                0,
                WorkspaceMember(user_id=record.ownerUserId, role="owner"),
            )

        tool_rows = list(getattr(record, "toolSelections", []) or [])
        selected_tool_ids = [
            getattr(tool_row, "toolConfigId", "")
            for tool_row in tool_rows
            if getattr(tool_row, "toolConfigId", None)
        ]

        return UserSpaceWorkspace(
            id=record.id,
            name=record.name,
            description=record.description,
            owner_user_id=record.ownerUserId,
            selected_tool_ids=selected_tool_ids,
            conversation_ids=[],
            members=members,
            created_at=record.createdAt,
            updated_at=record.updatedAt,
        )

    async def _ensure_metadata_synced(self) -> None:
        if self._metadata_synced:
            return

        async with self._metadata_sync_lock:
            if self._metadata_synced:
                return

            db = await get_db()
            if not self._workspaces_dir.exists():
                self._metadata_synced = True
                return

            for workspace_dir in self._workspaces_dir.iterdir():
                if not workspace_dir.is_dir():
                    continue

                meta_path = workspace_dir / "workspace.json"
                if not meta_path.exists():
                    continue

                try:
                    workspace = UserSpaceWorkspace.model_validate_json(
                        meta_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    logger.warning("Skipping invalid workspace metadata file %s", meta_path)
                    continue

                owner = await db.user.find_unique(where={"id": workspace.owner_user_id})
                if not owner:
                    logger.warning(
                        "Skipping workspace %s backfill: owner user %s not found",
                        workspace.id,
                        workspace.owner_user_id,
                    )
                    continue

                await db.workspace.upsert(
                    where={"id": workspace.id},
                    data={
                        "create": {
                            "id": workspace.id,
                            "name": workspace.name,
                            "description": workspace.description,
                            "ownerUserId": workspace.owner_user_id,
                            "createdAt": workspace.created_at,
                            "updatedAt": workspace.updated_at,
                        },
                        "update": {
                            "name": workspace.name,
                            "description": workspace.description,
                            "ownerUserId": workspace.owner_user_id,
                            "updatedAt": workspace.updated_at,
                        },
                    },
                )

                await db.workspacemember.delete_many(where={"workspaceId": workspace.id})

                normalized_members: dict[str, str] = {workspace.owner_user_id: "owner"}
                for member in workspace.members:
                    if member.user_id == workspace.owner_user_id:
                        continue
                    normalized_members[member.user_id] = (
                        "editor" if member.role == "owner" else member.role
                    )

                for member_user_id, member_role in normalized_members.items():
                    user = await db.user.find_unique(where={"id": member_user_id})
                    if not user:
                        continue
                    await db.workspacemember.create(
                        data={
                            "workspaceId": workspace.id,
                            "userId": member_user_id,
                            "role": member_role,
                        }
                    )

                await db.workspacetoolselection.delete_many(
                    where={"workspaceId": workspace.id}
                )
                for tool_id in workspace.selected_tool_ids:
                    tool = await db.toolconfig.find_unique(where={"id": tool_id})
                    if not tool or not tool.enabled:
                        continue
                    await db.workspacetoolselection.create(
                        data={
                            "workspaceId": workspace.id,
                            "toolConfigId": tool_id,
                        }
                    )

                if workspace.conversation_ids:
                    await db.conversation.update_many(
                        where={
                            "id": {"in": workspace.conversation_ids},
                            "workspaceId": None,
                        },
                        data={"workspaceId": workspace.id},
                    )

            self._metadata_synced = True

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
    ) -> UserSpaceWorkspace:
        await self._ensure_metadata_synced()
        db = await get_db()

        workspace = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

        user_role: str | None = None
        if workspace.ownerUserId == user_id:
            user_role = "owner"
        else:
            for member in list(getattr(workspace, "members", []) or []):
                if getattr(member, "userId", None) != user_id:
                    continue
                role_value = getattr(member, "role", "viewer")
                user_role = (
                    role_value
                    if isinstance(role_value, str)
                    else str(getattr(role_value, "value", role_value))
                )
                break

        if user_role is None:
            raise HTTPException(status_code=404, detail="Workspace not found")

        if required_role == "editor" and user_role not in {"owner", "editor"}:
            raise HTTPException(status_code=403, detail="Editor access required")
        if required_role == "owner" and user_role != "owner":
            raise HTTPException(status_code=403, detail="Owner access required")

        return self._workspace_from_record(workspace)

    async def list_workspaces(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> PaginatedWorkspacesResponse:
        await self._ensure_metadata_synced()
        db = await get_db()

        where_clause: dict[str, Any] = {
            "OR": [
                {"ownerUserId": user_id},
                {"members": {"some": {"userId": user_id}}},
            ]
        }

        rows = await db.workspace.find_many(
            where=where_clause,
            include={"members": True, "toolSelections": True},
            order={"updatedAt": "desc"},
            skip=offset,
            take=limit,
        )
        total = await db.workspace.count(where=where_clause)

        return PaginatedWorkspacesResponse(
            items=[self._workspace_from_record(row) for row in rows],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def create_workspace(
        self, request: CreateWorkspaceRequest, user_id: str
    ) -> UserSpaceWorkspace:
        await self._ensure_metadata_synced()
        db = await get_db()

        now = _utc_now()
        workspace_id = str(uuid4())

        await db.workspace.create(
            data={
                "id": workspace_id,
                "name": request.name,
                "description": request.description,
                "ownerUserId": user_id,
                "createdAt": now,
                "updatedAt": now,
            },
            include={"members": True, "toolSelections": True},
        )

        await db.workspacemember.create(
            data={
                "workspaceId": workspace_id,
                "userId": user_id,
                "role": "owner",
            }
        )

        for tool_id in request.selected_tool_ids:
            tool = await db.toolconfig.find_unique(where={"id": tool_id})
            if not tool or not tool.enabled:
                continue
            await db.workspacetoolselection.create(
                data={
                    "workspaceId": workspace_id,
                    "toolConfigId": tool_id,
                }
            )

        self._workspace_files_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_git_repo(workspace_id)

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
        )
        if not refreshed:
            raise HTTPException(status_code=500, detail="Failed to create workspace")

        return self._workspace_from_record(refreshed)

    async def get_workspace(self, workspace_id: str, user_id: str) -> UserSpaceWorkspace:
        return await self._enforce_workspace_access(workspace_id, user_id)

    async def delete_workspace(self, workspace_id: str, user_id: str) -> None:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="owner")
        db = await get_db()
        try:
            await db.workspace.delete(where={"id": workspace_id})
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

        workspace_dir = self._workspace_dir(workspace_id)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    async def enforce_workspace_role(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str,
    ) -> UserSpaceWorkspace:
        return await self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role=required_role,
        )

    async def update_workspace(
        self,
        workspace_id: str,
        request: UpdateWorkspaceRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        db = await get_db()

        update_data: dict[str, Any] = {"updatedAt": _utc_now()}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description

        await db.workspace.update(where={"id": workspace_id}, data=update_data)

        if request.selected_tool_ids is not None:
            await db.workspacetoolselection.delete_many(where={"workspaceId": workspace_id})
            for tool_id in request.selected_tool_ids:
                tool = await db.toolconfig.find_unique(where={"id": tool_id})
                if not tool or not tool.enabled:
                    continue
                await db.workspacetoolselection.create(
                    data={
                        "workspaceId": workspace_id,
                        "toolConfigId": tool_id,
                    }
                )

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
        )
        if not refreshed:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return self._workspace_from_record(refreshed)

    async def update_workspace_members(
        self,
        workspace_id: str,
        request: UpdateWorkspaceMembersRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        workspace = await self._enforce_workspace_access(
            workspace_id, user_id, required_role="owner"
        )
        db = await get_db()

        normalized_members: dict[str, WorkspaceMember] = {
            workspace.owner_user_id: WorkspaceMember(
                user_id=workspace.owner_user_id, role="owner"
            )
        }
        for member in request.members:
            if member.user_id == workspace.owner_user_id:
                continue
            normalized_role = "editor" if member.role == "owner" else member.role
            normalized_members[member.user_id] = WorkspaceMember(
                user_id=member.user_id,
                role=normalized_role,
            )

        await db.workspacemember.delete_many(where={"workspaceId": workspace_id})
        for member in normalized_members.values():
            user = await db.user.find_unique(where={"id": member.user_id})
            if not user:
                continue
            await db.workspacemember.create(
                data={
                    "workspaceId": workspace_id,
                    "userId": member.user_id,
                    "role": member.role,
                }
            )

        await db.workspace.update(
            where={"id": workspace_id},
            data={"updatedAt": _utc_now()},
        )

        refreshed = await db.workspace.find_unique(
            where={"id": workspace_id},
            include={"members": True, "toolSelections": True},
        )
        if not refreshed:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return self._workspace_from_record(refreshed)

    async def list_workspace_files(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceFileInfo]:
        await self._enforce_workspace_access(workspace_id, user_id)
        self._ensure_workspace_git_repo(workspace_id)
        files_dir = self._workspace_files_dir(workspace_id)
        if not files_dir.exists():
            return []

        files: list[UserSpaceFileInfo] = []
        for file_path in files_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = str(file_path.relative_to(files_dir))
            if self._is_reserved_internal_path(relative):
                continue
            stat = file_path.stat()
            files.append(
                UserSpaceFileInfo(
                    path=relative,
                    size_bytes=stat.st_size,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )

        files.sort(key=lambda item: item.path)
        return files

    async def upsert_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        request: UpsertWorkspaceFileRequest,
        user_id: str,
    ) -> UserSpaceFileResponse:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(request.content, encoding="utf-8")

        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        sidecar_payload: dict[str, Any] = {}
        if request.artifact_type is not None:
            sidecar_payload["artifact_type"] = request.artifact_type

        if request.live_data_connections is not None:
            sidecar_payload["live_data_connections"] = [
                connection.model_dump(mode="json")
                for connection in request.live_data_connections
            ]

        if sidecar_payload:
            sidecar.write_text(
                json.dumps(sidecar_payload),
                encoding="utf-8",
            )
        elif sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

        stat = file_path.stat()
        await self._touch_workspace(workspace_id)

        return UserSpaceFileResponse(
            path=relative_path,
            content=request.content,
            artifact_type=request.artifact_type,
            live_data_connections=request.live_data_connections,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    async def get_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        user_id: str,
    ) -> UserSpaceFileResponse:
        await self._enforce_workspace_access(workspace_id, user_id)
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        artifact_type = None
        live_data_connections: list[UserSpaceLiveDataConnection] | None = None
        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_value = sidecar_data.get("artifact_type")
                if sidecar_value == "module_ts":
                    artifact_type = cast(ArtifactType, sidecar_value)

                raw_connections = sidecar_data.get("live_data_connections")
                if isinstance(raw_connections, list):
                    parsed_connections: list[UserSpaceLiveDataConnection] = []
                    for item in raw_connections:
                        if not isinstance(item, dict):
                            continue
                        try:
                            parsed_connections.append(
                                UserSpaceLiveDataConnection.model_validate(item)
                            )
                        except Exception:
                            continue
                    live_data_connections = parsed_connections or None
            except Exception:
                artifact_type = None
                live_data_connections = None

        stat = file_path.stat()
        return UserSpaceFileResponse(
            path=relative_path,
            content=file_path.read_text(encoding="utf-8"),
            artifact_type=artifact_type,
            live_data_connections=live_data_connections,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    async def delete_workspace_file(
        self, workspace_id: str, relative_path: str, user_id: str
    ) -> None:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

        await self._touch_workspace(workspace_id)

    async def create_snapshot(
        self,
        workspace_id: str,
        user_id: str,
        message: str | None = None,
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)
        self._run_git(workspace_id, ["add", "-A"])

        normalized_message = (message or "Snapshot").strip()
        commit_subject = (
            normalized_message.splitlines()[0][:200]
            if normalized_message
            else "Snapshot"
        )
        self._run_git(workspace_id, ["commit", "--allow-empty", "-m", commit_subject])

        snapshot_id = self._run_git(workspace_id, ["rev-parse", "HEAD"]).stdout.strip()
        commit_ts = self._run_git(
            workspace_id,
            ["show", "-s", "--format=%ct", snapshot_id],
        ).stdout.strip()
        created_at = datetime.fromtimestamp(int(commit_ts), tz=timezone.utc)

        tracked_files = self._run_git(
            workspace_id,
            ["ls-tree", "-r", "--name-only", snapshot_id],
        ).stdout.splitlines()
        file_count = sum(
            1
            for file_name in tracked_files
            if not self._is_reserved_internal_path(file_name)
        )

        await self._touch_workspace(workspace_id, ts=created_at)

        return UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            message=commit_subject,
            created_at=created_at,
            file_count=file_count,
        )

    async def _get_snapshot_retention_cutoff(self) -> float | None:
        """Return a UNIX timestamp cutoff based on app_settings snapshot_retention_days.

        Returns None if retention is disabled (0 or unset).
        """
        try:
            from ragtime.indexer.repository import repository

            app_settings = await repository.get_settings()

            if not app_settings:
                return None
            retention_days = getattr(app_settings, "snapshot_retention_days", 0) or 0
            if retention_days <= 0:
                return None
            return _utc_now().timestamp() - (retention_days * 86400)
        except Exception:
            return None

    async def list_snapshots(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceSnapshot]:
        await self._enforce_workspace_access(workspace_id, user_id)
        self._ensure_workspace_git_repo(workspace_id)

        head_check = self._run_git(
            workspace_id, ["rev-parse", "--verify", "HEAD"], check=False
        )
        if head_check.returncode != 0:
            return []

        cutoff = await self._get_snapshot_retention_cutoff()

        snapshots: list[UserSpaceSnapshot] = []
        git_log = self._run_git(
            workspace_id,
            ["log", "--pretty=format:%H%x1f%ct%x1f%s"],
        ).stdout

        for line in git_log.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) != 3:
                continue

            commit_id, commit_ts, commit_subject = parts
            ts = int(commit_ts)
            if cutoff is not None and ts < cutoff:
                continue
            tracked_files = self._run_git(
                workspace_id,
                ["ls-tree", "-r", "--name-only", commit_id],
            ).stdout.splitlines()
            file_count = sum(
                1
                for file_name in tracked_files
                if not self._is_reserved_internal_path(file_name)
            )
            snapshots.append(
                UserSpaceSnapshot(
                    id=commit_id,
                    workspace_id=workspace_id,
                    message=commit_subject,
                    created_at=datetime.fromtimestamp(ts, tz=timezone.utc),
                    file_count=file_count,
                )
            )
        return snapshots

    async def restore_snapshot(
        self, workspace_id: str, snapshot_id: str, user_id: str
    ) -> UserSpaceSnapshot:
        await self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)

        commit_check = self._run_git(
            workspace_id,
            ["cat-file", "-e", f"{snapshot_id}^{{commit}}"],
            check=False,
        )
        if commit_check.returncode != 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        # Enforce retention window on restore
        cutoff = await self._get_snapshot_retention_cutoff()
        if cutoff is not None:
            commit_ts = self._run_git(
                workspace_id,
                ["show", "-s", "--format=%ct", snapshot_id],
            ).stdout.strip()
            if int(commit_ts) < cutoff:
                raise HTTPException(
                    status_code=410,
                    detail="Snapshot has expired and cannot be restored",
                )

        self._run_git(workspace_id, ["reset", "--hard", snapshot_id])
        self._run_git(workspace_id, ["clean", "-fd"])

        commit_ts = self._run_git(
            workspace_id,
            ["show", "-s", "--format=%ct", snapshot_id],
        ).stdout.strip()
        commit_subject = self._run_git(
            workspace_id,
            ["show", "-s", "--format=%s", snapshot_id],
        ).stdout.strip()
        tracked_files = self._run_git(
            workspace_id,
            ["ls-tree", "-r", "--name-only", snapshot_id],
        ).stdout.splitlines()
        file_count = sum(
            1
            for file_name in tracked_files
            if not self._is_reserved_internal_path(file_name)
        )

        snapshot = UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            message=commit_subject,
            created_at=datetime.fromtimestamp(int(commit_ts), tz=timezone.utc),
            file_count=file_count,
        )

        await self._touch_workspace(workspace_id)

        return snapshot


userspace_service = UserSpaceService()
