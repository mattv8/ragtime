from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.userspace.models import (
    ArtifactType,
    CreateWorkspaceRequest,
    PaginatedWorkspacesResponse,
    UpdateWorkspaceMembersRequest,
    UpdateWorkspaceRequest,
    UpsertWorkspaceFileRequest,
    UserSpaceFileInfo,
    UserSpaceFileResponse,
    UserSpaceSnapshot,
    UserSpaceWorkspace,
    WorkspaceMember,
)

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserSpaceService:
    def __init__(self) -> None:
        self._base_dir = Path(settings.index_data_path) / "_userspace"
        self._workspaces_dir = self._base_dir / "workspaces"
        self._workspaces_dir.mkdir(parents=True, exist_ok=True)

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

    def _read_workspace(self, workspace_id: str) -> UserSpaceWorkspace:
        path = self._workspace_meta_path(workspace_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        return UserSpaceWorkspace.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_workspace(self, workspace: UserSpaceWorkspace) -> None:
        workspace_dir = self._workspace_dir(workspace.id)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self._workspace_files_dir(workspace.id).mkdir(parents=True, exist_ok=True)
        self._workspace_meta_path(workspace.id).write_text(
            workspace.model_dump_json(indent=2),
            encoding="utf-8",
        )

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

    def _has_workspace_access(
        self, workspace: UserSpaceWorkspace, user_id: str
    ) -> bool:
        if workspace.owner_user_id == user_id:
            return True
        return any(member.user_id == user_id for member in workspace.members)

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

    def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
    ) -> UserSpaceWorkspace:
        workspace = self._read_workspace(workspace_id)
        if not self._has_workspace_access(workspace, user_id):
            raise HTTPException(status_code=404, detail="Workspace not found")

        if required_role is None:
            return workspace

        if workspace.owner_user_id == user_id:
            return workspace

        member_role = next(
            (member.role for member in workspace.members if member.user_id == user_id),
            None,
        )
        if member_role is None:
            raise HTTPException(status_code=403, detail="Access denied")

        if required_role == "editor" and member_role not in {"editor", "owner"}:
            raise HTTPException(status_code=403, detail="Editor access required")

        if required_role == "owner":
            raise HTTPException(status_code=403, detail="Owner access required")

        return workspace

    def list_workspaces(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> PaginatedWorkspacesResponse:
        all_results: list[UserSpaceWorkspace] = []
        for workspace_dir in (
            self._workspaces_dir.iterdir() if self._workspaces_dir.exists() else []
        ):
            if not workspace_dir.is_dir():
                continue
            meta_path = workspace_dir / "workspace.json"
            if not meta_path.exists():
                continue
            workspace = UserSpaceWorkspace.model_validate_json(
                meta_path.read_text(encoding="utf-8")
            )
            if self._has_workspace_access(workspace, user_id):
                all_results.append(workspace)

        all_results.sort(key=lambda item: item.updated_at, reverse=True)
        total = len(all_results)
        page = all_results[offset : offset + limit]
        return PaginatedWorkspacesResponse(
            items=page, total=total, offset=offset, limit=limit
        )

    def create_workspace(
        self, request: CreateWorkspaceRequest, user_id: str
    ) -> UserSpaceWorkspace:
        now = _utc_now()
        workspace = UserSpaceWorkspace(
            id=str(uuid4()),
            name=request.name,
            description=request.description,
            owner_user_id=user_id,
            selected_tool_ids=request.selected_tool_ids,
            members=[WorkspaceMember(user_id=user_id, role="owner")],
            created_at=now,
            updated_at=now,
        )
        self._write_workspace(workspace)
        self._ensure_workspace_git_repo(workspace.id)
        return workspace

    def get_workspace(self, workspace_id: str, user_id: str) -> UserSpaceWorkspace:
        return self._enforce_workspace_access(workspace_id, user_id)

    def delete_workspace(self, workspace_id: str, user_id: str) -> None:
        self._enforce_workspace_access(workspace_id, user_id, required_role="owner")
        workspace_dir = self._workspace_dir(workspace_id)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    def enforce_workspace_role(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str,
    ) -> UserSpaceWorkspace:
        return self._enforce_workspace_access(
            workspace_id,
            user_id,
            required_role=required_role,
        )

    def list_workspace_conversation_ids(
        self, workspace_id: str, user_id: str
    ) -> list[str]:
        workspace = self._enforce_workspace_access(workspace_id, user_id)
        return list(workspace.conversation_ids)

    def add_conversation_to_workspace(
        self,
        workspace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> UserSpaceWorkspace:
        workspace = self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )
        if conversation_id not in workspace.conversation_ids:
            workspace.conversation_ids.append(conversation_id)
            workspace.updated_at = _utc_now()
            self._write_workspace(workspace)
        return workspace

    def has_workspace_conversation_access(
        self,
        workspace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        workspace = self._enforce_workspace_access(workspace_id, user_id)
        return conversation_id in workspace.conversation_ids

    def update_workspace(
        self,
        workspace_id: str,
        request: UpdateWorkspaceRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        workspace = self._enforce_workspace_access(
            workspace_id, user_id, required_role="editor"
        )

        if request.name is not None:
            workspace.name = request.name
        if request.description is not None:
            workspace.description = request.description
        if request.selected_tool_ids is not None:
            workspace.selected_tool_ids = request.selected_tool_ids

        workspace.updated_at = _utc_now()
        self._write_workspace(workspace)
        return workspace

    def update_workspace_members(
        self,
        workspace_id: str,
        request: UpdateWorkspaceMembersRequest,
        user_id: str,
    ) -> UserSpaceWorkspace:
        workspace = self._enforce_workspace_access(
            workspace_id, user_id, required_role="owner"
        )

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

        workspace.members = list(normalized_members.values())
        workspace.updated_at = _utc_now()
        self._write_workspace(workspace)
        return workspace

    def list_workspace_files(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceFileInfo]:
        self._enforce_workspace_access(workspace_id, user_id)
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

    def upsert_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        request: UpsertWorkspaceFileRequest,
        user_id: str,
    ) -> UserSpaceFileResponse:
        self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(request.content, encoding="utf-8")

        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if request.artifact_type is not None:
            sidecar.write_text(
                json.dumps({"artifact_type": request.artifact_type}),
                encoding="utf-8",
            )
        elif sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

        stat = file_path.stat()
        workspace = self._read_workspace(workspace_id)
        workspace.updated_at = _utc_now()
        self._write_workspace(workspace)

        return UserSpaceFileResponse(
            path=relative_path,
            content=request.content,
            artifact_type=request.artifact_type,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def get_workspace_file(
        self,
        workspace_id: str,
        relative_path: str,
        user_id: str,
    ) -> UserSpaceFileResponse:
        self._enforce_workspace_access(workspace_id, user_id)
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=404, detail="File not found")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        artifact_type = None
        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_value = sidecar_data.get("artifact_type")
                if sidecar_value in {
                    "dashboard_json",
                    "report_markdown",
                    "report_html",
                    "module_js",
                    "module_ts",
                }:
                    artifact_type = cast(ArtifactType, sidecar_value)
            except Exception:
                artifact_type = None

        stat = file_path.stat()
        return UserSpaceFileResponse(
            path=relative_path,
            content=file_path.read_text(encoding="utf-8"),
            artifact_type=artifact_type,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def delete_workspace_file(
        self, workspace_id: str, relative_path: str, user_id: str
    ) -> None:
        self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)
        if self._is_reserved_internal_path(relative_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        file_path = self._resolve_workspace_file_path(workspace_id, relative_path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
        sidecar = file_path.with_suffix(file_path.suffix + ".artifact.json")
        if sidecar.exists() and sidecar.is_file():
            sidecar.unlink()

        workspace = self._read_workspace(workspace_id)
        workspace.updated_at = _utc_now()
        self._write_workspace(workspace)

    def create_snapshot(
        self,
        workspace_id: str,
        user_id: str,
        message: str | None = None,
    ) -> UserSpaceSnapshot:
        self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
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

        snapshot_meta = UserSpaceSnapshot(
            id=snapshot_id,
            workspace_id=workspace_id,
            message=commit_subject,
            created_at=created_at,
            file_count=file_count,
        )

        workspace = self._read_workspace(workspace_id)
        workspace.updated_at = created_at
        self._write_workspace(workspace)

        return snapshot_meta

    def _get_snapshot_retention_cutoff(self) -> float | None:
        """Return a UNIX timestamp cutoff based on app_settings snapshot_retention_days.

        Returns None if retention is disabled (0 or unset).
        """
        try:
            import asyncio

            from ragtime.indexer.repository import repository

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are inside an async context, can't block.
                # Use a thread to fetch settings.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    app_settings = pool.submit(
                        asyncio.run, repository.get_settings()
                    ).result()
            else:
                app_settings = asyncio.run(repository.get_settings())

            if not app_settings:
                return None
            retention_days = getattr(app_settings, "snapshot_retention_days", 0) or 0
            if retention_days <= 0:
                return None
            return _utc_now().timestamp() - (retention_days * 86400)
        except Exception:
            return None

    def list_snapshots(
        self, workspace_id: str, user_id: str
    ) -> list[UserSpaceSnapshot]:
        self._enforce_workspace_access(workspace_id, user_id)
        self._ensure_workspace_git_repo(workspace_id)

        head_check = self._run_git(
            workspace_id, ["rev-parse", "--verify", "HEAD"], check=False
        )
        if head_check.returncode != 0:
            return []

        cutoff = self._get_snapshot_retention_cutoff()

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

    def restore_snapshot(
        self, workspace_id: str, snapshot_id: str, user_id: str
    ) -> UserSpaceSnapshot:
        self._enforce_workspace_access(workspace_id, user_id, required_role="editor")
        self._ensure_workspace_git_repo(workspace_id)

        commit_check = self._run_git(
            workspace_id,
            ["cat-file", "-e", f"{snapshot_id}^{{commit}}"],
            check=False,
        )
        if commit_check.returncode != 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        # Enforce retention window on restore
        cutoff = self._get_snapshot_retention_cutoff()
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

        workspace = self._read_workspace(workspace_id)
        workspace.updated_at = _utc_now()
        self._write_workspace(workspace)

        return snapshot


userspace_service = UserSpaceService()
userspace_service = UserSpaceService()
