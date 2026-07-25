"""Curated, read-only planning context for external agents.

Wraps existing ``userspace_service`` operations (which enforce workspace
ACLs and path protections) and shapes their output into the versioned
planning contract. Never reads workspace content directly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.indexer.repository import repository
from ragtime.indexer.tool_selection import resolve_effective_tool_ids
from ragtime.userspace.planning_contract import (
    PLANNING_CONTRACT_VERSION,
    build_builder_contract,
    build_recommended_workflow,
)
from ragtime.userspace.workspace_tool_options import (
    load_workspace_tool_options,
    resolve_workspace_tool_write_access,
)

logger = get_logger(__name__)

_MAX_KEY_FILES = 40
_MAX_FILE_LIST_LIMIT = 500
_MAX_READ_LINES = 2000
_MAX_READ_CHARS = 100_000


def _is_visible_file(entry: Any) -> bool:
    if getattr(entry, "entry_type", "file") == "directory":
        return False
    return not str(entry.path).startswith(".ragtime/")


class UserSpacePlanningService:
    """Read-only planning surface. All access flows through userspace_service."""

    async def _is_admin(self, user_id: str) -> bool:
        db = await get_db()
        user = await db.user.find_unique(where={"id": user_id})
        return bool(user and getattr(user, "role", "") == "admin")

    async def _selected_tools(self, workspace: Any) -> list[dict[str, Any]]:
        from ragtime.userspace.service import userspace_service

        selected_ids = await resolve_effective_tool_ids(
            tool_selection_mode=getattr(workspace, "tool_selection_mode", ""),
            selected_tool_ids=getattr(workspace, "selected_tool_ids", []),
            selected_tool_group_ids=getattr(workspace, "selected_tool_group_ids", []),
            list_healthy_enabled_tool_ids=repository.list_healthy_enabled_tool_ids,
            get_tool_ids_for_groups=repository.get_tool_ids_for_groups,
        )
        selected_ids = await userspace_service.filter_tool_ids_for_workspace_owner(workspace, selected_ids)
        selected = set(selected_ids)
        owner_access = await userspace_service._resolve_workspace_owner_tool_access(workspace, selected_ids)
        configs = await repository.list_tool_configs(enabled_only=True)
        tool_options = getattr(workspace, "tool_options", {}) or {}
        tools: list[dict[str, Any]] = []
        for cfg in configs:
            if cfg.id not in selected:
                continue
            option_state = tool_options.get(cfg.id)
            raw_option = option_state.model_dump() if option_state is not None else None
            write_enabled = (
                resolve_workspace_tool_write_access(
                    bool(getattr(cfg, "allow_write", False)),
                    load_workspace_tool_options(raw_option),
                )
                and owner_access.get(cfg.id, "deny") == "read_write"
            )
            tools.append(
                {
                    "component_id": cfg.id,
                    "name": cfg.name,
                    "tool_type": getattr(cfg.tool_type, "value", str(cfg.tool_type)),
                    "description": cfg.description or "",
                    "server_write_enabled": write_enabled,
                }
            )
        return tools

    @staticmethod
    def _context_revision(updated_at: Any, file_count: int, entrypoint: Any, tool_ids: list[str]) -> str:
        basis = json.dumps(
            {
                "updated_at": str(updated_at),
                "file_count": file_count,
                "entrypoint": [
                    str(getattr(entrypoint, "state", "")),
                    str(getattr(entrypoint, "framework", "") or ""),
                    str(getattr(entrypoint, "command", "") or ""),
                ],
                "tools": sorted(tool_ids),
            },
            sort_keys=True,
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    async def get_workspace_context(self, workspace_id: str, user_id: str) -> dict[str, Any]:
        from ragtime.userspace.service import userspace_service

        is_admin = await self._is_admin(user_id)
        workspace = await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer", is_admin=is_admin)

        files = await userspace_service.list_workspace_files(workspace_id, user_id, is_admin=is_admin)
        visible = [entry for entry in files if _is_visible_file(entry)]
        key_files = sorted(entry.path for entry in visible)[:_MAX_KEY_FILES]

        entrypoint = userspace_service.get_workspace_entrypoint_status(workspace_id)
        is_default_static = bool(userspace_service.is_default_static_entrypoint(workspace_id, status=entrypoint))

        snapshot_summary: dict[str, Any] = {"count": 0, "last_message": None, "last_created_at": None}
        try:
            snapshots = await userspace_service.list_snapshots(workspace_id, user_id)
            if snapshots:
                snapshot_summary = {
                    "count": len(snapshots),
                    "last_message": getattr(snapshots[0], "message", None),
                    "last_created_at": getattr(snapshots[0], "created_at", None),
                }
        except Exception:
            logger.debug("Snapshot summary unavailable for workspace %s", workspace_id, exc_info=True)

        selected_tools = await self._selected_tools(workspace)
        sqlite_mode = str(getattr(workspace, "sqlite_persistence_mode", "include"))

        caller_role = "viewer"
        if getattr(workspace, "owner_user_id", None) == user_id:
            caller_role = "owner"
        else:
            for member in getattr(workspace, "members", []) or []:
                member_user_id = getattr(member, "user_id", None) or getattr(member, "userId", None)
                if member_user_id == user_id:
                    caller_role = str(getattr(member, "role", "viewer"))
                    break
            else:
                if is_admin:
                    caller_role = "admin"

        valid = entrypoint.state == "valid"
        payload: dict[str, Any] = {
            "contract_version": PLANNING_CONTRACT_VERSION,
            "workspace": {
                "id": workspace.id,
                "name": workspace.name,
                "description": workspace.description or "",
                "sqlite_persistence_mode": sqlite_mode,
                "caller_role": caller_role,
                "updated_at": getattr(workspace, "updated_at", None),
            },
            "architecture": {
                "entrypoint_state": entrypoint.state,
                "framework": entrypoint.framework if valid else None,
                "command": entrypoint.command if valid else None,
                "cwd": entrypoint.cwd if valid else None,
                "is_default_static": is_default_static,
                "file_count": len(visible),
                "key_files": key_files,
            },
            "selected_tools": selected_tools,
            "snapshot_summary": snapshot_summary,
            "builder_contract": build_builder_contract(
                sqlite_persistence_mode=sqlite_mode,
                has_live_data_tools=bool(selected_tools),
            ),
            "recommended_workflow": build_recommended_workflow(),
        }
        payload["context_revision"] = self._context_revision(
            getattr(workspace, "updated_at", None),
            len(visible),
            entrypoint,
            [tool["component_id"] for tool in selected_tools],
        )
        return payload

    async def list_files(
        self,
        workspace_id: str,
        user_id: str,
        prefix: str = "",
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        from ragtime.userspace.service import userspace_service

        limit = max(1, min(int(limit), _MAX_FILE_LIST_LIMIT))
        offset = max(0, int(offset))
        is_admin = await self._is_admin(user_id)
        files = await userspace_service.list_workspace_files(workspace_id, user_id, is_admin=is_admin)
        visible = [entry for entry in files if _is_visible_file(entry)]
        cleaned_prefix = (prefix or "").strip().lstrip("/")
        if cleaned_prefix:
            visible = [entry for entry in visible if str(entry.path).startswith(cleaned_prefix)]
        visible.sort(key=lambda entry: str(entry.path))
        window = visible[offset : offset + limit]
        return {
            "workspace_id": workspace_id,
            "total": len(visible),
            "offset": offset,
            "limit": limit,
            "files": [{"path": entry.path, "size_bytes": entry.size_bytes, "updated_at": entry.updated_at} for entry in window],
        }

    async def read_file(
        self,
        workspace_id: str,
        user_id: str,
        path: str,
        start_line: int = 1,
        max_lines: int = 400,
        max_chars: int = 30000,
    ) -> dict[str, Any]:
        from ragtime.userspace.service import userspace_service

        start_line = max(1, int(start_line))
        max_lines = max(1, min(int(max_lines), _MAX_READ_LINES))
        max_chars = max(200, min(int(max_chars), _MAX_READ_CHARS))

        is_admin = await self._is_admin(user_id)
        response = await userspace_service.get_workspace_file(workspace_id, path, user_id, is_admin=is_admin)

        lines = response.content.splitlines()
        total_lines = len(lines)
        window = lines[start_line - 1 : start_line - 1 + max_lines]
        selected: list[str] = []
        selected_chars = 0
        line_clipped = False
        for line in window:
            separator_chars = 1 if selected else 0
            if selected_chars + separator_chars + len(line) <= max_chars:
                selected.append(line)
                selected_chars += separator_chars + len(line)
                continue
            if not selected:
                selected.append(line[:max_chars])
                line_clipped = True
            break

        consumed_lines = len(selected)
        text = "\n".join(selected)
        end_line = min(start_line - 1 + consumed_lines, total_lines)
        truncated = line_clipped or consumed_lines < len(window) or end_line < total_lines
        next_start_line = end_line + 1 if end_line < total_lines else None
        return {
            "path": response.path,
            "total_lines": total_lines,
            "start_line": start_line,
            "end_line": end_line,
            "content": text,
            "truncated": truncated,
            "line_clipped": line_clipped,
            "next_start_line": next_start_line,
        }


planning_service = UserSpacePlanningService()
