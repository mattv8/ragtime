from __future__ import annotations

from typing import Any, Literal, Sequence, cast

from ragtime.core.database import get_db as _get_db

ToolSurface = Literal["chat", "workspace"]
ToolAccessLevel = Literal["deny", "read", "read_write"]

_POLICY_COLUMN_BY_SURFACE = {
    "chat": '"default_chat_access"',
    "workspace": '"default_workspace_access"',
}
_ACCESS_COLUMN_BY_SURFACE = {
    "chat": '"chat_access"',
    "workspace": '"workspace_access"',
}
_GRANT_PRIORITY: dict[ToolAccessLevel, int] = {
    "deny": 0,
    "read": 1,
    "read_write": 2,
}


def normalize_default_access_level(level: str | None, *, allow_write: bool) -> ToolAccessLevel:
    normalized: ToolAccessLevel = cast(ToolAccessLevel, level) if level in _GRANT_PRIORITY else "deny"
    if not allow_write and normalized == "read_write":
        return "read"
    return normalized


def _unique_tool_ids(tool_config_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(tool_id for tool_id in tool_config_ids if tool_id))


async def get_db() -> Any:
    return await _get_db()


def _resolve_matching_level(
    *,
    fallback: ToolAccessLevel,
    user_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> ToolAccessLevel:
    matching_levels = [str(row.get("access_level")) for row in [*user_rows, *group_rows] if row.get("access_level") in _GRANT_PRIORITY]
    if "deny" in matching_levels:
        return "deny"
    if "read_write" in matching_levels:
        return "read_write"
    if "read" in matching_levels:
        return "read"
    return fallback


async def resolve_tool_access(
    *,
    user_id: str,
    is_admin: bool,
    surface: ToolSurface,
    tool_config_ids: Sequence[str],
) -> dict[str, ToolAccessLevel]:
    """Batched. Returns level for every requested id ('deny' when no policy)."""
    unique_tool_ids = _unique_tool_ids(tool_config_ids)
    if not unique_tool_ids:
        return {}
    if is_admin:
        return {tool_id: "read_write" for tool_id in unique_tool_ids}

    db = await get_db()
    policy_column = _POLICY_COLUMN_BY_SURFACE[surface]
    access_column = _ACCESS_COLUMN_BY_SURFACE[surface]
    policy_rows: list[dict[str, Any]] = await db.query_raw(
        f"""
        SELECT
            policy."tool_config_id",
            {policy_column}::text AS default_access,
            tool."allow_write" AS allow_write
        FROM "tool_access_policies" policy
        JOIN "tool_configs" tool ON tool."id" = policy."tool_config_id"
        WHERE policy."tool_config_id" = ANY($1::text[])
        """,
        unique_tool_ids,
    )
    user_rows: list[dict[str, Any]] = await db.query_raw(
        f"""
        SELECT
            policy."tool_config_id",
            access.{access_column}::text AS access_level
        FROM "tool_user_access" access
        JOIN "tool_access_policies" policy ON policy."id" = access."policy_id"
        WHERE policy."tool_config_id" = ANY($1::text[])
          AND access."user_id" = $2::text
          AND access.{access_column} IS NOT NULL
        """,
        unique_tool_ids,
        user_id,
    )
    group_rows: list[dict[str, Any]] = await db.query_raw(
        f"""
        SELECT
            policy."tool_config_id",
            access.{access_column}::text AS access_level
        FROM "tool_auth_group_access" access
        JOIN "tool_access_policies" policy ON policy."id" = access."policy_id"
        JOIN "auth_group_memberships" membership ON membership."group_id" = access."auth_group_id"
        WHERE policy."tool_config_id" = ANY($1::text[])
          AND membership."user_id" = $2::text
          AND access.{access_column} IS NOT NULL
          AND (membership."expires_at" IS NULL OR membership."expires_at" > CURRENT_TIMESTAMP)
        """,
        unique_tool_ids,
        user_id,
    )

    fallback_by_tool: dict[str, ToolAccessLevel] = {
        str(row["tool_config_id"]): normalize_default_access_level(
            cast(str | None, row.get("default_access")),
            allow_write=bool(row.get("allow_write")),
        )
        for row in policy_rows
        if row.get("tool_config_id")
    }
    user_rows_by_tool: dict[str, list[dict[str, Any]]] = {tool_id: [] for tool_id in unique_tool_ids}
    group_rows_by_tool: dict[str, list[dict[str, Any]]] = {tool_id: [] for tool_id in unique_tool_ids}

    for row in user_rows:
        tool_id = str(row.get("tool_config_id") or "")
        if tool_id in user_rows_by_tool:
            user_rows_by_tool[tool_id].append(row)
    for row in group_rows:
        tool_id = str(row.get("tool_config_id") or "")
        if tool_id in group_rows_by_tool:
            group_rows_by_tool[tool_id].append(row)

    resolved: dict[str, ToolAccessLevel] = {}
    for tool_id in unique_tool_ids:
        fallback: ToolAccessLevel = fallback_by_tool.get(tool_id, "deny")
        resolved[tool_id] = _resolve_matching_level(
            fallback=fallback,
            user_rows=user_rows_by_tool[tool_id],
            group_rows=group_rows_by_tool[tool_id],
        )
    return resolved


async def filter_tool_ids_by_access(
    *,
    user_id: str,
    is_admin: bool,
    surface: ToolSurface,
    tool_config_ids: Sequence[str],
) -> list[str]:
    """Order-preserving ids with level != deny."""
    resolved = await resolve_tool_access(
        user_id=user_id,
        is_admin=is_admin,
        surface=surface,
        tool_config_ids=tool_config_ids,
    )
    return [tool_id for tool_id in tool_config_ids if resolved.get(tool_id, "deny") != "deny"]


async def ensure_tool_access_policy(tool_config_id: str) -> None:
    """Idempotent deny/deny policy creation (create/import/duplicate paths)."""
    db = await get_db()
    await db.toolaccesspolicy.upsert(
        where={"toolConfigId": tool_config_id},
        data=cast(
            Any,
            {
                "create": {
                    "toolConfig": {"connect": {"id": tool_config_id}},
                    "defaultChatAccess": "deny",
                    "defaultWorkspaceAccess": "deny",
                },
                "update": {},
            },
        ),
    )
