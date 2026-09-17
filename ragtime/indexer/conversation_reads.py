import json
from dataclasses import dataclass
from typing import Any

from ragtime.core.database import get_db


@dataclass(frozen=True)
class ConversationReadMember:
    userId: str
    role: str


@dataclass(frozen=True)
class ConversationReadMetadata:
    userId: str | None
    workspaceId: str | None
    toolSelectionMode: str | None
    disabledBuiltinToolIds: Any
    subagentsEnabled: bool
    members: list[ConversationReadMember]


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


async def get_conversation_read_metadata(conversation_id: str) -> ConversationReadMetadata | None:
    """Fetch only the metadata needed by conversation member/tool read routes."""
    db = await get_db()
    rows = await db.query_raw(
        """
        SELECT
            c."user_id" AS "userId",
            c."workspace_id" AS "workspaceId",
            c."tool_selection_mode" AS "toolSelectionMode",
            c."disabled_builtin_tool_ids" AS "disabledBuiltinToolIds",
            c."subagents_enabled" AS "subagentsEnabled",
            cm."user_id" AS "memberUserId",
            cm."role"::text AS "memberRole"
        FROM "conversations" c
        LEFT JOIN "conversation_members" cm ON cm."conversation_id" = c."id"
        WHERE c."id" = $1
        """,
        conversation_id,
    )
    if not rows:
        return None

    row = rows[0]
    members = [
        ConversationReadMember(userId=str(member["memberUserId"]), role=str(member["memberRole"]))
        for member in rows
        if member.get("memberUserId") is not None and member.get("memberRole") is not None
    ]
    return ConversationReadMetadata(
        userId=row.get("userId"),
        workspaceId=row.get("workspaceId"),
        toolSelectionMode=row.get("toolSelectionMode"),
        disabledBuiltinToolIds=_json_value(row.get("disabledBuiltinToolIds")),
        subagentsEnabled=bool(row.get("subagentsEnabled")),
        members=members,
    )
