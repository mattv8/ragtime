from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any, Literal

from fastapi import HTTPException

from ragtime.indexer.background_tasks import parse_message_content
from ragtime.indexer.models import ChatMessage
from ragtime.indexer.repository import repository
from ragtime.userspace.workspace_code_index_service import workspace_code_index_service

TranscriptDirection = Literal["forward", "backward"]
CodeSearchMode = Literal["semantic", "symbols", "hybrid"]


def _safe_attachment_label(value: Any) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "attachment"))
    text = " ".join(text.split())
    return PurePath(text.replace("\\", "/")).name[:120] or "attachment"


def _project_user_content(content: str) -> str:
    try:
        parsed = parse_message_content(content)
    except (AttributeError, TypeError):
        return content
    if isinstance(parsed, str):
        return parsed
    parts: list[str] = []
    for part in parsed:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type == "text":
            text = str(part.get("text") or "")
            if text:
                parts.append(text)
        elif part_type in {"image_url", "input_image"}:
            parts.append("[Image attachment]")
        elif part_type in {"file", "input_file"}:
            label = part.get("filename") or part.get("name") or "attachment"
            parts.append(f"[Attached file: {_safe_attachment_label(label)}]")
        else:
            label = part.get("filename") or part.get("name")
            parts.append(f"[Attachment: {_safe_attachment_label(label)}]" if label else "[Unsupported content omitted]")
    return "\n\n".join(parts) or "[Unsupported content omitted]"


def _project_visible_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        content = ""
        tool_calls: list[dict[str, Any]] = []
        if message.role == "user":
            content = _project_user_content(message.content)
        elif message.events:
            content = "".join(str(event.get("content") or "") for event in message.events if isinstance(event, dict) and event.get("type") == "content")
            tool_calls = [
                {"tool": str(event.get("tool") or "unknown"), "input": event.get("input")}
                for event in message.events
                if isinstance(event, dict) and event.get("type") == "tool"
            ]
        else:
            content = message.content
        if message.role == "assistant" and not content and not tool_calls:
            continue
        visible.append(
            {
                "message_index": len(visible),
                "message_id": message.message_id,
                "role": message.role,
                "timestamp": message.timestamp,
                "content": content,
                "tool_calls": tool_calls,
            }
        )
    return visible


def _page_visible_messages(
    messages: list[dict[str, Any]],
    *,
    direction: TranscriptDirection,
    cursor: int | None,
    limit: int,
) -> dict[str, Any]:
    total = len(messages)
    if cursor is not None and (cursor < 0 or cursor >= total):
        raise HTTPException(status_code=400, detail="Transcript cursor is out of range")
    if direction == "forward":
        start = 0 if cursor is None else cursor + 1
        end = min(total, start + limit)
    else:
        end = total if cursor is None else cursor
        start = max(0, end - limit)
    page = messages[start:end]
    return {
        "total_visible_messages": total,
        "previous_cursor": page[0]["message_index"] if start > 0 and page else None,
        "next_cursor": page[-1]["message_index"] if end < total and page else None,
        "has_more_before": start > 0,
        "has_more_after": end < total,
        "messages": page,
    }


class WorkspaceAgentReadService:
    async def _enforce_viewer(self, workspace_id: str, user_id: str, *, is_admin: bool) -> Any:
        from ragtime.userspace.service import userspace_service

        return await userspace_service.enforce_workspace_role(
            workspace_id,
            user_id,
            "viewer",
            is_admin=is_admin,
        )

    async def list_conversations(
        self,
        workspace_id: str,
        user_id: str,
        *,
        is_admin: bool,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 200))
        await self._enforce_viewer(workspace_id, user_id, is_admin=is_admin)
        summaries = await repository.list_conversation_summaries(
            user_id=user_id,
            include_all=is_admin,
            workspace_id=workspace_id,
        )
        window = summaries[offset : offset + limit]
        return {
            "total": len(summaries),
            "offset": offset,
            "limit": limit,
            "conversations": [
                {
                    "id": item.id,
                    "title": item.title,
                    "message_count": item.message_count,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "subagent_conversation_ids": item.subagent_conversation_ids,
                }
                for item in window
            ],
        }

    async def get_conversation_transcript(
        self,
        workspace_id: str,
        user_id: str,
        conversation_id: str,
        *,
        is_admin: bool,
        direction: TranscriptDirection,
        cursor: int | None,
        limit: int,
    ) -> dict[str, Any]:
        limit = max(1, int(limit))
        await self._enforce_viewer(workspace_id, user_id, is_admin=is_admin)
        has_access = await repository.check_conversation_access(
            conversation_id,
            user_id,
            is_admin=is_admin,
            workspace_id=workspace_id,
        )
        if not has_access:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        page = _page_visible_messages(
            _project_visible_messages(conversation.messages),
            direction=direction,
            cursor=cursor,
            limit=limit,
        )
        return {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "parent_conversation_id": conversation.parent_conversation_id,
                "subagent_conversation_ids": conversation.subagent_conversation_ids,
                "read_only": bool(conversation.parent_conversation_id),
            },
            **page,
        }

    async def search_code(
        self,
        workspace_id: str,
        user_id: str,
        *,
        is_admin: bool,
        query: str,
        mode: CodeSearchMode,
        max_results: int,
        max_chars_per_result: int,
    ) -> dict[str, Any]:
        await self._enforce_viewer(workspace_id, user_id, is_admin=is_admin)
        cleaned = str(query or "").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="query is required")
        raw = await workspace_code_index_service.search_workspace_code_read_only(
            workspace_id=workspace_id,
            query=cleaned,
            mode=mode,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
        )
        results = []
        for item in raw.get("results", []):
            projected = {
                "path": item.get("path"),
                "snippet": str(item.get("snippet") or "")[:max_chars_per_result],
                "score": item.get("score"),
                "source": item.get("source"),
            }
            if item.get("source") == "symbol":
                projected["kind"] = item.get("kind")
                projected["symbol"] = item.get("symbol")
            results.append(projected)
        response = {
            "status": raw.get("status"),
            "mode": mode,
            "query": cleaned,
            "result_count": len(results),
            "results": results,
        }
        allowed_errors = {
            "semantic search unavailable",
            "User Space code indexing is disabled",
        }
        error = raw.get("error")
        if error in allowed_errors:
            response["error"] = str(error)
        return response


agent_read_service = WorkspaceAgentReadService()
