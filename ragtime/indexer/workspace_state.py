
from __future__ import annotations

import asyncio
from typing import Optional

from ragtime.indexer.models import (
    ChatTask,
    ChatTaskResponse,
    Conversation,
    ConversationResponse,
    WorkspaceChatStateResponse,
)
from ragtime.indexer.repository import repository

def _to_conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        model=conversation.model,
        user_id=conversation.user_id,
        workspace_id=conversation.workspace_id,
        username=conversation.username,
        display_name=conversation.display_name,
        messages=conversation.messages,
        total_tokens=conversation.total_tokens,
        active_task_id=conversation.active_task_id,
        tool_output_mode=conversation.tool_output_mode,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _to_chat_task_response(task: ChatTask) -> ChatTaskResponse:
    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=task.streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


async def build_workspace_chat_state(
    *,
    workspace_id: str,
    user_id: Optional[str],
    is_admin: bool,
    selected_conversation_id: Optional[str] = None,
) -> WorkspaceChatStateResponse:
    conversations, interrupted_conversation_ids = await asyncio.gather(
        repository.list_conversations(
            user_id=user_id,
            include_all=is_admin,
            workspace_id=workspace_id,
        ),
        repository.get_interrupted_conversation_ids_for_workspace(workspace_id),
    )

    selected_id = None
    active_task = None
    interrupted_task = None

    if selected_conversation_id and any(
        conversation.id == selected_conversation_id for conversation in conversations
    ):
        selected_id = selected_conversation_id
        active_task = await repository.get_active_task_for_conversation(selected_id)
        if not active_task:
            interrupted_task = (
                await repository.get_last_interrupted_task_for_conversation(selected_id)
            )

    return WorkspaceChatStateResponse(
        conversations=[
            _to_conversation_response(conversation) for conversation in conversations
        ],
        interrupted_conversation_ids=interrupted_conversation_ids,
        selected_conversation_id=selected_id,
        active_task=_to_chat_task_response(active_task) if active_task else None,
        interrupted_task=(
            _to_chat_task_response(interrupted_task) if interrupted_task else None
        ),
    )
