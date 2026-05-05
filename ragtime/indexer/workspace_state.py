
from __future__ import annotations

import asyncio
from typing import Optional

from ragtime.indexer.models import (
    ChatTask,
    ChatTaskResponse,
    Conversation,
    ConversationResponse,
    ConversationSummaryResponse,
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
        message_count=len(conversation.messages),
        total_tokens=conversation.total_tokens,
        active_task_id=conversation.active_task_id,
        active_branch_id=conversation.active_branch_id,
        disabled_builtin_tool_ids=conversation.disabled_builtin_tool_ids,
        tool_output_mode=conversation.tool_output_mode,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _summary_to_conversation_response(
    summary: ConversationSummaryResponse,
) -> ConversationResponse:
    return ConversationResponse(
        id=summary.id,
        title=summary.title,
        model=summary.model,
        user_id=summary.user_id,
        workspace_id=summary.workspace_id,
        username=summary.username,
        display_name=summary.display_name,
        messages=[],
        message_count=summary.message_count,
        total_tokens=summary.total_tokens,
        active_task_id=summary.active_task_id,
        active_branch_id=summary.active_branch_id,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
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
    conversation_summaries, interrupted_conversation_ids = await asyncio.gather(
        repository.list_conversation_summaries(
            user_id=user_id,
            include_all=is_admin,
            workspace_id=workspace_id,
        ),
        repository.get_interrupted_conversation_ids_for_workspace(workspace_id),
    )

    selected_id = None
    selected_conversation = None
    active_task = None
    interrupted_task = None

    summary_ids = {summary.id for summary in conversation_summaries}
    if selected_conversation_id in summary_ids:
        selected_id = selected_conversation_id
    elif conversation_summaries:
        selected_id = conversation_summaries[0].id

    if selected_id:
        selected_conversation, active_task = await asyncio.gather(
            repository.get_conversation(selected_id),
            repository.get_active_task_for_conversation(selected_id),
        )
        if selected_conversation is None:
            selected_id = None
            active_task = None
        elif not active_task:
            interrupted_task = (
                await repository.get_last_interrupted_task_for_conversation(selected_id)
            )

    conversations = [
        _summary_to_conversation_response(summary) for summary in conversation_summaries
    ]
    if selected_conversation is not None:
        selected_response = _to_conversation_response(selected_conversation)
        conversations = [
            (
                selected_response
                if conversation.id == selected_response.id
                else conversation
            )
            for conversation in conversations
        ]

    return WorkspaceChatStateResponse(
        conversations=conversations,
        interrupted_conversation_ids=interrupted_conversation_ids,
        selected_conversation_id=selected_id,
        active_task=_to_chat_task_response(active_task) if active_task else None,
        interrupted_task=(
            _to_chat_task_response(interrupted_task) if interrupted_task else None
        ),
    )
