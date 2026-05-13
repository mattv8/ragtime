"""
Background task service for async chat message processing.

Handles running chat tasks in the background so they persist across
client disconnections, page switches, and server restarts.

Also handles scheduled filesystem re-indexing tasks.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragtime.core.app_settings import SettingsCache
from ragtime.core.event_bus import task_event_bus
from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import strip_table_metadata
from ragtime.core.usage_accounting import (
    _estimate_output_tokens,
    bind_usage_attempt_task,
    finalize_stale_attempts_for_tasks,
    finalize_usage_attempt,
)
from ragtime.indexer.chat_attachments import cleanup_expired_chat_attachments
from ragtime.indexer.chat_events import append_reasoning_event, finalize_reasoning_block
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.models import (
    ChatTaskStatus,
    FilesystemConnectionConfig,
    SchemaIndexConfig,
)
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import SCHEMA_INDEXER_CAPABLE_TYPES, schema_indexer
from ragtime.indexer.service import indexer
from ragtime.indexer.utils import safe_tool_name
from ragtime.rag import rag

logger = get_logger(__name__)


DEV_SERVER_INTERRUPT_MESSAGE = (
    "Chat run interrupted by dev server reload or shutdown before it finished."
)
WRAPPED_PROCESSING_ERROR_PREFIX = "I encountered an error processing your request:"


# Throttle "task_progress" events published to per-conversation SSE channels.
# Per-task token streaming continues to fan out at full rate on the
# task-scoped channel; conversation subscribers (especially anonymous shared
# chat viewers without per-task SSE access) only need a coarser progress
# heartbeat to know when to refresh.
_CONV_PROGRESS_THROTTLE_S = 0.5
_LAST_CONV_PROGRESS_TS: Dict[str, float] = {}


async def _notify_conversation_progress(conversation_id: str, task_id: str) -> None:
    now = time.monotonic()
    last = _LAST_CONV_PROGRESS_TS.get(conversation_id, 0.0)
    if now - last < _CONV_PROGRESS_THROTTLE_S:
        return
    _LAST_CONV_PROGRESS_TS[conversation_id] = now
    await task_event_bus.publish(
        f"conversation:{conversation_id}",
        {
            "event": "task_progress",
            "task_id": task_id,
            "conversation_id": conversation_id,
        },
    )


def _extract_wrapped_processing_error(full_response: str) -> Optional[str]:
    """Return wrapped internal error detail if response is the generic error shell."""
    text = (full_response or "").strip()
    if not text.startswith(WRAPPED_PROCESSING_ERROR_PREFIX):
        return None

    detail = text[len(WRAPPED_PROCESSING_ERROR_PREFIX) :].strip()
    return detail or "Internal processing error"


def _synthesize_incomplete_response(
    events: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    hit_max_iterations: bool,
) -> Optional[str]:
    """Build a visible fallback only when a run ended cleanly without final text."""
    had_tool_activity = bool(tool_calls) or any(
        ev.get("type") == "tool" for ev in events
    )
    had_reasoning_activity = any(ev.get("type") == "reasoning" for ev in events)

    if not (had_tool_activity or had_reasoning_activity):
        return None

    last_tool_name = next(
        (
            str(ev.get("tool"))
            for ev in reversed(events)
            if ev.get("type") == "tool" and ev.get("tool")
        ),
        "tool",
    )

    if hit_max_iterations:
        return (
            "I finished this run without emitting a final answer text "
            "before hitting the iteration limit. "
            "Please send Continue so I can resume from the current state."
        )

    if had_tool_activity:
        return (
            "I completed tool activity but did not emit a final answer text "
            f"(last tool: {last_tool_name}). "
            "Please send Continue and I will finalize the response."
        )

    return (
        "I emitted reasoning but no final answer text in this run. "
        "Please send Continue and I will complete the response."
    )


async def _persist_partial_assistant_message(
    conversation_id: str,
    full_response: str,
    events: list[dict[str, Any]],
) -> bool:
    """Persist any partial assistant turn that has text or structured events."""
    has_partial_text = bool(full_response.strip())
    has_partial_activity = bool(events)
    if not has_partial_text and not has_partial_activity:
        return False

    persisted = await repository.add_message(
        conversation_id,
        "assistant",
        full_response,
        events=events if events else None,
    )
    try:
        await repository.link_assistant_snapshot_tool_calls(
            persisted,
            getattr(persisted, "workspace_id", None) if persisted else None,
        )
    except Exception as link_err:
        logger.warning(
            f"Failed to link agent-created snapshot to assistant message: {link_err}"
        )
    return True


async def _close_stream_handles(
    stream_iter: Any,
    stream: Any,
    task_id: str,
) -> None:
    """Best-effort close of async stream handles to stop upstream generation."""
    closers: list[tuple[str, Any]] = []

    iter_aclose = getattr(stream_iter, "aclose", None)
    if callable(iter_aclose):
        closers.append(("stream iterator", iter_aclose))

    stream_aclose = getattr(stream, "aclose", None)
    if callable(stream_aclose) and stream is not stream_iter:
        closers.append(("stream", stream_aclose))

    for label, closer in closers:
        try:
            result = closer()
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                await result
        except Exception as close_err:
            logger.debug(
                "Failed to close %s for background task %s: %s",
                label,
                task_id,
                close_err,
            )


def parse_message_content(content: str) -> Union[str, List[Dict[str, Any]]]:
    """
    Parse message content that might be a JSON-encoded multimodal payload.

    Handles these shapes:
    - '[{"type":"text","text":"..."}, {"type":"image_url",...}]'
    - '{"content": [...]}'
    - '{"role": "user", "content": "[...]"}' (double-wrapped)
    - '{"type":"text","text":"..."}' (single part object)
    """
    if not content:
        return content

    trimmed = content.lstrip()
    if not trimmed or trimmed[0] not in "[{":
        return content

    try:
        parsed: Any = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content

    # Direct list of content parts
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict) and "type" in parsed[0]:
            logger.debug(
                f"Parsed multimodal content: {len(parsed)} parts, "
                f"types={[p.get('type') for p in parsed]}"
            )
            return parsed
        return content

    if isinstance(parsed, dict):
        # Single content part dict
        if parsed.get("type"):
            return [parsed]

        # Wrapped with content field
        if "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, list):
                return inner
            if isinstance(inner, str) and inner != content:
                inner_parsed = parse_message_content(inner)
                return inner_parsed

    return content


def strip_images_from_content(
    content: Union[str, List[Dict[str, Any]]],
) -> Union[str, List[Dict[str, Any]]]:
    """
    Strip image_url parts from multimodal content to reduce token usage in history.

    Images in chat history are generally not needed - only the current message
    needs the actual image data. This significantly reduces tokens sent to the LLM.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    # Filter out image_url parts, keep text and file references
    stripped = []
    image_count = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            image_count += 1
            # Replace with a placeholder text
            stripped.append({"type": "text", "text": "[image attachment]"})
        else:
            stripped.append(part)

    if image_count > 0:
        logger.debug(f"Stripped {image_count} images from chat history")

    return stripped if stripped else content


def _truncate_tool_output(output: str, max_chars: int = 5000) -> str:
    """Truncate tool output for inclusion in chat history."""
    if not output:
        return "(no output)"
    output = strip_table_metadata(output)
    if len(output) > max_chars:
        return output[:max_chars] + "... (truncated)"
    return output


def rebuild_tool_messages_from_events(
    events: list[dict[str, Any]],
    msg_idx: int,
    max_tool_output_chars: int = 5000,
) -> list[Any]:
    """Convert stored chronological events into native LangChain message objects.

    Produces ``AIMessage(tool_calls=[...])`` + ``ToolMessage(...)`` pairs for tool
    events, and regular ``AIMessage(content=...)`` for content blocks.  This format
    is what LangChain's agent executor expects and prevents the LLM from echoing
    tool-call metadata as conversational text.

    Args:
        events: Chronological event dicts from a stored assistant message.
        msg_idx: Index of the parent message in the conversation (used to
            generate deterministic ``tool_call_id`` values).
        max_tool_output_chars: Maximum characters per tool output before truncation.

    Returns:
        List of ``BaseMessage`` objects ready for the ``chat_history`` placeholder.
    """
    messages: list[Any] = []
    pending_content = ""
    tool_seq = 0

    for ev in events:
        ev_type = ev.get("type")

        if ev_type == "content":
            pending_content += ev.get("content", "")

        elif ev_type == "tool":
            # Flush accumulated text before this tool call
            if pending_content.strip():
                messages.append(AIMessage(content=pending_content))
                pending_content = ""

            tool_call_id = f"call_{msg_idx}_{tool_seq}"
            tool_seq += 1
            tool_name = ev.get("tool", "unknown")
            tool_args = ev.get("input") or {}
            tool_output = _truncate_tool_output(
                str(ev.get("output", "")), max_tool_output_chars
            )

            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "id": tool_call_id,
                        }
                    ],
                )
            )
            messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call_id,
                )
            )

        elif ev_type == "reasoning":
            # Reasoning blocks are internal; skip in history
            continue

    # Flush any trailing content
    if pending_content.strip():
        messages.append(AIMessage(content=pending_content))

    return messages


class BackgroundTaskService:
    """Service for managing background chat tasks."""

    def __init__(self):
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._filesystem_scheduler_task: Optional[asyncio.Task] = None
        self._schema_scheduler_task: Optional[asyncio.Task] = None
        self._git_scheduler_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background task service."""
        logger.info("Background task service starting")
        self._shutdown = False
        # Resume any pending/running tasks from previous session
        await self._resume_stale_tasks()
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        # Start filesystem re-indexing scheduler
        self._filesystem_scheduler_task = asyncio.create_task(
            self._filesystem_reindex_scheduler()
        )
        # Start schema re-indexing scheduler
        self._schema_scheduler_task = asyncio.create_task(
            self._schema_reindex_scheduler()
        )
        # Start git re-indexing scheduler
        self._git_scheduler_task = asyncio.create_task(self._git_reindex_scheduler())

    async def stop(self):
        """Stop the background task service."""
        logger.info("Background task service stopping")
        self._shutdown = True

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel filesystem scheduler
        if (
            self._filesystem_scheduler_task
            and not self._filesystem_scheduler_task.done()
        ):
            self._filesystem_scheduler_task.cancel()
            try:
                await self._filesystem_scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel schema scheduler
        if self._schema_scheduler_task and not self._schema_scheduler_task.done():
            self._schema_scheduler_task.cancel()
            try:
                await self._schema_scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel git scheduler
        if self._git_scheduler_task and not self._git_scheduler_task.done():
            self._git_scheduler_task.cancel()
            try:
                await self._git_scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all running tasks
        for _, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._running_tasks.clear()

    async def _resume_stale_tasks(self):
        """Mark tasks that were running when server restarted as interrupted.

        Instead of automatically resuming tasks (which could cause issues if the
        conversation was modified or cleared), we mark them as 'interrupted' so
        the frontend can show a Continue button to the user.
        """
        try:
            db = await repository._get_db()
            stale_tasks = await db.chattask.find_many(
                where={"status": {"in": ["pending", "running"]}}
            )

            if not stale_tasks:
                return

            count = 0
            for prisma_task in stale_tasks:
                # Mark as interrupted instead of resuming
                await db.chattask.update(
                    where={"id": prisma_task.id},
                    data={
                        "status": "interrupted",
                        "completedAt": datetime.utcnow(),
                        "lastUpdateAt": datetime.utcnow(),
                    },
                )
                # Clear the active task reference from the conversation
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None},
                )
                count += 1

            if count > 0:
                logger.info(f"Marked {count} stale task(s) as interrupted")

                # Finalize any open usage attempts linked to these stale tasks
                stale_task_ids = [t.id for t in stale_tasks]
                usage_count = await finalize_stale_attempts_for_tasks(stale_task_ids)
                if usage_count > 0:
                    logger.info(f"Finalized {usage_count} stale usage attempt(s)")

        except Exception as e:
            logger.error(f"Failed to mark stale tasks as interrupted: {e}")

    async def _cleanup_loop(self):
        """Periodically clean up stale tasks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await repository.cleanup_stale_tasks(max_age_seconds=600)
                await cleanup_expired_chat_attachments()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _filesystem_reindex_scheduler(self):
        """
        Periodically check filesystem indexer tools and trigger re-indexing.

        Runs every hour and checks each filesystem_indexer tool config to see
        if it's due for re-indexing based on its configured interval.
        """
        # Initial delay to let the system stabilize
        await asyncio.sleep(60)

        while not self._shutdown:
            try:
                await self._check_and_trigger_filesystem_reindex()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in filesystem reindex scheduler: {e}")

            # Check every hour
            await asyncio.sleep(3600)

    async def _check_and_trigger_filesystem_reindex(self):
        """Check filesystem indexer tools and trigger re-indexing if due."""
        try:
            db = await repository._get_db()

            # Get all enabled filesystem_indexer tool configs
            tool_configs = await db.toolconfig.find_many(
                where={
                    "toolType": "filesystem_indexer",
                    "enabled": True,
                }
            )

            for config in tool_configs:
                try:
                    connection_config = config.connectionConfig
                    if not isinstance(connection_config, dict):
                        continue

                    # Parse the connection config
                    fs_config = FilesystemConnectionConfig(**connection_config)

                    # Skip if manual-only (interval = 0)
                    if fs_config.reindex_interval_hours <= 0:
                        continue

                    # Check if re-indexing is due
                    last_indexed = fs_config.last_indexed_at
                    if last_indexed:
                        next_reindex = last_indexed + timedelta(
                            hours=fs_config.reindex_interval_hours
                        )
                        # Use timezone-aware comparison (handle both naive and aware)
                        now = datetime.now(timezone.utc)
                        # Make last_indexed tz-aware if needed
                        if last_indexed.tzinfo is None:
                            last_indexed = last_indexed.replace(tzinfo=timezone.utc)
                            next_reindex = last_indexed + timedelta(
                                hours=fs_config.reindex_interval_hours
                            )
                        if now < next_reindex:
                            continue  # Not due yet

                    # Trigger re-indexing
                    logger.info(
                        f"Triggering scheduled re-index for '{config.name}' "
                        f"(last indexed: {last_indexed or 'never'})"
                    )
                    await filesystem_indexer.trigger_index(
                        tool_config_id=config.id,
                        config=fs_config,
                        full_reindex=False,  # Incremental
                    )

                except Exception as e:
                    logger.warning(
                        f"Error checking filesystem tool '{config.name}': {e}"
                    )

        except Exception as e:
            logger.error(f"Error in filesystem reindex check: {e}")

    async def _schema_reindex_scheduler(self):
        """
        Periodically check SQL database tools and trigger schema re-indexing.

        Runs every hour and checks each schema-capable tool config to see
        if it's due for schema re-indexing based on its configured interval.
        """
        # Initial delay to let the system stabilize
        await asyncio.sleep(120)  # 2 minutes

        while not self._shutdown:
            try:
                await self._check_and_trigger_schema_reindex()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schema reindex scheduler: {e}")

            # Check every hour
            await asyncio.sleep(3600)

    async def _check_and_trigger_schema_reindex(self):
        """Check SQL database tools and trigger schema re-indexing if due."""
        try:
            # Use repository to get properly decrypted tool configs
            all_tools = await repository.list_tool_configs(enabled_only=True)

            # Filter for schema-capable tools (postgres, mssql, mysql)
            tool_configs = [
                t
                for t in all_tools
                if t.tool_type.value in SCHEMA_INDEXER_CAPABLE_TYPES
            ]

            for config in tool_configs:
                try:
                    connection_config = config.connection_config
                    if not isinstance(connection_config, dict):
                        continue

                    # Parse into SchemaIndexConfig to handle type coercion and validation
                    # This properly handles cases where numeric values are stored as strings
                    try:
                        schema_config = SchemaIndexConfig(**connection_config)
                    except Exception:
                        continue

                    if not schema_config.schema_index_enabled:
                        continue

                    if schema_config.schema_index_interval_hours <= 0:
                        continue  # Manual only

                    # Check if re-indexing is due
                    last_indexed = schema_config.last_schema_indexed_at
                    if last_indexed:
                        # Ensure timezone awareness for comparison
                        if last_indexed.tzinfo is None:
                            last_indexed = last_indexed.replace(tzinfo=timezone.utc)

                        next_reindex = last_indexed + timedelta(
                            hours=schema_config.schema_index_interval_hours
                        )
                        if datetime.now(timezone.utc) < next_reindex:
                            continue  # Not due yet

                    # Trigger schema re-indexing
                    logger.info(
                        f"Triggering scheduled schema re-index for '{config.name}' "
                        f"(last indexed: {last_indexed or 'never'})"
                    )
                    await schema_indexer.trigger_index(
                        tool_config_id=config.id,
                        tool_type=config.tool_type.value,
                        connection_config=connection_config,
                        full_reindex=False,  # Use hash-based change detection
                        tool_name=safe_tool_name(config.name) or None,
                    )

                except Exception as e:
                    logger.warning(
                        f"Error checking schema for tool '{config.name}': {e}"
                    )

        except Exception as e:
            logger.error(f"Error in schema reindex check: {e}")

    async def _git_reindex_scheduler(self):
        """
        Periodically check git-based indexes and trigger pull & re-indexing.

        Runs every hour and checks each git-based index to see
        if it's due for re-indexing based on its configured interval.
        """
        # Initial delay to let the system stabilize
        await asyncio.sleep(180)  # 3 minutes

        while not self._shutdown:
            try:
                await self._check_and_trigger_git_reindex()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in git reindex scheduler: {e}")

            # Check every hour
            await asyncio.sleep(3600)

    async def _check_and_trigger_git_reindex(self):
        """Check git-based indexes and trigger pull & re-indexing if due."""
        try:
            db = await repository._get_db()

            # Get all git-based index metadata
            git_indexes = await db.indexmetadata.find_many(where={"sourceType": "git"})

            for metadata in git_indexes:
                try:
                    # Get config snapshot
                    config_snapshot = metadata.configSnapshot
                    if not isinstance(config_snapshot, dict):
                        continue

                    # Get reindex interval (default 0 = manual only)
                    interval_hours = config_snapshot.get("reindex_interval_hours", 0)
                    if interval_hours <= 0:
                        continue  # Manual only

                    # Check if re-indexing is due
                    last_modified = metadata.lastModified
                    if last_modified:
                        next_reindex = last_modified + timedelta(hours=interval_hours)
                        if datetime.utcnow() < next_reindex.replace(tzinfo=None):
                            continue  # Not due yet

                    # Check if there's already a running job for this index
                    running_job = await db.indexjob.find_first(
                        where={
                            "name": metadata.name,
                            "status": {"in": ["pending", "processing"]},
                        }
                    )
                    if running_job:
                        logger.debug(
                            f"Skipping scheduled git reindex for '{metadata.name}': "
                            f"job {running_job.id} already running"
                        )
                        continue

                    # Get stored token (decrypt if needed)
                    from ragtime.core.encryption import decrypt_secret

                    git_token = (
                        decrypt_secret(metadata.gitToken) if metadata.gitToken else None
                    )

                    # Build config from snapshot
                    from ragtime.indexer.models import IndexConfig, OcrMode, OcrProvider

                    # Handle legacy enable_ocr field and convert to ocr_mode
                    ocr_mode_str = config_snapshot.get("ocr_mode", "disabled")
                    if ocr_mode_str == "disabled" and config_snapshot.get(
                        "enable_ocr", False
                    ):
                        ocr_mode_str = "tesseract"  # Legacy compatibility

                    config = IndexConfig(
                        name=metadata.name,
                        description=metadata.description or "",
                        file_patterns=config_snapshot.get("file_patterns", ["**/*"]),
                        exclude_patterns=config_snapshot.get(
                            "exclude_patterns",
                            ["**/test/**", "**/tests/**", "**/__pycache__/**"],
                        ),
                        chunk_size=config_snapshot.get("chunk_size", 1000),
                        chunk_overlap=config_snapshot.get("chunk_overlap", 200),
                        max_file_size_kb=config_snapshot.get("max_file_size_kb", 500),
                        ocr_mode=OcrMode(ocr_mode_str),
                        ocr_provider=(
                            OcrProvider(config_snapshot["ocr_provider"])
                            if config_snapshot.get("ocr_provider")
                            else None
                        ),
                        ocr_vision_model=config_snapshot.get("ocr_vision_model"),
                        git_clone_timeout_minutes=config_snapshot.get(
                            "git_clone_timeout_minutes", 5
                        ),
                        git_history_depth=config_snapshot.get("git_history_depth", 1),
                        reindex_interval_hours=interval_hours,
                    )

                    # Trigger re-indexing
                    logger.info(
                        f"Triggering scheduled git pull & re-index for '{metadata.name}' "
                        f"(last indexed: {last_modified or 'never'})"
                    )
                    await indexer.create_index_from_git(
                        git_url=metadata.source or "",
                        branch=metadata.gitBranch or "main",
                        config=config,
                        git_token=git_token,
                    )

                except Exception as e:
                    logger.warning(f"Error checking git index '{metadata.name}': {e}")

        except Exception as e:
            logger.error(f"Error in git reindex check: {e}")

    def start_task(
        self,
        conversation_id: str,
        user_message: str,
        existing_task_id: Optional[str] = None,
        blocked_tool_names: Optional[set[str]] = None,
        workspace_context: Optional[dict[str, str]] = None,
        disabled_builtin_tool_ids: Optional[set[str]] = None,
        usage_attempt_id: Optional[str] = None,
    ) -> str:
        """
        Start a background task for processing a chat message.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message to process
            existing_task_id: Optional existing task ID to resume
            usage_attempt_id: Optional usage tracking attempt ID

        Returns:
            The task ID
        """
        task_id = existing_task_id or ""

        async def run():
            nonlocal task_id
            full_response = ""
            events: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            partial_message_persisted = False
            reasoning_block_started_at: Optional[datetime] = None
            try:
                # Create or get the task
                if not task_id:
                    task = await repository.create_chat_task(
                        conversation_id, user_message
                    )
                    task_id = task.id
                else:
                    task = await repository.get_chat_task(task_id)
                    if not task:
                        logger.error(f"Task {task_id} not found")
                        return

                # Update status to running
                await repository.update_chat_task_status(
                    task_id, ChatTaskStatus.running
                )

                # Notify conversation subscribers (SSE) that a task has started.
                # Both authenticated ChatPanel and PublicSharedChatView listen on
                # `conversation:{id}` so they can react in real time without
                # frequent polling.
                await task_event_bus.publish(
                    f"conversation:{conversation_id}",
                    {
                        "event": "task_started",
                        "task_id": task_id,
                        "conversation_id": conversation_id,
                    },
                )

                # Bind usage attempt to this task
                if usage_attempt_id:
                    await bind_usage_attempt_task(usage_attempt_id, task_id)

                # Get conversation for chat history
                conv = await repository.get_conversation(conversation_id)
                if not conv:
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, "Conversation not found"
                    )
                    return

                # Build chat history (exclude the user message we're about to process)
                # Include tool call information so the LLM has full context
                app_settings = await SettingsCache.get_instance().get_settings()
                max_tool_output_chars = int(
                    app_settings.get("max_tool_output_chars", 5000)
                )

                chat_history = []
                for msg_idx, msg in enumerate(
                    conv.messages[:-1]
                ):  # Exclude last (current user) message
                    if msg.role == "user":
                        # Parse content in case it's a JSON-encoded multimodal array
                        parsed_content = parse_message_content(msg.content)
                        if not isinstance(parsed_content, str):
                            parsed_content, attachment_stats = (
                                await rag.preprocess_message_content_async(
                                    parsed_content,
                                    conversation_id=conversation_id,
                                    user_id=conv.user_id,
                                    workspace_id=getattr(conv, "workspace_id", None),
                                    model_id=conv.model,
                                )
                            )
                            if attachment_stats:
                                logger.debug(
                                    "Expanded chat history attachments for conversation=%s files=%d included_chunks=%d",
                                    conversation_id,
                                    attachment_stats.get("file_count", 0),
                                    attachment_stats.get("included_chunk_count", 0),
                                )
                        # Strip images from history - they consume tokens but add little value
                        parsed_content = strip_images_from_content(parsed_content)
                        chat_history.append(HumanMessage(content=parsed_content))
                    elif msg.role == "assistant":
                        if msg.events:
                            # Reconstruct native AIMessage(tool_calls) + ToolMessage pairs
                            chat_history.extend(
                                rebuild_tool_messages_from_events(
                                    msg.events, msg_idx, max_tool_output_chars
                                )
                            )
                        elif msg.content and msg.content.strip():
                            chat_history.append(AIMessage(content=msg.content))

                if not rag.is_ready:
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, "RAG service not ready"
                    )
                    await task_event_bus.publish(
                        task_id,
                        {
                            "completed": True,
                            "status": "failed",
                            "error": "RAG service not ready",
                        },
                    )
                    await task_event_bus.publish(
                        f"conversation:{conversation_id}",
                        {
                            "event": "task_completed",
                            "task_id": task_id,
                            "status": "failed",
                            "error": "RAG service not ready",
                        },
                    )
                    return

                # Track running tools by run_id -> index in events list
                running_tool_indices: dict[str, int] = {}
                hit_max_iterations = False  # Track if we hit the iteration limit
                last_update = datetime.utcnow()
                current_version = 0  # Version counter for efficient client polling

                # Use UI agent (with chart tool and enhanced prompt)
                # Parse user message in case it's a JSON-encoded multimodal array
                parsed_user_message = parse_message_content(user_message)

                # Inactivity timeout: if the LLM/agent stream yields no
                # events for this many seconds the task is considered stuck
                # (e.g. hung LLM API call during heavy server load).
                # Must exceed the maximum tool timeout (default 300s) since
                # tool execution blocks the event stream between tool_start
                # and tool_end.
                _STREAM_INACTIVITY_TIMEOUT = 600  # 10 minutes

                _stream = rag.process_query_stream(
                    parsed_user_message,
                    chat_history,
                    is_ui=True,
                    blocked_tool_names=blocked_tool_names,
                    workspace_context=workspace_context,
                    conversation_model=conv.model,
                    conversation_id=conversation_id,
                    user_id=conv.user_id,
                    chat_task_id=task_id,
                    message_index=len(conv.messages),
                    disabled_builtin_tool_ids=disabled_builtin_tool_ids,
                )
                _stream_iter = _stream.__aiter__()
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(
                                _stream_iter.__anext__(),
                                timeout=_STREAM_INACTIVITY_TIMEOUT,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            if full_response.strip():
                                # Edge case: some provider streams can stall after
                                # emitting usable assistant text (and occasionally
                                # after a final tool start), never sending an
                                # explicit stream end event.
                                logger.warning(
                                    "Task %s stream inactive for %ss after partial output; "
                                    "treating as graceful stream end "
                                    "(content_chars=%d, events=%d, running_tools=%d)",
                                    task_id,
                                    _STREAM_INACTIVITY_TIMEOUT,
                                    len(full_response),
                                    len(events),
                                    len(running_tool_indices),
                                )

                                # Close any unresolved running tool rows so the
                                # stored event log does not leave a perpetual
                                # "running" tool bubble.
                                for tool_idx in list(running_tool_indices.values()):
                                    if (
                                        0 <= tool_idx < len(events)
                                        and events[tool_idx].get("type") == "tool"
                                        and "output" not in events[tool_idx]
                                    ):
                                        events[tool_idx][
                                            "output"
                                        ] = "Tool completed or stream ended without a final tool_end event."
                                running_tool_indices.clear()
                                break

                            raise TimeoutError(
                                f"LLM/agent stream produced no output for "
                                f"{_STREAM_INACTIVITY_TIMEOUT}s - possible hung API call"
                            ) from exc

                        if self._shutdown:
                            await repository.update_chat_task_status(
                                task_id,
                                ChatTaskStatus.interrupted,
                                error_message=DEV_SERVER_INTERRUPT_MESSAGE,
                            )
                            await task_event_bus.publish(
                                task_id,
                                {
                                    "completed": True,
                                    "status": "interrupted",
                                    "error": DEV_SERVER_INTERRUPT_MESSAGE,
                                },
                            )
                            await task_event_bus.publish(
                                f"conversation:{conversation_id}",
                                {
                                    "event": "task_completed",
                                    "task_id": task_id,
                                    "status": "interrupted",
                                    "error": DEV_SERVER_INTERRUPT_MESSAGE,
                                },
                            )
                            return

                        if not isinstance(event, dict):
                            # Text token
                            token = event
                            full_response += token

                            if reasoning_block_started_at is not None:
                                finalize_reasoning_block(
                                    events, reasoning_block_started_at
                                )
                                reasoning_block_started_at = None

                            # Add to events
                            if events and events[-1].get("type") == "content":
                                events[-1]["content"] += token
                            else:
                                events.append(
                                    {
                                        "type": "content",
                                        "channel": "final",
                                        "content": token,
                                    }
                                )

                            # Update streaming state less frequently (every 400ms) for text tokens
                            # Tool events are still updated immediately above
                            now = datetime.utcnow()
                            if (now - last_update).total_seconds() > 0.4:
                                result = (
                                    await repository.update_chat_task_streaming_state(
                                        task_id,
                                        full_response,
                                        events,
                                        tool_calls,
                                        hit_max_iterations,
                                        current_version,
                                    )
                                )
                                if result and result.streaming_state:
                                    current_version = result.streaming_state.version
                                    await task_event_bus.publish(
                                        task_id, result.streaming_state.dict()
                                    )
                                    await _notify_conversation_progress(
                                        conversation_id, task_id
                                    )
                                last_update = now
                            continue

                        event_type = event.get("type")

                        if event_type == "tool_start":
                            run_id = event.get("run_id", "")
                            tool_name = event.get("tool")
                            # Check if a ghost generating event already exists
                            # for this tool (created by tool_generating before
                            # on_tool_start fires).
                            ghost_idx = None
                            for gi in range(len(events) - 1, -1, -1):
                                gev = events[gi]
                                if (
                                    gev.get("type") == "tool"
                                    and gev.get("tool") == tool_name
                                    and "output" not in gev
                                    and "input" not in gev
                                ):
                                    ghost_idx = gi
                                    break

                            if ghost_idx is not None:
                                # Upgrade the ghost event with full tool_start data
                                events[ghost_idx]["input"] = event.get("input")
                                events[ghost_idx]["connection"] = event.get(
                                    "connection"
                                )
                                if event.get("presentation") is not None:
                                    events[ghost_idx]["presentation"] = event.get(
                                        "presentation"
                                    )
                                if run_id:
                                    running_tool_indices[run_id] = ghost_idx
                            else:
                                # Add tool call to events immediately (without output)
                                # This allows the frontend to show the hourglass/running state
                                tool_event = {
                                    "type": "tool",
                                    "channel": "commentary",
                                    "tool": tool_name,
                                    "input": event.get("input"),
                                    "connection": event.get("connection"),
                                    # No "output" key = running state
                                }
                                if event.get("presentation") is not None:
                                    tool_event["presentation"] = event.get(
                                        "presentation"
                                    )
                                events.append(tool_event)
                                # Track this tool's index by run_id
                                if run_id:
                                    running_tool_indices[run_id] = len(events) - 1

                            # Force immediate update so the UI shows the running tool
                            result = await repository.update_chat_task_streaming_state(
                                task_id,
                                full_response,
                                events,
                                tool_calls,
                                hit_max_iterations,
                                current_version,
                            )
                            if result and result.streaming_state:
                                current_version = result.streaming_state.version
                                await task_event_bus.publish(
                                    task_id, result.streaming_state.dict()
                                )
                                await _notify_conversation_progress(
                                    conversation_id, task_id
                                )
                            last_update = datetime.utcnow()

                        elif event_type == "tool_end":
                            run_id = event.get("run_id", "")
                            # Find the matching tool_start by run_id
                            tool_idx = running_tool_indices.pop(run_id, None)
                            if tool_idx is not None:
                                # Update the existing tool event with output
                                events[tool_idx]["output"] = event.get("output")
                                if event.get("presentation") is not None:
                                    events[tool_idx]["presentation"] = event.get(
                                        "presentation"
                                    )
                                tool_calls.append(
                                    {
                                        "tool": events[tool_idx]["tool"],
                                        "input": events[tool_idx].get("input"),
                                        "output": events[tool_idx].get("output"),
                                        "connection": events[tool_idx].get(
                                            "connection"
                                        ),
                                        "presentation": events[tool_idx].get(
                                            "presentation"
                                        ),
                                    }
                                )

                                # Force immediate update so the UI shows the completed tool
                                result = (
                                    await repository.update_chat_task_streaming_state(
                                        task_id,
                                        full_response,
                                        events,
                                        tool_calls,
                                        hit_max_iterations,
                                        current_version,
                                    )
                                )
                                if result and result.streaming_state:
                                    current_version = result.streaming_state.version
                                    await task_event_bus.publish(
                                        task_id, result.streaming_state.dict()
                                    )
                                    await _notify_conversation_progress(
                                        conversation_id, task_id
                                    )
                                last_update = datetime.utcnow()

                        elif event_type == "max_iterations_reached":
                            hit_max_iterations = True
                            logger.info(f"Task {task_id} hit max iterations limit")
                            # Force immediate update
                            result = await repository.update_chat_task_streaming_state(
                                task_id,
                                full_response,
                                events,
                                tool_calls,
                                hit_max_iterations,
                                current_version,
                            )
                            if result and result.streaming_state:
                                current_version = result.streaming_state.version
                                await task_event_bus.publish(
                                    task_id, result.streaming_state.dict()
                                )
                                await _notify_conversation_progress(
                                    conversation_id, task_id
                                )
                            last_update = datetime.utcnow()

                        elif event_type == "tool_generating":
                            # LLM is streaming tool call arguments - update
                            # the latest running tool event with line progress.
                            gen_tool = event.get("tool", "")
                            gen_lines = event.get("lines", 0)
                            if gen_tool and gen_lines:
                                # Find the last running tool event matching this name
                                for i in range(len(events) - 1, -1, -1):
                                    ev = events[i]
                                    if (
                                        ev.get("type") == "tool"
                                        and ev.get("tool") == gen_tool
                                        and "output" not in ev
                                    ):
                                        ev["generating_lines"] = gen_lines
                                        break
                                else:
                                    # No matching running tool - emit a ghost
                                    # tool event so the frontend shows progress
                                    events.append(
                                        {
                                            "type": "tool",
                                            "channel": "commentary",
                                            "tool": gen_tool,
                                            "generating_lines": gen_lines,
                                            **(
                                                {
                                                    "presentation": event.get(
                                                        "presentation"
                                                    )
                                                }
                                                if event.get("presentation") is not None
                                                else {}
                                            ),
                                        }
                                    )

                            # Throttle updates to avoid flooding the SSE bus
                            now = datetime.utcnow()
                            if (now - last_update).total_seconds() > 0.6:
                                result = (
                                    await repository.update_chat_task_streaming_state(
                                        task_id,
                                        full_response,
                                        events,
                                        tool_calls,
                                        hit_max_iterations,
                                        current_version,
                                    )
                                )
                                if result and result.streaming_state:
                                    current_version = result.streaming_state.version
                                    await task_event_bus.publish(
                                        task_id, result.streaming_state.dict()
                                    )
                                    await _notify_conversation_progress(
                                        conversation_id, task_id
                                    )
                                last_update = now

                        elif event_type == "reasoning":
                            # Reasoning/thinking content from LLM
                            reasoning_text = event.get("content", "")
                            if reasoning_text:
                                reasoning_block_started_at = append_reasoning_event(
                                    events,
                                    reasoning_text,
                                    reasoning_block_started_at,
                                )
                finally:
                    await _close_stream_handles(_stream_iter, _stream, task_id)

                # Task completed successfully - save final state
                if not full_response.strip():
                    synthesized_response = _synthesize_incomplete_response(
                        events,
                        tool_calls,
                        hit_max_iterations,
                    )

                    if synthesized_response:
                        full_response = synthesized_response
                        events.append(
                            {
                                "type": "content",
                                "channel": "final",
                                "content": full_response,
                            }
                        )
                        logger.warning(
                            "Background task %s ended without assistant text; "
                            "synthesized fallback response (events=%d, tool_calls=%d, hit_max_iterations=%s)",
                            task_id,
                            len(events),
                            len(tool_calls),
                            hit_max_iterations,
                        )
                    else:
                        error_message = (
                            "Model stream ended without assistant text output"
                        )
                        logger.warning(
                            "Background task %s ended with no assistant text "
                            "(events=%d, tool_calls=%d, hit_max_iterations=%s)",
                            task_id,
                            len(events),
                            len(tool_calls),
                            hit_max_iterations,
                        )
                        await repository.update_chat_task_status(
                            task_id,
                            ChatTaskStatus.failed,
                            error_message,
                        )
                        await task_event_bus.publish(
                            task_id,
                            {
                                "completed": True,
                                "status": "failed",
                                "error": error_message,
                            },
                        )
                        if usage_attempt_id:
                            await finalize_usage_attempt(
                                usage_attempt_id,
                                status="failed",
                                failure_reason=error_message,
                                output_tokens=_estimate_output_tokens(
                                    full_response, events
                                ),
                            )
                        return

                wrapped_error_detail = _extract_wrapped_processing_error(full_response)
                if wrapped_error_detail is not None:
                    if reasoning_block_started_at is not None:
                        finalize_reasoning_block(events, reasoning_block_started_at)
                        reasoning_block_started_at = None
                    if (
                        not partial_message_persisted
                        and await _persist_partial_assistant_message(
                            conversation_id,
                            full_response,
                            events,
                        )
                    ):
                        partial_message_persisted = True
                    await repository.update_chat_task_status(
                        task_id,
                        ChatTaskStatus.interrupted,
                        wrapped_error_detail,
                    )
                    await task_event_bus.publish(
                        task_id,
                        {
                            "completed": True,
                            "status": "interrupted",
                            "error": wrapped_error_detail,
                            "content": full_response,
                            "events": events,
                        },
                    )
                    await task_event_bus.publish(
                        f"conversation:{conversation_id}",
                        {
                            "event": "task_completed",
                            "task_id": task_id,
                            "status": "interrupted",
                            "error": wrapped_error_detail,
                        },
                    )
                    logger.warning(
                        "Background task %s produced wrapped internal error; "
                        "marking interrupted (%s)",
                        task_id,
                        wrapped_error_detail,
                    )
                    if usage_attempt_id:
                        await finalize_usage_attempt(
                            usage_attempt_id,
                            status="interrupted",
                            failure_reason=wrapped_error_detail,
                            output_tokens=_estimate_output_tokens(
                                full_response,
                                events,
                            ),
                        )
                    return

                if reasoning_block_started_at is not None:
                    finalize_reasoning_block(events, reasoning_block_started_at)
                    reasoning_block_started_at = None

                await repository.complete_chat_task(
                    task_id,
                    full_response,
                    events,
                    tool_calls,
                    hit_max_iterations,
                    current_version,
                )

                # Add the assistant response to the conversation
                persisted_conv = await repository.add_message(
                    conversation_id,
                    "assistant",
                    full_response,
                    events=events if events else None,
                )
                try:
                    await repository.link_assistant_snapshot_tool_calls(
                        persisted_conv,
                        (
                            getattr(persisted_conv, "workspace_id", None)
                            if persisted_conv
                            else None
                        ),
                    )
                except Exception as link_err:
                    logger.warning(
                        f"Failed to link agent-created snapshot to assistant message: {link_err}"
                    )
                partial_message_persisted = True

                # Notify completion
                await task_event_bus.publish(
                    task_id,
                    {
                        "completed": True,
                        "status": "completed",
                        "content": full_response,
                        "events": events,
                    },
                )
                await task_event_bus.publish(
                    f"conversation:{conversation_id}",
                    {
                        "event": "task_completed",
                        "task_id": task_id,
                        "status": "completed",
                    },
                )

                logger.info(f"Background task {task_id} completed successfully")

                # Finalize usage attempt on success
                if usage_attempt_id:
                    await finalize_usage_attempt(
                        usage_attempt_id,
                        status="completed",
                        output_tokens=_estimate_output_tokens(full_response, events),
                    )

            except asyncio.CancelledError:
                # Save partial response before cancelling
                try:
                    if reasoning_block_started_at is not None:
                        finalize_reasoning_block(events, reasoning_block_started_at)
                        reasoning_block_started_at = None
                    if (
                        not partial_message_persisted
                        and await _persist_partial_assistant_message(
                            conversation_id,
                            full_response,
                            events,
                        )
                    ):
                        partial_message_persisted = True
                        logger.info(
                            "Saved partial assistant message for task %s (%s)",
                            task_id,
                            "cancelled",
                        )

                    # If shutting down (hot-reload/restart), mark as interrupted
                    # so user sees "continue?" prompt. Otherwise mark as cancelled.
                    if self._shutdown:
                        await repository.update_chat_task_status(
                            task_id,
                            ChatTaskStatus.interrupted,
                            error_message=DEV_SERVER_INTERRUPT_MESSAGE,
                        )
                        await task_event_bus.publish(
                            task_id,
                            {
                                "completed": True,
                                "status": "interrupted",
                                "error": DEV_SERVER_INTERRUPT_MESSAGE,
                            },
                        )
                        logger.info(f"Task {task_id} interrupted by shutdown")
                        if usage_attempt_id:
                            await finalize_usage_attempt(
                                usage_attempt_id,
                                status="interrupted",
                                failure_reason="Server shutdown",
                                output_tokens=_estimate_output_tokens(
                                    full_response, events
                                ),
                            )
                    else:
                        await repository.cancel_chat_task(task_id)
                        await task_event_bus.publish(
                            task_id, {"completed": True, "status": "cancelled"}
                        )
                        if usage_attempt_id:
                            await finalize_usage_attempt(
                                usage_attempt_id,
                                status="cancelled",
                                output_tokens=_estimate_output_tokens(
                                    full_response, events
                                ),
                            )
                except Exception as db_err:
                    logger.warning(
                        f"Task {task_id}: Could not update task status (database may be disconnected): {db_err}"
                    )
                raise
            except Exception as e:
                logger.exception(f"Background task {task_id} failed")
                try:
                    if reasoning_block_started_at is not None:
                        finalize_reasoning_block(events, reasoning_block_started_at)
                        reasoning_block_started_at = None
                    if (
                        not partial_message_persisted
                        and await _persist_partial_assistant_message(
                            conversation_id,
                            full_response,
                            events,
                        )
                    ):
                        partial_message_persisted = True
                        logger.info(
                            "Saved partial assistant message for task %s (%s)",
                            task_id,
                            "failed",
                        )
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, str(e)
                    )
                    await task_event_bus.publish(
                        task_id,
                        {"completed": True, "status": "failed", "error": str(e)},
                    )
                    await task_event_bus.publish(
                        f"conversation:{conversation_id}",
                        {
                            "event": "task_completed",
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(e),
                        },
                    )
                    if usage_attempt_id:
                        await finalize_usage_attempt(
                            usage_attempt_id,
                            status="failed",
                            failure_reason=str(e),
                            output_tokens=_estimate_output_tokens(
                                full_response, events
                            ),
                        )
                except Exception as db_err:
                    logger.warning(
                        f"Task {task_id}: Could not update task status (database may be disconnected): {db_err}"
                    )
            finally:
                self._running_tasks.pop(task_id, None)

        # Create and track the task
        asyncio_task = asyncio.create_task(run())

        # We need to get task_id synchronously, so we'll use a placeholder
        # The actual task ID will be set inside the coroutine
        # For now, we return empty and the caller should poll for it
        if existing_task_id:
            self._running_tasks[existing_task_id] = asyncio_task
            return existing_task_id
        else:
            # Return placeholder - caller should create task first
            return ""

    async def start_task_async(
        self,
        conversation_id: str,
        user_message: str,
        blocked_tool_names: Optional[set[str]] = None,
        workspace_context: Optional[dict[str, str]] = None,
        disabled_builtin_tool_ids: Optional[set[str]] = None,
        usage_attempt_id: Optional[str] = None,
    ) -> str:
        """
        Start a background task asynchronously.

        Creates the task record first, then starts processing.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message to process
            usage_attempt_id: Optional usage tracking attempt ID

        Returns:
            The task ID
        """
        # Create the task record first
        task = await repository.create_chat_task(conversation_id, user_message)

        # Start processing in background
        self.start_task(
            conversation_id,
            user_message,
            task.id,
            blocked_tool_names=blocked_tool_names,
            workspace_context=workspace_context,
            disabled_builtin_tool_ids=disabled_builtin_tool_ids,
            usage_attempt_id=usage_attempt_id,
        )

        return task.id

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: The task ID to cancel

        Returns:
            True if task was found and cancelled
        """
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            if not task.done():
                task.cancel()
                return True
        return False

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is currently running."""
        if task_id in self._running_tasks:
            return not self._running_tasks[task_id].done()
        return False


# Global service instance
background_task_service = BackgroundTaskService()
