"""
Background task service for async chat message processing.

Handles running chat tasks in the background so they persist across
client disconnections, page switches, and server restarts.

Also handles scheduled filesystem re-indexing tasks.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from ragtime.core.logging import get_logger
from ragtime.indexer.models import ChatTaskStatus, FilesystemConnectionConfig
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import SCHEMA_INDEXER_CAPABLE_TYPES
from ragtime.indexer.utils import safe_tool_name

logger = get_logger(__name__)


class BackgroundTaskService:
    """Service for managing background chat tasks."""

    def __init__(self):
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._filesystem_scheduler_task: Optional[asyncio.Task] = None
        self._schema_scheduler_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background task service."""
        logger.info("Background task service starting")
        self._shutdown = False
        # Resume any pending/running tasks from previous session
        await self._resume_stale_tasks()
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        # Start filesystem re-indexing scheduler
        self._filesystem_scheduler_task = asyncio.create_task(
            self._filesystem_reindex_scheduler()
        )
        # Start schema re-indexing scheduler
        self._schema_scheduler_task = asyncio.create_task(
            self._schema_reindex_scheduler()
        )

    async def stop(self):
        """Stop the background task service."""
        logger.info("Background task service stopping")
        self._shutdown = True

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

        except Exception as e:
            logger.error(f"Failed to mark stale tasks as interrupted: {e}")

    async def _cleanup_loop(self):
        """Periodically clean up stale tasks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await repository.cleanup_stale_tasks(max_age_seconds=3600)
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
        from ragtime.indexer.filesystem_service import filesystem_indexer

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
                        if datetime.utcnow() < next_reindex:
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
        from ragtime.indexer.schema_service import schema_indexer

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

                    # Check if schema indexing is enabled
                    schema_index_enabled = connection_config.get(
                        "schema_index_enabled", False
                    )
                    if not schema_index_enabled:
                        continue

                    # Get interval (default 24 hours)
                    interval_hours = connection_config.get(
                        "schema_index_interval_hours", 24
                    )
                    if interval_hours <= 0:
                        continue  # Manual only

                    # Check if re-indexing is due
                    last_indexed_str = connection_config.get("last_schema_indexed_at")
                    if last_indexed_str:
                        try:
                            # Parse ISO format datetime
                            if isinstance(last_indexed_str, str):
                                last_indexed = datetime.fromisoformat(
                                    last_indexed_str.replace("Z", "+00:00")
                                )
                            else:
                                last_indexed = last_indexed_str

                            next_reindex = last_indexed + timedelta(
                                hours=interval_hours
                            )
                            if datetime.now(last_indexed.tzinfo or None) < next_reindex:
                                continue  # Not due yet
                        except (ValueError, TypeError):
                            pass  # Invalid date, trigger reindex

                    # Trigger schema re-indexing
                    logger.info(
                        f"Triggering scheduled schema re-index for '{config.name}' "
                        f"(last indexed: {last_indexed_str or 'never'})"
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

    def start_task(
        self,
        conversation_id: str,
        user_message: str,
        existing_task_id: Optional[str] = None,
    ) -> str:
        """
        Start a background task for processing a chat message.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message to process
            existing_task_id: Optional existing task ID to resume

        Returns:
            The task ID
        """
        task_id = existing_task_id or ""

        async def run():
            nonlocal task_id
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

                # Get conversation for chat history
                conv = await repository.get_conversation(conversation_id)
                if not conv:
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, "Conversation not found"
                    )
                    return

                # Build chat history (exclude the user message we're about to process)
                # Include tool call information so the LLM has full context
                chat_history = []
                for msg in conv.messages[:-1]:  # Exclude last (current user) message
                    if msg.role == "user":
                        chat_history.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        # Build content that includes tool call context
                        content_parts = []

                        # If message has events (interleaved content and tools), reconstruct
                        if msg.events:
                            for event in msg.events:
                                if event.get("type") == "content":
                                    content_parts.append(event.get("content", ""))
                                elif event.get("type") == "tool":
                                    tool_name = event.get("tool", "unknown")
                                    tool_input = event.get("input", {})
                                    tool_output = event.get("output", "")
                                    # Format tool call for context - use natural language
                                    # to avoid the model mimicking markup patterns
                                    input_str = ""
                                    if isinstance(tool_input, dict):
                                        # Extract query/code from common field names
                                        for field in [
                                            "query",
                                            "sql",
                                            "code",
                                            "command",
                                            "python_code",
                                        ]:
                                            if field in tool_input:
                                                input_str = str(tool_input[field])
                                                break
                                        if not input_str:
                                            input_str = str(tool_input)
                                    else:
                                        input_str = str(tool_input)
                                    # Use prose format instead of markup to prevent mimicry
                                    content_parts.append(
                                        f"\n(I used {tool_name} with: {input_str[:200]}{'...' if len(input_str) > 200 else ''} - Result: {tool_output[:500]}{'...' if len(str(tool_output)) > 500 else ''})\n"
                                    )
                            full_content = "".join(content_parts)
                        else:
                            # Fallback to just content if no events
                            full_content = msg.content

                        if full_content.strip():
                            chat_history.append(AIMessage(content=full_content))

                # Process the message with streaming
                from ragtime.rag import rag

                if not rag.is_ready:
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, "RAG service not ready"
                    )
                    return

                full_response = ""
                events = []
                tool_calls = []
                # Track running tools by run_id -> index in events list
                running_tool_indices: dict[str, int] = {}
                hit_max_iterations = False  # Track if we hit the iteration limit
                last_update = datetime.utcnow()
                current_version = 0  # Version counter for efficient client polling

                # Use UI agent (with chart tool and enhanced prompt)
                async for event in rag.process_query_stream(
                    user_message, chat_history, is_ui=True
                ):
                    if self._shutdown:
                        await repository.cancel_chat_task(task_id)
                        return

                    if isinstance(event, dict):
                        event_type = event.get("type")

                        if event_type == "tool_start":
                            run_id = event.get("run_id", "")
                            # Add tool call to events immediately (without output)
                            # This allows the frontend to show the hourglass/running state
                            tool_event = {
                                "type": "tool",
                                "tool": event.get("tool"),
                                "input": event.get("input"),
                                # No "output" key = running state
                            }
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
                            last_update = datetime.utcnow()

                        elif event_type == "tool_end":
                            run_id = event.get("run_id", "")
                            # Find the matching tool_start by run_id
                            tool_idx = running_tool_indices.pop(run_id, None)
                            if tool_idx is not None:
                                # Update the existing tool event with output
                                events[tool_idx]["output"] = event.get("output")
                                tool_calls.append(
                                    {
                                        "tool": events[tool_idx]["tool"],
                                        "input": events[tool_idx].get("input"),
                                        "output": events[tool_idx].get("output"),
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
                            last_update = datetime.utcnow()
                    else:
                        # Text token
                        token = event
                        full_response += token

                        # Add to events
                        if events and events[-1].get("type") == "content":
                            events[-1]["content"] += token
                        else:
                            events.append({"type": "content", "content": token})

                    # Update streaming state less frequently (every 400ms) for text tokens
                    # Tool events are still updated immediately above
                    now = datetime.utcnow()
                    if (now - last_update).total_seconds() > 0.4:
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
                        last_update = now

                # Task completed successfully - save final state
                await repository.complete_chat_task(
                    task_id,
                    full_response,
                    events,
                    tool_calls,
                    hit_max_iterations,
                    current_version,
                )

                # Add the assistant response to the conversation
                await repository.add_message(
                    conversation_id,
                    "assistant",
                    full_response,
                    tool_calls=tool_calls if tool_calls else None,
                    events=events if events else None,
                )

                # Auto-generate title if needed
                conv = await repository.get_conversation(conversation_id)
                if conv and conv.title == "New Chat" and len(conv.messages) >= 2:
                    first_msg = conv.messages[0].content[:50]
                    new_title = first_msg + (
                        "..." if len(conv.messages[0].content) > 50 else ""
                    )
                    await repository.update_conversation_title(
                        conversation_id, new_title
                    )

                logger.info(f"Background task {task_id} completed successfully")

            except asyncio.CancelledError:
                # Save partial response before cancelling
                try:
                    if full_response.strip():
                        await repository.add_message(
                            conversation_id,
                            "assistant",
                            full_response,
                            tool_calls=tool_calls if tool_calls else None,
                            events=events if events else None,
                        )
                        logger.info(
                            f"Saved partial response for cancelled task {task_id}"
                        )

                    # If shutting down (hot-reload/restart), mark as interrupted
                    # so user sees "continue?" prompt. Otherwise mark as cancelled.
                    if self._shutdown:
                        await repository.update_chat_task_status(
                            task_id, ChatTaskStatus.interrupted
                        )
                        logger.info(f"Task {task_id} interrupted by shutdown")
                    else:
                        await repository.cancel_chat_task(task_id)
                except Exception as db_err:
                    logger.warning(
                        f"Task {task_id}: Could not update task status (database may be disconnected): {db_err}"
                    )
                raise
            except Exception as e:
                logger.exception(f"Background task {task_id} failed")
                try:
                    await repository.update_chat_task_status(
                        task_id, ChatTaskStatus.failed, str(e)
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

    async def start_task_async(self, conversation_id: str, user_message: str) -> str:
        """
        Start a background task asynchronously.

        Creates the task record first, then starts processing.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message to process

        Returns:
            The task ID
        """
        # Create the task record first
        task = await repository.create_chat_task(conversation_id, user_message)

        # Start processing in background
        self.start_task(conversation_id, user_message, task.id)

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
