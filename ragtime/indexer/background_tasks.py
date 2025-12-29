"""
Background task service for async chat message processing.

Handles running chat tasks in the background so they persist across
client disconnections, page switches, and server restarts.
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Set

from langchain_core.messages import HumanMessage, AIMessage

from ragtime.core.logging import get_logger
from ragtime.indexer.repository import repository
from ragtime.indexer.models import ChatTaskStatus

logger = get_logger(__name__)


class BackgroundTaskService:
    """Service for managing background chat tasks."""

    def __init__(self):
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False

    async def start(self):
        """Start the background task service."""
        logger.info("Background task service starting")
        self._shutdown = False
        # Resume any pending/running tasks from previous session
        await self._resume_stale_tasks()
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop the background task service."""
        logger.info("Background task service stopping")
        self._shutdown = True

        # Cancel all running tasks
        for task_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._running_tasks.clear()

    async def _resume_stale_tasks(self):
        """Resume tasks that were running when server restarted."""
        from ragtime.rag import rag

        # Wait for RAG to be ready
        for _ in range(30):  # Wait up to 30 seconds
            if rag.is_ready:
                break
            await asyncio.sleep(1)

        if not rag.is_ready:
            logger.warning("RAG not ready, skipping task resume")
            return

        try:
            db = await repository._get_db()
            stale_tasks = await db.chattask.find_many(
                where={
                    "status": {"in": ["pending", "running"]}
                }
            )

            for prisma_task in stale_tasks:
                task = repository._prisma_task_to_model(prisma_task)
                logger.info(f"Resuming stale task {task.id} for conversation {task.conversation_id}")
                self.start_task(task.conversation_id, task.user_message, task.id)

        except Exception as e:
            logger.error(f"Failed to resume stale tasks: {e}")

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

    def start_task(
        self,
        conversation_id: str,
        user_message: str,
        existing_task_id: Optional[str] = None
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
                    task = await repository.create_chat_task(conversation_id, user_message)
                    task_id = task.id
                else:
                    task = await repository.get_chat_task(task_id)
                    if not task:
                        logger.error(f"Task {task_id} not found")
                        return

                # Update status to running
                await repository.update_chat_task_status(task_id, ChatTaskStatus.running)

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
                                    # Format tool call for context
                                    input_str = ""
                                    if isinstance(tool_input, dict):
                                        # Extract query/code from common field names
                                        for field in ["query", "sql", "code", "command", "python_code"]:
                                            if field in tool_input:
                                                input_str = str(tool_input[field])
                                                break
                                        if not input_str:
                                            input_str = str(tool_input)
                                    else:
                                        input_str = str(tool_input)
                                    content_parts.append(
                                        f"\n[Tool: {tool_name}]\nQuery: {input_str}\nResult: {tool_output}\n"
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

                async for event in rag.process_query_stream(user_message, chat_history):
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
                            await repository.update_chat_task_streaming_state(
                                task_id, full_response, events, tool_calls, hit_max_iterations
                            )
                            last_update = datetime.utcnow()

                        elif event_type == "tool_end":
                            run_id = event.get("run_id", "")
                            # Find the matching tool_start by run_id
                            tool_idx = running_tool_indices.pop(run_id, None)
                            if tool_idx is not None:
                                # Update the existing tool event with output
                                events[tool_idx]["output"] = event.get("output")
                                tool_calls.append({
                                    "tool": events[tool_idx]["tool"],
                                    "input": events[tool_idx].get("input"),
                                    "output": events[tool_idx].get("output"),
                                })

                                # Force immediate update so the UI shows the completed tool
                                await repository.update_chat_task_streaming_state(
                                    task_id, full_response, events, tool_calls, hit_max_iterations
                                )
                                last_update = datetime.utcnow()

                        elif event_type == "max_iterations_reached":
                            hit_max_iterations = True
                            logger.info(f"Task {task_id} hit max iterations limit")
                            # Force immediate update
                            await repository.update_chat_task_streaming_state(
                                task_id, full_response, events, tool_calls, hit_max_iterations
                            )
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

                    # Update streaming state frequently (every 200ms) for real-time display
                    now = datetime.utcnow()
                    if (now - last_update).total_seconds() > 0.2:
                        await repository.update_chat_task_streaming_state(
                            task_id, full_response, events, tool_calls, hit_max_iterations
                        )
                        last_update = now

                # Task completed successfully - save final state
                await repository.complete_chat_task(
                    task_id, full_response, events, tool_calls, hit_max_iterations
                )

                # Add the assistant response to the conversation
                await repository.add_message(
                    conversation_id,
                    "assistant",
                    full_response,
                    tool_calls=tool_calls if tool_calls else None,
                    events=events if events else None
                )

                # Auto-generate title if needed
                conv = await repository.get_conversation(conversation_id)
                if conv and conv.title == "New Chat" and len(conv.messages) >= 2:
                    first_msg = conv.messages[0].content[:50]
                    new_title = first_msg + ("..." if len(conv.messages[0].content) > 50 else "")
                    await repository.update_conversation_title(conversation_id, new_title)

                logger.info(f"Background task {task_id} completed successfully")

            except asyncio.CancelledError:
                # Save partial response before cancelling
                if full_response.strip():
                    await repository.add_message(
                        conversation_id,
                        "assistant",
                        full_response,
                        tool_calls=tool_calls if tool_calls else None,
                        events=events if events else None
                    )
                    logger.info(f"Saved partial response for cancelled task {task_id}")

                await repository.cancel_chat_task(task_id)
                raise
            except Exception as e:
                logger.exception(f"Background task {task_id} failed")
                await repository.update_chat_task_status(
                    task_id, ChatTaskStatus.failed, str(e)
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
        user_message: str
    ) -> str:
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
