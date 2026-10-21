import asyncio
import contextvars
import unittest
from unittest import mock

from ragtime.indexer import background_tasks


class QueueCompletionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_handler_is_detached_and_uses_a_fresh_context(self) -> None:
        inherited_marker = contextvars.ContextVar("inherited_marker", default="clean")
        observed: list[str] = []

        async def handler(conversation_id: str) -> None:
            await asyncio.sleep(0)
            observed.append(f"{conversation_id}:{inherited_marker.get()}")

        previous = background_tasks._successful_chat_completion_handler
        try:
            background_tasks.register_successful_chat_completion_handler(handler)
            inherited_marker.set("parent-selected-tool")
            background_tasks._schedule_successful_completion_handler("conversation-1")
            await asyncio.gather(*background_tasks._detached_completion_handlers)
        finally:
            background_tasks._successful_chat_completion_handler = previous

        self.assertEqual(observed, ["conversation-1:clean"])

    async def test_completion_handler_exception_is_consumed_after_detach(self) -> None:
        async def handler(_conversation_id: str) -> None:
            raise RuntimeError("dispatch failure")

        previous = background_tasks._successful_chat_completion_handler
        try:
            background_tasks.register_successful_chat_completion_handler(handler)
            with mock.patch.object(background_tasks.logger, "exception") as logged:
                background_tasks._schedule_successful_completion_handler("conversation-1")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            logged.assert_called_once()
        finally:
            background_tasks._successful_chat_completion_handler = previous

    async def test_service_stop_cancels_tracked_completion_handler(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(_conversation_id: str) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        service = background_tasks.BackgroundTaskService()
        previous = background_tasks._successful_chat_completion_handler
        try:
            background_tasks.register_successful_chat_completion_handler(handler)
            background_tasks._schedule_successful_completion_handler("conversation-1", service)
            await started.wait()
            await service.stop()
        finally:
            background_tasks._successful_chat_completion_handler = previous

        self.assertTrue(cancelled.is_set())

    async def test_service_does_not_schedule_completion_handler_after_shutdown(self) -> None:
        observed: list[str] = []

        async def handler(conversation_id: str) -> None:
            observed.append(conversation_id)

        service = background_tasks.BackgroundTaskService()
        previous = background_tasks._successful_chat_completion_handler
        try:
            background_tasks.register_successful_chat_completion_handler(handler)
            await service.stop()
            background_tasks._schedule_successful_completion_handler("conversation-1", service)
            await asyncio.sleep(0)
        finally:
            background_tasks._successful_chat_completion_handler = previous

        self.assertEqual(observed, [])
