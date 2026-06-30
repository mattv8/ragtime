"""Regression tests for subagent stream isolation.

A parent agent spawns subagents from *inside* its own LangChain
``astream_events`` context. If a child background task inherits the parent's
context, LangChain's ``var_child_runnable_config`` contextvar carries the
parent's event-collector callback into the child, so the child's streamed
tokens and its ``submit_subagent_handoff`` tool call leak into the parent's
``response_content`` / events.

``background_task_service.start_task`` runs every task in a context where that
contextvar is cleared. These tests validate that mechanism without needing an
LLM, and demonstrate the leak that occurs without the fix.
"""

import asyncio
import contextvars
import unittest
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig, var_child_runnable_config


def _sanitized_task_context() -> contextvars.Context:
    """Mirror background_task_service.start_task's context handling."""
    task_context = contextvars.copy_context()
    task_context.run(var_child_runnable_config.set, None)
    return task_context


class SubagentStreamIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_task_does_not_inherit_parent_runnable_config(self) -> None:
        parent_config = cast(RunnableConfig, {"callbacks": ["parent-astream-events-handler"]})
        token = var_child_runnable_config.set(parent_config)
        try:
            observed: dict[str, Any] = {}

            async def child() -> None:
                observed["config"] = var_child_runnable_config.get()

            task = asyncio.get_running_loop().create_task(child(), context=_sanitized_task_context())
            await task

            self.assertIsNone(
                observed["config"],
                "A background/subagent task must not inherit the parent's astream_events callback context.",
            )
        finally:
            var_child_runnable_config.reset(token)

    async def test_plain_create_task_leaks_parent_context(self) -> None:
        # Guard test: documents the exact leak the sanitized context prevents.
        parent_config = cast(RunnableConfig, {"callbacks": ["parent-astream-events-handler"]})
        token = var_child_runnable_config.set(parent_config)
        try:
            observed: dict[str, Any] = {}

            async def child() -> None:
                observed["config"] = var_child_runnable_config.get()

            task = asyncio.get_running_loop().create_task(child())
            await task

            self.assertEqual(
                observed["config"],
                parent_config,
                "Without the sanitized context, a child task inherits the parent's stream context (the leak).",
            )
        finally:
            var_child_runnable_config.reset(token)


if __name__ == "__main__":
    unittest.main()
