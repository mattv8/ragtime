from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock


def _install_background_task_test_stubs() -> None:
    ragtime_pkg: Any = sys.modules.setdefault("ragtime", types.ModuleType("ragtime"))
    ragtime_pkg.__path__ = getattr(ragtime_pkg, "__path__", [])

    core_pkg: Any = sys.modules.setdefault(
        "ragtime.core", types.ModuleType("ragtime.core")
    )
    core_pkg.__path__ = getattr(core_pkg, "__path__", [])

    class _StubSettingsCache:
        @staticmethod
        def get_instance() -> Any:
            class _Instance:
                async def get_settings(self) -> dict[str, Any]:
                    return {}

            return _Instance()

    core_app_settings: Any = types.ModuleType("ragtime.core.app_settings")
    core_app_settings.SettingsCache = _StubSettingsCache
    sys.modules["ragtime.core.app_settings"] = core_app_settings

    core_event_bus: Any = types.ModuleType("ragtime.core.event_bus")

    class _StubTaskEventBus:
        async def publish(self, *_args, **_kwargs) -> None:
            return None

        async def subscribe(self, *_args, **_kwargs):
            return asyncio.Queue()

        def unsubscribe(self, *_args, **_kwargs) -> None:
            return None

    core_event_bus.task_event_bus = _StubTaskEventBus()
    sys.modules["ragtime.core.event_bus"] = core_event_bus

    core_logging: Any = types.ModuleType("ragtime.core.logging")

    class _Logger:
        def debug(self, *_args, **_kwargs) -> None:
            return None

        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, *_args, **_kwargs) -> None:
            return None

        def error(self, *_args, **_kwargs) -> None:
            return None

        def exception(self, *_args, **_kwargs) -> None:
            return None

    core_logging.get_logger = lambda _name: _Logger()
    sys.modules["ragtime.core.logging"] = core_logging

    core_sql_utils: Any = types.ModuleType("ragtime.core.sql_utils")
    core_sql_utils.strip_table_metadata = lambda value: value
    sys.modules["ragtime.core.sql_utils"] = core_sql_utils

    core_usage: Any = types.ModuleType("ragtime.core.usage_accounting")

    async def _noop_async(*_args, **_kwargs) -> None:
        return None

    core_usage._estimate_output_tokens = lambda *_args, **_kwargs: 0
    core_usage.bind_usage_attempt_task = _noop_async
    core_usage.finalize_stale_attempts_for_tasks = _noop_async
    core_usage.finalize_usage_attempt = _noop_async
    sys.modules["ragtime.core.usage_accounting"] = core_usage

    indexer_pkg: Any = sys.modules.setdefault(
        "ragtime.indexer", types.ModuleType("ragtime.indexer")
    )
    indexer_pkg.__path__ = getattr(indexer_pkg, "__path__", [])

    chat_attachments: Any = types.ModuleType("ragtime.indexer.chat_attachments")
    chat_attachments.cleanup_expired_chat_attachments = _noop_async
    sys.modules["ragtime.indexer.chat_attachments"] = chat_attachments

    chat_events: Any = types.ModuleType("ragtime.indexer.chat_events")
    chat_events.append_reasoning_event = lambda *_args, **_kwargs: None
    chat_events.finalize_reasoning_block = lambda *_args, **_kwargs: None
    sys.modules["ragtime.indexer.chat_events"] = chat_events

    filesystem_service: Any = types.ModuleType("ragtime.indexer.filesystem_service")
    filesystem_service.filesystem_indexer = SimpleNamespace()
    sys.modules["ragtime.indexer.filesystem_service"] = filesystem_service

    class _ChatTaskStatus(str, Enum):
        pending = "pending"
        running = "running"
        completed = "completed"
        failed = "failed"
        cancelled = "cancelled"
        interrupted = "interrupted"

    indexer_models: Any = types.ModuleType("ragtime.indexer.models")
    indexer_models.ChatTaskStatus = _ChatTaskStatus
    indexer_models.FilesystemConnectionConfig = object
    indexer_models.SchemaIndexConfig = object
    sys.modules["ragtime.indexer.models"] = indexer_models

    repository_mod: Any = types.ModuleType("ragtime.indexer.repository")
    repository_mod.repository = SimpleNamespace()
    sys.modules["ragtime.indexer.repository"] = repository_mod

    schema_service: Any = types.ModuleType("ragtime.indexer.schema_service")
    schema_service.SCHEMA_INDEXER_CAPABLE_TYPES = set()
    schema_service.schema_indexer = SimpleNamespace()
    sys.modules["ragtime.indexer.schema_service"] = schema_service

    indexer_service: Any = types.ModuleType("ragtime.indexer.service")
    indexer_service.indexer = SimpleNamespace()
    sys.modules["ragtime.indexer.service"] = indexer_service

    indexer_utils: Any = types.ModuleType("ragtime.indexer.utils")
    indexer_utils.safe_tool_name = lambda name: str(name or "")
    sys.modules["ragtime.indexer.utils"] = indexer_utils

    rag_pkg: Any = types.ModuleType("ragtime.rag")
    rag_pkg.rag = SimpleNamespace(
        is_ready=True,
        preprocess_message_content_async=_noop_async,
        process_query_stream=lambda *_args, **_kwargs: None,
    )
    sys.modules["ragtime.rag"] = rag_pkg

    lc_pkg: Any = sys.modules.setdefault(
        "langchain_core", types.ModuleType("langchain_core")
    )
    lc_pkg.__path__ = getattr(lc_pkg, "__path__", [])
    lc_messages: Any = types.ModuleType("langchain_core.messages")

    class _BaseMessage:
        def __init__(self, content: Any = "", **kwargs: Any) -> None:
            self.content = content
            self.kwargs = kwargs

    class AIMessage(_BaseMessage):
        pass

    class HumanMessage(_BaseMessage):
        pass

    class ToolMessage(_BaseMessage):
        pass

    lc_messages.AIMessage = AIMessage
    lc_messages.HumanMessage = HumanMessage
    lc_messages.ToolMessage = ToolMessage
    sys.modules["langchain_core.messages"] = lc_messages


def _load_background_tasks_module():
    _install_background_task_test_stubs()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "ragtime"
        / "indexer"
        / "background_tasks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ragtime_indexer_background_tasks_under_test",
        module_path,
    )
    assert spec and spec.loader, f"failed to build spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


background_tasks = _load_background_tasks_module()


class _HangingAsyncStream:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._never = asyncio.Event()
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_called = True


class BackgroundTaskCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_task_closes_stream_iterator(self) -> None:
        service = background_tasks.BackgroundTaskService()
        stream = _HangingAsyncStream()

        fake_conversation = SimpleNamespace(
            messages=[
                SimpleNamespace(
                    role="user",
                    content="hello",
                    events=None,
                )
            ],
            user_id="user-1",
            model="lmstudio::openai/gpt-oss-20b",
            workspace_id=None,
        )

        fake_repository = SimpleNamespace(
            create_chat_task=mock.AsyncMock(),
            get_chat_task=mock.AsyncMock(return_value=SimpleNamespace(id="task-1")),
            update_chat_task_status=mock.AsyncMock(),
            get_conversation=mock.AsyncMock(return_value=fake_conversation),
            update_chat_task_streaming_state=mock.AsyncMock(),
            cancel_chat_task=mock.AsyncMock(),
            complete_chat_task=mock.AsyncMock(),
            add_message=mock.AsyncMock(),
            link_assistant_snapshot_tool_calls=mock.AsyncMock(),
            update_chat_task=mock.AsyncMock(),
        )

        fake_event_bus = SimpleNamespace(publish=mock.AsyncMock())
        fake_rag = SimpleNamespace(
            is_ready=True,
            process_query_stream=mock.Mock(return_value=stream),
        )
        fake_settings_cache = SimpleNamespace(
            get_settings=mock.AsyncMock(return_value={"max_tool_output_chars": 5000})
        )

        with (
            mock.patch.object(background_tasks, "repository", fake_repository),
            mock.patch.object(background_tasks, "task_event_bus", fake_event_bus),
            mock.patch.object(background_tasks, "rag", fake_rag),
            mock.patch.object(
                background_tasks.SettingsCache,
                "get_instance",
                return_value=fake_settings_cache,
            ),
        ):
            task_id = service.start_task(
                conversation_id="conv-1",
                user_message="hello",
                existing_task_id="task-1",
            )
            self.assertEqual(task_id, "task-1")

            running_task = service._running_tasks["task-1"]
            await asyncio.wait_for(stream.started.wait(), timeout=1.0)

            self.assertTrue(service.cancel_task("task-1"))

            with self.assertRaises(asyncio.CancelledError):
                await running_task

        self.assertTrue(stream.aclose_called)


if __name__ == "__main__":
    unittest.main()
