from __future__ import annotations

import asyncio
import logging
import unittest
from unittest import mock

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from ragtime.core.performance import (
    PerformanceSettings,
    SlowRequestMiddleware,
    get_request_id,
    timed_operation,
    track_operation,
)


class PerformanceSettingsTests(unittest.TestCase):
    def test_default_thresholds_are_hardcoded(self) -> None:
        settings = PerformanceSettings()
        self.assertEqual(settings.slow_request_threshold_seconds, 1.0)
        self.assertEqual(settings.slow_operation_threshold_seconds, 1.0)
        self.assertEqual(settings.event_loop_lag_threshold_seconds, 0.5)

    def test_settings_can_be_overridden_for_testing(self) -> None:
        settings = PerformanceSettings(
            slow_request_threshold_seconds=0.5,
            slow_operation_threshold_seconds=0.2,
            event_loop_lag_threshold_seconds=0.0,
        )
        self.assertEqual(settings.slow_request_threshold_seconds, 0.5)
        self.assertEqual(settings.slow_operation_threshold_seconds, 0.2)
        self.assertEqual(settings.event_loop_lag_threshold_seconds, 0.0)


class PerformanceMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def _scope(self) -> dict[str, object]:
        return {"type": "http", "method": "PATCH", "path": "/private?secret=no", "headers": []}

    async def test_tracks_ttfb_final_body_operations_and_hides_raw_request_data(self) -> None:
        observed_ids: list[str | None] = []
        sent: list[Message] = []

        async def collect(message: Message) -> None:
            sent.append(message)

        async def app(scope: object, receive: object, send: object) -> None:
            observed_ids.append(get_request_id())
            with track_operation("database.read"):
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"one", "more_body": True})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"two", "more_body": False})  # type: ignore[misc]

        middleware = SlowRequestMiddleware(
            app,
            service="test-service",
            settings=PerformanceSettings(0.1, 0.1, 0.0),
        )
        times = iter((10.0, 10.01, 10.02, 10.2, 10.7))
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await middleware(self._scope(), _receive, collect)

        self.assertEqual([message["body"] for message in sent if message["type"] == "http.response.body"], [b"one", b"two"])
        self.assertEqual(len(observed_ids), 1)
        self.assertIsNotNone(observed_ids[0])
        self.assertIsNone(get_request_id())
        rendered = "\n".join(logs.output)
        self.assertIn("ttfb=0.200s", rendered)
        self.assertIn("elapsed=0.700s", rendered)
        self.assertIn("database.read", rendered)
        self.assertNotIn("/private", rendered)
        self.assertNotIn("secret=no", rendered)

    async def test_all_http_methods_are_logged(self) -> None:
        """Test that logs include HTTP method for all verbs."""
        app = FastAPI()

        @app.api_route("/items/{item_id}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
        async def item(item_id: str) -> dict[str, str]:
            return {"item_id": item_id}

        for method in ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
            with self.subTest(method=method), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
                await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "method": method,
                        "path": "/items/secret",
                        "query_string": b"",
                        "headers": [],
                    },
                    _receive,
                    _discard_send,
                )
            rendered = "\n".join(logs.output)
            self.assertIn(f"method={method}", rendered)
            self.assertIn("route=/items/{item_id}", rendered)

    async def test_operation_threshold_independent_from_request_threshold(self) -> None:
        """Test that operation threshold uses request-level settings, independent from request threshold."""

        async def app(scope: object, receive: object, send: object) -> None:
            with track_operation("slow-operation"):
                pass
            try:
                with track_operation("failed-operation"):
                    raise ValueError("do not log this value")
            except ValueError:
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        times = iter((1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.5, 1.5))
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times), self.assertLogs("ragtime.performance", level="WARNING") as logs:
            await SlowRequestMiddleware(app, service="ops-test", settings=PerformanceSettings(99.0, 0.1, 0.0))(self._scope(), _receive, _discard_send)
        rendered = "\n".join(logs.output)
        self.assertIn("slow-operation", rendered)
        self.assertIn("outcome=success", rendered)
        self.assertIn("failed-operation", rendered)
        self.assertIn("outcome=failed", rendered)
        self.assertNotIn("do not log this value", rendered)

    async def test_zero_operation_threshold_disables_operation_warnings(self) -> None:
        """Test that zero operation threshold disables slow operation warnings."""

        async def app(scope: object, receive: object, send: object) -> None:
            with track_operation("operation-1"):
                pass
            with track_operation("operation-2"):
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        times_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times_list), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(99.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)
        rendered = "\n".join(logs.output)
        # Operations are still aggregated in the summary, but no slow warnings
        self.assertIn("operation-1", rendered)
        self.assertIn("operation-2", rendered)
        self.assertNotIn("slow operation", rendered)

    async def test_zero_request_threshold_disables_request_warnings(self) -> None:
        """Test that zero request threshold disables slow request warnings."""

        async def app(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        times = iter((1.0, 1.5, 5.0))
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)
        rendered = "\n".join(logs.output)
        # No slow request warnings even though elapsed time is large
        self.assertNotIn("slow response start", rendered)
        self.assertIn("request complete", rendered)

    async def test_concurrent_requests_are_isolated(self) -> None:
        """Test that concurrent requests maintain isolated operation aggregation."""
        request_ids: list[str | None] = []
        request_events: dict[str, asyncio.Event] = {
            "req1_enter": asyncio.Event(),
            "req2_enter": asyncio.Event(),
            "req1_proceed": asyncio.Event(),
            "req2_proceed": asyncio.Event(),
        }

        async def app1(scope: object, receive: object, send: object) -> None:
            request_ids.append(get_request_id())
            request_events["req1_enter"].set()
            await request_events["req1_proceed"].wait()
            with track_operation("operation-A"):
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        async def app2(scope: object, receive: object, send: object) -> None:
            request_ids.append(get_request_id())
            request_events["req2_enter"].set()
            await request_events["req2_proceed"].wait()
            with track_operation("operation-B"):
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        middleware1 = SlowRequestMiddleware(app1, settings=PerformanceSettings(0.0, 0.0, 0.0))
        middleware2 = SlowRequestMiddleware(app2, settings=PerformanceSettings(0.0, 0.0, 0.0))

        async def run_both() -> None:
            task1 = asyncio.create_task(middleware1(self._scope(), _receive, _discard_send))
            task2 = asyncio.create_task(middleware2(self._scope(), _receive, _discard_send))

            await request_events["req1_enter"].wait()
            await request_events["req2_enter"].wait()

            request_events["req1_proceed"].set()
            request_events["req2_proceed"].set()

            await asyncio.gather(task1, task2)

        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await run_both()

        # Both requests should have unique IDs
        self.assertEqual(len(request_ids), 2)
        self.assertNotEqual(request_ids[0], request_ids[1])
        self.assertIsNotNone(request_ids[0])
        self.assertIsNotNone(request_ids[1])
        summaries = {getattr(record, "performance")["request_id"]: getattr(record, "performance") for record in logs.records}
        self.assertEqual(set(summaries[request_ids[0]]["operations"]), {"operation-A"})
        self.assertEqual(set(summaries[request_ids[1]]["operations"]), {"operation-B"})

    async def test_detached_child_task_cannot_mutate_closed_request_state(self) -> None:
        """Test that operations in detached child tasks after request closes don't mutate aggregates."""
        child_task_created = asyncio.Event()

        async def app(scope: object, receive: object, send: object) -> None:
            async def child_operation() -> None:
                child_task_created.set()
                await asyncio.sleep(0.01)
                with track_operation("detached-operation"):
                    pass

            asyncio.create_task(child_operation())
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)

        # Give child task time to run
        await asyncio.sleep(0.05)

        rendered = "\n".join(logs.output)
        # Detached operation should NOT be in the request summary (because state is closed)
        lines = rendered.split("\n")
        request_lines = [l for l in lines if "request complete" in l]
        self.assertTrue(len(request_lines) > 0)
        self.assertNotIn("detached-operation", request_lines[0])

    async def test_closed_context_retains_service_request_id_for_operation_warnings(self) -> None:
        """Test that slow operations in closed request context retain service and request_id."""
        request_ids: list[str | None] = []

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            request_ids.append(get_request_id())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            with track_operation("detached-slow-op"):
                pass

        middleware = SlowRequestMiddleware(app, service="test-service", settings=PerformanceSettings(0.0, 0.05, 0.0))
        with (
            mock.patch("ragtime.core.performance.monotonic", side_effect=[0.0, 0.01, 0.02, 0.02, 0.12]),
            self.assertLogs("ragtime.performance", level="DEBUG") as logs,
        ):
            await middleware(self._scope(), _receive, _discard_send)

        completion, operation = [getattr(record, "performance") for record in logs.records]
        self.assertEqual(completion["operations"], {})
        self.assertEqual(operation["operation"], "detached-slow-op")
        self.assertEqual(operation["service"], "test-service")
        self.assertIsNotNone(request_ids[0])
        self.assertEqual(operation["request_id"], request_ids[0])
        self.assertIsNone(get_request_id())

    async def test_sync_and_async_decorator_preserves_exception_and_cancellation(self) -> None:
        """Test that @timed_operation preserves exception and cancellation for sync and async functions."""

        @timed_operation("sync-function")
        def sync_func() -> None:
            raise RuntimeError("sync error")

        @timed_operation("async-function")
        async def async_func() -> None:
            raise RuntimeError("async error")

        @timed_operation("async-cancelled")
        async def async_cancel_func() -> None:
            await asyncio.sleep(10)

        # Test sync function exception
        with self.assertRaisesRegex(RuntimeError, "sync error"):
            sync_func()

        # Test async function exception
        with self.assertRaisesRegex(RuntimeError, "async error"):
            await async_func()

        # Test async function cancellation
        task = asyncio.create_task(async_cancel_func())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_operation_name_capped_at_32(self) -> None:
        """Test that operation names beyond 32 are not stored."""

        async def app(scope: object, receive: object, send: object) -> None:
            for index in range(40):
                with track_operation(f"operation-{index}"):
                    pass
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        middleware = SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))
        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await middleware(self._scope(), _receive, _discard_send)
        rendered = "\n".join(logs.output)
        self.assertEqual(rendered.count("operation-"), 32)

    async def test_sse_completion_is_debug_not_warning(self) -> None:
        """Test that SSE completion is logged at DEBUG level, not WARNING."""

        async def streaming(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"data: test\n\n", "more_body": False})  # type: ignore[misc]

        times = iter((1.0, 1.1, 5.0))  # Large elapsed time
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(streaming, settings=PerformanceSettings(1.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)

        rendered = "\n".join(logs.output)
        # SSE completion should be DEBUG even with large elapsed time
        self.assertIn("sse request complete", rendered)
        # Should be at DEBUG level (lower severity than WARNING)
        self.assertTrue(any("DEBUG" in line for line in logs.output))

    async def test_streaming_response_client_disconnect_no_final_body_logs_normal(self) -> None:
        """Test real StreamingResponse ASGI2.3 disconnect after first chunk - regression test."""
        first_chunk = asyncio.Event()

        async def mock_receive() -> Message:
            await first_chunk.wait()
            return {"type": "http.disconnect"}

        sent_messages: list[Message] = []

        async def mock_send(message: Message) -> None:
            sent_messages.append(message)
            if message["type"] == "http.response.body":
                first_chunk.set()

        async def chunks():
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        response = StreamingResponse(chunks(), media_type="text/event-stream")
        scope = self._scope() | {"asgi": {"version": "3.0", "spec_version": "2.3"}}
        with (
            mock.patch("ragtime.core.performance.monotonic", side_effect=[0.0, 0.1, 5.0]),
            self.assertLogs("ragtime.performance", level="DEBUG") as logs,
        ):
            await SlowRequestMiddleware(response)(scope, mock_receive, mock_send)

        self.assertTrue(first_chunk.is_set())
        self.assertFalse(any(message["type"] == "http.response.body" and not message.get("more_body") for message in sent_messages))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelno, logging.DEBUG)
        self.assertIn("outcome=disconnected", logs.output[0])

    async def test_non_sse_incomplete_without_disconnect_warns(self) -> None:
        """Test that non-SSE incomplete response without client disconnect still warns."""

        async def app(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            # App returns without sending final body and without disconnect

        with self.assertLogs("ragtime.performance", level="WARNING") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)

        rendered = "\n".join(logs.output)
        self.assertIn("request ended without final body", rendered)
        self.assertNotIn("(client disconnect)", rendered)
        self.assertIn("outcome=incomplete", rendered)

    async def test_non_sse_disconnect_respects_slow_threshold(self) -> None:
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await receive()

        for threshold, expected_level in ((1.0, logging.WARNING), (0.0, logging.DEBUG)):
            with (
                self.subTest(threshold=threshold),
                mock.patch("ragtime.core.performance.monotonic", side_effect=[0.0, 0.1, 2.0]),
                self.assertLogs("ragtime.performance", level="DEBUG") as logs,
            ):
                await SlowRequestMiddleware(app, settings=PerformanceSettings(threshold, 0.0, 0.0))(self._scope(), _receive, _discard_send)
            self.assertEqual(len(logs.records), 1)
            self.assertEqual(logs.records[0].levelno, expected_level)
            self.assertIn("outcome=disconnected", logs.output[0])

    async def test_send_backpressure_elapsed_measured_after_await(self) -> None:
        """Test that final body elapsed time is measured after send (includes backpressure)."""
        clock = [1.0]

        async def slow_send(message: Message) -> None:
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await asyncio.sleep(0)
                clock[0] = 1.2

        async def app(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        with mock.patch("ragtime.core.performance.monotonic", side_effect=lambda: clock[0]), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, slow_send)

        rendered = "\n".join(logs.output)
        # Elapsed should reflect the time after send completes (1.2 - 1.0 = 0.2s)
        self.assertIn("elapsed=0.200s", rendered)

    async def test_background_error_has_distinct_label_and_outcome(self) -> None:
        """Test that errors after body completion have distinct labels and outcomes."""

        async def app(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 202, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]
            raise RuntimeError("background error")

        async def blocked_send(message: Message) -> None:
            if message["type"] == "http.response.body":
                return None

        times = iter((1.0, 1.1, 1.2, 2.0, 2.0))
        with mock.patch("ragtime.core.performance.monotonic", side_effect=times), self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            with self.assertRaisesRegex(RuntimeError, "background error"):
                await SlowRequestMiddleware(app, settings=PerformanceSettings(0.5, 0.0, 0.0))(self._scope(), _receive, blocked_send)
        rendered = "\n".join(logs.output)
        self.assertIn("background/app failure", rendered)
        self.assertIn("status=202", rendered)

    async def test_background_cancellation_has_distinct_label(self) -> None:
        """Test that cancellation after body completion has distinct label."""
        sent: list[Message] = []
        entered = asyncio.Event()

        async def collect(message: Message) -> None:
            sent.append(message)

        async def streaming(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"first", "more_body": False})  # type: ignore[misc]
            entered.set()
            await asyncio.Event().wait()

        middleware = SlowRequestMiddleware(streaming, settings=PerformanceSettings(0.01, 0.0, 0.0))
        task = asyncio.create_task(middleware(self._scope(), _receive, collect))
        await entered.wait()
        task.cancel()
        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await task
        rendered = "\n".join(logs.output)
        self.assertIn("background/app cancellation", rendered)
        self.assertIsNone(get_request_id())

    async def test_failure_cancellation_sse_and_context_are_handled_without_buffering(self) -> None:
        """Test failure and cancellation handling without buffering."""
        sent: list[Message] = []
        entered = asyncio.Event()

        async def collect(message: Message) -> None:
            sent.append(message)

        async def streaming(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"first", "more_body": True})  # type: ignore[misc]
            entered.set()
            await asyncio.Event().wait()

        middleware = SlowRequestMiddleware(streaming, settings=PerformanceSettings(0.01, 0.0, 0.0))
        task = asyncio.create_task(middleware(self._scope(), _receive, collect))
        await entered.wait()
        task.cancel()
        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(sent[1]["body"], b"first")
        self.assertTrue(any("cancelled" in line for line in logs.output))
        self.assertIsNone(get_request_id())

        # Test app failure
        async def failing(scope: object, receive: object, send: object) -> None:
            raise RuntimeError("boom")

        with self.assertLogs("ragtime.performance", level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await SlowRequestMiddleware(failing)(self._scope(), _receive, collect)
        self.assertTrue(any("failed" in line for line in logs.output))

    async def test_operation_decorator_and_child_task_aggregate_then_close(self) -> None:
        """Test that operation decorator works and child tasks aggregate before close."""

        @timed_operation("decorated")
        async def decorated() -> None:
            with track_operation("nested"):
                await asyncio.sleep(0)

        child_done = asyncio.Event()

        async def app(scope: object, receive: object, send: object) -> None:
            async def child() -> None:
                await decorated()
                child_done.set()

            await asyncio.create_task(child())
            await send({"type": "http.response.start", "status": 204, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)
        self.assertTrue(child_done.is_set())
        self.assertIn("decorated", "\n".join(logs.output))
        self.assertIn("nested", "\n".join(logs.output))

    async def test_lifespan_starts_and_stops_single_monitor(self) -> None:
        """Test that lifespan starts and stops a single lag monitor."""
        received = iter(({"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}))
        sent: list[Message] = []
        monitor_tasks: list[asyncio.Task[None]] = []

        async def collect(message: Message) -> None:
            sent.append(message)
            if message["type"] == "lifespan.startup.complete":
                task = middleware._lag_monitor_task
                self.assertIsNotNone(task)
                if task is not None:
                    monitor_tasks.append(task)
                middleware._start_lag_monitor()
                self.assertIs(middleware._lag_monitor_task, task)

        async def lifespan_app(scope: object, receive: object, send: object) -> None:
            self.assertEqual((await receive())["type"], "lifespan.startup")  # type: ignore[misc]
            await send({"type": "lifespan.startup.complete"})  # type: ignore[misc]
            self.assertEqual((await receive())["type"], "lifespan.shutdown")  # type: ignore[misc]
            await send({"type": "lifespan.shutdown.complete"})  # type: ignore[misc]

        async def receive() -> Message:
            return next(received)

        middleware = SlowRequestMiddleware(lifespan_app, settings=PerformanceSettings(0.0, 0.0, 0.5))
        await middleware({"type": "lifespan"}, receive, collect)
        self.assertEqual(
            [message["type"] for message in sent],
            ["lifespan.startup.complete", "lifespan.shutdown.complete"],
        )
        self.assertIsNone(middleware._lag_monitor_task)
        self.assertEqual(len(monitor_tasks), 1)
        self.assertTrue(monitor_tasks[0].done())

    async def test_zero_lag_threshold_skips_lag_monitor_start(self) -> None:
        """Test that zero lag threshold skips starting the lag monitor."""
        received = iter(({"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}))
        sent: list[Message] = []

        async def collect(message: Message) -> None:
            sent.append(message)

        async def lifespan_app(scope: object, receive: object, send: object) -> None:
            self.assertEqual((await receive())["type"], "lifespan.startup")  # type: ignore[misc]
            await send({"type": "lifespan.startup.complete"})  # type: ignore[misc]
            self.assertEqual((await receive())["type"], "lifespan.shutdown")  # type: ignore[misc]
            await send({"type": "lifespan.shutdown.complete"})  # type: ignore[misc]

        async def receive() -> Message:
            return next(received)

        middleware = SlowRequestMiddleware(lifespan_app, settings=PerformanceSettings(0.0, 0.0, 0.0))
        await middleware({"type": "lifespan"}, receive, collect)
        # After startup complete, monitor should not be created when threshold is 0
        self.assertIsNone(middleware._lag_monitor_task)

    async def test_lag_monitor_creates_task_when_threshold_positive(self) -> None:
        """Test that lag monitor creates a task when threshold is positive."""
        middleware = SlowRequestMiddleware(
            lambda scope, receive, send: None,  # type: ignore[misc]
            settings=PerformanceSettings(0.0, 0.0, 0.1),
        )
        middleware._start_lag_monitor()
        self.assertIsNotNone(middleware._lag_monitor_task)
        await middleware._stop_lag_monitor()

    async def test_debug_disabled_avoids_json_allocation(self) -> None:
        """Test that JSON payload is not built when DEBUG is disabled."""

        async def app(scope: object, receive: object, send: object) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[misc]
            await send({"type": "http.response.body", "body": b"", "more_body": False})  # type: ignore[misc]

        with mock.patch("ragtime.core.performance.json.dumps") as dumps, self.assertNoLogs("ragtime.performance", level="WARNING"):
            await SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))(self._scope(), _receive, _discard_send)
        dumps.assert_not_called()

    async def test_warning_disabled_avoids_slow_operation_json_and_pid(self) -> None:
        """Test that suppressed slow-operation warnings skip payload construction."""
        with (
            mock.patch("ragtime.core.performance._LOGGER.isEnabledFor", return_value=False),
            mock.patch("ragtime.core.performance.json.dumps") as dumps,
            mock.patch("ragtime.core.performance.os.getpid") as getpid,
            mock.patch("ragtime.core.performance.monotonic", side_effect=[0.0, 1.5]),
        ):
            with track_operation("standalone-slow"):
                pass

        dumps.assert_not_called()
        getpid.assert_not_called()

    async def test_standalone_operation_outside_request_uses_default_settings(self) -> None:
        """Test that standalone slow operations outside request context warn with default settings."""
        with mock.patch("ragtime.core.performance.monotonic", side_effect=[0.0, 1.5]), self.assertLogs("ragtime.performance", level="WARNING") as logs:
            with track_operation("standalone-slow"):
                pass

        rendered = "\n".join(logs.output)
        self.assertIn("standalone-slow", rendered)
        self.assertIn("outcome=success", rendered)
        self.assertIn("request_id=none", rendered)


async def _receive() -> Message:
    return {"type": "http.disconnect"}


async def _discard_send(message: Message) -> None:
    del message
