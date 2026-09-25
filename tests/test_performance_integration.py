"""Integration coverage for payload-free ASGI request timing."""

import asyncio
import logging
import subprocess
import sys
import unittest

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from ragtime.core.logging import RequestCorrelationFilter
from ragtime.core.performance import PerformanceSettings, SlowRequestMiddleware, get_request_id
from runtime.manager.api import create_app as create_manager_app
from runtime.worker.api import create_app as create_worker_app


class SlowRequestMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_logs_route_template_without_raw_query_and_clears_correlation(self) -> None:
        async def application(scope, receive, send) -> None:
            self.assertIsNotNone(get_request_id())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = SlowRequestMiddleware(
            application,
            service="test-service",
            settings=PerformanceSettings(0, 0, 0),
        )
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger("ragtime.performance")
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        async def send(_message: object) -> None:
            return None

        try:
            await middleware(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/accounts/secret-value",
                    "query_string": b"token=secret-value",
                    "route": type("Route", (), {"path_format": "/accounts/{account_id}"})(),
                },
                receive,
                send,
            )
        finally:
            logger.setLevel(previous_level)
            logger.removeHandler(handler)

        self.assertIsNone(get_request_id())
        self.assertEqual(len(records), 1)
        message = records[0].getMessage()
        self.assertIn("service=test-service", message)
        self.assertIn("route=/accounts/{account_id}", message)
        self.assertNotIn("secret-value", message)

    async def test_runtime_manager_and_worker_apps_emit_slow_request_logs(self) -> None:
        for create_app, service, path in (
            (create_manager_app, "runtime-manager", "/__timing_manager__"),
            (create_worker_app, "runtime-worker", "/__timing_worker__"),
        ):
            app = create_app()
            for middleware in app.user_middleware:
                if middleware.cls is SlowRequestMiddleware:
                    middleware.kwargs["settings"] = PerformanceSettings(0.001, 1.0, 0.0)

            @app.get(path)
            async def slow_endpoint() -> dict[str, bool]:
                await asyncio.sleep(0.01)
                return {"ok": True}

            records: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = records.append  # type: ignore[method-assign]
            logger = logging.getLogger("ragtime.performance")
            previous_level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get(path)
            finally:
                logger.setLevel(previous_level)
                logger.removeHandler(handler)

            self.assertEqual(response.status_code, 200)
            messages = [record.getMessage() for record in records]
            self.assertTrue(any(f"service={service}" in message for message in messages))
            self.assertTrue(any(f"method=GET route={path}" in message for message in messages))

    def test_main_app_emits_slow_request_log_in_isolated_process(self) -> None:
        script = """
import asyncio
import logging
import httpx
from ragtime.main import app
from ragtime.core.performance import PerformanceSettings, SlowRequestMiddleware

for middleware in app.user_middleware:
    if middleware.cls is SlowRequestMiddleware:
        middleware.kwargs["settings"] = PerformanceSettings(0.001, 1.0, 0.0)

records = []
handler = logging.Handler()
handler.emit = records.append
logger = logging.getLogger("ragtime.performance")
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

@app.get("/__timing_main__")
async def slow_endpoint():
    await asyncio.sleep(0.01)
    return {"ok": True}

async def run():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/__timing_main__")
    assert response.status_code == 200

asyncio.run(run())
print("|".join(record.getMessage() for record in records))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("service=ragtime", result.stdout)
        self.assertIn("method=GET route=/__timing_main__", result.stdout)

    async def test_base_http_middleware_preserves_content_protection_correlation(self) -> None:
        class ContentProtectionLogMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                logging.getLogger("ragtime.content_protection.service").info(
                    "content_protection_timing",
                    extra={"content_protection": {"direction": "inbound", "outcome": "permitted"}},
                )
                return await call_next(request)

        app = FastAPI()
        app.add_middleware(ContentProtectionLogMiddleware)
        app.add_middleware(SlowRequestMiddleware, service="correlation-test", settings=PerformanceSettings(0, 0, 0))

        @app.get("/__correlation__")
        async def endpoint() -> Response:
            return Response()

        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        handler.addFilter(RequestCorrelationFilter())
        logger = logging.getLogger("ragtime.content_protection.service")
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/__correlation__")
        finally:
            logger.setLevel(previous_level)
            logger.removeHandler(handler)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(records), 1)
        self.assertNotEqual(getattr(records[0], "request_id"), "-")


if __name__ == "__main__":
    unittest.main()
