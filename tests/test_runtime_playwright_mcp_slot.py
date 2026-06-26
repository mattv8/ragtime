from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

worker_service: Any | None
runtime_import_error: ImportError | None
try:
    worker_service = importlib.import_module("runtime.worker.service")
except ImportError as exc:
    worker_service = None
    runtime_import_error = exc
else:
    runtime_import_error = None


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolResult:
    def __init__(self, payload: dict[str, Any], *, is_error: bool = False) -> None:
        self.content = [_FakeContent(json.dumps(payload))]
        self.isError = is_error


class _FakeListToolsResult:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools


class _FakeClient:
    """Stand-in for mcp.ClientSession's entered client."""

    def __init__(self, call_impl: Any, list_impl: Any) -> None:
        self._call_impl = call_impl
        self._list_impl = list_impl
        self.initialize_count = 0

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeToolResult:
        return await self._call_impl(name, arguments)

    async def list_tools(self) -> _FakeListToolsResult:
        return await self._list_impl()


def _install_fake_mcp(state: dict[str, Any], call_impl: Any, list_impl: Any | None = None):
    """Patch the MCP stdio client + session the owner task imports lazily."""

    @contextlib.asynccontextmanager
    async def fake_stdio_client(_params: Any):
        state["stdio_starts"] = state.get("stdio_starts", 0) + 1
        try:
            yield (object(), object())
        finally:
            state["stdio_exits"] = state.get("stdio_exits", 0) + 1

    async def default_list_impl() -> _FakeListToolsResult:
        return _FakeListToolsResult([])

    fake_client = _FakeClient(call_impl, list_impl or default_list_impl)
    state["client"] = fake_client

    class FakeClientSession:
        def __init__(self, _read: Any, _write: Any) -> None:
            self._client = fake_client

        async def __aenter__(self) -> _FakeClient:
            return self._client

        async def __aexit__(self, *_exc: Any) -> bool:
            return False

    return [
        mock.patch("mcp.client.stdio.stdio_client", fake_stdio_client),
        mock.patch("mcp.ClientSession", FakeClientSession),
    ]


@unittest.skipIf(worker_service is None, f"runtime worker unavailable: {runtime_import_error}")
class RuntimeMcpServerRegistryTests(unittest.IsolatedAsyncioTestCase):
    def _make_service(self, pool_size: int = 2):
        assert worker_service is not None
        with mock.patch.dict(os.environ, {"RUNTIME_MCP_SERVER_POOL_SIZE": str(pool_size)}):
            return worker_service.WorkerService()

    async def _invoke(self, service: Any, url: str = "https://example.com") -> Any:
        # playwright_external_browse is the one sessionless built-in tool.
        return await service._invoke_mcp_tool(
            None,
            "playwright",
            "playwright_external_browse",
            {"url": url},
            timeout_ms=2000,
        )

    async def test_warm_connection_is_reused_across_calls(self) -> None:
        state: dict[str, Any] = {}

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            return _FakeToolResult({"ok": True, "url": "https://example.com"})

        # Pin to a single slot so sequential calls deterministically reuse the
        # same warm connection instead of round-robining across the pool.
        service = self._make_service(pool_size=1)
        patches = _install_fake_mcp(state, call_impl)
        with patches[0], patches[1]:
            first = await self._invoke(service)
            second = await self._invoke(service)
            await service._terminate_mcp_brokers()

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.server_id, "runtime-playwright")
        self.assertEqual(first.server_name, "Runtime Playwright")
        # The MCP server subprocess is started once and reused (warm Chromium).
        self.assertEqual(state["stdio_starts"], 1)
        self.assertEqual(state["client"].initialize_count, 1)
        # Internal context keys must never leak into the surfaced request.
        self.assertNotIn("__ragtime_preview_base_url", first.request)

    async def test_terminate_brokers_tears_down_without_leak(self) -> None:
        state: dict[str, Any] = {}

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            return _FakeToolResult({"ok": True})

        service = self._make_service()
        patches = _install_fake_mcp(state, call_impl)
        with patches[0], patches[1]:
            await self._invoke(service)
            await service._terminate_mcp_brokers()

        # The stdio context manager must have exited (subprocess torn down),
        # and every slot owner task must be cleared.
        self.assertEqual(state["stdio_exits"], state["stdio_starts"])
        pool = service._mcp_pools.get("playwright")
        self.assertIsNotNone(pool)
        self.assertTrue(all(slot.owner_task is None for slot in pool.slots))

    async def test_concurrency_capped_by_pool_size(self) -> None:
        state: dict[str, Any] = {}
        release = asyncio.Event()
        in_flight = 0
        max_in_flight = 0

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await release.wait()
            finally:
                in_flight -= 1
            return _FakeToolResult({"ok": True})

        service = self._make_service(pool_size=2)
        patches = _install_fake_mcp(state, call_impl)
        with patches[0], patches[1]:
            tasks = [asyncio.create_task(self._invoke(service)) for _ in range(3)]
            for _ in range(50):
                await asyncio.sleep(0.01)
                if max_in_flight >= 2:
                    break
            self.assertEqual(max_in_flight, 2)
            release.set()
            results = await asyncio.gather(*tasks)
            await service._terminate_mcp_brokers()

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(max_in_flight, 2)

    async def test_transient_call_failure_is_retried_with_fresh_connection(self) -> None:
        state: dict[str, Any] = {}
        attempts = 0

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("stdio transport broke")
            return _FakeToolResult({"ok": True})

        service = self._make_service()
        patches = _install_fake_mcp(state, call_impl)
        with patches[0], patches[1]:
            result = await self._invoke(service)
            await service._terminate_mcp_brokers()

        self.assertTrue(result.ok)
        self.assertEqual(attempts, 2)
        self.assertEqual(state["stdio_starts"], 2)

    async def test_unknown_server_and_tool_are_rejected(self) -> None:
        service = self._make_service()
        dummy_session = SimpleNamespace(devserver_port=12345, workspace_id="w")

        # Unknown server -> 400 (resolved before anything else).
        with self.assertRaises(worker_service.HTTPException) as server_ctx:  # type: ignore[union-attr]
            await service._invoke_mcp_tool(dummy_session, "totally_unknown", "x", {}, timeout_ms=1000)
        self.assertEqual(server_ctx.exception.status_code, 400)

        # Known server, removed/unknown tool -> 400.
        for tool_name in ("playwright_console_messages", "totally_unknown"):
            with self.assertRaises(worker_service.HTTPException) as tool_ctx:  # type: ignore[union-attr]
                await service._invoke_mcp_tool(dummy_session, "playwright", tool_name, {"path": ""}, timeout_ms=1000)
            self.assertEqual(tool_ctx.exception.status_code, 400)

    async def test_list_server_tools_maps_schema_and_caches(self) -> None:
        assert worker_service is not None
        state: dict[str, Any] = {}
        list_calls = 0

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            return _FakeToolResult({"ok": True})

        async def list_impl() -> _FakeListToolsResult:
            nonlocal list_calls
            list_calls += 1
            return _FakeListToolsResult(
                [
                    SimpleNamespace(
                        name="pylance_hover",
                        description="Hover info",
                        inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}},
                    )
                ]
            )

        # A custom (non-excluded) server spec to exercise the generic path.
        spec = worker_service.McpServerSpec(
            name="pylance-test",
            server_id="pylance",
            display_name="Pylance",
            command="x",
            args=[],
            agent_excluded_tools=frozenset(),
        )
        service = self._make_service()
        patches = _install_fake_mcp(state, call_impl, list_impl)
        with patches[0], patches[1]:
            infos = await service._list_server_tools(spec)
            infos_again = await service._list_server_tools(spec)
            await service._terminate_mcp_brokers()

        self.assertEqual(len(infos), 1)
        info = infos[0]
        self.assertEqual(info.name, "pylance_hover")
        self.assertEqual(info.server_name, "pylance-test")
        self.assertFalse(info.agent_excluded)
        self.assertEqual(info.input_schema.get("properties", {}).get("symbol", {}).get("type"), "string")
        # Cached: the second call does not re-list over the wire.
        self.assertEqual(list_calls, 1)
        self.assertIs(infos_again, infos)

    async def test_playwright_builtin_tools_are_agent_excluded(self) -> None:
        assert worker_service is not None
        state: dict[str, Any] = {}

        async def call_impl(_name: str, _args: dict[str, Any]) -> _FakeToolResult:
            return _FakeToolResult({"ok": True})

        async def list_impl() -> _FakeListToolsResult:
            return _FakeListToolsResult([SimpleNamespace(name="playwright_debug_steps", description="", inputSchema={})])

        spec = worker_service._BUILTIN_MCP_SERVERS["playwright"]
        service = self._make_service()
        patches = _install_fake_mcp(state, call_impl, list_impl)
        with patches[0], patches[1]:
            infos = await service._list_server_tools(spec)
            await service._terminate_mcp_brokers()

        self.assertEqual(len(infos), 1)
        # playwright_debug_steps is intentionally allowed so the dynamic binder
        # can bind it natively without a hardcoded Python wrapper.
        self.assertFalse(infos[0].agent_excluded)


@unittest.skipIf(worker_service is None, "runtime worker unavailable")
class ComponentsHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.components = importlib.import_module("ragtime.rag.components")
        except Exception as exc:  # pragma: no cover - heavy optional import
            self.skipTest(f"ragtime.rag.components unavailable: {exc}")

    def test_compact_mcp_metadata_bounds_payload(self) -> None:
        compact = self.components._compact_mcp_metadata(
            {
                "ok": True,
                "server_id": "runtime-playwright",
                "server_name": "Runtime Playwright",
                "tool_name": "playwright_capture_screenshot",
                "request": {"path": "dashboard/main.ts"},
                "response": {
                    "preview_image_url": "/indexes/userspace/runtime/workspaces/w/screenshots/a.png",
                    "huge": "x" * 5000,
                    "items": list(range(100)),
                },
            }
        )
        assert compact is not None
        self.assertEqual(compact["tool_name"], "playwright_capture_screenshot")
        self.assertEqual(
            compact["response"]["preview_image_url"],
            "/indexes/userspace/runtime/workspaces/w/screenshots/a.png",
        )
        self.assertLess(len(compact["response"]["huge"]), 5000)
        self.assertTrue(compact["response"]["huge"].endswith("...(truncated)"))
        self.assertLessEqual(len(compact["response"]["items"]), 26)

    def test_compact_mcp_metadata_returns_none_for_non_dict(self) -> None:
        self.assertIsNone(self.components._compact_mcp_metadata(None))
        self.assertIsNone(self.components._compact_mcp_metadata("nope"))

    def test_json_schema_to_pydantic_required_and_optional(self) -> None:
        model = self.components._json_schema_to_pydantic(
            "pylance_hover",
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol to inspect"},
                    "line": {"type": "integer"},
                    "deep": {"type": "boolean", "default": False},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbol"],
            },
        )
        # Required field present validates fine; optionals get defaults.
        instance = model(symbol="foo")
        dumped = instance.model_dump()
        self.assertEqual(dumped["symbol"], "foo")
        self.assertIsNone(dumped["line"])
        self.assertEqual(dumped["deep"], False)
        # Missing required field raises.
        with self.assertRaises(Exception):
            model()

    def test_json_schema_to_pydantic_permissive_fallback(self) -> None:
        model = self.components._json_schema_to_pydantic("freeform", {"type": "object"})
        instance = model(anything="goes", count=3)
        dumped = instance.model_dump()
        self.assertEqual(dumped.get("anything"), "goes")
        self.assertEqual(dumped.get("count"), 3)


if __name__ == "__main__":
    unittest.main()
