from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

SERVER_NAME = "runtime-playwright"

# Single source of truth for debug-step bounds on the MCP server side. The
# Node broker (playwright_broker.js) re-clamps authoritatively, and the RAG
# agent binds this tool dynamically from the MCP JSON schema below.
MAX_DEBUG_STEPS = 25
DEBUG_TIMEOUT_MS_BOUNDS = (1000, 60000)
DEBUG_WIDTH_BOUNDS = (320, 1920)
DEBUG_HEIGHT_BOUNDS = (240, 1600)

_broker_process: asyncio.subprocess.Process | None = None
_broker_lock = asyncio.Lock()
_request_counter = 0


def _preview_url(path: str, base_url: str, *, cache_bust_key: str | None = None) -> str:
    normalized = str(path or "").strip().lstrip("/")
    base = base_url.rstrip("/") + "/"
    url = f"{base}{normalized}" if normalized else base
    if cache_bust_key:
        url = f"{url}{'&' if '?' in url else '?'}_ragtime_mcp_ts={cache_bust_key}"
    return url


async def _invoke_playwright_broker(request: dict[str, Any], *, timeout_ms: int) -> dict[str, Any]:
    global _broker_process, _request_counter

    async with _broker_lock:
        if _broker_process is None or _broker_process.returncode is not None:
            broker_path = os.getenv("RAGTIME_PLAYWRIGHT_BROKER_JS_PATH", "")
            if not broker_path:
                broker_path = str(Path(__file__).parent / "templates" / "playwright_broker.js")
            if not Path(broker_path).exists():
                raise RuntimeError(f"Playwright broker script not found: {broker_path}")

            _broker_process = await asyncio.create_subprocess_exec(
                "node",
                broker_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "NODE_PATH": "/usr/local/lib/node_modules"},
            )
            assert _broker_process.stdout is not None
            ready_line = await asyncio.wait_for(_broker_process.stdout.readline(), timeout=20)
            if not ready_line:
                _broker_process.kill()
                _broker_process = None
                raise RuntimeError("Playwright MCP runtime failed to start")
            ready = json.loads(ready_line.decode("utf-8", errors="replace"))
            if ready.get("type") != "ready":
                _broker_process.kill()
                _broker_process = None
                raise RuntimeError("Invalid Playwright MCP runtime handshake")

        _request_counter += 1
        request_id = f"mcp-{_request_counter}-{uuid.uuid4().hex[:8]}"
        payload = {"id": request_id, **request}

        assert _broker_process.stdin is not None
        assert _broker_process.stdout is not None

        _broker_process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        await _broker_process.stdin.drain()

        try:
            response_line = await asyncio.wait_for(
                _broker_process.stdout.readline(),
                timeout=max(5.0, timeout_ms / 1000.0 + 5.0),
            )
        except Exception:
            _broker_process.kill()
            _broker_process = None
            raise

        if not response_line:
            _broker_process.kill()
            _broker_process = None
            raise RuntimeError("Playwright MCP runtime closed before returning a response")

        response = json.loads(response_line.decode("utf-8", errors="replace"))
        if response.get("id") != request_id:
            _broker_process.kill()
            _broker_process = None
            raise RuntimeError("Playwright MCP runtime returned a mismatched response")

        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Playwright MCP runtime failed"))

        result = response.get("result")
        return result if isinstance(result, dict) else {}


def _text(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))]


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _prepare_debug_steps(steps: Any, screenshot_dir: Path | None) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        raise RuntimeError("steps must be an array")
    if len(steps) > MAX_DEBUG_STEPS:
        raise RuntimeError(f"Playwright debug runs are limited to {MAX_DEBUG_STEPS} steps")
    prepared: list[dict[str, Any]] = []
    for entry in steps:
        if not isinstance(entry, dict):
            raise RuntimeError("Each Playwright debug step must be an object")
        step = dict(entry)
        action = str(step.get("action") or "").strip().lower()
        if action == "screenshot":
            if not screenshot_dir:
                raise RuntimeError("Screenshot directory is unavailable")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            step["output_path"] = str(screenshot_dir / f"{uuid.uuid4().hex}.png")
        prepared.append(step)
    return prepared


async def _tool_content_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    base_url = arguments.pop("__ragtime_preview_base_url", "http://127.0.0.1:0")
    timeout_ms = int(arguments.get("timeout_ms") or 15000)
    return await _invoke_playwright_broker(
        {
            "type": "content_probe",
            "url": _preview_url(str(arguments.get("path") or ""), base_url),
            "timeout_ms": timeout_ms,
            "wait_after_load_ms": int(arguments.get("wait_after_load_ms") or 2000),
            "inject_mock_context": bool(arguments.get("inject_mock_context") or False),
        },
        timeout_ms=timeout_ms,
    )


async def _tool_capture_screenshot(arguments: dict[str, Any]) -> dict[str, Any]:
    base_url = arguments.pop("__ragtime_preview_base_url", "http://127.0.0.1:0")
    screenshot_dir_str = arguments.pop("__ragtime_screenshot_dir", "")
    screenshot_dir = Path(screenshot_dir_str) if screenshot_dir_str else None
    if not screenshot_dir:
        raise RuntimeError("Screenshot directory is unavailable")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    output_path = screenshot_dir / f"{uuid.uuid4().hex}.png"
    timeout_ms = int(arguments.get("timeout_ms") or 25000)
    cache_key = str(int(time.time() * 1000))
    return await _invoke_playwright_broker(
        {
            "type": "screenshot",
            "url": _preview_url(str(arguments.get("path") or ""), base_url, cache_bust_key=cache_key),
            "output_path": str(output_path),
            "viewport_width": int(arguments.get("width") or arguments.get("viewport_width") or 1440),
            "viewport_height": int(arguments.get("height") or arguments.get("viewport_height") or 900),
            "capture_full_page": bool(arguments.get("full_page", True)),
            "timeout_ms": timeout_ms,
            "wait_for_selector": str(arguments.get("wait_for_selector") or "body"),
            "capture_element": bool(arguments.get("capture_element") or False),
            "clip_padding_px": int(arguments.get("clip_padding_px") or 16),
            "wait_after_load_ms": int(arguments.get("wait_after_load_ms") or 1800),
            "refresh_before_capture": bool(arguments.get("refresh_before_capture", True)),
            "max_pixels": _bounded_int(os.getenv("RAGTIME_MCP_MAX_SCREENSHOT_PIXELS", 1_440_000), 1_440_000, minimum=1000, maximum=10_000_000),
        },
        timeout_ms=timeout_ms,
    )


async def _tool_external_browse(arguments: dict[str, Any]) -> dict[str, Any]:
    arguments.pop("__ragtime_preview_base_url", None)
    timeout_ms = int(arguments.get("timeout_ms") or 20000)
    return await _invoke_playwright_broker(
        {
            "type": "external_browse",
            "url": str(arguments.get("url") or ""),
            "timeout_ms": timeout_ms,
            "wait_after_load_ms": int(arguments.get("wait_after_load_ms") or 1500),
            "extract_links": bool(arguments.get("extract_links", True)),
            "max_text_chars": int(arguments.get("max_text_chars") or 4000),
            "max_links": int(arguments.get("max_links") or 20),
            "user_agent": str(arguments.get("user_agent") or ""),
        },
        timeout_ms=timeout_ms,
    )


async def _tool_debug_steps(arguments: dict[str, Any]) -> dict[str, Any]:
    base_url = arguments.pop("__ragtime_preview_base_url", "http://127.0.0.1:0")
    screenshot_dir_str = arguments.pop("__ragtime_screenshot_dir", "")
    screenshot_dir = Path(screenshot_dir_str) if screenshot_dir_str else None
    timeout_ms = _bounded_int(arguments.get("timeout_ms"), 25000, minimum=DEBUG_TIMEOUT_MS_BOUNDS[0], maximum=DEBUG_TIMEOUT_MS_BOUNDS[1])
    return await _invoke_playwright_broker(
        {
            "type": "debug_steps",
            "url": _preview_url(str(arguments.get("path") or ""), base_url),
            "steps": _prepare_debug_steps(arguments.get("steps") or [], screenshot_dir),
            "timeout_ms": timeout_ms,
            "viewport_width": _bounded_int(arguments.get("width"), 1280, minimum=DEBUG_WIDTH_BOUNDS[0], maximum=DEBUG_WIDTH_BOUNDS[1]),
            "viewport_height": _bounded_int(arguments.get("height"), 900, minimum=DEBUG_HEIGHT_BOUNDS[0], maximum=DEBUG_HEIGHT_BOUNDS[1]),
            "allow_external_navigation": bool(arguments.get("allow_external_navigation") or False),
        },
        timeout_ms=timeout_ms,
    )


TOOL_HANDLERS = {
    "playwright_content_probe": _tool_content_probe,
    "playwright_capture_screenshot": _tool_capture_screenshot,
    "playwright_external_browse": _tool_external_browse,
    "playwright_debug_steps": _tool_debug_steps,
}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


TOOLS = [
    Tool(
        name="playwright_content_probe",
        description="Render the active workspace preview in Playwright and return title, visible-text metrics, and console errors.",
        inputSchema=_schema(
            {
                "path": {"type": "string", "description": "Workspace preview path, relative to /.", "default": ""},
                "timeout_ms": {"type": "integer", "description": "Navigation timeout in milliseconds.", "default": 15000},
                "wait_after_load_ms": {"type": "integer", "description": "Post-load settle wait in milliseconds.", "default": 2000},
                "inject_mock_context": {"type": "boolean", "description": "Inject mock Ragtime live-data context before rendering.", "default": False},
            }
        ),
    ),
    Tool(
        name="playwright_capture_screenshot",
        description="Capture a Playwright screenshot of the active workspace preview and store it as a workspace runtime artifact.",
        inputSchema=_schema(
            {
                "path": {"type": "string", "description": "Workspace preview path, relative to /.", "default": ""},
                "width": {"type": "integer", "description": "Viewport width.", "default": 1440},
                "height": {"type": "integer", "description": "Viewport height.", "default": 900},
                "full_page": {"type": "boolean", "description": "Capture full page when possible.", "default": True},
                "wait_for_selector": {"type": "string", "description": "Selector to wait for or capture.", "default": "body"},
                "capture_element": {"type": "boolean", "description": "Capture only the unique visible matched element.", "default": False},
                "clip_padding_px": {"type": "integer", "description": "Padding around element clip.", "default": 16},
                "timeout_ms": {"type": "integer", "description": "Navigation timeout in milliseconds.", "default": 25000},
                "wait_after_load_ms": {"type": "integer", "description": "Post-load settle wait in milliseconds.", "default": 1800},
                "refresh_before_capture": {"type": "boolean", "description": "Refresh before capture.", "default": True},
            }
        ),
    ),
    Tool(
        name="playwright_external_browse",
        description="Browse a public URL through the workspace runtime Playwright browser and return rendered text, links, and console errors.",
        inputSchema=_schema(
            {
                "url": {"type": "string", "description": "Absolute http/https URL to browse."},
                "timeout_ms": {"type": "integer", "description": "Navigation timeout in milliseconds.", "default": 20000},
                "wait_after_load_ms": {"type": "integer", "description": "Post-load settle wait in milliseconds.", "default": 1500},
                "extract_links": {"type": "boolean", "description": "Whether to extract top links from the page.", "default": True},
                "max_text_chars": {"type": "integer", "description": "Maximum visible text chars to return.", "default": 4000},
                "max_links": {"type": "integer", "description": "Maximum links to return.", "default": 20},
                "user_agent": {"type": "string", "description": "Optional User-Agent override.", "default": ""},
            },
            required=["url"],
        ),
    ),
    Tool(
        name="playwright_debug_steps",
        description=(
            "Run a bounded Playwright debug step sequence against the active workspace preview. "
            "This accepts structured actions only; it cannot run arbitrary JavaScript or Node.js code. "
            "Private/local external hosts are blocked; external top-level navigation requires allow_external_navigation=true."
        ),
        inputSchema=_schema(
            {
                "path": {"type": "string", "description": "Initial workspace preview path, relative to /.", "default": ""},
                "steps": {
                    "type": "array",
                    "description": (
                        f"Up to {MAX_DEBUG_STEPS} steps. Supported actions: goto, click, fill, type, press, select_option, "
                        "wait_for_selector, wait_for_timeout, query, content, screenshot."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "Step action name."},
                            "selector": {"type": "string", "description": "CSS selector for selector-based actions."},
                            "value": {"type": "string", "description": "Value for fill/select_option/type actions."},
                            "text": {"type": "string", "description": "Text for type actions."},
                            "key": {"type": "string", "description": "Keyboard key for press actions."},
                            "path": {"type": "string", "description": "Relative path for goto actions."},
                            "url": {"type": "string", "description": "Absolute URL for goto actions when external navigation is allowed."},
                            "attributes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "For action=query: element attribute names to read (max 12), e.g. ['href','aria-label'].",
                            },
                            "ms": {"type": "integer", "description": "Milliseconds for wait_for_timeout, capped at 5000."},
                            "state": {"type": "string", "description": "Selector wait state, defaults to visible."},
                            "full_page": {"type": "boolean", "description": "Screenshot full page when action=screenshot."},
                            "stop_on_error": {"type": "boolean", "description": "Defaults true. Set false to continue after a failed step."},
                        },
                        "required": ["action"],
                    },
                    "maxItems": MAX_DEBUG_STEPS,
                },
                "timeout_ms": {"type": "integer", "description": "Per-run/action timeout in milliseconds.", "default": 25000},
                "width": {"type": "integer", "description": "Viewport width.", "default": 1280},
                "height": {"type": "integer", "description": "Viewport height.", "default": 900},
                "allow_external_navigation": {
                    "type": "boolean",
                    "description": "Allow top-level goto to public external http(s) URLs. Private/local hosts remain blocked.",
                    "default": False,
                },
            },
            required=["steps"],
        ),
    ),
]


async def main() -> None:
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return _text({"ok": False, "error": f"Unknown tool: {name}"})
        try:
            result = await handler(arguments or {})
            result.setdefault("ok", True)
            return _text(result)
        except Exception as exc:
            return _text({"ok": False, "error": str(exc)})

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
