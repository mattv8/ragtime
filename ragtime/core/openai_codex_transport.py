"""HTTP request building for the OpenAI Codex subscription Responses API."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ragtime.core.model_providers import OPENAI_CODEX_RESPONSES_PAYLOAD_KEYS
from ragtime.core.openai_codex_auth import OPENAI_CODEX_RESPONSES_ENDPOINT


def codex_response_headers(account_id: str, user_agent: str = "ragtime") -> dict[str, str]:
    """Build non-secret headers accepted by the Codex subscription endpoint."""
    headers = {"User-Agent": user_agent}
    if account_id.strip():
        headers["ChatGPT-Account-Id"] = account_id.strip()
    return headers


def normalize_codex_responses_payload(content: bytes) -> bytes:
    """Remove Responses fields unsupported by the Codex subscription backend."""
    if not content:
        return content
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content
    if not isinstance(payload, dict):
        return content

    normalized = {key: value for key, value in payload.items() if key in OPENAI_CODEX_RESPONSES_PAYLOAD_KEYS}
    normalized["store"] = False
    _strip_input_item_ids(normalized)
    return content if normalized == payload else json.dumps(normalized, separators=(",", ":")).encode("utf-8")


def build_codex_responses_request(request: httpx.Request, content: bytes) -> httpx.Request:
    """Rewrite an OpenAI-compatible Responses request for Codex."""
    target_url = request.url
    path = request.url.path.lower()
    if "/v1/responses" in path or path.endswith("/responses") or "/chat/completions" in path:
        target_url = httpx.URL(OPENAI_CODEX_RESPONSES_ENDPOINT)
        content = normalize_codex_responses_payload(content)

    headers = httpx.Headers(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    return httpx.Request(request.method, target_url, headers=headers, content=content, extensions=request.extensions)


def _strip_input_item_ids(payload: dict[str, Any]) -> None:
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return
    payload["input"] = [{key: value for key, value in item.items() if key != "id"} if isinstance(item, dict) else item for item in input_items]


class OpenAICodexTransport(httpx.BaseTransport):
    """Rewrite OpenAI SDK requests to the Codex subscription backend."""

    def __init__(self) -> None:
        self._delegate = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._delegate.handle_request(build_codex_responses_request(request, request.read()))

    def close(self) -> None:
        self._delegate.close()


class OpenAICodexAsyncTransport(httpx.AsyncBaseTransport):
    """Async variant of :class:`OpenAICodexTransport`."""

    def __init__(self) -> None:
        self._delegate = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._delegate.handle_async_request(build_codex_responses_request(request, await request.aread()))

    async def aclose(self) -> None:
        await self._delegate.aclose()
