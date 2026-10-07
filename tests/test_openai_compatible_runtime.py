from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ragtime.core.openai_compatible import CompatibleProviderError
from ragtime.core.openai_compatible_client import CompatibleChatOpenAI, compatible_chat_options
from ragtime.rag.components import ChatContextWindowExceededError, RAGComponents, RequestLLMResolution


class OpenAICompatibleRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _rag(self) -> RAGComponents:
        rag = RAGComponents()
        rag._app_settings = {
            "llm_max_tokens": 4096,
            "openai_compatible_base_url": "https://compatible.test/api/root",
            "openai_compatible_api_key": "endpoint-key",
            "openai_api_key": "must-not-be-used",
        }
        return rag

    async def test_generic_build_streams_fragmented_tool_call_and_round_trips_result(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/root/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer endpoint-key")
            parsed_payload = json.loads(request.content)
            self.assertIsInstance(parsed_payload, dict)
            payload: dict[str, Any] = parsed_payload
            requests.append(payload)
            if len(requests) == 1:
                streaming = payload.get("stream")
                assert isinstance(streaming, bool)
                self.assertTrue(streaming)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        'data: {"id":"first","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"value\\":"}}]}}]}\n\n'
                        'data: {"id":"first","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                )
            streaming = payload.get("stream")
            assert isinstance(streaming, bool)
            self.assertFalse(streaming)
            messages = payload.get("messages")
            assert isinstance(messages, list)
            self.assertTrue(any(isinstance(message, dict) and message.get("role") == "tool" for message in messages))
            return httpx.Response(
                200,
                json={
                    "id": "second",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "tool result"}, "finish_reason": "stop"}],
                },
            )

        metadata = SimpleNamespace(id="same-id", context_limit=2048, max_output_tokens=1024)
        transport = httpx.MockTransport(handler)
        with (
            mock.patch("ragtime.rag.components.get_compatible_model", new=mock.AsyncMock(return_value=metadata)),
            mock.patch(
                "ragtime.core.openai_compatible_client.compatible_http_clients",
                return_value=(httpx.Client(transport=transport, follow_redirects=False), httpx.AsyncClient(transport=transport, follow_redirects=False)),
            ),
        ):
            client = await self._rag()._build_llm("openai_compatible", "same-id", 1024)

        self.assertIsInstance(client, ChatOpenAI)
        assert isinstance(client, ChatOpenAI)
        self.assertFalse(getattr(client, "use_responses_api", False))

        @tool
        def lookup(value: str) -> str:
            """Look up a value."""
            return value.upper()

        bound = client.bind_tools([lookup])
        chunks = [chunk async for chunk in bound.astream([HumanMessage(content="look up x")])]
        first: AIMessageChunk | None = cast(AIMessageChunk, chunks[0]) if chunks else None
        for chunk in chunks[1:]:
            if first is not None:
                first = cast(AIMessageChunk, first + cast(BaseMessageChunk, chunk))

        assert first is not None
        result = lookup.invoke(first.tool_calls[0]["args"])
        nonstreaming_bound = client.model_copy(update={"streaming": False}).bind_tools([lookup])
        final = await nonstreaming_bound.ainvoke(
            [HumanMessage(content="look up x"), cast(AIMessageChunk, first), ToolMessage(content=result, tool_call_id="call_1")]
        )
        self.assertEqual(final.content, "tool result")
        self.assertEqual(len(requests), 2)

    async def test_keyless_client_does_not_inherit_openai_environment_or_rewrite_max_tokens(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                },
            )

        transport = httpx.MockTransport(handler)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret", "OPENAI_ORG_ID": "env-org", "OPENAI_PROJECT_ID": "env-project"}):
            options = compatible_chat_options("")
            options["http_client"]._transport = transport
            options["http_async_client"]._transport = transport
            client_kwargs: dict[str, Any] = {"max_tokens": 17, "use_responses_api": False}
            client_kwargs.update(options)
            client = CompatibleChatOpenAI(model="keyless", base_url="https://compatible.test/v1", **client_kwargs)
            await client.ainvoke([HumanMessage(content="hello")])
            options["http_client"].close()
            await options["http_async_client"].aclose()

        headers = seen["headers"]
        payload = seen["payload"]
        assert isinstance(headers, dict) and isinstance(payload, dict)
        self.assertNotIn("authorization", headers)
        self.assertNotIn("openai-organization", headers)
        self.assertNotIn("openai-project", headers)
        self.assertEqual(payload["max_tokens"], 17)
        self.assertNotIn("max_completion_tokens", payload)

    async def test_generic_runtime_caps_override_and_never_falls_back(self) -> None:
        rag = self._rag()
        metadata = SimpleNamespace(id="gpt-4o", context_limit=2048, max_output_tokens=1024)
        with mock.patch("ragtime.rag.components.get_compatible_model", new=mock.AsyncMock(return_value=metadata)):
            self.assertEqual(await rag._resolve_llm_max_tokens("openai_compatible", "gpt-4o"), 1024)
            self.assertEqual(await rag._resolve_chat_context_limit("openai_compatible", "gpt-4o"), 2048)
            resolution = RequestLLMResolution(llm=object(), provider="openai_compatible", model="gpt-4o", max_tokens=9000)
            capped = (
                await rag._prepare_chat_context_window(
                    llm_resolution=resolution, system_prompt="", tool_scope_prompt="", turn_system_content="", chat_history=[], user_content="x", tools=[]
                )
            )[0].max_tokens
            assert isinstance(capped, int)
            self.assertLessEqual(capped, 1024)
            self.assertLess(capped, 9000)
        self.assertEqual(
            rag._ordered_llm_candidate_providers(requested_provider="openai_compatible", model_id="gpt-4o", provider_override="openai_compatible"),
            ["openai_compatible"],
        )

    async def test_generic_metadata_failure_does_not_abort_initialization_or_other_provider_fallback(self) -> None:
        rag = self._rag()
        assert rag._app_settings is not None
        rag._app_settings["llm_provider"] = "openai_compatible"
        with mock.patch("ragtime.rag.components.get_compatible_model", new=mock.AsyncMock(side_effect=CompatibleProviderError("authentication", "safe"))):
            await rag._init_llm()
        self.assertIsNone(rag.llm)

        assert rag._app_settings is not None
        rag._app_settings.update({"llm_provider": "openai", "allowed_chat_models": ["openai::same", "openai_compatible::same"]})
        self.assertEqual(rag._ordered_llm_candidate_providers(requested_provider="openai", model_id="same", provider_override="openai"), ["openai"])

    async def test_generic_unknown_context_and_metadata_error_never_use_legacy_default(self) -> None:
        rag = self._rag()
        unknown = SimpleNamespace(id="same-id", context_limit=None, max_output_tokens=None)
        with mock.patch("ragtime.rag.components.get_compatible_model", new=mock.AsyncMock(return_value=unknown)):
            with self.assertRaisesRegex(ChatContextWindowExceededError, "configure its documented context limit"):
                await rag._resolve_chat_context_limit("openai_compatible", "same-id")
        with mock.patch(
            "ragtime.rag.components.get_compatible_model",
            new=mock.AsyncMock(side_effect=CompatibleProviderError("unavailable", "upstream body")),
        ):
            self.assertEqual(await rag._build_context_headroom_prompt(chat_history=[], user_content="", provider="openai_compatible"), "")

    def test_generic_errors_are_safe_and_do_not_change_openrouter_credit_state(self) -> None:
        error = httpx.HTTPStatusError(
            "raw upstream body with secret",
            request=httpx.Request("POST", "https://compatible.test/chat/completions"),
            response=httpx.Response(429, json={}),
        )
        rag = self._rag()
        resolution = RequestLLMResolution(llm=object(), provider="openai_compatible", model="same-id")
        with mock.patch("ragtime.rag.components.note_openrouter_payment_required") as note:
            message = rag._chat_runtime_error_message(error, resolution)
        self.assertEqual(message, "The configured OpenAI-compatible provider is rate limited. Please retry shortly.")
        note.assert_not_called()
