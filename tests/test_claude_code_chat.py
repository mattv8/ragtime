"""Tests for the native Claude Code LangChain chat adapter."""

import asyncio
import json
import os
from typing import cast
from unittest import IsolatedAsyncioTestCase, mock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from ragtime.core.claude_code_chat import ClaudeCodeChatModel


class ClaudeCodeChatModelTests(IsolatedAsyncioTestCase):
    def _process(self, output: dict[str, object], returncode: int = 0) -> mock.AsyncMock:
        process = mock.AsyncMock()
        process.communicate.return_value = (
            json.dumps({"is_error": False, "subtype": "success", "terminal_reason": "completed", **output}).encode(),
            b"",
        )
        process.returncode = returncode
        return process

    async def test_ainvoke_returns_cli_text_response(self) -> None:
        process = self._process({"result": "Hello", "usage": {"input_tokens": 2, "output_tokens": 1, "cache_read_input_tokens": 3}})

        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            response = await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")])

        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.content, "Hello")
        message = cast(AIMessage, response)
        self.assertIsNotNone(message.usage_metadata)
        usage = message.usage_metadata or {}
        self.assertEqual(usage.get("input_token_details", {}).get("cache_read"), 3)

    async def test_serializes_ordered_conversation_and_isolates_cli(self) -> None:
        process = self._process({"result": "Done"})
        environment = {"ANTHROPIC_API_KEY": "secret", "CLAUDE_CODE_OAUTH_TOKEN": "oauth"}
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create,
        ):
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke(
                [
                    HumanMessage(content="first"),
                    AIMessage(content="calling", tool_calls=[{"name": "lookup", "args": {"x": 1}, "id": "call_1"}]),
                    ToolMessage(content="result", tool_call_id="call_1"),
                ]
            )

        args, kwargs = create.call_args
        self.assertIn("--safe-mode", args)
        self.assertNotIn("--bare", args)
        self.assertEqual(kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"], "oauth")
        self.assertNotIn("ANTHROPIC_API_KEY", kwargs["env"])
        payload = json.loads(process.communicate.call_args.args[0])
        self.assertEqual([message["role"] for message in payload["messages"]], ["user", "assistant", "tool"])
        self.assertEqual(payload["messages"][2]["tool_call_id"], "call_1")

    async def test_max_tokens_overrides_child_environment_only(self) -> None:
        process = self._process({"result": "Done"})
        with (
            mock.patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "99"}, clear=True),
            mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create,
        ):
            await ClaudeCodeChatModel("claude-opus-4-6", max_tokens=1234).ainvoke([HumanMessage(content="Hi")])
            self.assertEqual(os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "99")

        self.assertEqual(create.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "1234")

    async def test_reasoning_effort_overrides_child_environment_only(self) -> None:
        process = self._process({"result": "Done"})
        with (
            mock.patch.dict(os.environ, {"CLAUDE_CODE_EFFORT_LEVEL": "low"}, clear=True),
            mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create,
        ):
            await ClaudeCodeChatModel("claude-opus-4-6", reasoning_effort="high").ainvoke([HumanMessage(content="Hi")])
            self.assertEqual(os.environ["CLAUDE_CODE_EFFORT_LEVEL"], "low")

        self.assertEqual(create.call_args.kwargs["env"]["CLAUDE_CODE_EFFORT_LEVEL"], "high")

    async def test_bound_tools_convert_valid_response(self) -> None:
        process = self._process({"structured_output": {"content": "", "tool_calls": [{"name": "lookup", "arguments": {"query": "a"}}]}})
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a record.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
        model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool], tool_choice="lookup")

        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create:
            response = await model.ainvoke([HumanMessage(content="Hi")])

        self.assertIsInstance(response, AIMessage)
        message = cast(AIMessage, response)
        self.assertEqual(message.tool_calls[0]["name"], "lookup")
        self.assertEqual(message.tool_calls[0]["args"], {"query": "a"})
        self.assertIn("--json-schema", create.call_args.args)
        request = json.loads(process.communicate.call_args.args[0])
        self.assertEqual(request["tools"][0]["description"], "Find a record.")

    async def test_tool_choice_enforces_responses_and_schema(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
        ]
        valid_calls = {
            "none": [],
            "any": [{"name": "lookup", "arguments": {}}],
            "lookup": [{"name": "lookup", "arguments": {}}],
        }
        invalid_calls = {
            "none": [{"name": "lookup", "arguments": {}}],
            "any": [],
            "lookup": [{"name": "search", "arguments": {}}],
        }
        for choice, calls in valid_calls.items():
            process = self._process({"structured_output": {"content": "ok", "tool_calls": calls}})
            model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools(tools, tool_choice=choice)
            with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create:
                response = await model.ainvoke([HumanMessage(content="Hi")])
            self.assertEqual(response.content, "ok")
            schema = json.loads(create.call_args.args[create.call_args.args.index("--json-schema") + 1])
            calls_schema = schema["properties"]["tool_calls"]
            if choice == "none":
                self.assertEqual(calls_schema["maxItems"], 0)
            else:
                self.assertEqual(calls_schema["minItems"], 1)
            if choice == "lookup":
                self.assertEqual(calls_schema["items"]["oneOf"][0]["properties"]["name"]["const"], "lookup")
        for choice, calls in invalid_calls.items():
            process = self._process({"structured_output": {"content": "ok", "tool_calls": calls}})
            model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools(tools, tool_choice=choice)
            with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
                with self.assertRaises(RuntimeError):
                    await model.ainvoke([HumanMessage(content="Hi")])

    async def test_bound_tools_reject_free_text_result(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        process = self._process({"result": '{"content":"ok","tool_calls":[]}'})
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "invalid response"):
                await ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool]).ainvoke([HumanMessage(content="Hi")])

    async def test_usage_includes_cache_tokens_and_omits_invalid_counts(self) -> None:
        process = self._process(
            {"result": "ok", "usage": {"input_tokens": 3, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 500, "output_tokens": 10}}
        )
        invalid_process = self._process({"result": "ok", "usage": {"input_tokens": True, "output_tokens": -1}})
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", side_effect=[process, invalid_process]):
            response = cast(AIMessage, await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")]))
            invalid_response = cast(AIMessage, await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")]))
        usage = response.usage_metadata
        self.assertIsNotNone(usage)
        self.assertEqual((usage or {})["input_tokens"], 1503)
        self.assertEqual((usage or {})["total_tokens"], 1513)
        self.assertIsNone(invalid_response.usage_metadata)

    async def test_nonzero_json_failures_have_safe_exact_categories(self) -> None:
        cases = [
            ({"terminal_reason": "failed", "subtype": "error", "api_error_status": 401, "result": "secret"}, "authentication failed"),
            ({"terminal_reason": "failed", "subtype": "error", "api_error_status": 403, "result": "secret"}, "authentication failed"),
            ({"terminal_reason": "failed", "subtype": "error", "api_error_status": 429, "result": "secret"}, "rate or usage limit"),
            ({"terminal_reason": "failed", "subtype": "error", "result": "secret"}, "request failed"),
            ({"terminal_reason": "authentication_failed", "subtype": "error", "result": "secret"}, "request failed"),
            ({"terminal_reason": "rate_limit", "subtype": "rate_limit_error", "result": "secret"}, "request failed"),
            ({"terminal_reason": "blocking_limit", "subtype": "error", "result": "secret"}, "request failed"),
            ({"terminal_reason": "rapid_refill_breaker", "subtype": "error", "result": "secret"}, "request failed"),
        ]
        for output, message in cases:
            process = self._process(output, returncode=1)
            with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
                with self.assertRaisesRegex(RuntimeError, message) as error:
                    await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")])
            self.assertNotIn("secret", str(error.exception))

    async def test_actual_request_limit_codes_have_safe_reduce_request_category(self) -> None:
        cases = [
            {"terminal_reason": "prompt_too_long", "subtype": "error"},
            {"terminal_reason": "max_turns", "subtype": "error_max_turns"},
            {"terminal_reason": "budget_exhausted", "subtype": "error_max_budget_usd"},
            {"terminal_reason": "structured_output_retry_exhausted", "subtype": "error_max_structured_output_retries"},
            {"terminal_reason": "failed", "subtype": "error_max_budget_usd"},
            {"terminal_reason": "failed", "subtype": "error_max_structured_output_retries"},
        ]
        for output in cases:
            process = self._process({**output, "result": "secret"}, returncode=1)
            with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "request limit reached") as error:
                    await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")])
            self.assertNotIn("secret", str(error.exception))

    async def test_text_blocks_preserve_separators_and_images_fail(self) -> None:
        process = self._process({"result": "ok"})
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])])
        payload = json.loads(process.communicate.call_args.args[0])
        self.assertEqual(payload["messages"][0]["content"], "one\ntwo")
        with self.assertRaisesRegex(ValueError, "text-only"):
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content=[{"type": "image", "source": "x"}])])

    async def test_system_prompt_file_is_private_framed_and_removed(self) -> None:
        process = self._process({"result": "ok"})
        captured: dict[str, object] = {}

        async def communicate(payload: bytes) -> tuple[bytes, bytes]:
            create_args = create.call_args.args
            prompt_path = create_args[create_args.index("--system-prompt-file") + 1]
            captured["prompt_path"] = prompt_path
            captured["mode"] = os.stat(prompt_path).st_mode & 0o777
            captured["text"] = open(prompt_path, encoding="utf-8").read()
            captured["payload"] = json.loads(payload)
            return process.communicate.return_value

        process.communicate.side_effect = communicate
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create:
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([SystemMessage(content="system authority"), HumanMessage(content="Hi")])
        self.assertEqual(captured["mode"], 0o600)
        self.assertEqual(captured["text"], "system authority")
        payload = cast(dict[str, object], captured["payload"])
        self.assertEqual(
            payload["instruction"], "Answer the latest supplied user message. Use supplied tool schemas for tool-call requests only; never execute tools."
        )
        self.assertEqual(cast(list[dict[str, str]], payload["messages"]), [{"role": "user", "content": "Hi"}])
        self.assertFalse(os.path.exists(cast(str, captured["prompt_path"])))

    async def test_no_system_message_uses_protocol_system_prompt(self) -> None:
        process = self._process({"result": "ok"})
        captured: dict[str, str] = {}

        async def communicate(payload: bytes) -> tuple[bytes, bytes]:
            args = create.call_args.args
            prompt_path = args[args.index("--system-prompt-file") + 1]
            captured["text"] = open(prompt_path, encoding="utf-8").read()
            return process.communicate.return_value

        process.communicate.side_effect = communicate
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process) as create:
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")])
        self.assertTrue(captured["text"])

    async def test_oversized_tool_schema_fails_before_subprocess_spawn(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "description": "x" * 120_000}}}
        model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool])
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec") as create:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                await model.ainvoke([HumanMessage(content="Hi")])
        create.assert_not_called()

    async def test_tool_call_round_trip_preserves_assistant_and_tool_history(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        first = self._process({"structured_output": {"content": "", "tool_calls": [{"name": "lookup", "arguments": {"query": "a"}}]}})
        second = self._process({"structured_output": {"content": "Found it", "tool_calls": []}})
        model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool])
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", side_effect=[first, second]):
            tool_request = cast(AIMessage, await model.ainvoke([HumanMessage(content="Find a")]))
            response = await model.ainvoke(
                [HumanMessage(content="Find a"), tool_request, ToolMessage(content="answer", tool_call_id=tool_request.tool_calls[0]["id"])]
            )

        self.assertEqual(response.content, "Found it")
        history = json.loads(second.communicate.call_args.args[0])["messages"]
        self.assertEqual([message["role"] for message in history], ["user", "assistant", "tool"])
        self.assertEqual(history[2]["tool_call_id"], tool_request.tool_calls[0]["id"])

    async def test_public_astream_returns_one_complete_bound_tool_chunk(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        process = self._process(
            {
                "structured_output": {"content": "", "tool_calls": [{"name": "lookup", "arguments": {"query": "a"}}]},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
        model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool])
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            chunks = [chunk async for chunk in model.astream([HumanMessage(content="Find a")])]

        self.assertEqual(len(chunks), 1)
        chunk = cast(AIMessageChunk, chunks[0])
        self.assertEqual(chunk.tool_calls[0]["name"], "lookup")
        self.assertEqual((chunk.usage_metadata or {})["total_tokens"], 3)

    def test_public_stream_returns_one_complete_bound_tool_chunk(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        process = self._process({"structured_output": {"content": "", "tool_calls": [{"name": "lookup", "arguments": {"query": "a"}}]}})
        model = ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool])
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            chunks = list(model.stream([HumanMessage(content="Find a")]))

        self.assertEqual(len(chunks), 1)
        chunk = cast(AIMessageChunk, chunks[0])
        self.assertEqual(chunk.tool_calls[0]["name"], "lookup")

    def test_tool_choice_accepts_named_none_and_any(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        model = ClaudeCodeChatModel("claude-opus-4-6")
        self.assertEqual(model.bind_tools([tool], tool_choice="none").tool_choice, "none")
        self.assertEqual(model.bind_tools([tool], tool_choice="any").tool_choice, "any")
        self.assertEqual(model.bind_tools([tool], tool_choice="lookup").tool_choice, "lookup")
        with self.assertRaisesRegex(ValueError, "Unknown tool_choice"):
            model.bind_tools([tool], tool_choice="missing")

    async def test_rejects_invalid_responses_and_unsupported_content(self) -> None:
        malformed = self._process({"structured_output": "{"})
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=malformed):
            with self.assertRaisesRegex(RuntimeError, "invalid tool response"):
                await ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool]).ainvoke([HumanMessage(content="Hi")])
        with self.assertRaisesRegex(ValueError, "text-only"):
            await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content=[{"type": "image"}])])

    async def test_rejects_failed_cli_result_even_with_success_exit_code(self) -> None:
        process = self._process({"result": "ignored", "is_error": True, "subtype": "error", "terminal_reason": "failed"})
        with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                await ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")])

    async def test_rejects_unknown_tool_and_invalid_arguments(self) -> None:
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        for call, error in (({"name": "missing", "arguments": {}}, "unknown tool"), ({"name": "lookup", "arguments": "bad"}, "invalid tool response")):
            process = self._process({"structured_output": {"content": "", "tool_calls": [call]}})
            with mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process):
                with self.assertRaisesRegex(RuntimeError, error):
                    await ClaudeCodeChatModel("claude-opus-4-6").bind_tools([tool]).ainvoke([HumanMessage(content="Hi")])

    async def test_timeout_and_cancellation_reap_the_process_group(self) -> None:
        process = mock.AsyncMock()
        process.returncode = None
        process.communicate.side_effect = asyncio.TimeoutError
        with (
            mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process),
            mock.patch("ragtime.core.claude_code_chat._terminate_process", new=mock.AsyncMock()) as terminate,
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await ClaudeCodeChatModel("claude-opus-4-6", timeout=0.01).ainvoke([HumanMessage(content="Hi")])
        terminate.assert_awaited_once_with(process, process_group=os.name != "nt")

    async def test_cancellation_reaps_the_process_group(self) -> None:
        process = mock.AsyncMock()
        process.returncode = None
        started = asyncio.Event()

        async def wait_for_input(_payload: bytes) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.Event().wait()
            return b"", b""

        process.communicate.side_effect = wait_for_input
        with (
            mock.patch("ragtime.core.claude_code_chat.asyncio.create_subprocess_exec", return_value=process),
            mock.patch("ragtime.core.claude_code_chat._terminate_process", new=mock.AsyncMock()) as terminate,
        ):
            task = asyncio.create_task(ClaudeCodeChatModel("claude-opus-4-6").ainvoke([HumanMessage(content="Hi")]))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        terminate.assert_awaited_once_with(process, process_group=os.name != "nt")
