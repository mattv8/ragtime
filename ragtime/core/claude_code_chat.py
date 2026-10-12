"""LangChain chat adapter backed by the locally authenticated Claude Code CLI."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from ragtime.core.claude_code import _terminate_process, build_claude_code_subscription_env

_SUPPORTED_TOOL_CHOICES = {"auto", "none", "any", "required"}
_SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high", "max"}
_MAX_SCHEMA_BYTES = 120_000
_REQUEST_FRAMING = "Answer the latest supplied user message. Use supplied tool schemas for tool-call requests only; never execute tools."
_REQUEST_LIMIT_TERMINAL_REASONS = {"prompt_too_long", "max_turns", "budget_exhausted", "structured_output_retry_exhausted"}
_REQUEST_LIMIT_SUBTYPES = {"error_max_budget_usd", "error_max_structured_output_retries", "error_max_turns"}


class ClaudeCodeChatModel(BaseChatModel):
    """Run Claude Code without exporting its credentials into this process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    command: str = "claude"
    timeout: float = 300
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    bound_tools: tuple[dict[str, Any], ...] = Field(default_factory=tuple, exclude=True)
    tool_choice: str | None = Field(default=None, exclude=True)

    def __init__(
        self,
        model: str,
        command: str = "claude",
        timeout: float = 300,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in _SUPPORTED_REASONING_EFFORTS:
            raise ValueError("reasoning_effort must be one of: low, medium, high, max.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided.")
        model_data = {"model": model, "command": command, "timeout": timeout, "max_tokens": max_tokens, "reasoning_effort": reasoning_effort, **kwargs}
        super().__init__(**model_data)

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "command": self.command}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | dict[str, Any] | None = None, **kwargs: Any) -> ClaudeCodeChatModel:
        if kwargs:
            raise ValueError(f"Unsupported bind_tools parameters: {', '.join(sorted(kwargs))}.")
        normalized_tools = tuple(convert_to_openai_tool(tool) for tool in tools)
        tool_names = {tool["function"]["name"] for tool in normalized_tools}
        normalized_choice = self._normalize_tool_choice(tool_choice, tool_names)
        return self.model_copy(update={"bound_tools": normalized_tools, "tool_choice": normalized_choice})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if kwargs:
            raise ValueError(f"Unsupported invocation parameters: {', '.join(sorted(kwargs))}.")
        try:
            return asyncio.run(self._agenerate(messages, stop=stop))
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" in str(exc):
                raise RuntimeError("Use ainvoke from an asynchronous context.") from exc
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if kwargs:
            raise ValueError(f"Unsupported invocation parameters: {', '.join(sorted(kwargs))}.")
        payload, system_prompt = self._serialize_messages(messages)
        response = await self._invoke_cli(payload, system_prompt)
        message = self._response_message(response, stop)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield ChatGenerationChunk(message=self._message_chunk(result.generations[0].message))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        yield ChatGenerationChunk(message=self._message_chunk(result.generations[0].message))

    @staticmethod
    def _message_chunk(message: BaseMessage) -> AIMessageChunk:
        if not isinstance(message, AIMessage):
            raise RuntimeError("Claude Code returned an invalid response.")
        return AIMessageChunk(
            content=message.content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            name=message.name,
            id=message.id,
            tool_calls=message.tool_calls,
            invalid_tool_calls=message.invalid_tool_calls,
            usage_metadata=message.usage_metadata,
        )

    @staticmethod
    def _normalize_tool_choice(choice: str | dict[str, Any] | None, tool_names: set[str]) -> str | None:
        if choice is None:
            return None
        if isinstance(choice, str):
            if choice in _SUPPORTED_TOOL_CHOICES:
                return "any" if choice == "required" else choice
            if choice in tool_names:
                return choice
            raise ValueError(f"Unknown tool_choice: {choice}.")
        function = choice.get("function") if isinstance(choice, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name in tool_names:
            return name
        raise ValueError("tool_choice must be auto, none, any, or a bound tool name.")

    def _serialize_messages(self, messages: Sequence[BaseMessage]) -> tuple[bytes, str]:
        serialized: list[dict[str, Any]] = []
        system_messages: list[str] = []
        for message in messages:
            content = self._text_content(message.content)
            if isinstance(message, SystemMessage):
                system_messages.append(content)
                continue
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ToolMessage):
                serialized.append({"role": "tool", "content": content, "tool_call_id": message.tool_call_id})
                continue
            else:
                raise ValueError(f"Unsupported message type: {type(message).__name__}.")
            entry: dict[str, Any] = {"role": role, "content": content}
            if isinstance(message, AIMessage) and message.tool_calls:
                entry["tool_calls"] = message.tool_calls
            serialized.append(entry)
        request: dict[str, Any] = {"instruction": _REQUEST_FRAMING, "messages": serialized}
        if self.bound_tools:
            request["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {"type": "object"}),
                }
                for tool in self.bound_tools
            ]
            request["tool_choice"] = self.tool_choice or "auto"
        return json.dumps(request, separators=(",", ":")).encode(), "\n\n".join(system_messages) or _REQUEST_FRAMING

    @staticmethod
    def _text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise ValueError("Claude Code adapter supports text-only message content.")
        text: list[str] = []
        for block in content:
            if isinstance(block, str):
                text.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text.append(block["text"])
            else:
                raise ValueError("Claude Code adapter supports text-only message content.")
        return "\n".join(text)

    async def _invoke_cli(self, payload: bytes, system_prompt: str) -> dict[str, Any]:
        process: asyncio.subprocess.Process | None = None
        process_group = os.name != "nt"
        subprocess_kwargs: dict[str, Any] = {}
        if process_group:
            subprocess_kwargs["start_new_session"] = True
        with tempfile.TemporaryDirectory(prefix="ragtime-claude-") as working_directory:
            mcp_config = os.path.join(working_directory, "mcp.json")
            with open(mcp_config, "w", encoding="utf-8") as config_file:
                json.dump({"mcpServers": {}}, config_file)
            system_prompt_file = os.path.join(working_directory, "system-prompt.txt")
            with open(system_prompt_file, "w", encoding="utf-8") as prompt_file:
                prompt_file.write(system_prompt)
            os.chmod(system_prompt_file, 0o600)
            try:
                process = await asyncio.create_subprocess_exec(
                    self.command,
                    "--print",
                    "--output-format",
                    "json",
                    "--model",
                    self.model,
                    "--system-prompt-file",
                    system_prompt_file,
                    "--safe-mode",
                    "--setting-sources",
                    "",
                    "--tools",
                    "",
                    "--strict-mcp-config",
                    "--mcp-config",
                    mcp_config,
                    "--no-chrome",
                    "--disable-slash-commands",
                    "--no-session-persistence",
                    *self._tool_args(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_directory,
                    env=self._child_environment(),
                    **subprocess_kwargs,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(payload), timeout=self.timeout)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if process is not None:
                    await _terminate_process(process, process_group=process_group)
                raise RuntimeError("Claude Code request timed out.") from exc
            except asyncio.CancelledError:
                if process is not None:
                    await _terminate_process(process, process_group=process_group)
                raise
            except OSError as exc:
                raise RuntimeError("Claude Code CLI is unavailable.") from exc
        try:
            parsed = json.loads(stdout.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if process.returncode not in (0, None):
                raise RuntimeError("Claude Code request failed.") from exc
            raise RuntimeError("Claude Code returned an invalid response.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Claude Code returned an invalid response.")
        if parsed.get("is_error") is not False or parsed.get("subtype") != "success" or parsed.get("terminal_reason") != "completed":
            raise RuntimeError(self._safe_failure_message(parsed))
        if process.returncode not in (0, None):
            raise RuntimeError("Claude Code request failed.")
        return parsed

    @staticmethod
    def _safe_failure_message(response: dict[str, Any]) -> str:
        terminal_reason = response.get("terminal_reason")
        subtype = response.get("subtype")
        api_error_status = response.get("api_error_status")
        if api_error_status in {401, 403}:
            return "Claude Code authentication failed. Reconnect Claude Code and try again."
        if api_error_status == 429:
            return "Claude Code rate or usage limit reached. Try again later."
        if terminal_reason in _REQUEST_LIMIT_TERMINAL_REASONS or subtype in _REQUEST_LIMIT_SUBTYPES:
            return "Claude Code request limit reached. Reduce the request and try again."
        return "Claude Code request failed."

    def _tool_args(self) -> tuple[str, ...]:
        if not self.bound_tools:
            return ()
        schema_json = json.dumps(self._tool_response_schema(), separators=(",", ":"))
        if len(schema_json.encode()) > _MAX_SCHEMA_BYTES:
            raise ValueError(f"Claude Code tool schema exceeds the {_MAX_SCHEMA_BYTES}-byte CLI argument limit.")
        return ("--json-schema", schema_json)

    def _child_environment(self) -> dict[str, str]:
        environment = build_claude_code_subscription_env()
        if self.max_tokens is not None:
            environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_tokens)
        if self.reasoning_effort is not None:
            environment["CLAUDE_CODE_EFFORT_LEVEL"] = self.reasoning_effort
        return environment

    def _tool_response_schema(self) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        for tool in self.bound_tools:
            function = tool["function"]
            calls.append(
                {
                    "type": "object",
                    "properties": {"name": {"const": function["name"]}, "arguments": function.get("parameters", {"type": "object"})},
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
            )
        choice = self.tool_choice or "auto"
        call_schema: dict[str, Any] = {"type": "array", "items": {"oneOf": calls}}
        if choice == "none":
            call_schema["maxItems"] = 0
        elif choice in {"any"} or choice in {tool["function"]["name"] for tool in self.bound_tools}:
            call_schema["minItems"] = 1
            if choice not in {"any"}:
                call_schema["items"] = {"oneOf": [call for call in calls if call["properties"]["name"]["const"] == choice]}
        return {
            "type": "object",
            "properties": {"content": {"type": "string"}, "tool_calls": call_schema},
            "required": ["content", "tool_calls"],
            "additionalProperties": False,
        }

    def _response_message(self, response: dict[str, Any], stop: list[str] | None) -> AIMessage:
        raw_result = response.get("structured_output") if self.bound_tools else response.get("result", response.get("structured_output"))
        if not isinstance(raw_result, (str, dict)):
            raise RuntimeError("Claude Code returned an invalid response.")
        if self.bound_tools:
            try:
                envelope = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            except json.JSONDecodeError as exc:
                raise RuntimeError("Claude Code returned an invalid tool response.") from exc
            if not isinstance(envelope, dict) or not isinstance(envelope.get("content"), str) or not isinstance(envelope.get("tool_calls"), list):
                raise RuntimeError("Claude Code returned an invalid tool response.")
            content = envelope["content"]
            tool_calls = self._validated_tool_calls(envelope["tool_calls"])
        else:
            content = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
            tool_calls = []
        if stop:
            content = self._apply_stop_sequences(content, stop)
        usage_metadata = self._usage_metadata(response.get("usage"))
        metadata = {key: response[key] for key in ("model", "stop_reason") if response.get(key) is not None}
        return AIMessage(content=content, tool_calls=tool_calls, usage_metadata=usage_metadata, response_metadata=metadata)

    @staticmethod
    def _usage_metadata(usage: Any) -> dict[str, Any] | None:
        if not isinstance(usage, dict):
            return None
        counts = {key: usage.get(key, 0) for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            return None
        input_tokens = counts["input_tokens"] + counts["cache_read_input_tokens"] + counts["cache_creation_input_tokens"]
        output_tokens = counts["output_tokens"]
        metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        metadata["input_token_details"] = {"cache_read": counts["cache_read_input_tokens"], "cache_creation": counts["cache_creation_input_tokens"]}
        return metadata

    def _validated_tool_calls(self, calls: list[Any]) -> list[dict[str, Any]]:
        names = {tool["function"]["name"] for tool in self.bound_tools}
        choice = self.tool_choice or "auto"
        if choice == "none" and calls:
            raise RuntimeError("Claude Code returned tool calls when tools were disabled.")
        if choice == "any" and not calls:
            raise RuntimeError("Claude Code did not return a required tool call.")
        if choice in names:
            if not calls:
                raise RuntimeError("Claude Code did not return the required tool call.")
            if any(not isinstance(call, dict) or call.get("name") != choice for call in calls):
                raise RuntimeError("Claude Code selected an unrequested tool.")
        validated = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                raise RuntimeError("Claude Code returned an invalid tool response.")
            if call["name"] not in names:
                raise RuntimeError("Claude Code selected an unknown tool.")
            validated.append({"name": call["name"], "args": call["arguments"], "id": f"call_{uuid.uuid4().hex}", "type": "tool_call"})
        return validated

    @staticmethod
    def _apply_stop_sequences(content: str, stop: list[str]) -> str:
        positions = [content.find(sequence) for sequence in stop if sequence]
        positions = [position for position in positions if position >= 0]
        return content[: min(positions)] if positions else content
