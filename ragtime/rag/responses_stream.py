"""Stable Responses API stream conversion for interleaved output items."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from langchain_core.outputs import ChatGenerationChunk

# This private LangChain seam preserves its metadata, usage, and reasoning conversion.
# Adapter tests intentionally pin its behavior without pinning the dependency version.
from langchain_openai.chat_models.base import _convert_responses_chunk_to_generation_chunk


class _StableChunkConverter:
    """Assign each Responses output block one index for the request lifetime."""

    def __init__(self) -> None:
        self._next_index = 0
        self._indices: dict[tuple[int, int | None], int] = {}

    @staticmethod
    def _block_key(chunk: Any) -> tuple[int, int | None] | None:
        if not hasattr(chunk, "output_index"):
            return None
        if chunk.type in {"response.output_text.delta", "response.output_text.annotation.added"}:
            return chunk.output_index, chunk.content_index
        if chunk.type == "response.output_text.done":
            return chunk.output_index, getattr(chunk, "content_index", None)
        if chunk.type == "response.output_item.added" and chunk.item.type == "message":
            return None
        if chunk.type in {
            "response.output_item.added",
            "response.output_item.done",
            "response.function_call_arguments.delta",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
        }:
            return chunk.output_index, None
        return None

    def convert(
        self,
        chunk: Any,
        *,
        schema: Any,
        metadata: dict[str, Any],
        has_reasoning: bool,
        output_version: Literal["v0", "responses/v1"],
    ) -> tuple[ChatGenerationChunk | None, bool]:
        block_key = self._block_key(chunk)
        if block_key is None:
            state = (self._next_index - 1, -1, -1)
        elif (index := self._indices.get(block_key)) is not None:
            state = (index, block_key[0], block_key[1] if block_key[1] is not None else -1)
        elif chunk.type == "response.output_text.done":
            state = (self._next_index, block_key[0], block_key[1] if block_key[1] is not None else -1)
        else:
            state = (self._next_index - 1, -1, -1)

        current_index, _, _, generation_chunk = _convert_responses_chunk_to_generation_chunk(
            chunk,
            *state,
            schema=schema,
            metadata=metadata,
            has_reasoning=has_reasoning,
            output_version=output_version,
        )
        if block_key is not None and generation_chunk is not None and block_key not in self._indices:
            self._indices[block_key] = current_index
            self._next_index = current_index + 1
        return generation_chunk, bool(generation_chunk and "reasoning" in generation_chunk.message.additional_kwargs)


def stream_responses(
    model: Any,
    messages: list[Any],
    stop: list[str] | None = None,
    run_manager: Any = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """Stream Responses chunks while preserving output identities across interleaving."""
    kwargs["stream"] = True
    payload = model._get_request_payload(messages, stop=stop, **kwargs)
    if model.include_response_headers:
        raw_context_manager = model.root_client.with_raw_response.responses.create(**payload)
        context_manager = raw_context_manager.parse()
        headers = {"headers": dict(raw_context_manager.headers)}
    else:
        context_manager = model.root_client.responses.create(**payload)
        headers = {}
    schema = kwargs.get("response_format")
    converter = _StableChunkConverter()
    is_first_chunk = True
    has_reasoning = False

    with context_manager as response:
        for chunk in response:
            generation_chunk, is_reasoning = converter.convert(
                chunk,
                schema=schema,
                metadata=headers if is_first_chunk else {},
                has_reasoning=has_reasoning,
                output_version=model.output_version,
            )
            if generation_chunk:
                if run_manager:
                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                is_first_chunk = False
                has_reasoning = has_reasoning or is_reasoning
                yield generation_chunk


async def astream_responses(
    model: Any,
    messages: list[Any],
    stop: list[str] | None = None,
    run_manager: Any = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]:
    """Async equivalent of :func:`stream_responses`."""
    kwargs["stream"] = True
    payload = model._get_request_payload(messages, stop=stop, **kwargs)
    if model.include_response_headers:
        raw_context_manager = await model.root_async_client.with_raw_response.responses.create(**payload)
        context_manager = raw_context_manager.parse()
        headers = {"headers": dict(raw_context_manager.headers)}
    else:
        context_manager = await model.root_async_client.responses.create(**payload)
        headers = {}
    schema = kwargs.get("response_format")
    converter = _StableChunkConverter()
    is_first_chunk = True
    has_reasoning = False

    async with context_manager as response:
        async for chunk in response:
            generation_chunk, is_reasoning = converter.convert(
                chunk,
                schema=schema,
                metadata=headers if is_first_chunk else {},
                has_reasoning=has_reasoning,
                output_version=model.output_version,
            )
            if generation_chunk:
                if run_manager:
                    await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                is_first_chunk = False
                has_reasoning = has_reasoning or is_reasoning
                yield generation_chunk
