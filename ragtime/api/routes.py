"""
API route definitions.
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage

from ragtime import __version__
from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.app_settings import get_app_settings
from ragtime.models import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    Usage,
    ModelInfo,
    ModelsResponse,
    HealthResponse,
)
from ragtime.rag import rag

logger = get_logger(__name__)

router = APIRouter()


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if settings.api_key:
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")

        # Support both "Bearer <key>" and raw key
        key = authorization.replace("Bearer ", "").strip()
        if key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    app_settings = await get_app_settings()
    return HealthResponse(
        status="healthy" if rag.is_ready else "initializing",
        version=__version__,
        indexes_loaded=list(rag.retrievers.keys()),
        model=app_settings.get("llm_model", "gpt-4-turbo"),
        llm_provider=app_settings.get("llm_provider", "openai")
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    now = int(time.time())
    return ModelsResponse(
        data=[
            ModelInfo(
                id="ragtime",
                created=now,
                owned_by="ragtime",
                root="ragtime",
                parent=None
            ),
        ]
    )


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """
    Main chat endpoint with RAG and tool calling.
    OpenAI API compatible for use with OpenWebUI and similar tools.
    """
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="Service initializing, please retry")

    # Extract the latest user message
    user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None
    )

    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    logger.info(f"Processing query: {user_msg[:100]}...")

    # Build chat history for context
    chat_history = []
    for msg in request.messages[:-1]:  # Exclude the current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    # Handle streaming response - use true LLM streaming
    if request.stream:
        return StreamingResponse(
            _stream_response_tokens(user_msg, chat_history, request.model),
            media_type="text/event-stream"
        )

    # Non-streaming: process the query normally
    answer = await rag.process_query(user_msg, chat_history)

    logger.info(f"Response generated ({len(answer)} chars)")

    # Standard JSON response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop"
            )
        ]
    )


async def _stream_response_tokens(user_msg: str, chat_history: list, model: str):
    """
    Generate true streaming response by yielding tokens from the LLM.

    Streams tokens as they're generated, supporting <think> tags and other
    structured output without filtering. Tool calls are formatted as readable
    text for OpenAI API compatibility with external clients.
    """
    chunk_id = f"chatcmpl-{int(time.time())}"

    def make_chunk(content: str, finish_reason: str | None = None) -> str:
        """Create an SSE chunk with the given content."""
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # Track active tool for formatting
    current_tool: str | None = None

    # Stream tokens from the RAG agent
    async for event in rag.process_query_stream(user_msg, chat_history):
        # Handle structured events (tool calls)
        if isinstance(event, dict):
            event_type = event.get("type")

            if event_type == "tool_start":
                tool_name = event.get("tool", "unknown")
                tool_input = event.get("input", {})
                current_tool = tool_name

                # Format tool start as readable text
                # Use a clean format that renders well in markdown
                input_display = ""
                if tool_input:
                    # Extract the most relevant input field
                    for field in ["query", "sql", "code", "command", "python_code"]:
                        if field in tool_input and tool_input[field]:
                            input_display = str(tool_input[field])
                            break
                    if not input_display:
                        input_display = json.dumps(tool_input, indent=2)

                header = f"\n\n**Using {tool_name}**\n```\n{input_display}\n```\n"
                yield make_chunk(header)

            elif event_type == "tool_end":
                tool_name = event.get("tool", current_tool or "unknown")
                tool_output = event.get("output", "")
                current_tool = None

                # Format tool output - truncate if very long
                if len(str(tool_output)) > 500:
                    output_display = str(tool_output)[:500] + "...(truncated)"
                else:
                    output_display = str(tool_output)

                footer = f"\n**Result:**\n```\n{output_display}\n```\n\n"
                yield make_chunk(footer)

            elif event_type == "max_iterations_reached":
                yield make_chunk("\n\n*[Reached maximum tool iterations]*\n")

        else:
            # Plain string content - stream directly
            yield make_chunk(str(event))

    # Final chunk with finish_reason
    yield make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"
