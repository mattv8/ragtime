"""
API route definitions.
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from ragtime import __version__
from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    IndexLoadingDetail,
    MemoryStats,
    Message,
    ModelInfo,
    ModelsResponse,
    Usage,
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
    """Health check endpoint.

    Returns progressive loading status:
    - status: "healthy" when core is ready (LLM/settings loaded)
    - indexes_loading: True while FAISS indexes are loading in background
    - indexes_ready: True when all indexes are loaded
    - memory: Real-time process memory statistics
    - index_details: Per-index loading status with timing info
    """
    import psutil

    app_settings = await get_app_settings()
    loading_status = rag.loading_status

    # Determine overall status
    if rag.is_ready:
        status = "healthy"
    else:
        status = "initializing"

    # Get real-time memory stats
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()

    memory = MemoryStats(
        rss_mb=mem_info.rss / (1024 * 1024),
        vms_mb=mem_info.vms / (1024 * 1024),
        percent=process.memory_percent(),
        available_mb=virtual_mem.available / (1024 * 1024),
        total_mb=virtual_mem.total / (1024 * 1024),
    )

    # Convert index details to response format
    index_details = [
        IndexLoadingDetail(
            name=d["name"],
            status=d["status"],
            size_mb=d.get("size_mb"),
            chunk_count=d.get("chunk_count"),
            load_time_seconds=d.get("load_time_seconds"),
            error=d.get("error"),
        )
        for d in loading_status.get("index_details", [])
    ]

    return HealthResponse(
        status=status,
        version=__version__,
        indexes_loaded=loading_status["retrievers_available"],
        model=app_settings.get("llm_model", "gpt-4-turbo"),
        llm_provider=app_settings.get("llm_provider", "openai"),
        indexes_ready=loading_status["indexes_ready"],
        indexes_loading=loading_status["indexes_loading"],
        indexes_total=loading_status["indexes_total"],
        indexes_loaded_count=loading_status["indexes_loaded"],
        memory=memory,
        index_details=index_details if index_details else None,
        sequential_loading=loading_status.get("sequential_loading", False),
        loading_index=loading_status.get("loading_index"),
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    now = int(time.time())
    app_settings = await get_app_settings()
    server_name = app_settings.get("server_name", "Ragtime")
    # Use lowercase for API model ID
    model_id = server_name.lower().replace(" ", "-")
    return ModelsResponse(
        data=[
            ModelInfo(
                id=model_id, created=now, owned_by=model_id, root=model_id, parent=None
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
        raise HTTPException(
            status_code=503, detail="Service initializing, please retry"
        )

    # Extract the latest user message
    user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), None
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
            media_type="text/event-stream",
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
                finish_reason="stop",
            )
        ],
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
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # Track active tool for formatting
    current_tool: str | None = None

    def _detect_code_language(tool_name: str, tool_input: dict) -> str:
        """Detect the appropriate code language for syntax highlighting."""
        if "sql" in tool_input:
            return "sql"
        if "python_code" in tool_input or "code" in tool_input:
            return "python"
        if tool_name and "postgres" in tool_name.lower():
            return "sql"
        if tool_name and "odoo" in tool_name.lower():
            return "python"
        return ""

    def _format_tool_output(output: str, max_length: int = 1500) -> str:
        """Format tool output, truncating if needed."""
        if not output:
            return "(no output)"
        # Truncate very long outputs
        if len(output) > max_length:
            output = output[:max_length] + "\n... (truncated)"
        return output

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
                input_display = ""
                if tool_input:
                    # Extract the most relevant input field
                    for field in ["query", "sql", "code", "command", "python_code"]:
                        if field in tool_input and tool_input[field]:
                            input_display = str(tool_input[field])
                            break
                    if not input_display:
                        input_display = json.dumps(tool_input, indent=2)

                # Detect language for syntax highlighting
                lang = _detect_code_language(tool_name, tool_input)

                # Simple blockquote format - universally compatible
                header = (
                    f"\n\n> **Using {tool_name}**\n\n```{lang}\n{input_display}\n```\n"
                )
                yield make_chunk(header)

            elif event_type == "tool_end":
                tool_name = event.get("tool", current_tool or "unknown")
                tool_output = event.get("output", "")
                current_tool = None

                output_display = _format_tool_output(str(tool_output))

                # Simple format for result
                footer = f"\n> **Result:**\n\n```\n{output_display}\n```\n\n"
                yield make_chunk(footer)

            elif event_type == "max_iterations_reached":
                yield make_chunk("\n\n> **Note:** Reached maximum tool iterations\n")

        else:
            # Plain string content - stream directly
            yield make_chunk(str(event))

    # Final chunk with finish_reason
    yield make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"
