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
            ModelInfo(
                id="ragtime-tools",
                created=now,
                owned_by="ragtime",
                root="ragtime-tools",
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
    structured output without filtering. Tool calls are executed first,
    then the response is streamed.
    """
    chunk_id = f"chatcmpl-{int(time.time())}"

    # Stream tokens directly from the LLM/agent
    async for token in rag.process_query_stream(user_msg, chat_history):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with finish_reason
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
