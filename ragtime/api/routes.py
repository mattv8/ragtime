"""
API route definitions.
"""

import asyncio
import json
import time
from typing import Optional

import psutil
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragtime import __version__
from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.indexer.routes import get_available_chat_models
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
)
from ragtime.rag import rag

logger = get_logger(__name__)

router = APIRouter()

_MODELS_CACHE_TTL_SECONDS = 30.0
_models_cache_lock = asyncio.Lock()
_models_cache: dict[tuple, tuple[float, list[str]]] = {}


# Canonical provider aliases accepted on input.
_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "oa": "openai",
    "anthropic": "anthropic",
    "an": "anthropic",
    "ollama": "ollama",
    "ol": "ollama",
    "github": "github_copilot",
    "gh": "github_copilot",
    "copilot": "github_copilot",
    "github_copilot": "github_copilot",
    "github_models": "github_copilot",
}

# Short provider tokens emitted by /v1/models to keep IDs compact.
_OPENAPI_PROVIDER_TOKENS: dict[str, str] = {
    "openai": "oa",
    "anthropic": "an",
    "ollama": "ol",
    "github_copilot": "gh",
}


def _canonical_provider_name(provider: str) -> str:
    token = (provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(token, token)


def _openapi_provider_token(provider: str) -> str:
    canonical = _canonical_provider_name(provider)
    return _OPENAPI_PROVIDER_TOKENS.get(canonical, canonical)


def _owned_by_from_openapi_model_id(default_provider: str, model_id: str) -> str:
    """Resolve ModelInfo.owned_by from a scoped or bare OpenAPI model ID."""
    raw = (model_id or "").strip()
    if "::" in raw:
        scope, _, _ = raw.partition("::")
        return _canonical_provider_name(scope)
    return _canonical_provider_name(default_provider)


def _split_openapi_model_id(default_provider: str, model_id: str) -> tuple[str, str]:
    """Split OpenAPI model ID into (canonical_provider, model_slug)."""
    raw = (model_id or "").strip()
    if "::" in raw:
        scope, _, slug = raw.partition("::")
        return _canonical_provider_name(scope), slug.strip()
    return _canonical_provider_name(default_provider), raw


def _models_cache_key(app_settings: dict, default_provider: str) -> tuple:
    """Build a stable cache key for /v1/models responses."""
    sync_chat = bool(app_settings.get("openapi_sync_chat_models", True))
    allowed_openapi = tuple(
        str(v).strip()
        for v in (app_settings.get("allowed_openapi_models") or [])
        if str(v).strip()
    )
    allowed_chat = tuple(
        str(v).strip()
        for v in (app_settings.get("allowed_chat_models") or [])
        if str(v).strip()
    )
    llm_model = str(app_settings.get("llm_model", "") or "").strip()
    return (
        default_provider,
        sync_chat,
        allowed_openapi,
        allowed_chat,
        llm_model,
    )


def _get_cached_model_ids(cache_key: tuple) -> Optional[list[str]]:
    entry = _models_cache.get(cache_key)
    if not entry:
        return None

    expires_at, model_ids = entry
    if expires_at <= time.monotonic():
        _models_cache.pop(cache_key, None)
        return None

    return list(model_ids)


def _set_cached_model_ids(cache_key: tuple, model_ids: list[str]) -> None:
    _models_cache[cache_key] = (
        time.monotonic() + _MODELS_CACHE_TTL_SECONDS,
        list(model_ids),
    )


def _configured_openapi_model(app_settings: dict) -> str:
    """Return the model ID configured for OpenAI-compatible clients."""
    return str(app_settings.get("llm_model", "") or "").strip()


def _configured_openapi_models(app_settings: dict) -> list[str]:
    """Return OpenAPI-visible model IDs in stable order."""
    configured = _configured_openapi_model(app_settings)
    allowed = [
        str(value).strip()
        for value in (app_settings.get("allowed_chat_models") or [])
        if str(value).strip()
    ]

    ordered: list[str] = []
    seen: set[str] = set()

    def _push(model_id: str) -> None:
        if model_id and model_id not in seen:
            seen.add(model_id)
            ordered.append(model_id)

    # Keep the configured OpenAPI model first when present.
    _push(configured)
    for model_id in allowed:
        _push(model_id)

    return ordered


def _normalize_runtime_model(provider: str, model: str) -> str:
    """Normalize model IDs before forwarding to provider APIs."""
    model_id = (model or "").strip()
    if not model_id:
        return ""

    if "::" in model_id:
        scoped_provider, _, scoped_model = model_id.partition("::")
        scoped_provider = _canonical_provider_name(scoped_provider)
        scoped_model = scoped_model.strip()
        if not scoped_provider or not scoped_model:
            return model_id
        normalized_scoped_model = _normalize_runtime_model(
            scoped_provider, scoped_model
        )
        return f"{scoped_provider}::{normalized_scoped_model}"

    provider_name = _canonical_provider_name(provider)
    model_id = model_id.lstrip("/")

    # Copilot chat endpoints expect bare model slugs (e.g. "gpt-4.1"),
    # not publisher-prefixed forms (e.g. "openai/gpt-4.1").
    if provider_name == "github_copilot" and "/" in model_id:
        _, _, remainder = model_id.partition("/")
        if remainder:
            return remainder

    return model_id


def _resolve_effective_model(requested_model: Optional[str], app_settings: dict) -> str:
    """Resolve the provider runtime model from OpenAPI request model aliases."""
    provider = str(app_settings.get("llm_provider", "openai") or "openai")
    configured_openapi_model = _configured_openapi_model(app_settings)
    configured_runtime_model = _normalize_runtime_model(
        provider, configured_openapi_model
    )

    requested = (requested_model or "").strip()
    if not requested:
        return configured_runtime_model

    requested_runtime_model = _normalize_runtime_model(provider, requested)

    if configured_runtime_model:
        configured_aliases = {
            configured_openapi_model.lower(),
            configured_runtime_model.lower(),
        }
        # Configured OpenAPI model aliases all map to the configured runtime model.
        if requested.lower() in configured_aliases:
            return configured_runtime_model

    return requested_runtime_model


def _normalize_openapi_model_id(default_provider: str, model_id: str) -> str:
    """Normalize to deterministic scoped OpenAPI ID with compact provider token."""
    raw = (model_id or "").strip()
    if not raw:
        return ""
    runtime_id = _normalize_runtime_model(default_provider, raw)
    if not runtime_id:
        return ""

    if "::" in runtime_id:
        provider, _, scoped_model = runtime_id.partition("::")
        provider_token = _openapi_provider_token(provider)
        return f"{provider_token}::{scoped_model}"

    provider_token = _openapi_provider_token(default_provider)
    return f"{provider_token}::{runtime_id}"


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
            type=d.get("type"),
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
    default_provider = _canonical_provider_name(
        str(app_settings.get("llm_provider", "openai") or "openai")
    )
    cache_key = _models_cache_key(app_settings, default_provider)

    cached_model_ids = _get_cached_model_ids(cache_key)
    if cached_model_ids is not None:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=model_id,
                    created=now,
                    owned_by=_owned_by_from_openapi_model_id(
                        default_provider, model_id
                    ),
                    root=model_id,
                    parent=None,
                )
                for model_id in cached_model_ids
            ]
        )

    async with _models_cache_lock:
        cached_model_ids = _get_cached_model_ids(cache_key)
        if cached_model_ids is not None:
            return ModelsResponse(
                data=[
                    ModelInfo(
                        id=model_id,
                        created=now,
                        owned_by=_owned_by_from_openapi_model_id(
                            default_provider, model_id
                        ),
                        root=model_id,
                        parent=None,
                    )
                    for model_id in cached_model_ids
                ]
            )

        sync_chat = app_settings.get("openapi_sync_chat_models", True)
        allowed_openapi = (
            [
                str(v).strip()
                for v in (app_settings.get("allowed_openapi_models") or [])
                if str(v).strip()
            ]
            if not sync_chat
            else []
        )
        allowed_openapi_scoped_set: Optional[set[str]] = None
        allowed_openapi_bare_set: Optional[set[str]] = None
        if allowed_openapi:
            allowed_openapi_scoped_set = set()
            allowed_openapi_bare_set = set()
            for allowed_model in allowed_openapi:
                normalized_openapi_id = _normalize_openapi_model_id(
                    default_provider, allowed_model
                )
                if normalized_openapi_id:
                    allowed_openapi_scoped_set.add(normalized_openapi_id)

                normalized_runtime_id = _normalize_runtime_model(
                    default_provider, allowed_model
                )
                if normalized_runtime_id and "::" not in normalized_runtime_id:
                    allowed_openapi_bare_set.add(normalized_runtime_id)

        allowed_openapi_bare_lookup = allowed_openapi_bare_set or set()

        model_ids: list[str] = []
        # In synced mode, collapse duplicate slugs across providers to avoid
        # duplicate choices in OpenWebUI model discovery.
        collapse_cross_provider_duplicates = allowed_openapi_scoped_set is None
        selected_by_slug: dict[str, str] = {}
        selected_slug_order: list[str] = []

        def _append_or_select(normalized_model_id: str) -> None:
            if not normalized_model_id:
                return
            if not collapse_cross_provider_duplicates:
                if normalized_model_id not in model_ids:
                    model_ids.append(normalized_model_id)
                return

            candidate_provider, candidate_slug = _split_openapi_model_id(
                default_provider, normalized_model_id
            )
            if not candidate_slug:
                return

            current = selected_by_slug.get(candidate_slug)
            if current is None:
                selected_by_slug[candidate_slug] = normalized_model_id
                selected_slug_order.append(candidate_slug)
                return

            current_provider, _ = _split_openapi_model_id(default_provider, current)
            # Prefer the configured default provider when the slug exists from
            # multiple providers (e.g. openai and github-copilot catalogs).
            if (
                current_provider != default_provider
                and candidate_provider == default_provider
            ):
                selected_by_slug[candidate_slug] = normalized_model_id

        try:
            available = await get_available_chat_models()

            for model in available.models:
                base_model_id = str(model.id or "").strip()
                model_provider = _canonical_provider_name(
                    str(model.provider or default_provider).strip().lower()
                    or default_provider
                )
                if not base_model_id:
                    continue

                scoped_openapi_id = _normalize_openapi_model_id(
                    model_provider, f"{model_provider}::{base_model_id}"
                )
                runtime_base_id = _normalize_runtime_model(
                    model_provider, base_model_id
                )

                # When using a separate OpenAPI list, filter to only allowed models.
                if allowed_openapi_scoped_set is not None:
                    if (
                        scoped_openapi_id not in allowed_openapi_scoped_set
                        and runtime_base_id not in allowed_openapi_bare_lookup
                    ):
                        continue

                # Deterministic contract: always emit provider-scoped model IDs.
                normalized = scoped_openapi_id
                _append_or_select(normalized)
        except Exception as exc:
            logger.warning(
                "Failed to load available chat models for /v1/models: %s", exc
            )

        if collapse_cross_provider_duplicates and selected_slug_order:
            model_ids = [selected_by_slug[slug] for slug in selected_slug_order]

        if not model_ids:
            for model_id in _configured_openapi_models(app_settings):
                _append_or_select(
                    _normalize_openapi_model_id(default_provider, model_id)
                )
            if collapse_cross_provider_duplicates and selected_slug_order:
                model_ids = [selected_by_slug[slug] for slug in selected_slug_order]
            else:
                model_ids = [model_id for model_id in model_ids if model_id]

        if not model_ids:
            # Last-resort fallback for partially configured instances.
            model_ids = ["gpt-4.1"]

        _set_cached_model_ids(cache_key, model_ids)

    return ModelsResponse(
        data=[
            ModelInfo(
                id=model_id,
                created=now,
                owned_by=_owned_by_from_openapi_model_id(default_provider, model_id),
                root=model_id,
                parent=None,
            )
            for model_id in model_ids
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

    app_settings = await get_app_settings()
    default_provider = _canonical_provider_name(
        str(app_settings.get("llm_provider", "openai") or "openai")
    )
    effective_model = _resolve_effective_model(request.model, app_settings)
    if not effective_model:
        raise HTTPException(
            status_code=400,
            detail="No model configured. Set an LLM model in Settings.",
        )
    response_model = _normalize_openapi_model_id(
        default_provider, request.model or effective_model
    ) or _normalize_openapi_model_id(default_provider, effective_model)
    tool_output_mode = (
        request.agent_options.tool_output_mode
        if request.agent_options and request.agent_options.tool_output_mode is not None
        else app_settings.get("tool_output_mode", "default")
    )
    suppress_tool_output = tool_output_mode == "hide"

    # Extract the latest user message (full message, including multimodal content)
    user_message = next(
        (m for m in reversed(request.messages) if m.role == "user"),
        None,
    )

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    user_text = user_message.get_text_content()
    logger.info(f"Processing query with model {effective_model}: {user_text[:100]}...")

    # Build chat history for context (convert to LangChain format)
    chat_history = []
    for msg in request.messages[:-1]:  # Exclude the current message
        if msg.role == "user":
            chat_history.append(
                HumanMessage(content=await rag._convert_message_to_langchain_async(msg))
            )
        elif msg.role == "assistant":
            if msg.tool_calls:
                # Reconstruct native AIMessage with tool_calls
                chat_history.append(
                    AIMessage(
                        content=msg.get_text_content(),
                        tool_calls=[
                            {
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments or "{}"),
                                "id": tc.id,
                            }
                            for tc in msg.tool_calls
                        ],
                    )
                )
            else:
                chat_history.append(AIMessage(content=msg.get_text_content()))
        elif msg.role == "tool":
            chat_history.append(
                ToolMessage(
                    content=msg.get_text_content(),
                    tool_call_id=msg.tool_call_id or "",
                )
            )

    # Handle streaming response - use true LLM streaming
    if request.stream:
        return StreamingResponse(
            _stream_response_tokens(
                user_message,
                chat_history,
                effective_model,
                response_model=response_model,
                suppress_tool_output=suppress_tool_output,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: process the query normally
    answer = await rag.process_query(
        user_message,
        chat_history,
        conversation_model=effective_model,
    )

    logger.info(f"Response generated ({len(answer)} chars)")

    # Standard JSON response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=response_model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
    )


async def _stream_response_tokens(
    user_message,
    chat_history: list,
    model: str,
    response_model: Optional[str] = None,
    suppress_tool_output: bool = False,
):
    """
    Generate true streaming response by yielding tokens from the LLM.

    Streams tokens as they're generated, supporting <think> tags and other
    structured output without filtering. Tool calls are formatted as readable
    text for OpenAI API compatibility with external clients.

    Args:
        user_message: Message object (can contain multimodal content)
        chat_history: Previous messages
        model: Model name string
        suppress_tool_output: Whether to hide tool-call/result blocks in stream
    """
    chunk_id = f"chatcmpl-{int(time.time())}"

    def make_chunk(content: str, finish_reason: str | None = None) -> str:
        """Create an SSE chunk with the given content."""
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": response_model or model,
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
    async for event in rag.process_query_stream(
        user_message, chat_history, conversation_model=model
    ):
        # Handle structured events (tool calls)
        if isinstance(event, dict):
            event_type = event.get("type")

            if event_type == "tool_start":
                tool_name = event.get("tool", "unknown")
                tool_input = event.get("input", {})
                current_tool = tool_name

                # Skip tool output if suppressed
                if suppress_tool_output:
                    continue

                # Format tool input for immediate display
                input_display = ""
                if tool_input:
                    for field in ["query", "sql", "code", "command", "python_code"]:
                        if field in tool_input and tool_input[field]:
                            input_display = str(tool_input[field])
                            break
                    if not input_display:
                        input_display = json.dumps(tool_input, indent=2)

                lang = _detect_code_language(tool_name, tool_input)

                # Emit collapsible tool call block
                block = (
                    f'\n\n<details type="tool_call">\n'
                    f"<summary>Calling {tool_name}...</summary>\n\n"
                    f"```{lang}\n{input_display}\n```\n\n"
                    f"</details>\n"
                )
                yield make_chunk(block)

            elif event_type == "tool_end":
                tool_name = event.get("tool", current_tool or "unknown")
                tool_output = event.get("output", "")
                current_tool = None

                # Skip tool output if suppressed
                if suppress_tool_output:
                    continue

                output_display = _format_tool_output(str(tool_output))

                # Emit result in a collapsible details block
                block = (
                    f'\n<details type="tool_result">\n'
                    f"<summary>Result</summary>\n\n"
                    f"```\n{output_display}\n```\n\n"
                    f"</details>\n\n"
                )
                yield make_chunk(block)

            elif event_type == "max_iterations_reached":
                yield make_chunk("\n\n> **Note:** Reached maximum tool iterations\n")

            elif event_type == "reasoning":
                # Reasoning/thinking content - emit as collapsible details block
                reasoning_text = event.get("content", "")
                if reasoning_text:
                    block = (
                        f'\n<details type="reasoning">\n'
                        f"<summary>Thinking...</summary>\n\n"
                        f"{reasoning_text}\n\n"
                        f"</details>\n"
                    )
                    yield make_chunk(block)

        else:
            # Plain string content - stream directly
            yield make_chunk(str(event))

    # Final chunk with finish_reason
    yield make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"
