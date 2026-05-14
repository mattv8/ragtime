"""
API route definitions.
"""

import asyncio
from dataclasses import dataclass, replace
import hmac
import json
import time
from typing import Optional

import psutil
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragtime import __version__
from ragtime.config import settings
from ragtime.core.api_accounting import log_api_request
from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import (
    compose_model_display_label,
)
from ragtime.core.model_providers import (
    get_provider_label,
    normalize_provider_name,
)
from ragtime.core.security import get_current_user_optional
from ragtime.indexer.background_tasks import rebuild_tool_messages_from_events
from ragtime.indexer.routes import AvailableModel, get_available_chat_models
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


@dataclass(frozen=True)
class _OpenAPIModelEntry:
    id: str
    runtime_model: str
    provider: str
    owned_by: str
    root: str
    selection_keys: tuple[str, ...]


_models_cache: dict[tuple, tuple[float, list[_OpenAPIModelEntry]]] = {}


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
    default_chat_model = str(app_settings.get("default_chat_model", "") or "").strip()
    return (
        default_provider,
        sync_chat,
        allowed_openapi,
        allowed_chat,
        llm_model,
        default_chat_model,
    )


def _get_cached_model_entries(cache_key: tuple) -> Optional[list[_OpenAPIModelEntry]]:
    entry = _models_cache.get(cache_key)
    if not entry:
        return None

    expires_at, model_entries = entry
    if expires_at <= time.monotonic():
        _models_cache.pop(cache_key, None)
        return None

    return list(model_entries)


def _set_cached_model_entries(
    cache_key: tuple,
    model_entries: list[_OpenAPIModelEntry],
) -> None:
    _models_cache[cache_key] = (
        time.monotonic() + _MODELS_CACHE_TTL_SECONDS,
        list(model_entries),
    )


def _configured_openapi_model(app_settings: dict) -> str:
    """Return the model ID configured for OpenAI-compatible clients."""
    override = str(app_settings.get("default_chat_model", "") or "").strip()
    if override:
        return override
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
        scoped_provider = normalize_provider_name(scoped_provider)
        scoped_model = scoped_model.strip()
        if not scoped_provider or not scoped_model:
            return model_id
        normalized_scoped_model = _normalize_runtime_model(
            scoped_provider, scoped_model
        )
        return f"{scoped_provider}::{normalized_scoped_model}"

    provider_name = normalize_provider_name(provider)
    model_id = model_id.lstrip("/")

    # Copilot chat endpoints expect bare model slugs (e.g. "gpt-4.1"),
    # not publisher-prefixed forms (e.g. "openai/gpt-4.1").
    if provider_name == "github_copilot" and "/" in model_id:
        _, _, remainder = model_id.partition("/")
        if remainder:
            return remainder

    return model_id


def _normalize_label_part(value: Optional[str]) -> str:
    return " ".join(str(value or "").strip().split())


def _split_runtime_model(default_provider: str, runtime_model: str) -> tuple[str, str]:
    raw = str(runtime_model or "").strip()
    if "::" in raw:
        provider, _, model_id = raw.partition("::")
        return normalize_provider_name(provider), model_id.strip()
    return normalize_provider_name(default_provider), raw


def _openapi_display_id_for_parts(
    *,
    provider: str,
    model_id: str,
    model_provider_label: Optional[str] = None,
    family_label: Optional[str] = None,
    display_name: Optional[str] = None,
) -> str:
    return compose_model_display_label(
        model_id=model_id,
        provider_label=_normalize_label_part(model_provider_label)
        or get_provider_label(provider),
        family_label=family_label,
        display_name=display_name,
    )


def _openapi_display_id_from_available_model(
    model: AvailableModel,
    provider: str,
) -> str:
    if getattr(model, "selector_label", None):
        return _normalize_label_part(model.selector_label)
    return _openapi_display_id_for_parts(
        provider=provider,
        model_id=model.id,
        model_provider_label=model.model_provider_label,
        family_label=model.model_family or model.group,
        display_name=model.display_name or model.name,
    )


def _selection_key_variants(*values: Optional[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    variants: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        folded = raw.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        variants.append(raw)
    return tuple(variants)


def _build_openapi_model_entry(model: AvailableModel, default_provider: str) -> _OpenAPIModelEntry:
    provider = normalize_provider_name(str(model.provider or default_provider))
    base_model_id = str(model.id or "").strip()
    runtime_model = _normalize_runtime_model(provider, f"{provider}::{base_model_id}")
    _runtime_provider, runtime_model_id = _split_runtime_model(provider, runtime_model)
    display_id = _openapi_display_id_from_available_model(model, provider)
    owned_by = _normalize_label_part(model.model_provider_label) or get_provider_label(
        provider
    )
    provider_scoped_model_id = f"{provider}::{base_model_id}"
    return _OpenAPIModelEntry(
        id=display_id,
        runtime_model=runtime_model,
        provider=provider,
        owned_by=owned_by,
        root=display_id,
        selection_keys=_selection_key_variants(
            display_id,
            provider_scoped_model_id,
            runtime_model,
            base_model_id,
            runtime_model_id,
        ),
    )


def _build_fallback_openapi_model_entry(
    default_provider: str,
    model_id: str,
) -> _OpenAPIModelEntry:
    runtime_model = _normalize_runtime_model(default_provider, model_id)
    provider, runtime_model_id = _split_runtime_model(default_provider, runtime_model)
    display_id = _openapi_display_id_for_parts(
        provider=provider,
        model_id=runtime_model_id,
    )
    provider_scoped_model_id = f"{provider}::{runtime_model_id}"
    return _OpenAPIModelEntry(
        id=display_id,
        runtime_model=runtime_model,
        provider=provider,
        owned_by=get_provider_label(provider),
        root=display_id,
        selection_keys=_selection_key_variants(
            display_id,
            provider_scoped_model_id,
            runtime_model,
            model_id,
            runtime_model_id,
        ),
    )


def _entry_duplicate_slug(entry: _OpenAPIModelEntry) -> str:
    _provider, model_id = _split_runtime_model(entry.provider, entry.runtime_model)
    return model_id.casefold()


def _entry_matches_allowed(entry: _OpenAPIModelEntry, allowed_models: set[str]) -> bool:
    if not allowed_models:
        return True
    folded_allowed = {value.casefold() for value in allowed_models}
    return any(key.casefold() in folded_allowed for key in entry.selection_keys)


def _ensure_unique_openapi_model_ids(
    entries: list[_OpenAPIModelEntry],
) -> list[_OpenAPIModelEntry]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.id.casefold()
        counts[key] = counts.get(key, 0) + 1

    used: set[str] = set()
    result: list[_OpenAPIModelEntry] = []
    for entry in entries:
        display_id = entry.id
        if counts[entry.id.casefold()] > 1:
            base = f"{entry.id} (via {get_provider_label(entry.provider)})"
            display_id = base
            suffix = 2
            while display_id.casefold() in used:
                display_id = f"{base} {suffix}"
                suffix += 1

        used.add(display_id.casefold())
        if display_id == entry.id:
            result.append(entry)
            continue
        result.append(
            replace(
                entry,
                id=display_id,
                root=display_id,
                selection_keys=_selection_key_variants(display_id, *entry.selection_keys),
            )
        )
    return result


def _model_entry_to_info(entry: _OpenAPIModelEntry, created: int) -> ModelInfo:
    return ModelInfo(
        id=entry.id,
        created=created,
        owned_by=entry.owned_by,
        root=entry.root,
        parent=None,
    )


def _normalize_openapi_model_id(default_provider: str, model_id: str) -> str:
    """Return the pretty OpenAI-compatible ID advertised for a fallback model."""
    runtime_id = _normalize_runtime_model(default_provider, model_id)
    if not runtime_id:
        return ""
    provider, runtime_model_id = _split_runtime_model(default_provider, runtime_id)
    return _openapi_display_id_for_parts(provider=provider, model_id=runtime_model_id)


async def _get_openapi_model_entries(
    app_settings: dict,
    default_provider: str,
) -> list[_OpenAPIModelEntry]:
    cache_key = _models_cache_key(app_settings, default_provider)

    cached_model_entries = _get_cached_model_entries(cache_key)
    if cached_model_entries is not None:
        return cached_model_entries

    async with _models_cache_lock:
        cached_model_entries = _get_cached_model_entries(cache_key)
        if cached_model_entries is not None:
            return cached_model_entries

        sync_chat = app_settings.get("openapi_sync_chat_models", True)
        allowed_openapi = {
            str(value).strip()
            for value in (app_settings.get("allowed_openapi_models") or [])
            if str(value).strip()
        } if not sync_chat else set()

        entries: list[_OpenAPIModelEntry] = []
        collapse_cross_provider_duplicates = not allowed_openapi
        selected_by_slug: dict[str, _OpenAPIModelEntry] = {}
        selected_slug_order: list[str] = []

        def _append_or_select(entry: _OpenAPIModelEntry) -> None:
            if not collapse_cross_provider_duplicates:
                entries.append(entry)
                return

            slug = _entry_duplicate_slug(entry)
            if not slug:
                return

            current = selected_by_slug.get(slug)
            if current is None:
                selected_by_slug[slug] = entry
                selected_slug_order.append(slug)
                return

            if current.provider != default_provider and entry.provider == default_provider:
                selected_by_slug[slug] = entry

        try:
            available = await get_available_chat_models()

            for model in available.models:
                if not str(model.id or "").strip():
                    continue
                entry = _build_openapi_model_entry(model, default_provider)

                if allowed_openapi and not _entry_matches_allowed(entry, allowed_openapi):
                    continue

                _append_or_select(entry)
        except Exception as exc:
            logger.warning(
                "Failed to load available chat models for /v1/models: %s", exc
            )

        if collapse_cross_provider_duplicates and selected_slug_order:
            entries = [selected_by_slug[slug] for slug in selected_slug_order]

        if not entries:
            for model_id in _configured_openapi_models(app_settings):
                entry = _build_fallback_openapi_model_entry(default_provider, model_id)
                if allowed_openapi and not _entry_matches_allowed(entry, allowed_openapi):
                    continue
                _append_or_select(entry)
            if collapse_cross_provider_duplicates and selected_slug_order:
                entries = [selected_by_slug[slug] for slug in selected_slug_order]

        if not entries:
            entries = [_build_fallback_openapi_model_entry(default_provider, "gpt-4.1")]

        entries = _ensure_unique_openapi_model_ids(entries)
        _set_cached_model_entries(cache_key, entries)
        return entries


def _find_openapi_model_entry(
    entries: list[_OpenAPIModelEntry],
    requested_model: str,
) -> Optional[_OpenAPIModelEntry]:
    requested = str(requested_model or "").strip().casefold()
    if not requested:
        return None
    for entry in entries:
        if entry.id.casefold() == requested:
            return entry
    return None


async def _resolve_effective_model(
    requested_model: Optional[str],
    app_settings: dict,
    default_provider: str,
) -> tuple[str, str]:
    """Resolve an advertised OpenAPI model ID to a provider runtime model."""
    entries = await _get_openapi_model_entries(app_settings, default_provider)

    requested = (requested_model or "").strip()
    if requested:
        entry = _find_openapi_model_entry(entries, requested)
        if entry:
            return entry.runtime_model, entry.id

        runtime_model = _normalize_runtime_model(default_provider, requested)
        advertised_model = _normalize_openapi_model_id(default_provider, runtime_model)
        return runtime_model, advertised_model or requested

    configured_openapi_model = _configured_openapi_model(app_settings)
    configured_runtime_model = _normalize_runtime_model(
        default_provider,
        configured_openapi_model,
    )
    if configured_runtime_model:
        for entry in entries:
            if entry.runtime_model.casefold() == configured_runtime_model.casefold():
                return entry.runtime_model, entry.id
            if configured_openapi_model.casefold() in (
                key.casefold() for key in entry.selection_keys
            ):
                return entry.runtime_model, entry.id
        advertised_model = _normalize_openapi_model_id(
            default_provider,
            configured_runtime_model,
        )
        return configured_runtime_model, advertised_model

    first_entry = entries[0] if entries else None
    if first_entry:
        return first_entry.runtime_model, first_entry.id

    return "", ""


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if settings.api_key:
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")

        # Support both "Bearer <key>" and raw key
        key = authorization.replace("Bearer ", "").strip()
        if not hmac.compare_digest(key, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Health check endpoint.

    Returns progressive loading status:
    - status: "healthy" when core is ready (LLM/settings loaded)
    - indexes_loading: True while FAISS indexes are loading in background
    - indexes_ready: True when all indexes are loaded
    - memory: Real-time process memory statistics
    - index_details: Per-index loading status with timing info

    Unauthenticated callers receive only readiness-focused fields. Detailed
    model/provider/index/memory information is returned only to
    authenticated users to avoid disclosing deployment posture publicly.
    """
    loading_status = rag.loading_status

    # Determine overall status
    if rag.is_ready:
        status = "healthy"
    else:
        status = "initializing"

    if current_user is None:
        # Minimal public readiness signal.
        return HealthResponse(
            status=status,
            version=__version__,
            indexes_loaded=[],
            model="",
            llm_provider="",
        )

    app_settings = await get_app_settings()

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
    default_provider = normalize_provider_name(
        str(app_settings.get("llm_provider", "openai") or "openai")
    )
    model_entries = await _get_openapi_model_entries(app_settings, default_provider)
    return ModelsResponse(
        data=[_model_entry_to_info(entry, now) for entry in model_entries]
    )


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """
    Main chat endpoint with RAG and tool calling.
    OpenAI API compatible for use with OpenWebUI and similar tools.
    """
    if not rag.is_ready:
        asyncio.ensure_future(
            log_api_request(
                endpoint="/v1/chat/completions",
                http_method="POST",
                status_code=503,
            )
        )
        raise HTTPException(
            status_code=503, detail="Service initializing, please retry"
        )

    app_settings = await get_app_settings()
    default_provider = normalize_provider_name(
        str(app_settings.get("llm_provider", "openai") or "openai")
    )
    effective_model, response_model = await _resolve_effective_model(
        request.model,
        app_settings,
        default_provider,
    )
    if not effective_model:
        asyncio.ensure_future(
            log_api_request(
                endpoint="/v1/chat/completions",
                http_method="POST",
                status_code=400,
            )
        )
        raise HTTPException(
            status_code=400,
            detail="No model configured. Set an LLM model in Settings.",
        )
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
        asyncio.ensure_future(
            log_api_request(
                provider=default_provider,
                model=effective_model,
                endpoint="/v1/chat/completions",
                http_method="POST",
                status_code=400,
            )
        )
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
            message_events = getattr(msg, "events", None)
            if message_events:
                chat_history.extend(
                    rebuild_tool_messages_from_events(message_events, 0)
                )
            elif msg.tool_calls:
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
        asyncio.ensure_future(
            log_api_request(
                provider=default_provider,
                model=effective_model,
                endpoint="/v1/chat/completions",
                http_method="POST",
                status_code=200,
                streaming=True,
            )
        )
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

    asyncio.ensure_future(
        log_api_request(
            provider=default_provider,
            model=effective_model,
            endpoint="/v1/chat/completions",
            http_method="POST",
            status_code=200,
            streaming=False,
        )
    )

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
