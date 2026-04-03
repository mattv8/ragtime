"""
Model context and capability limits fetched from models.dev.

Source: https://models.dev/api.json
"""

import asyncio
import re

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# models.dev provider/model catalog (updated frequently)
MODELS_DEV_API_URL = "https://models.dev/api.json"

# Cache for model limits (populated on first request)
_model_limits_cache: dict[str, int] = {}
# Cache for model output limits
_model_output_limits_cache: dict[str, int] = {}
# Cache for function calling support
_model_supports_function_calling: dict[str, bool] = {}
# Cache for reasoning/thinking support
_model_supports_reasoning: dict[str, bool] = {}
# Cache for thinking budget support
_model_supports_thinking_budget: dict[str, bool] = {}
# Cache for models requiring Responses API (populated from Copilot /models)
_model_requires_responses_api: dict[str, bool] = {}
# Cache for models that support Responses API (including dual-endpoint models)
_model_supports_responses_api: dict[str, bool] = {}
# Cache for provider-reported reasoning capabilities (authoritative)
_provider_supports_reasoning: dict[str, bool] = {}
# Cache for provider-reported thinking-budget capabilities (authoritative)
_provider_supports_thinking_budget: dict[str, bool] = {}
_cache_lock = asyncio.Lock()
_cache_loaded = False

# Default when model not found
DEFAULT_CONTEXT_LIMIT = 8192


# Model family grouping patterns for UI organization
# Format: {provider: [(regex_pattern, group_name_or_None)]}
# If group_name is None, it uses the first capture group of the regex
# IMPORTANT: Patterns are matched in order, so more specific patterns must come first
MODEL_FAMILY_PATTERNS = {
    "openai": [
        # O-series models (reasoning models) - must come before gpt patterns
        (r"^o\d+-", "O-Series"),
        (r"^o\d+$", "O-Series"),
        # GPT-4o/turbo aliases should stay explicit before numeric captures.
        (r"^gpt-4o", "GPT-4o"),
        (r"^gpt-4-turbo", "GPT-4 Turbo"),
        # Codex family must come before generic numeric captures.
        (r"^gpt-\d+(?:\.\d+)?-codex", "Codex"),
        # Numeric GPT families are captured dynamically (e.g. 5.4 -> GPT-5.4).
        (r"^gpt-(\d+\.\d+)(?:$|[-_])", None),
        (r"^gpt-(\d+)(?:$|[-_])", None),
    ],
    "anthropic": [
        # Haiku models grouped together (all versions) - must be BEFORE general claude-3.5/3 patterns
        (r"claude-haiku-4-5", "Haiku"),
        (r"claude-haiku-4\.5", "Haiku"),
        (r"claude-haiku-4", "Haiku"),
        (r"claude-(3-5|3\.5|3)-haiku", "Haiku"),
        # Opus and Sonnet families
        (r"claude-opus-4", "Claude Opus 4"),
        (r"claude-sonnet-4", "Claude Sonnet 4"),
        (r"claude-4", "Claude 4"),
        (r"claude-(3-5|3\.5)-sonnet", "Claude 3.5 Sonnet"),
        (r"claude-(3-5|3\.5)", "Claude 3.5"),
        (r"claude-3-opus", "Claude 3 Opus"),
        (r"claude-3-sonnet", "Claude 3 Sonnet"),
        (r"claude-3", "Claude 3"),
        (r"claude-2", "Claude 2"),
    ],
    "ollama": [(r"^([a-z0-9]+)", None)],
    "github_copilot": [
        # GitHub-hosted OpenAI families
        (r"^(openai/)?gpt-4o", "GPT-4o"),
        # Codex family must come before generic numeric captures.
        (r"^(?:openai/)?gpt-\d+(?:\.\d+)?-codex", "Codex"),
        (r"^(?:openai/)?gpt-(\d+\.\d+)(?:$|[-_])", None),
        (r"^(?:openai/)?gpt-(\d+)(?:$|[-_])", None),
        # Claude families (supports both prefixed and unprefixed ids)
        (r"(anthropic/)?claude-haiku-4-5", "Haiku"),
        (r"(anthropic/)?claude-haiku-4\.5", "Haiku"),
        (r"(anthropic/)?claude-haiku-4", "Haiku"),
        (r"(anthropic/)?claude-(3-5|3\.5|3)-haiku", "Haiku"),
        (r"(anthropic/)?claude-opus-4", "Claude Opus 4"),
        (r"(anthropic/)?claude-sonnet-4", "Claude Sonnet 4"),
        (r"(anthropic/)?claude-4", "Claude 4"),
        (r"(anthropic/)?claude-(3-5|3\.5)-sonnet", "Claude 3.5 Sonnet"),
        (r"(anthropic/)?claude-(3-5|3\.5)", "Claude 3.5"),
        (r"(anthropic/)?claude-3-opus", "Claude 3 Opus"),
        (r"(anthropic/)?claude-3-sonnet", "Claude 3 Sonnet"),
        (r"(anthropic/)?claude-3", "Claude 3"),
        (r"(anthropic/)?claude-2", "Claude 2"),
        (r"claude", "Claude"),
        # Gemini families
        (r"(google/)?gemini-2\.5", "Gemini 2.5"),
        (r"(google/)?gemini-2", "Gemini 2"),
        (r"(google/)?gemini-1\.5", "Gemini 1.5"),
        (r"gemini", "Gemini"),
        (r"(xai/)?grok", "Grok"),
        (r"o\d", "O-Series"),
    ],
    "github_models": [
        # Codex family must come before generic numeric captures.
        (r"^openai/gpt-\d+(?:\.\d+)?-codex", "Codex"),
        (r"^openai/gpt-(\d+\.\d+)(?:$|[-_])", None),
        (r"^openai/gpt-(\d+)(?:$|[-_])", None),
        (r"^anthropic/claude-haiku-4-5", "Haiku"),
        (r"^anthropic/claude-haiku-4\.5", "Haiku"),
        (r"^anthropic/claude-haiku-4", "Haiku"),
        (r"^anthropic/claude-(3-5|3\.5|3)-haiku", "Haiku"),
        (r"^anthropic/claude-opus-4", "Claude Opus 4"),
        (r"^anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        (r"^anthropic/claude", "Claude"),
        (r"^google/gemini", "Gemini"),
        (r"^xai/grok", "Grok"),
        (r"^openai/o\d", "O-Series"),
    ],
}


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _expand_model_keys(model_id: str, provider: str) -> set[str]:
    """Generate key aliases so prefixed and unprefixed lookups both work."""
    keys = {model_id}

    if "/" in model_id:
        _, _, short_id = model_id.partition("/")
        if short_id:
            keys.add(short_id)
    elif provider in {
        "openai",
        "anthropic",
        "google",
        "xai",
        "github-copilot",
        "github_models",
        "github_copilot",
    }:
        keys.add(f"{provider}/{model_id}")

    return keys


def _infer_thinking_budget_support(model_info: dict[str, object]) -> bool | None:
    """Infer thinking-budget support from models.dev metadata.

    Only returns True when explicit budget-related fields are present.
    The generic ``reasoning`` flag is NOT used as a fallback because many
    reasoning-capable models (e.g. GPT-5.x) do not accept a
    ``thinking_budget`` parameter and will reject it with
    ``invalid_thinking_budget``.
    """
    explicit_budget_keys = [
        "thinking_budget",
        "supports_thinking_budget",
        "thinkingBudget",
        "max_thinking_budget",
        "min_thinking_budget",
    ]
    for key in explicit_budget_keys:
        value = model_info.get(key)
        if isinstance(value, bool):
            return value
        if _coerce_int(value) is not None:
            return True

    provider_obj = model_info.get("provider")
    if isinstance(provider_obj, dict):
        for key in explicit_budget_keys:
            value = provider_obj.get(key)
            if isinstance(value, bool):
                return value
            if _coerce_int(value) is not None:
                return True

    return None


def _candidate_lookup_keys(model_id: str) -> list[str]:
    """Build ranked lookup candidates for provider/version-normalized model IDs."""
    raw = str(model_id or "").strip()
    if not raw:
        return []

    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(raw)

    # Drop provider prefix for fallback matching.
    if "/" in raw:
        _, _, short_id = raw.partition("/")
        add(short_id)

    # Strip common date/version suffixes often appended by providers.
    raw_no_suffix = re.sub(r"[-_@](?:20\d{6}|v\d+:\d+)$", "", raw)
    add(raw_no_suffix)

    # Anthropic family normalization between forms like:
    # - claude-haiku-4-5-20251001
    # - claude-4-5-haiku
    for base in list(candidates):
        short = base.split("/", 1)[1] if "/" in base else base
        m = re.match(
            r"^(claude)-(haiku|sonnet|opus)-(\d+)(?:-(\d+))?(?:-\d{8})?$",
            short,
        )
        if m:
            family, major, minor = m.group(2), m.group(3), m.group(4)
            version = f"{major}-{minor}" if minor else major
            add(f"claude-{version}-{family}")
            add(f"claude-{family}-{version}")

        m2 = re.match(
            r"^(claude)-(\d+)(?:-(\d+))?-(haiku|sonnet|opus)(?:-\d{8})?$",
            short,
        )
        if m2:
            major, minor, family = m2.group(2), m2.group(3), m2.group(4)
            version = f"{major}-{minor}" if minor else major
            add(f"claude-{family}-{version}")
            add(f"claude-{version}-{family}")

    return candidates


def _best_match_value(cache: dict[str, int], model_id: str) -> int | None:
    """Return best cached value using deterministic ranked matching."""
    if not cache:
        return None

    candidates = _candidate_lookup_keys(model_id)
    if not candidates:
        return None

    best: tuple[int, int, int] | None = None
    best_value: int | None = None

    for candidate in candidates:
        c = candidate.lower()
        for key, value in cache.items():
            k = key.lower()
            score = 0
            if c == k:
                score = 1000
            elif c.startswith(k):
                score = 900
            elif k.startswith(c):
                score = 800
            elif k in c or c in k:
                score = 600
            if score == 0:
                continue

            rank = (score, len(k), len(c))
            if best is None or rank > best:
                best = rank
                best_value = value

    return best_value


def _best_match_flag(cache: dict[str, bool], model_id: str) -> bool | None:
    """Return best cached boolean flag using same lookup strategy as limits."""
    if not cache:
        return None

    candidates = _candidate_lookup_keys(model_id)
    if not candidates:
        return None

    best: tuple[int, int, int] | None = None
    best_value: bool | None = None

    for candidate in candidates:
        c = candidate.lower()
        for key, value in cache.items():
            k = key.lower()
            score = 0
            if c == k:
                score = 1000
            elif c.startswith(k):
                score = 900
            elif k.startswith(c):
                score = 800
            elif k in c or c in k:
                score = 600
            if score == 0:
                continue

            rank = (score, len(k), len(c))
            if best is None or rank > best:
                best = rank
                best_value = value

    return best_value


async def _fetch_models_dev_data() -> tuple[dict[str, int], dict[str, int]]:
    """Fetch model limits and capabilities from models.dev."""
    global _model_supports_function_calling, _model_supports_reasoning, _model_supports_thinking_budget
    limits: dict[str, int] = {}
    output_limits: dict[str, int] = {}
    function_calling: dict[str, bool] = {}
    reasoning_support: dict[str, bool] = {}
    thinking_budget_support: dict[str, bool] = {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(MODELS_DEV_API_URL)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                logger.warning("models.dev payload was not a dictionary")
                return {}, {}

            for provider, provider_payload in data.items():
                if not isinstance(provider_payload, dict):
                    continue
                models_obj = provider_payload.get("models", {})
                if not isinstance(models_obj, dict):
                    continue

                for fallback_id, model_info in models_obj.items():
                    if not isinstance(model_info, dict):
                        continue

                    model_id = str(model_info.get("id") or fallback_id or "").strip()
                    if not model_id:
                        continue

                    limit_info = model_info.get("limit", {})
                    if not isinstance(limit_info, dict):
                        limit_info = {}

                    context_limit = _coerce_int(limit_info.get("context"))
                    output_limit = _coerce_int(limit_info.get("output"))

                    supports_fc = model_info.get("tool_call")
                    supports_reasoning_flag = model_info.get("reasoning")
                    supports_thinking_budget_flag = _infer_thinking_budget_support(
                        model_info
                    )

                    key_variants = _expand_model_keys(model_id, str(provider).lower())

                    for key in key_variants:
                        if context_limit is not None:
                            limits[key] = context_limit
                        if output_limit is not None:
                            output_limits[key] = output_limit
                        if isinstance(supports_fc, bool):
                            function_calling[key] = supports_fc
                        if isinstance(supports_reasoning_flag, bool):
                            reasoning_support[key] = supports_reasoning_flag
                        if isinstance(supports_thinking_budget_flag, bool):
                            thinking_budget_support[key] = supports_thinking_budget_flag

            _model_supports_function_calling = function_calling
            _model_supports_reasoning = reasoning_support
            _model_supports_thinking_budget = thinking_budget_support

            logger.info(
                "Loaded %s context limits, %s output limits, %s function-calling flags, %s reasoning flags, %s thinking-budget flags from models.dev",
                len(limits),
                len(output_limits),
                len(_model_supports_function_calling),
                len(_model_supports_reasoning),
                len(_model_supports_thinking_budget),
            )
            return limits, output_limits

    except Exception as e:
        logger.warning(f"Failed to fetch models.dev model data: {e}")
        return {}, {}


async def _ensure_cache_loaded() -> None:
    """Ensure the cache is loaded (thread-safe)."""
    global _cache_loaded, _model_limits_cache, _model_output_limits_cache

    if _cache_loaded:
        return

    async with _cache_lock:
        # Double-check after acquiring lock
        if _cache_loaded:
            return

        # Try to fetch from models.dev
        fetched_limits, fetched_output = await _fetch_models_dev_data()

        if fetched_limits:
            _model_limits_cache = fetched_limits
            _model_output_limits_cache = fetched_output
        else:
            logger.info("Using empty cache as fetch failed")

        _cache_loaded = True


async def get_context_limit(model_id: str) -> int:
    """
    Get the context limit for a model.

    Tries exact match first, then partial match, then returns default.
    """
    await _ensure_cache_loaded()

    matched = _best_match_value(_model_limits_cache, model_id)
    if matched is not None:
        return matched

    return DEFAULT_CONTEXT_LIMIT


async def get_context_limits_batch(model_ids: list[str]) -> dict[str, int]:
    """Get context limits for multiple models at once."""
    await _ensure_cache_loaded()

    result = {}
    for model_id in model_ids:
        result[model_id] = await get_context_limit(model_id)
    return result


async def get_output_limit(model_id: str) -> int | None:
    """
    Get the output token limit for a model.

    Returns None if not found (let caller decide default).
    """
    await _ensure_cache_loaded()

    matched = _best_match_value(_model_output_limits_cache, model_id)
    if matched is not None:
        return matched

    return None


def update_model_limit(model_id: str, limit: int) -> None:
    """Update the context limit for a model in the runtime cache."""
    _model_limits_cache[model_id] = limit


def update_model_output_limit(model_id: str, limit: int) -> None:
    """Update the output limit for a model in the runtime cache."""
    _model_output_limits_cache[model_id] = limit


def update_model_function_calling(model_id: str, supports: bool) -> None:
    """Update function calling support for a model in the runtime cache."""
    _model_supports_function_calling[model_id] = supports


def invalidate_cache() -> None:
    """Invalidate the cache (forces re-fetch on next request)."""
    global _cache_loaded
    _cache_loaded = False
    _model_limits_cache.clear()
    _model_output_limits_cache.clear()
    _model_supports_function_calling.clear()
    _model_supports_reasoning.clear()
    _model_supports_thinking_budget.clear()
    _model_requires_responses_api.clear()
    _provider_supports_reasoning.clear()
    _provider_supports_thinking_budget.clear()


async def supports_function_calling(model_id: str) -> bool:
    """
    Check if a model supports function calling (indicates it's a chat model).

    Returns True if the model supports function calling, False otherwise.
    Uses models.dev metadata when available.
    """
    await _ensure_cache_loaded()

    matched = _best_match_flag(_model_supports_function_calling, model_id)
    if matched is not None:
        return matched

    # Default heuristics if not in LiteLLM data
    # OpenAI: gpt-* and o-series models support function calling (except whisper, dall-e, tts, embeddings)
    if (
        model_id.startswith("gpt-")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
    ):
        return not any(
            x in model_id.lower() for x in ["whisper", "dall-e", "tts", "embedding"]
        )

    # Anthropic: all claude models support function calling
    if "claude" in model_id.lower():
        return True

    # Conservative default: assume no function calling support
    return False


async def supports_reasoning(model_id: str) -> bool:
    """
    Check if a model supports reasoning/thinking tokens.

    Returns True when provider-reported metadata confirms support.
    """
    await _ensure_cache_loaded()

    matched = _best_match_flag(_provider_supports_reasoning, model_id)
    if matched is not None:
        return matched

    # Do not infer by model-name heuristics. Reasoning support must come from
    # provider-reported structured capability metadata.
    return False


async def supports_thinking_budget(model_id: str) -> bool:
    """Check if a model supports Copilot/OpenAI-style `thinking_budget`.

    Only returns True when provider-reported metadata confirms support.
    """
    await _ensure_cache_loaded()

    matched = _best_match_flag(_provider_supports_thinking_budget, model_id)
    if matched is not None:
        return matched

    return False


def register_model_supported_endpoints(
    model_id: str, supported_endpoints: list[str]
) -> None:
    """Register supported API endpoints for a model from provider API responses.

    Called when model metadata is fetched from provider-specific APIs
    (e.g. Copilot ``/models`` endpoint) that report which endpoints each model
    supports.
    """
    needs_responses = (
        "/responses" in supported_endpoints
        and "/chat/completions" not in supported_endpoints
    )
    supports_responses = "/responses" in supported_endpoints

    key_variants = {str(model_id).strip()}
    if "/" in model_id:
        _, _, short_id = model_id.partition("/")
        if short_id:
            key_variants.add(short_id)

    for key in key_variants:
        _model_requires_responses_api[key] = needs_responses
        _model_supports_responses_api[key] = supports_responses


def register_model_reasoning_capabilities(
    model_id: str,
    *,
    reasoning_supported: bool = False,
    thinking_budget_supported: bool = False,
) -> None:
    """Register reasoning-related capability flags from provider metadata.

    This supplements models.dev-derived flags with provider-native capabilities
    (for example Copilot ``/models`` payloads).
    """
    if not reasoning_supported and not thinking_budget_supported:
        return

    key_variants = {str(model_id).strip()}
    if "/" in model_id:
        _, _, short_id = model_id.partition("/")
        if short_id:
            key_variants.add(short_id)

    for key in key_variants:
        if reasoning_supported:
            _provider_supports_reasoning[key] = True
        if thinking_budget_supported:
            _provider_supports_thinking_budget[key] = True
        if reasoning_supported:
            _model_supports_reasoning[key] = True
        if thinking_budget_supported:
            _model_supports_thinking_budget[key] = True


async def requires_responses_api(model_id: str) -> bool:
    """Check if a model requires the OpenAI Responses API instead of Chat Completions.

    Returns True if the model is known to only support ``/responses`` and not
    ``/chat/completions``.  Data comes from provider model APIs (e.g. Copilot
    ``/models`` endpoint's ``supportedEndpoints`` field) or from a previous
    runtime fallback cached by ``register_model_supported_endpoints``.

    If no data is available, returns False — the caller should use
    ``/chat/completions`` and rely on the runtime auto-fallback in
    ``_CopilotChatOpenAI`` to catch ``unsupported_api_for_model`` errors and
    switch transparently.
    """
    await _ensure_cache_loaded()

    matched = _best_match_flag(_model_requires_responses_api, model_id)
    if matched is not None:
        return matched

    return False


async def supports_responses_api(model_id: str) -> bool:
    """Check if a model is known to support the Responses API endpoint."""
    await _ensure_cache_loaded()

    matched = _best_match_flag(_model_supports_responses_api, model_id)
    if matched is not None:
        return matched

    return False
