"""
Model context and capability limits fetched from models.dev.

Source: https://models.dev/api.json
"""

import asyncio
import re
from datetime import datetime, timezone

import httpx

from ragtime.core.logging import get_logger
from ragtime.core.model_providers import get_provider, normalize_provider_name

logger = get_logger(__name__)

# models.dev provider/model catalog (updated frequently)
MODELS_DEV_API_URL = "https://models.dev/api.json"

# Cache for model limits (populated on first request)
_model_limits_cache: dict[str, int] = {}
# Cache for model output limits
_model_output_limits_cache: dict[str, int] = {}
# Cache for model family/provider/display metadata
_model_family_labels_cache: dict[str, str] = {}
_model_provider_labels_cache: dict[str, str] = {}
_model_display_names_cache: dict[str, str] = {}
_model_freshness_cache: dict[str, int] = {}
# Cache for function calling support
_model_supports_function_calling: dict[str, bool] = {}
# Cache for reasoning/thinking support
_model_supports_reasoning: dict[str, bool] = {}
# Cache for reasoning effort parameter support
_model_supports_reasoning_effort: dict[str, bool] = {}
# Cache for thinking budget support
_model_supports_thinking_budget: dict[str, bool] = {}
# Cache for models requiring Responses API (populated from Copilot /models)
_model_requires_responses_api: dict[str, bool] = {}
# Cache for models that support Responses API (including dual-endpoint models)
_model_supports_responses_api: dict[str, bool] = {}
# Cache for provider-reported reasoning capabilities (authoritative)
_provider_supports_reasoning: dict[str, bool] = {}
# Cache for provider-reported reasoning effort support (authoritative)
_provider_supports_reasoning_effort: dict[str, bool] = {}
# Cache for provider-reported thinking-budget capabilities (authoritative)
_provider_supports_thinking_budget: dict[str, bool] = {}
# Cache for provider-reported or live-probed image input capabilities (authoritative)
_provider_supports_image_input: dict[str, bool] = {}
_cache_lock = asyncio.Lock()
_cache_loaded = False

# Default when model not found
DEFAULT_CONTEXT_LIMIT = 8192


PROVIDER_LABEL_OVERRIDES: dict[str, str] = {
    "ai21": "AI21",
    "arcee-ai": "Arcee AI",
    "baidu": "Baidu",
    "bytedance": "ByteDance",
    "bytedance-seed": "ByteDance Seed",
    "cohere": "Cohere",
    "cognitivecomputations": "Cognitive Computations",
    "deepseek": "DeepSeek",
    "deepcogito": "Deep Cogito",
    "essentialai": "Essential AI",
    "google": "Google",
    "ibm-granite": "IBM",
    "inclusionai": "Inclusion AI",
    "liquid": "LiquidAI",
    "meta-llama": "Meta",
    "minimax": "MiniMax",
    "mistralai": "Mistral",
    "moonshotai": "Moonshot AI",
    "nousresearch": "Nous Research",
    "nvidia": "NVIDIA",
    "perplexity": "Perplexity",
    "qwen": "Qwen",
    "rekaai": "Reka AI",
    "sao10k": "Sao10K",
    "thedrummer": "TheDrummer",
    "x-ai": "xAI",
    "xai": "xAI",
    "z-ai": "Z AI",
}

FAMILY_LABEL_OVERRIDES: dict[str, str] = {
    "ai21": "AI21",
    "arcee-ai": "Arcee AI",
    "claude": "Claude",
    "claude-haiku": "Claude Haiku",
    "claude-opus": "Claude Opus",
    "claude-sonnet": "Claude Sonnet",
    "cohere": "Cohere",
    "command": "Command",
    "codex": "Codex",
    "deepseek": "DeepSeek",
    "gemma": "Gemma",
    "glm": "GLM",
    "gpt": "GPT",
    "gpt-codex": "Codex",
    "gpt-mini": "GPT",
    "gpt-nano": "GPT",
    "gpt-oss": "GPT OSS",
    "gpt-pro": "GPT",
    "grok": "Grok",
    "granite": "Granite",
    "inflection": "Inflection",
    "kimi": "Moonshot",
    "laguna": "Laguna",
    "llama": "Llama",
    "lfm": "LFM",
    "mimo": "MiMo",
    "minimax": "MiniMax",
    "mistral": "Mistral",
    "mistral-large": "Mistral Large",
    "mistral-medium": "Mistral Medium",
    "mistral-nemo": "Mistral Nemo",
    "mistral-small": "Mistral Small",
    "moonshot": "Moonshot",
    "nemotron": "Nemotron",
    "nova": "Nova",
    "o": "O-Series",
    "o-mini": "O-Series",
    "o-pro": "O-Series",
    "o-series": "O-Series",
    "phi": "Phi",
    "qianfan": "Qianfan",
    "qwen": "Qwen",
    "reka": "Reka",
    "relace": "Relace",
    "router": "Router",
    "seed": "Seed",
}


HIGH_CONFIDENCE_TEXT_TAXONOMY_RULES: tuple[
    tuple[re.Pattern[str], str | None, str | None], ...
] = (
    (re.compile(r"\bcodex\b"), "OpenAI", "Codex"),
    (re.compile(r"\bqianfan\b"), "Baidu", "Qianfan"),
    (
        re.compile(r"\bbytedance[-\s]?seed[:/]\s*seed\b|\bbytedance-seed/seed\b"),
        "ByteDance Seed",
        "Seed",
    ),
    (re.compile(r"\bgranite\b"), "IBM", "Granite"),
    (re.compile(r"\blfm[-\s]?\d|\blfm2(?:\b|[-_.])"), "LiquidAI", "LFM"),
    (re.compile(r"\blaguna\b"), "Poolside", "Laguna"),
    (re.compile(r"\breka(?:[-\s]|$)"), "Reka AI", "Reka"),
    (re.compile(r"\brelace\b"), "Relace", "Relace"),
    (re.compile(r"\baion(?:[-\s]|labs|$)"), "Aion Labs", "Aion"),
    (re.compile(r"\bmimo\b"), "Xiaomi", "MiMo"),
    (re.compile(r"\binflection(?:[-\s]|/|$)"), "Inflection", "Inflection"),
    (re.compile(r"\bmorph(?:[-\s]|/|$)"), "Morph", "Morph"),
    (re.compile(r"\bmicrosoft/phi\b|\bphi[-\s]?\d"), "Microsoft", "Phi"),
)


TEXT_TAXONOMY_RULES: tuple[tuple[re.Pattern[str], str | None, str | None], ...] = (
    (re.compile(r"\bclaude\b.*\bhaiku\b|\bhaiku\b.*\bclaude\b"), "Anthropic", "Claude Haiku"),
    (re.compile(r"\bclaude\b.*\bopus\b|\bopus\b.*\bclaude\b"), "Anthropic", "Claude Opus"),
    (re.compile(r"\bclaude\b.*\bsonnet\b|\bsonnet\b.*\bclaude\b"), "Anthropic", "Claude Sonnet"),
    (re.compile(r"\bclaude\b"), "Anthropic", "Claude"),
    (re.compile(r"\b(?:gpt-\d|chatgpt-|openai/gpt)"), "OpenAI", "GPT"),
    (re.compile(r"\bo\d+(?:\b|[-_])"), "OpenAI", "O-Series"),
    (re.compile(r"\bgemini\b"), "Google", "Gemini"),
    (re.compile(r"\bgemma\b"), "Google", "Gemma"),
    (re.compile(r"\bmistral[-\s]small\b"), "Mistral", "Mistral Small"),
    (re.compile(r"\bmistral[-\s]medium\b"), "Mistral", "Mistral Medium"),
    (re.compile(r"\bmistral[-\s]large\b"), "Mistral", "Mistral Large"),
    (re.compile(r"\bmistral[-\s]nemo\b"), "Mistral", "Mistral Nemo"),
    (re.compile(r"\bmistral\b"), "Mistral", "Mistral"),
    (re.compile(r"\bmoonshot\b|\bkimi\b"), "Moonshot AI", "Moonshot"),
    (re.compile(r"\bqwen\b"), "Qwen", "Qwen"),
    (re.compile(r"\bdeepseek\b"), "DeepSeek", "DeepSeek"),
    (re.compile(r"\bllama\b"), "Meta", "Llama"),
    (re.compile(r"\bgrok\b|\bx-ai/|\bxai/"), "xAI", "Grok"),
    (re.compile(r"\bminimax\b|\bmini-max\b"), None, "MiniMax"),
    (re.compile(r"\bnemotron\b"), "NVIDIA", "Nemotron"),
    (re.compile(r"\bglm\b"), None, "GLM"),
    (re.compile(r"\bnova\b"), None, "Nova"),
    (re.compile(r"\bcohere\b|\bcommand-r\b"), "Cohere", "Cohere"),
    (re.compile(r"\barcee\b"), "Arcee AI", "Arcee AI"),
    (re.compile(r"\brouter\b|\bopenrouter/"), "OpenRouter", "Router"),
)


# Model family grouping patterns for UI organization
# Format: {provider: [(regex_pattern, group_name_or_None)]}
# If group_name is None, it uses the first capture group of the regex
# IMPORTANT: Patterns are matched in order, so more specific patterns must come first
MODEL_FAMILY_PATTERNS: dict[str, list[tuple[str, str | None]]] = {
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
        (r"claude-opus-4(?:[.-]\d+)?(?:\b|$)", "Claude Opus 4"),
        (r"claude-sonnet-4(?:[.-]\d+)?(?:\b|$)", "Claude Sonnet 4"),
        (r"claude-4", "Claude 4"),
        (r"claude-(3-5|3\.5)-sonnet", "Claude 3.5 Sonnet"),
        (r"claude-(3-5|3\.5)", "Claude 3.5"),
        (r"claude-3-opus", "Claude 3 Opus"),
        (r"claude-3-sonnet", "Claude 3 Sonnet"),
        (r"claude-3", "Claude 3"),
        (r"claude-2", "Claude 2"),
        (r"claude-([a-z0-9]+)-(\d+)(?:[.-]\d+)?(?:\b|$)", None),
        (r"claude", "Claude"),
    ],
    "ollama": [(r"^([a-z0-9]+)", None)],
    "llama_cpp": [(r"^([a-z0-9]+)", None)],
    "lmstudio": [(r"^([a-z0-9]+)", None)],
    "github_copilot": [
        # GitHub-hosted OpenAI families
        (r"^(openai/)?gpt-4o", "GPT-4o"),
        (r"^(?:openai/)?gpt-41-copilot", "GPT-4.1"),
        # Codex family must come before generic numeric captures.
        (r"^(?:openai/)?gpt-\d+(?:\.\d+)?-codex", "Codex"),
        (r"^(?:openai/)?gpt-(\d+\.\d+)(?:$|[-_])", None),
        (r"^(?:openai/)?gpt-(\d+)(?:$|[-_])", None),
        # Claude families (supports both prefixed and unprefixed ids)
        (r"(anthropic/)?claude-haiku-4-5", "Haiku"),
        (r"(anthropic/)?claude-haiku-4\.5", "Haiku"),
        (r"(anthropic/)?claude-haiku-4", "Haiku"),
        (r"(anthropic/)?claude-(3-5|3\.5|3)-haiku", "Haiku"),
        (r"(anthropic/)?claude-opus-4(?:[.-]\d+)?(?:\b|$)", "Claude Opus 4"),
        (r"(anthropic/)?claude-sonnet-4(?:[.-]\d+)?(?:\b|$)", "Claude Sonnet 4"),
        (r"(anthropic/)?claude-4", "Claude 4"),
        (r"(anthropic/)?claude-(3-5|3\.5)-sonnet", "Claude 3.5 Sonnet"),
        (r"(anthropic/)?claude-(3-5|3\.5)", "Claude 3.5"),
        (r"(anthropic/)?claude-3-opus", "Claude 3 Opus"),
        (r"(anthropic/)?claude-3-sonnet", "Claude 3 Sonnet"),
        (r"(anthropic/)?claude-3", "Claude 3"),
        (r"(anthropic/)?claude-2", "Claude 2"),
        (r"(?:anthropic/)?claude-([a-z0-9]+)-(\d+)(?:[.-]\d+)?(?:\b|$)", None),
        (r"claude", "Claude"),
        # Gemini families are captured by major version (e.g. 3.1 -> Gemini 3).
        (r"(?:google/)?gemini-(\d+)(?:\.\d+)?(?:$|[-_])", None),
        (r"gemini", "Gemini"),
        (r"(xai/)?grok", "Grok"),
        (r"^(?:openai/)?o\d+(?:$|[-_])", "O-Series"),
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
        (r"^anthropic/claude-opus-4(?:[.-]\d+)?(?:\b|$)", "Claude Opus 4"),
        (r"^anthropic/claude-sonnet-4(?:[.-]\d+)?(?:\b|$)", "Claude Sonnet 4"),
        (r"^anthropic/claude-([a-z0-9]+)-(\d+)(?:[.-]\d+)?(?:\b|$)", None),
        (r"^anthropic/claude", "Claude"),
        (r"^google/gemini-(\d+)(?:\.\d+)?(?:$|[-_])", None),
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


def _cache_key(value: str) -> str:
    return str(value or "").strip().lstrip("~").lower()


def _provider_label_from_slug(value: str | None) -> str | None:
    slug = str(value or "").strip().lstrip("~").lower()
    if not slug:
        return None
    descriptor = get_provider(slug)
    if descriptor and descriptor.label:
        return descriptor.label
    if slug in PROVIDER_LABEL_OVERRIDES:
        return PROVIDER_LABEL_OVERRIDES[slug]
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", slug) if part)


def _label_prefix_pattern(label: str) -> str | None:
    parts = [part for part in re.split(r"[-_\s]+", str(label or "").strip()) if part]
    if not parts:
        return None
    flexible_label = r"[-_.\s]*".join(re.escape(part) for part in parts)
    return rf"^{flexible_label}(?:\s*[:/-]\s*|[-_\s]+|(?=\d)|$)"


def _compact_label_key(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _strip_model_label_prefix(value: str, label: str) -> str:
    cleaned = value.strip()
    label_key = _compact_label_key(label)
    colon_match = re.match(r"^([^:]{1,40}):\s*", cleaned)
    if colon_match and label_key:
        prefix_key = _compact_label_key(colon_match.group(1))
        if prefix_key and (
            label_key.startswith(prefix_key) or prefix_key.startswith(label_key)
        ):
            stripped = cleaned[colon_match.end() :].strip()
            return stripped or cleaned

    pattern = _label_prefix_pattern(label)
    if not pattern:
        return cleaned
    stripped = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
    return stripped or cleaned


def _family_slug_to_label(value: str | None) -> str | None:
    slug = str(value or "").strip().lower().replace("_", "-")
    if not slug:
        return None
    if slug.startswith("gemini"):
        return "Gemini"
    if slug in FAMILY_LABEL_OVERRIDES:
        return FAMILY_LABEL_OVERRIDES[slug]
    return " ".join(
        part.upper() if part in {"gpt", "glm"} else part.capitalize()
        for part in slug.split("-")
    )


def _parse_date_rank(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _metadata_lookup_candidates(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> list[str]:
    candidates: list[str] = []

    def add(value: object) -> None:
        raw = str(value or "").strip()
        if not raw:
            return
        raw = raw.lstrip("~")
        variants = [raw]
        if raw.endswith(":free"):
            variants.append(raw.removesuffix(":free"))
        if "/" in raw:
            variants.append(raw.split("/", 1)[1])
        for variant in variants:
            for key in _candidate_lookup_keys(variant):
                normalized = _cache_key(key)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)

    add(model_id)
    if metadata:
        add(metadata.get("id"))
        add(metadata.get("canonical_slug"))
    add(name)
    return candidates


def _lookup_metadata_value(
    cache: dict[str, str] | dict[str, int],
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
    *,
    allow_fuzzy: bool = True,
) -> str | int | None:
    if not cache:
        return None

    candidates = _metadata_lookup_candidates(model_id, name, metadata)
    for candidate in candidates:
        if candidate in cache:
            return cache[candidate]

    if not allow_fuzzy:
        return None

    best: tuple[int, int] | None = None
    best_value: str | int | None = None
    for candidate in candidates:
        if len(candidate) < 4:
            continue
        for key, value in cache.items():
            if len(key) < 4:
                continue
            score = 0
            if candidate.startswith(key):
                score = 900
            elif key.startswith(candidate):
                score = 800
            elif key in candidate or candidate in key:
                score = 500
            if score == 0:
                continue
            rank = (score, min(len(key), len(candidate)))
            if best is None or rank > best:
                best = rank
                best_value = value

    return best_value


def _format_family_label(
    family_slug: str,
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    label = _family_slug_to_label(family_slug)
    if not label:
        return None
    high_confidence = _infer_high_confidence_taxonomy_from_text(model_id, name, metadata)[1]
    if high_confidence and label in {"GPT", "Other", "Liquid"}:
        return high_confidence
    if label == "Gemini":
        major = _extract_major_version("gemini", model_id, name, metadata)
        if major:
            return f"Gemini {major}"
    return label


def _infer_high_confidence_taxonomy_from_text(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[str | None, str | None]:
    text = " ".join(_metadata_text_values(model_id, name, metadata)).lower()
    if not text:
        return None, None

    for pattern, provider_label, family_label in HIGH_CONFIDENCE_TEXT_TAXONOMY_RULES:
        if pattern.search(text):
            return provider_label, family_label

    return None, None


def _metadata_text_values(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> list[str]:
    values = [model_id, name or ""]
    if metadata:
        values.extend(
            str(metadata.get(key) or "") for key in ("id", "name", "canonical_slug")
        )
    return [value.strip() for value in values if value and str(value).strip()]


def _extract_major_version(
    family: str,
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    family_pattern = re.escape(family.lower())
    pattern = re.compile(rf"\b{family_pattern}[-\s]?(\d+)(?:\.\d+)?")
    for value in _metadata_text_values(model_id, name, metadata):
        match = pattern.search(value.lower())
        if match:
            return match.group(1)
    return None


def _infer_taxonomy_from_text(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[str | None, str | None]:
    high_confidence = _infer_high_confidence_taxonomy_from_text(model_id, name, metadata)
    if high_confidence != (None, None):
        return high_confidence

    text = " ".join(_metadata_text_values(model_id, name, metadata)).lower()
    if not text:
        return None, None

    for pattern, provider_label, family_label in TEXT_TAXONOMY_RULES:
        if not pattern.search(text):
            continue
        if family_label == "Gemini":
            major = _extract_major_version("gemini", model_id, name, metadata)
            family_label = f"Gemini {major}" if major else "Gemini"
        return provider_label, family_label

    return None, None


def _infer_family_label_from_text(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    _provider_label, family_label = _infer_taxonomy_from_text(model_id, name, metadata)
    return family_label


def _infer_provider_label_from_text(
    model_id: str,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    provider_label, _family_label = _infer_taxonomy_from_text(model_id, name, metadata)
    return provider_label


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


def _capability_lookup_keys(model_id: str) -> list[str]:
    """Build exact alias keys for capability/endpoint cache lookups.

    Capability flags such as reasoning support and supported endpoints should
    not bleed across nearby model IDs like ``claude-haiku-4.5`` and
    ``claude-haiku-4.5-fast``.  Use only exact semantic aliases derived from
    normalization, not fuzzy prefix/substring matching.
    """
    return _candidate_lookup_keys(model_id)


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


def _lookup_capability_flag(cache: dict[str, bool], model_id: str) -> bool | None:
    """Return a cached capability flag using exact normalized alias matching."""
    if not cache:
        return None

    for candidate in _capability_lookup_keys(model_id):
        matched = cache.get(candidate)
        if isinstance(matched, bool):
            return matched

        lower_candidate = candidate.lower()
        for key, value in cache.items():
            if key.lower() == lower_candidate:
                return value

    return None


async def _fetch_models_dev_data() -> tuple[dict[str, int], dict[str, int]]:
    """Fetch model limits and capabilities from models.dev."""
    global _model_supports_function_calling, _model_supports_reasoning, _model_supports_reasoning_effort, _model_supports_thinking_budget
    global _model_family_labels_cache, _model_provider_labels_cache, _model_display_names_cache, _model_freshness_cache
    limits: dict[str, int] = {}
    output_limits: dict[str, int] = {}
    family_labels: dict[str, str] = {}
    provider_labels: dict[str, str] = {}
    display_names: dict[str, str] = {}
    freshness: dict[str, int] = {}
    function_calling: dict[str, bool] = {}
    reasoning_support: dict[str, bool] = {}
    reasoning_effort_support: dict[str, bool] = {}
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
                    family_slug = str(model_info.get("family") or "").strip()
                    model_name = str(model_info.get("name") or "").strip()
                    provider_label = _provider_label_from_slug(str(provider))
                    freshness_rank = _parse_date_rank(
                        model_info.get("release_date")
                    ) or _parse_date_rank(model_info.get("last_updated"))

                    key_variants = _expand_model_keys(model_id, str(provider).lower())
                    key_variants.add(str(fallback_id or ""))
                    if model_name:
                        key_variants.add(model_name)

                    for key in key_variants:
                        normalized_key = _cache_key(key)
                        if not normalized_key:
                            continue
                        if context_limit is not None:
                            limits[key] = context_limit
                        if output_limit is not None:
                            output_limits[key] = output_limit
                        if family_slug:
                            family_labels[normalized_key] = family_slug
                        if provider_label:
                            provider_labels[normalized_key] = provider_label
                        if model_name:
                            display_names[normalized_key] = model_name
                        if freshness_rank is not None:
                            freshness[normalized_key] = freshness_rank
                        if isinstance(supports_fc, bool):
                            function_calling[key] = supports_fc
                        if isinstance(supports_reasoning_flag, bool):
                            reasoning_support[key] = supports_reasoning_flag
                            reasoning_effort_support[key] = supports_reasoning_flag
                        if isinstance(supports_thinking_budget_flag, bool):
                            thinking_budget_support[key] = supports_thinking_budget_flag

            _model_supports_function_calling = function_calling
            _model_supports_reasoning = reasoning_support
            _model_supports_reasoning_effort = reasoning_effort_support
            _model_supports_thinking_budget = thinking_budget_support
            _model_family_labels_cache = family_labels
            _model_provider_labels_cache = provider_labels
            _model_display_names_cache = display_names
            _model_freshness_cache = freshness

            logger.info(
                "Loaded %s context limits, %s output limits, %s family labels, %s freshness entries, %s function-calling flags, %s reasoning flags, %s thinking-budget flags from models.dev",
                len(limits),
                len(output_limits),
                len(_model_family_labels_cache),
                len(_model_freshness_cache),
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


async def ensure_model_metadata_loaded() -> None:
    """Load models.dev metadata used for grouping, display labels, and freshness."""
    await _ensure_cache_loaded()


def resolve_model_provider_label(
    model_id: str,
    name: str | None = None,
    *,
    provider: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    """Resolve the model publisher/provider label from catalog metadata."""
    if metadata:
        raw_name = str(metadata.get("name") or name or "").strip()
        if ":" in raw_name:
            prefix, _, _rest = raw_name.partition(":")
            prefix = prefix.strip()
            if prefix and len(prefix) <= 40:
                return prefix

    raw_id = str(model_id or "").strip().lstrip("~")
    if "/" in raw_id:
        publisher, _, _short_id = raw_id.partition("/")
        label = _provider_label_from_slug(publisher)
        if label:
            return label

    descriptor = get_provider(provider)
    if descriptor and provider in {"openai", "anthropic"}:
        return descriptor.label

    inferred = _infer_provider_label_from_text(model_id, name, metadata)
    if inferred:
        return inferred

    cached = _lookup_metadata_value(
        _model_provider_labels_cache,
        model_id,
        name,
        metadata,
        allow_fuzzy=False,
    )
    if isinstance(cached, str) and cached:
        return cached

    return None


def resolve_model_family_label(
    model_id: str,
    name: str | None = None,
    *,
    provider: str | None = None,
    metadata: dict[str, object] | None = None,
) -> str | None:
    """Resolve a stable model-family label for UI grouping."""
    high_confidence = _infer_high_confidence_taxonomy_from_text(model_id, name, metadata)[1]
    if high_confidence:
        return high_confidence

    cached = _lookup_metadata_value(_model_family_labels_cache, model_id, name, metadata)
    if isinstance(cached, str) and cached:
        formatted = _format_family_label(cached, model_id, name, metadata)
        if formatted:
            return formatted

    inferred = _infer_family_label_from_text(model_id, name, metadata)
    if inferred:
        return inferred

    descriptor = get_provider(provider)
    if descriptor and descriptor.model_family_tokenizer_labels and metadata:
        architecture = metadata.get("architecture")
        tokenizer = architecture.get("tokenizer") if isinstance(architecture, dict) else None
        normalized_tokenizer = str(tokenizer or "").strip().lower()
        for token_value, label in descriptor.model_family_tokenizer_labels:
            if normalized_tokenizer == token_value.lower():
                return label

    return None


def get_model_freshness_rank(
    model_id: str,
    created: int | None = None,
    *,
    name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> int | None:
    """Return a comparable freshness rank from models.dev or provider metadata."""
    cached = _lookup_metadata_value(_model_freshness_cache, model_id, name, metadata)
    if isinstance(cached, int):
        return cached
    if metadata:
        for key in ("created", "release_date", "last_updated"):
            parsed = _parse_date_rank(metadata.get(key))
            if parsed is not None:
                return parsed
    if isinstance(created, int) and created > 0:
        return created
    return None


def clean_model_display_name(
    name: str,
    *,
    model_id: str | None = None,
    provider_label: str | None = None,
    family_label: str | None = None,
) -> str:
    """Strip redundant provider/family prefixes from a model display name."""
    cleaned = str(name or model_id or "").strip()
    if not cleaned:
        return str(model_id or "").strip()

    labels = [provider_label or ""]
    for label in labels:
        label = label.strip()
        if not label:
            continue
        next_cleaned = _strip_model_label_prefix(cleaned, label)
        if next_cleaned != cleaned:
            cleaned = next_cleaned
            continue

    family = str(family_label or "").strip()
    if family:
        remainder = _strip_model_label_prefix(cleaned, family)
        if remainder == cleaned:
            suffix_pattern = rf"(?:^|[-_.\s]+){re.escape(family)}(?P<suffix>\s*\([^)]*\))?$"
            if re.search(suffix_pattern, cleaned, flags=re.IGNORECASE):
                remainder = re.sub(
                    suffix_pattern,
                    lambda match: match.group("suffix") or "",
                    cleaned,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()
        if remainder == cleaned:
            family_parts = [part for part in re.split(r"[-_\s]+", family) if part]
            if len(family_parts) > 1:
                without_first = _strip_model_label_prefix(cleaned, family_parts[0])
                last_part = family_parts[-1]
                suffix_pattern = rf"(?:^|[-_.\s]+){re.escape(last_part)}(?P<suffix>\s*\([^)]*\))?$"
                if without_first != cleaned and re.search(
                    suffix_pattern,
                    without_first,
                    flags=re.IGNORECASE,
                ):
                    remainder = re.sub(
                        suffix_pattern,
                        lambda match: match.group("suffix") or "",
                        without_first,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()
        if remainder and not remainder.startswith("."):
            cleaned = remainder

    return cleaned or str(model_id or name or "").strip()


def _normalize_model_label_part(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def _model_labels_match(existing: str | None, candidate: str | None) -> bool:
    existing_key = _compact_label_key(existing)
    candidate_key = _compact_label_key(candidate)
    if not existing_key or not candidate_key:
        return False
    return existing_key == candidate_key or (
        len(candidate_key) > 2 and existing_key.endswith(candidate_key)
    )


def _append_distinct_model_label(parts: list[str], value: str | None) -> None:
    label = _normalize_model_label_part(value)
    if not label or label.casefold().startswith("other"):
        return
    if any(_model_labels_match(existing, label) for existing in parts):
        return
    parts.append(label)


def _model_slug_display_name(model_id: str | None) -> str:
    raw = str(model_id or "").strip().lstrip("/")
    if not raw:
        return ""
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return format_model_display_label(raw.replace(":", " "), model_id=raw)


def compose_model_display_label(
    *,
    model_id: str | None = None,
    provider_label: str | None = None,
    family_label: str | None = None,
    display_name: str | None = None,
    fallback_name: str | None = None,
) -> str:
    """Compose provider, family, and display metadata into one stable label."""
    provider = _normalize_model_label_part(provider_label)
    raw_family = _normalize_model_label_part(family_label)
    raw_display = (
        _normalize_model_label_part(display_name)
        or _normalize_model_label_part(fallback_name)
        or _model_slug_display_name(model_id)
    )

    family = ""
    if raw_family:
        family = format_model_display_label(raw_family, provider_label=provider)
        provider_key = _compact_label_key(provider)
        family_key = _compact_label_key(family)
        display_key = _compact_label_key(raw_display)
        family_parts = [part for part in family.split() if part]
        if provider_key and family_key and (
            provider_key.startswith(family_key) or provider_key.endswith(family_key)
        ):
            family = ""
        elif (
            len(family_parts) > 1
            and display_key.startswith(_compact_label_key(family_parts[0]))
            and not display_key.startswith(family_key)
        ):
            family = ""

    display = clean_model_display_name(
        raw_display,
        model_id=model_id,
        provider_label=provider,
        family_label=raw_family,
    )
    if family:
        display = clean_model_display_name(display, model_id=model_id, family_label=family)
    display = format_model_display_label(display, model_id=model_id)

    parts: list[str] = []
    _append_distinct_model_label(parts, provider)
    _append_distinct_model_label(parts, family)
    _append_distinct_model_label(parts, display)
    return _normalize_model_label_part(" ".join(parts)) or _model_slug_display_name(model_id)


def format_model_display_label(
    name: str,
    *,
    model_id: str | None = None,
    provider_label: str | None = None,
    family_label: str | None = None,
) -> str:
    """Return a human-readable model label after shared prefix cleanup."""
    cleaned = clean_model_display_name(
        name,
        model_id=model_id,
        provider_label=provider_label,
        family_label=family_label,
    )
    if not cleaned:
        cleaned = str(model_id or name or "").strip()
    return _format_model_display_text(cleaned)


def _format_model_display_text(value: str) -> str:
    raw = str(value or "")
    dates: dict[str, str] = {}

    def _protect_date(match: re.Match[str]) -> str:
        placeholder = f"DATEPLACEHOLDER{len(dates)}"
        dates[placeholder] = match.group(0)
        return placeholder

    raw = re.sub(r"20\d{2}-\d{2}-\d{2}", _protect_date, raw)
    normalized = re.sub(r"\s+", " ", raw.replace("_", " ").replace(":", " ")).strip()
    if not normalized:
        return ""

    tokens: list[str] = []
    for raw_token in normalized.split(" "):
        token = raw_token.strip()
        if not token:
            continue
        if token in dates:
            tokens.append(dates[token])
            continue
        pieces = [piece for piece in re.split(r"[-/]+", token) if piece]
        tokens.extend(dates.get(piece, piece) for piece in (pieces or [token]))

    return " ".join(_format_model_variant_part(token) for token in tokens if token)


def clean_model_variant_label(
    name: str,
    *,
    model_id: str | None = None,
    provider_label: str | None = None,
    family_label: str | None = None,
) -> str:
    """Return a compact variant label for grouping model versions in selectors."""
    variant = clean_model_display_name(
        name,
        model_id=model_id,
        provider_label=provider_label,
        family_label=family_label,
    )
    variant = re.sub(r"^(?:gpt|claude)[-_\s]+", "", variant, flags=re.IGNORECASE)
    variant = re.sub(
        r"[-_\s]*(?:20\d{2}-\d{2}-\d{2}|20\d{6})$",
        "",
        variant,
        flags=re.IGNORECASE,
    )
    variant = re.sub(r"[-_\s]+latest$", "", variant, flags=re.IGNORECASE)

    if str(family_label or "").strip().lower() == "codex":
        variant = re.sub(
            r"(^|[-_\s])codex(?=$|[-_\s])",
            r"\1",
            variant,
            flags=re.IGNORECASE,
        )

    variant = re.sub(r"^[-_\s]+|[-_\s]+$", "", variant).strip()
    if not variant:
        variant = str(model_id or name or "").strip()

    return " ".join(
        _format_model_variant_part(part) for part in re.split(r"[-_\s]+", variant) if part
    )


def _format_model_variant_part(value: str) -> str:
    lower = value.lower()
    if lower in {"api", "glm", "gpt", "lfm", "llm", "mlx", "ocr", "oss", "ui", "vl"}:
        return lower.upper()
    if re.match(r"^qwen\d", value, flags=re.IGNORECASE):
        return "Qwen" + value[4:]
    if lower == "mimo":
        return "MiMo"
    if lower == "codellama":
        return "CodeLLaMa"
    if lower == "deepresearch":
        return "Deep Research"
    if re.match(r"^o\d", value, flags=re.IGNORECASE):
        return lower
    if re.match(r"^\d+(?:b|k|m|t)$", value, flags=re.IGNORECASE):
        return value.upper()
    if re.match(r"^[a-z]+\d+[a-z0-9]*$", value, flags=re.IGNORECASE):
        return value.upper()
    if re.match(r"^(?:\d|o\d)", value, flags=re.IGNORECASE):
        return value
    return lower[:1].upper() + lower[1:]


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


def extract_openrouter_model_limits(
    row: dict[str, object]
) -> tuple[int | None, int | None]:
    """Extract OpenRouter context/output limits from structured API fields."""
    top_provider = row.get("top_provider")
    top_provider = top_provider if isinstance(top_provider, dict) else {}

    context_limit = _coerce_int(row.get("context_length")) or _coerce_int(
        top_provider.get("context_length")
    )
    output_limit = _coerce_int(top_provider.get("max_completion_tokens")) or _coerce_int(
        row.get("max_completion_tokens")
    )
    return context_limit, output_limit


def register_openrouter_model_limits(
    row: dict[str, object]
) -> tuple[int | None, int | None]:
    """Register OpenRouter model limits from the provider catalog response."""
    model_id = str(row.get("id") or "").strip()
    context_limit, output_limit = extract_openrouter_model_limits(row)

    if model_id and context_limit is not None:
        update_model_limit(model_id, context_limit)
    if model_id and output_limit is not None:
        update_model_output_limit(model_id, output_limit)

    return context_limit, output_limit


def update_model_function_calling(model_id: str, supports: bool) -> None:
    """Update function calling support for a model in the runtime cache."""
    _model_supports_function_calling[model_id] = supports


def invalidate_cache() -> None:
    """Invalidate the cache (forces re-fetch on next request)."""
    global _cache_loaded
    _cache_loaded = False
    _model_limits_cache.clear()
    _model_output_limits_cache.clear()
    _model_family_labels_cache.clear()
    _model_provider_labels_cache.clear()
    _model_display_names_cache.clear()
    _model_freshness_cache.clear()
    _model_supports_function_calling.clear()
    _model_supports_reasoning.clear()
    _model_supports_reasoning_effort.clear()
    _model_supports_thinking_budget.clear()
    _model_requires_responses_api.clear()
    _provider_supports_reasoning.clear()
    _provider_supports_reasoning_effort.clear()
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

    matched = _lookup_capability_flag(_provider_supports_reasoning, model_id)
    if matched is not None:
        return matched

    # Do not infer by model-name heuristics. Reasoning support must come from
    # provider-reported structured capability metadata.
    return False


async def supports_reasoning_effort(model_id: str) -> bool:
    """Check if a model supports the ``reasoning_effort`` request parameter."""
    await _ensure_cache_loaded()

    matched = _lookup_capability_flag(_provider_supports_reasoning_effort, model_id)
    if matched is not None:
        return matched

    matched = _lookup_capability_flag(_model_supports_reasoning_effort, model_id)
    if matched is not None:
        return matched

    return False


async def supports_thinking_budget(model_id: str) -> bool:
    """Check if a model supports Copilot/OpenAI-style `thinking_budget`.

    Only returns True when provider-reported metadata confirms support.
    """
    await _ensure_cache_loaded()

    matched = _lookup_capability_flag(_provider_supports_thinking_budget, model_id)
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
    reasoning_effort_supported: bool = False,
    thinking_budget_supported: bool = False,
) -> None:
    """Register reasoning-related capability flags from provider metadata.

    This supplements models.dev-derived flags with provider-native capabilities
    (for example Copilot ``/models`` payloads).
    """
    if (
        not reasoning_supported
        and not reasoning_effort_supported
        and not thinking_budget_supported
    ):
        return

    key_variants = {str(model_id).strip()}
    if "/" in model_id:
        _, _, short_id = model_id.partition("/")
        if short_id:
            key_variants.add(short_id)

    for key in key_variants:
        if reasoning_supported:
            _provider_supports_reasoning[key] = True
        if reasoning_effort_supported:
            _provider_supports_reasoning_effort[key] = True
        elif reasoning_supported:
            _provider_supports_reasoning_effort[key] = False
        if thinking_budget_supported:
            _provider_supports_thinking_budget[key] = True
        if reasoning_supported:
            _model_supports_reasoning[key] = True
        if reasoning_effort_supported:
            _model_supports_reasoning_effort[key] = True
        if thinking_budget_supported:
            _model_supports_thinking_budget[key] = True


def register_model_image_input_capability(
    model_id: str,
    supported: bool,
) -> None:
    """Register image-input capability from provider metadata or a live probe.

    This intentionally does not infer from model-name patterns. Callers should
    only pass data reported by a provider API or confirmed by a request that
    includes an image input.
    """
    normalized_model_id = str(model_id or "").strip()
    if not normalized_model_id:
        return

    key_variants = {normalized_model_id}
    if "/" in normalized_model_id:
        _, _, short_id = normalized_model_id.partition("/")
        if short_id:
            key_variants.add(short_id)

    for key in key_variants:
        _provider_supports_image_input[key] = bool(supported)


def register_model_input_modalities(
    model_id: str,
    modalities: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Register provider-reported input modalities for a model."""
    normalized = {str(modality or "").strip().lower() for modality in modalities}
    if not normalized:
        return
    register_model_image_input_capability(
        model_id,
        bool(normalized & {"image", "images", "vision"}),
    )


async def supports_image_input(model_id: str) -> bool:
    """Check if provider metadata or a live probe confirms image input support."""
    await _ensure_cache_loaded()

    matched = _lookup_capability_flag(_provider_supports_image_input, model_id)
    if matched is not None:
        return matched

    return False


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

    matched = _lookup_capability_flag(_model_requires_responses_api, model_id)
    if matched is not None:
        return matched

    return False


async def supports_responses_api(model_id: str) -> bool:
    """Check if a model is known to support the Responses API endpoint."""
    await _ensure_cache_loaded()

    matched = _lookup_capability_flag(_model_supports_responses_api, model_id)
    if matched is not None:
        return matched

    return False
