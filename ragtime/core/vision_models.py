"""
Vision model detection and OCR functionality.

Provides functions to:
- Detect vision-capable models from an Ollama server via capabilities API
- Detect vision-capable models from provider or catalog metadata
- Perform OCR using multimodal vision models
- Image preprocessing (resize, format conversion) for efficient processing
- Image classification for non-text images
- Manage vision model metadata

This module uses centralized capability detection from ollama.py.
Detection uses Ollama's /api/show 'capabilities' array which is the authoritative
source for model capabilities (e.g., ["completion", "vision", "tools"]).
"""

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from langchain_core.messages import HumanMessage
from PIL import Image

from ragtime.core import llama_cpp, lmstudio, omlx, openrouter
from ragtime.core.app_setting_defaults import DEFAULT_OCR_CONCURRENCY_LIMIT
from ragtime.core.app_settings import get_app_settings
from ragtime.core.claude_code_chat import ClaudeCodeChatModel
from ragtime.core.file_constants import RAW_CAMERA_EXTENSIONS
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import get_output_limit, register_model_image_input_capability
from ragtime.core.model_providers import normalize_provider_name, resolve_provider_api_key
from ragtime.core.ollama import (
    extract_capabilities,
    get_model_details,
    is_vision_model_by_capability,
    supports_structured_output,
)
from ragtime.core.openai_codex_auth import OPENAI_CODEX_DEFAULT_BASE_URL, ensure_openai_codex_token_fresh
from ragtime.core.openai_codex_transport import build_codex_responses_request, codex_response_headers

logger = get_logger(__name__)

# Default max dimension for image resizing (pixels on longest edge)
DEFAULT_MAX_IMAGE_DIMENSION = 2048
# JPEG quality for resized images (balance between size and quality)
DEFAULT_JPEG_QUALITY = 85
ANTHROPIC_MAX_ENCODED_IMAGE_BYTES = 5 * 1024 * 1024
OPENAI_COMPATIBLE_MAX_ENCODED_IMAGE_BYTES = 20 * 1024 * 1024
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_COMPATIBLE_VISION_PROVIDERS = {"openai", "openrouter", "omlx", "lmstudio", "llama_cpp"}
MODELS_DEV_API_URL = "https://models.dev/api.json"
MODELS_DEV_VISION_CACHE_TTL_SECONDS = 3600.0
VISION_CAPABILITY_TOKENS = {"vision", "image", "images", "image_input", "multimodal"}
VISION_MODEL_TYPE_TOKENS = {"vlm"}
TINY_PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
ANTHROPIC_VERSION = "2023-06-01"
OCR_OUTPUT_BUDGET_CEILING = 8192
OCR_FALLBACK_OUTPUT_BUDGET = 4096
OCR_OUTPUT_BUDGET_CACHE_TTL_SECONDS = MODELS_DEV_VISION_CACHE_TTL_SECONDS
OLLAMA_OCR_CONTEXT_LENGTH = 16384
OLLAMA_OCR_THINK = False
DEFAULT_VISION_OCR_TIMEOUT = 180.0
# Per-model budgets; memoized so per-page OCR never re-triggers catalog lookups when models.dev is unavailable.
_ocr_output_budget_cache: dict[str, tuple[float, int]] = {}


async def _resolve_ocr_output_budget(model: str) -> int:
    """Bound OCR output by the model's known output cap, or use the conservative fallback."""
    now = time.monotonic()
    cached = _ocr_output_budget_cache.get(model)
    if cached is not None and now - cached[0] < OCR_OUTPUT_BUDGET_CACHE_TTL_SECONDS:
        return cached[1]
    budget = await _lookup_ocr_output_budget(model)
    _ocr_output_budget_cache[model] = (now, budget)
    return budget


async def _lookup_ocr_output_budget(model: str) -> int:
    try:
        known_limit = await get_output_limit(model)
    except Exception:
        logger.debug("Output limit lookup failed for %s; using OCR fallback budget", model, exc_info=True)
        return OCR_FALLBACK_OUTPUT_BUDGET
    if known_limit is None or known_limit <= 0:
        return OCR_FALLBACK_OUTPUT_BUDGET
    return min(OCR_OUTPUT_BUDGET_CEILING, known_limit)


class _VisionOcrLimiter:
    """A capacity-adjustable, cancellation-safe limiter for OCR requests."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.active = 0
        self.condition = asyncio.Condition()

    async def acquire(self, capacity: int) -> None:
        async with self.condition:
            self.capacity = capacity
            while self.active >= self.capacity:
                await self.condition.wait()
            self.active += 1

    async def release(self) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()


_vision_ocr_limiter: _VisionOcrLimiter | None = None


# =============================================================================
# PROMPTS
# =============================================================================

# Prompt for classification-only (used by classify_image)
CLASSIFICATION_PROMPT = """Analyze this image and provide a structured description for search indexing. Return valid JSON.

Include these fields:
- image_type: photo, screenshot, diagram, chart, illustration, logo, etc.
- subject: main subject or focus
- description: brief factual description
- objects: list of identifiable objects, people, animals
- colors: list of dominant colors
- setting: location, environment, or context
- tags: list of search keywords

Be factual and objective. Include specific details useful for search."""

# JSON schema for structured classification output
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_type": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "colors": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["image_type", "subject", "description", "objects", "colors", "setting", "tags"],
}

# Prompt for OCR with classification (structured JSON output)
OCR_WITH_CLASSIFICATION_PROMPT = """Analyze this image. Extract all visible text AND describe the image for search indexing. Return valid JSON.

For extracted_text: list each text block with its type (Title, NarrativeText, ListItem, Table, Code, Handwritten, Uncategorized) and the exact text content.
For tables, emit Markdown with a header row, separator row, every source row in order, and empty cells retained. Preserve exact digits, units, clause identifiers, and uncertain text using explicit `[illegible]` markers.

For image analysis: describe the image type, subject, description, objects, setting, and search tags.

If there is no text in the image, return an empty extracted_text array."""

# JSON schema for OCR with classification
OCR_WITH_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extracted_text": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "Title",
                            "NarrativeText",
                            "ListItem",
                            "Table",
                            "Code",
                            "Handwritten",
                            "Uncategorized",
                        ],
                    },
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        },
        "image_type": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["extracted_text", "image_type", "subject", "description", "objects", "setting", "tags"],
}

# Prompt for OCR only (no classification)
OCR_ONLY_PROMPT = """Extract all visible text from this image with semantic structure. Return valid JSON.

For each text block, identify its type:
- Title: titles, headings, section headers
- NarrativeText: paragraphs, prose, body text
- ListItem: bullet points, numbered items
- Table: tabular data
- Code: code snippets, commands
- Handwritten: handwritten text
- Uncategorized: other text

Preserve text exactly. Maintain reading order.
For tables, emit Markdown with a header row, separator row, every source row in order, and empty cells retained. Preserve exact digits, units, clause identifiers, and uncertain text using explicit `[illegible]` markers.
If there is no text, return an empty array."""

# JSON schema for OCR only
OCR_ONLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extracted_text": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "Title",
                            "NarrativeText",
                            "ListItem",
                            "Table",
                            "Code",
                            "Header",
                            "Footer",
                            "FigureCaption",
                            "Formula",
                            "Handwritten",
                            "Annotation",
                            "Uncategorized",
                        ],
                    },
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        },
    },
    "required": ["extracted_text"],
}


def format_classification_metadata(
    image_type: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    objects: list[str] | None = None,
    colors: list[str] | None = None,
    setting: str | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    """
    Format image classification metadata as labeled lines.

    Centralizes the formatting logic used across classify_image,
    extract_text_with_vision, VisionOcrResult, etc.

    Args:
        image_type: Type of image (photo, screenshot, diagram, etc.)
        subject: Main subject or focus
        description: Brief factual description
        objects: List of identifiable objects
        colors: List of dominant colors
        setting: Location, environment, or context
        tags: List of search keywords

    Returns:
        List of formatted lines with [Label] prefixes
    """
    lines = []
    if image_type:
        lines.append(f"[ImageType] {image_type}")
    if subject:
        lines.append(f"[Subject] {subject}")
    if description:
        lines.append(f"[Description] {description}")
    if objects:
        lines.append(f"[Objects] {', '.join(objects)}")
    if colors:
        lines.append(f"[Colors] {', '.join(colors)}")
    if setting:
        lines.append(f"[Setting] {setting}")
    if tags:
        lines.append(f"[Tags] {', '.join(tags)}")
    return lines


def format_classification_from_dict(result: dict) -> list[str]:
    """
    Format classification metadata from a parsed JSON dict.

    Convenience wrapper for format_classification_metadata that extracts
    fields from a dict (e.g., LLM response).
    """
    return format_classification_metadata(
        image_type=result.get("image_type"),
        subject=result.get("subject"),
        description=result.get("description"),
        objects=result.get("objects"),
        colors=result.get("colors"),
        setting=result.get("setting"),
        tags=result.get("tags"),
    )


@dataclass
class VisionModelInfo:
    """Information about a provider-reported or live-probed vision model."""

    name: str
    provider: str = "ollama"
    modified_at: Optional[str] = None
    size: Optional[int] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    capabilities: list[str] | None = None
    context_limit: Optional[int] = None
    loaded: Optional[bool] = None


@dataclass
class _ModelsDevVisionCacheEntry:
    expires_at: float
    models_by_provider: dict[str, list[VisionModelInfo]]


_MODELS_DEV_VISION_CACHE: _ModelsDevVisionCacheEntry | None = None


def is_vision_capable(details: dict, model_name: str = "") -> bool:
    """
    Determine if a model supports vision/multimodal inputs.

    Uses centralized capability detection from ollama.py.
    Checks 'capabilities' array from /api/show response for "vision".

    Args:
        details: Full response from /api/show
        model_name: The model name (unused, kept for API compatibility)

    Returns:
        True if the model supports vision inputs
    """
    return is_vision_model_by_capability(details)


def _normalize_capability_tokens(capabilities: list[str] | None) -> set[str]:
    return {str(item or "").strip().lower() for item in capabilities or [] if item}


def _dedupe_capability_tokens(capabilities: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for capability in capabilities:
        if not capability or capability in seen:
            continue
        seen.add(capability)
        deduped.append(capability)
    return deduped


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _provider_metadata_reports_vision(
    *,
    capabilities: list[str] | None = None,
    model_type: str | None = None,
    engine_type: str | None = None,
) -> bool:
    """Return True only when provider metadata explicitly reports vision support."""
    capability_tokens = _normalize_capability_tokens(capabilities)
    type_tokens = {str(value or "").strip().lower() for value in (model_type, engine_type) if str(value or "").strip()}
    return bool(capability_tokens & VISION_CAPABILITY_TOKENS or type_tokens & VISION_MODEL_TYPE_TOKENS)


def _vision_model_info_from_provider_metadata(
    model: Any,
    provider: str,
) -> VisionModelInfo | None:
    normalized_provider = normalize_provider_name(provider)
    model_id = str(getattr(model, "id", getattr(model, "name", "")) or "").strip()
    if not model_id:
        return None

    capabilities = getattr(model, "capabilities", None)
    if not _provider_metadata_reports_vision(
        capabilities=capabilities,
        model_type=getattr(model, "model_type", None),
        engine_type=getattr(model, "engine_type", None),
    ):
        return None

    register_model_image_input_capability(model_id, True)
    return VisionModelInfo(
        name=model_id,
        provider=normalized_provider,
        capabilities=capabilities,
        context_limit=getattr(model, "context_limit", None),
        loaded=getattr(model, "loaded", None),
    )


def _metadata_input_modalities(row: dict[str, Any]) -> set[str]:
    input_modalities = None
    modalities_obj = row.get("modalities")
    if isinstance(modalities_obj, dict):
        input_modalities = modalities_obj.get("input")
    if input_modalities is None:
        input_modalities = row.get("supported_input_modalities")
    if not isinstance(input_modalities, list):
        return set()
    return {str(item or "").strip().lower() for item in input_modalities if item}


def _metadata_output_modalities(row: dict[str, Any]) -> set[str]:
    output_modalities = None
    modalities_obj = row.get("modalities")
    if isinstance(modalities_obj, dict):
        output_modalities = modalities_obj.get("output")
    if output_modalities is None:
        output_modalities = row.get("supported_output_modalities")
    if not isinstance(output_modalities, list):
        return set()
    return {str(item or "").strip().lower() for item in output_modalities if item}


def _metadata_capabilities(row: dict[str, Any]) -> list[str]:
    capabilities: list[str] = []
    capabilities_obj = row.get("capabilities")
    if isinstance(capabilities_obj, list):
        capabilities.extend(str(item).strip().lower() for item in capabilities_obj if item)
    elif isinstance(capabilities_obj, dict):
        supports_obj = capabilities_obj.get("supports")
        if isinstance(supports_obj, list):
            capabilities.extend(str(item).strip().lower() for item in supports_obj if item)
        elif isinstance(supports_obj, dict):
            capabilities.extend(str(flag).strip().lower() for flag, enabled in supports_obj.items() if enabled)
        capabilities.extend(str(flag).strip().lower() for flag, enabled in capabilities_obj.items() if enabled is True and flag != "supports")

    if _metadata_input_modalities(row) & {"image", "images", "vision"}:
        capabilities.append("image_input")

    return _dedupe_capability_tokens(capabilities)


def _metadata_row_reports_vision(row: dict[str, Any]) -> bool:
    output_modalities = _metadata_output_modalities(row)
    if output_modalities and "text" not in output_modalities:
        return False
    return _provider_metadata_reports_vision(
        capabilities=_metadata_capabilities(row),
        model_type=str(row.get("model_type") or row.get("type") or ""),
        engine_type=str(row.get("engine_type") or ""),
    )


def _models_dev_vision_models_from_payload(
    payload: dict[str, Any],
    provider: str,
    available_model_ids: set[str] | None = None,
) -> list[VisionModelInfo]:
    provider_key = normalize_provider_name(provider).replace("_", "-")
    provider_payload = payload.get(provider_key)
    if not isinstance(provider_payload, dict):
        provider_payload = payload.get(normalize_provider_name(provider))
    if not isinstance(provider_payload, dict):
        return []

    models_obj = provider_payload.get("models")
    if not isinstance(models_obj, dict):
        return []

    models: list[VisionModelInfo] = []
    for fallback_id, row in models_obj.items():
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or fallback_id or "").strip()
        if not model_id:
            continue
        if available_model_ids is not None and model_id not in available_model_ids:
            continue
        if not _metadata_row_reports_vision(row):
            continue

        capabilities = _metadata_capabilities(row)
        register_model_image_input_capability(model_id, True)
        models.append(
            VisionModelInfo(
                name=model_id,
                provider=normalize_provider_name(provider),
                capabilities=capabilities or None,
                context_limit=_coerce_positive_int(row.get("limit", {}).get("context") if isinstance(row.get("limit"), dict) else None),
            )
        )

    models.sort(key=lambda model: model.name.lower())
    return models


async def _list_models_dev_vision_models(provider: str, available_model_ids: set[str] | None = None) -> list[VisionModelInfo]:
    global _MODELS_DEV_VISION_CACHE

    now = time.monotonic()
    cached = _MODELS_DEV_VISION_CACHE
    normalized_provider = normalize_provider_name(provider)
    if cached and cached.expires_at > now:
        models = cached.models_by_provider.get(normalized_provider, [])
        if available_model_ids is None:
            return list(models)
        return [model for model in models if model.name in available_model_ids]

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(MODELS_DEV_API_URL)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        return []

    models_by_provider: dict[str, list[VisionModelInfo]] = {}
    for provider_key in payload:
        if not isinstance(provider_key, str):
            continue
        normalized_key = normalize_provider_name(provider_key.replace("-", "_"))
        models_by_provider[normalized_key] = _models_dev_vision_models_from_payload(
            payload,
            provider_key,
        )

    _MODELS_DEV_VISION_CACHE = _ModelsDevVisionCacheEntry(
        expires_at=now + MODELS_DEV_VISION_CACHE_TTL_SECONDS,
        models_by_provider=models_by_provider,
    )
    models = models_by_provider.get(normalized_provider, [])
    if available_model_ids is None:
        return list(models)
    return [model for model in models if model.name in available_model_ids]


async def _fetch_openai_available_model_ids(api_key: str | None) -> set[str] | None:
    key = str(api_key or "").strip()
    if not key:
        return None

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        return set()
    rows = payload.get("data")
    if not isinstance(rows, list):
        return set()
    return {model_id for row in rows if isinstance(row, dict) if (model_id := str(row.get("id") or "").strip())}


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = str(api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


_VISION_DEFAULT_BASE_URLS = {
    "openai": OPENAI_DEFAULT_BASE_URL,
    "openrouter": openrouter.DEFAULT_BASE_URL,
    "openai_codex": OPENAI_CODEX_DEFAULT_BASE_URL,
}


def _vision_http_endpoint(provider: str, base_url: str | None) -> str:
    """Return the endpoint an HTTP OCR provider calls, or fail when it has none."""
    endpoint = str(base_url or "").strip().rstrip("/") or _VISION_DEFAULT_BASE_URLS.get(provider, "")
    if not endpoint:
        raise ValueError(f"Vision OCR provider '{provider}' requires a base URL.")
    return endpoint


def _openai_compatible_base_url(provider: str, base_url: str | None = None) -> str:
    normalized_provider = normalize_provider_name(provider)
    if normalized_provider == "openai":
        return (base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")
    if normalized_provider == "openrouter":
        return (base_url or openrouter.DEFAULT_BASE_URL).rstrip("/")
    return str(base_url or "").strip().rstrip("/")


def _openai_compatible_chat_url(provider: str, base_url: str | None = None) -> str:
    normalized_base = _openai_compatible_base_url(provider, base_url)
    if normalized_base.endswith("/v1"):
        return f"{normalized_base}/chat/completions"
    return f"{normalized_base}/v1/chat/completions"


def _extract_openai_compatible_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


async def _post_openai_compatible_vision_request(
    *,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    model: str,
    prompt: str,
    image_b64: str,
    image_mime_type: str,
    schema: dict | None,
    timeout: float,
    max_tokens: int = OCR_FALLBACK_OUTPUT_BUDGET,
) -> dict[str, Any]:
    chat_url = _openai_compatible_chat_url(provider, base_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
        **_openai_vision_generation_parameters(model, max_tokens),
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ragtime_vision_ocr_result",
                "schema": schema,
                "strict": True,
            },
        }

    headers = _auth_headers(api_key)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(chat_url, json=payload, headers=headers)
        if response.status_code == 400 and schema is not None and _response_rejects_response_format(response):
            payload.pop("response_format", None)
            response = await client.post(chat_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


def _openai_vision_generation_parameters(model: str, max_tokens: int) -> dict[str, Any]:
    """Use the compatible token field for reasoning and GPT-5 Responses models."""
    model_name = model.lower()
    if model_name.startswith(("o1", "o3", "o4", "gpt-5")):
        return {"max_completion_tokens": max_tokens}
    return {"temperature": 0.1, "max_tokens": max_tokens}


def _response_rejects_response_format(response: httpx.Response) -> bool:
    try:
        detail = response.text.lower()
    except Exception:
        return False
    return "response_format" in detail or "json_schema" in detail


async def probe_openai_compatible_vision_capability(
    provider: str,
    base_url: str | None,
    model: str,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> bool:
    """Probe image input support with a real tiny image request."""
    model_id = str(model or "").strip()
    if not model_id:
        return False
    chat_url = _openai_compatible_chat_url(provider, base_url)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with OK only."},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                ],
            }
        ],
        "max_tokens": 3,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                chat_url,
                json=payload,
                headers=_auth_headers(api_key),
            )
            if response.is_success:
                register_model_image_input_capability(model_id, True)
                return True
            if response.status_code in {400, 404, 415, 422}:
                register_model_image_input_capability(model_id, False)
                return False
            response.raise_for_status()
            register_model_image_input_capability(model_id, True)
            return True
    except Exception as exc:
        logger.debug("Vision probe failed for %s/%s: %s", provider, model_id, exc)
        return False


async def list_vision_models(
    base_url: str,
    timeout: float = 10.0,
) -> list[VisionModelInfo]:
    """
    List vision-capable models from an Ollama server.

    Queries Ollama's /api/show endpoint for each model and checks the
    'capabilities' array for "vision" capability.

    Args:
        base_url: Ollama server base URL (e.g., 'http://localhost:11434')
        timeout: Request timeout in seconds

    Returns:
        List of VisionModelInfo objects for models that support vision
    """
    models = []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Get list of all models
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                model_name = model.get("name", "")
                if not model_name:
                    continue

                # Get detailed info to check capabilities
                details = await get_model_details(model_name, base_url, client)

                if is_vision_capable(details, model_name):
                    detail_info = details.get("details", {})
                    capabilities = extract_capabilities(details)
                    models.append(
                        VisionModelInfo(
                            name=model_name,
                            modified_at=model.get("modified_at"),
                            size=model.get("size"),
                            family=detail_info.get("family"),
                            parameter_size=detail_info.get("parameter_size"),
                            capabilities=capabilities if capabilities else None,
                        )
                    )

    except httpx.ConnectError:
        logger.warning(f"Cannot connect to Ollama at {base_url}")
        raise
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to Ollama at {base_url}")
        raise
    except Exception as e:
        logger.warning(f"Error listing Ollama vision models: {e}")
        raise

    return models


async def list_provider_vision_models(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    candidate_models: list[str] | None = None,
    available_image_model_ids: set[str] | None = None,
    timeout: float = 20.0,
) -> list[VisionModelInfo]:
    """List vision-capable models using provider metadata.

    Model IDs are never parsed for capability. A model is included only when
    provider or catalog metadata explicitly reports image/VLM support.
    """
    normalized_provider = normalize_provider_name(provider)

    if normalized_provider == "ollama":
        if not base_url:
            return []
        return await list_vision_models(base_url=base_url, timeout=timeout)

    if normalized_provider == "lmstudio":
        if not base_url:
            return []
        lmstudio_models = await lmstudio.list_chat_models(base_url, api_key=api_key)
        return [info for model in lmstudio_models if (info := _vision_model_info_from_provider_metadata(model, normalized_provider)) is not None]

    if normalized_provider == "omlx":
        if not base_url:
            return []
        omlx_models = await omlx.list_status_models(base_url, api_key=api_key)
        if not omlx_models:
            omlx_models = await omlx.list_models(base_url, api_key=api_key)
        return [info for model in omlx_models if (info := _vision_model_info_from_provider_metadata(model, normalized_provider)) is not None]

    if normalized_provider == "llama_cpp":
        if not base_url:
            return []
        llama_cpp_models = await llama_cpp.list_chat_models(base_url)
        return [info for model in llama_cpp_models if (info := _vision_model_info_from_provider_metadata(model, normalized_provider)) is not None]

    if normalized_provider == "openai":
        _ = (base_url, candidate_models)
        available_model_ids = await _fetch_openai_available_model_ids(api_key)
        return await _list_models_dev_vision_models(
            normalized_provider,
            available_model_ids=available_model_ids,
        )

    if normalized_provider in {"anthropic", "claude_code", "openai_codex"}:
        # Routes supply authenticated provider catalogs.  Do not repeat catalog
        # requests here: Codex catalog metadata in particular is authoritative.
        trusted_ids = available_image_model_ids or set()
        return [VisionModelInfo(name=model_id, provider=normalized_provider, capabilities=["image_input"]) for model_id in sorted(trusted_ids)]

    if normalized_provider == "openrouter":
        _ = base_url
        if not api_key:
            return []
        candidate_set = {str(model).strip() for model in candidate_models or [] if str(model).strip()}
        models: list[VisionModelInfo] = []
        for row in await openrouter.list_models(api_key, timeout=timeout):
            if not openrouter.supports_vision(row):
                continue
            model_id = openrouter.model_id(row)
            if not model_id or (candidate_set and model_id not in candidate_set):
                continue
            register_model_image_input_capability(model_id, True)
            models.append(
                VisionModelInfo(
                    name=model_id,
                    provider=normalized_provider,
                    capabilities=openrouter.capabilities(row),
                    context_limit=openrouter.context_limit(row),
                )
            )
        return models

    return []


def _load_image(image_content: bytes, source_format: str | None) -> tuple[Any, str]:
    """
    Helper to load image from bytes, handling raw formats.

    Args:
        image_content: Raw image bytes
        source_format: Optional source format hint

    Returns:
        Tuple of (PIL.Image or None, status string)
    """
    is_raw = source_format and source_format.lower() in RAW_CAMERA_EXTENSIONS

    if not is_raw:
        return Image.open(io.BytesIO(image_content)), "standard"

    # Try to process raw format with rawpy
    try:
        import rawpy
    except ImportError:
        logger.warning(f"rawpy not installed, cannot process {source_format}. Install with: pip install rawpy")
        return None, "raw_error"

    try:
        with rawpy.imread(io.BytesIO(image_content)) as raw:
            # Use camera white balance and auto brightness
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
            )
        return Image.fromarray(rgb), "raw"
    except Exception as e:
        logger.warning(f"Failed to process raw format {source_format}: {e}")
        return None, "raw_error"


def _process_image_content(img: Any, max_dimension: int, quality: int) -> tuple[bytes, str]:
    """
    Helper to resize and compress image.

    Args:
        img: PIL Image object
        max_dimension: Max pixel dimension
        quality: JPEG quality

    Returns:
        Tuple of (bytes, format)
    """
    # Convert to RGB if necessary (handles RGBA, palette, etc.)
    if img.mode in ("RGBA", "LA", "P"):
        # Create white background for transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Calculate resize dimensions
    width, height = img.size
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # Use high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    # Convert to JPEG bytes
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue(), "jpeg"


def preprocess_image(
    image_content: bytes,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    quality: int = DEFAULT_JPEG_QUALITY,
    source_format: str | None = None,
) -> tuple[bytes, str]:
    """
    Preprocess image for vision model processing.

    Resizes large images to reduce processing time and converts resized images
    to JPEG for efficient transfer. Below-limit standard images remain byte
    identical so text-bearing PNGs do not incur a lossy conversion.

    Args:
        image_content: Raw image bytes
        max_dimension: Maximum pixels on longest edge (default 2048)
        quality: JPEG compression quality 1-100 (default 85)
        source_format: Optional source format hint (e.g., '.cr2', '.nef')

    Returns:
        Tuple of (processed_image_bytes, format_used)
    """
    try:
        original_size = len(image_content)

        # 1. Load Image
        img, status = _load_image(image_content, source_format)

        if img is None:
            # Failed to load (raw error or unsupported)
            return image_content, status

        if status == "standard" and max(img.size) <= max_dimension:
            return image_content, _image_mime_type(img, source_format)

        # 2. Process Image (Resize & Convert)
        processed_bytes, fmt = _process_image_content(img, max_dimension, quality)

        # Log reduction
        new_size = len(processed_bytes)
        if new_size < original_size:
            reduction = ((original_size - new_size) / original_size) * 100
            logger.debug(f"Image preprocessing: {original_size:,} -> {new_size:,} bytes ({reduction:.1f}% reduction)")
        else:
            logger.debug(f"Image preprocessing: {original_size:,} -> {new_size:,} bytes")

        return processed_bytes, f"image/{fmt}"

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}, using original")
        return image_content, _source_mime_type(source_format)


def _source_mime_type(source_format: str | None) -> str:
    extension = str(source_format or "").strip().lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(extension, "application/octet-stream")


def _image_mime_type(image: Any, source_format: str | None) -> str:
    image_format = str(getattr(image, "format", "") or "").lower()
    return _source_mime_type(image_format or source_format)


def _prepare_vision_image_content(
    image_content: bytes,
    *,
    provider: str,
    preprocess: bool,
    max_dimension: int,
    source_format: str | None,
) -> tuple[bytes, str]:
    processed, mime_type = (
        preprocess_image(
            image_content,
            max_dimension=max_dimension,
            source_format=source_format,
        )
        if preprocess
        else (image_content, _source_mime_type(source_format))
    )
    if mime_type == "raw_error":
        raise ValueError("Vision OCR image could not be decoded.")
    encoded_limit = _encoded_image_limit(provider)
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"} or _base64_size(processed) > encoded_limit:
        image, status = _load_image(image_content, source_format)
        if image is None:
            raise ValueError("Vision OCR image could not be decoded.")
        processed, image_format = _bounded_vision_image(image, max_dimension, encoded_limit)
        mime_type = f"image/{image_format}"
    if _base64_size(processed) > encoded_limit:
        raise ValueError("Vision OCR image exceeds the provider request limit.")
    return processed, mime_type


def _encoded_image_limit(provider: str) -> int:
    if normalize_provider_name(provider) in {"anthropic", "claude_code"}:
        return ANTHROPIC_MAX_ENCODED_IMAGE_BYTES
    return OPENAI_COMPATIBLE_MAX_ENCODED_IMAGE_BYTES


def _base64_size(content: bytes) -> int:
    return 4 * ((len(content) + 2) // 3)


def _bounded_vision_image(image: Any, max_dimension: int, encoded_limit: int) -> tuple[bytes, str]:
    dimension = max_dimension
    while dimension >= 64:
        processed, image_format = _process_image_content(image, dimension, DEFAULT_JPEG_QUALITY)
        if _base64_size(processed) <= encoded_limit:
            return processed, image_format
        dimension //= 2
    return _process_image_content(image, 64, DEFAULT_JPEG_QUALITY)


async def _supports_structured_vision_output(
    provider: str,
    model: str,
    base_url: str | None,
) -> bool:
    normalized_provider = normalize_provider_name(provider)
    if normalized_provider == "ollama":
        ollama_endpoint = _vision_http_endpoint(normalized_provider, base_url)
        try:
            model_details = await get_model_details(model, ollama_endpoint)
            if model_details and supports_structured_output(model_details):
                logger.debug("Model %s supports structured output", model)
                return True

            capabilities = extract_capabilities(model_details) if model_details else []
            logger.debug(
                "Model %s does not support structured output (capabilities: %s), using text prompts",
                model,
                capabilities,
            )
        except Exception as exc:
            logger.debug("Could not check model capabilities: %s, using text prompts", exc)
        return False

    return normalized_provider in OPENAI_COMPATIBLE_VISION_PROVIDERS | {"anthropic", "claude_code", "openai_codex"}


def _resolve_vision_prompt_and_schema(
    *,
    prompt: str | None,
    include_classification: bool,
    use_structured_output: bool,
) -> tuple[str, dict[str, Any] | None]:
    if prompt is not None:
        return prompt, None

    if include_classification:
        return (
            OCR_WITH_CLASSIFICATION_PROMPT,
            OCR_WITH_CLASSIFICATION_SCHEMA if use_structured_output else None,
        )

    return (
        OCR_ONLY_PROMPT,
        OCR_ONLY_SCHEMA if use_structured_output else None,
    )


def _extract_ollama_response_text(data: dict[str, Any]) -> str:
    return str(data.get("response") or "").strip()


async def _request_vision_response_text(
    *,
    provider: str,
    base_url: str | None,
    model: str,
    api_key: str | None,
    prompt: str,
    schema: dict[str, Any] | None,
    processed_content: bytes,
    image_mime_type: str,
    timeout: float,
) -> str:
    normalized_provider = normalize_provider_name(provider)
    image_b64 = base64.b64encode(processed_content).decode("utf-8")

    limiter = await _get_vision_ocr_limiter()
    await limiter.acquire(await _vision_ocr_limit())
    try:
        # Queueing is intentionally outside each page's provider deadline.
        async with asyncio.timeout(timeout):
            return await _retry_vision_request(
                lambda: _request_vision_response_once(
                    normalized_provider=normalized_provider,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    prompt=prompt,
                    schema=schema,
                    image_b64=image_b64,
                    image_mime_type=image_mime_type,
                    timeout=timeout,
                ),
                timeout,
            )
    finally:
        await limiter.release()


async def _request_vision_response_once(
    *,
    normalized_provider: str,
    base_url: str | None,
    model: str,
    api_key: str | None,
    prompt: str,
    schema: dict[str, Any] | None,
    image_b64: str,
    image_mime_type: str,
    timeout: float,
) -> str:
    if normalized_provider == "ollama":
        ollama_endpoint = _vision_http_endpoint(normalized_provider, base_url)
        output_budget = await _resolve_ocr_output_budget(model)
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_json = {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "think": OLLAMA_OCR_THINK,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": OLLAMA_OCR_CONTEXT_LENGTH,
                    "num_predict": output_budget,
                },
            }
            if schema is not None:
                request_json["format"] = schema
            response = await client.post(
                f"{ollama_endpoint}/api/generate",
                json=request_json,
            )
            response.raise_for_status()
            data = response.json()
        _reject_ollama_incomplete_response(data)
        return _extract_ollama_response_text(data)

    if normalized_provider in OPENAI_COMPATIBLE_VISION_PROVIDERS:
        data = await _post_openai_compatible_vision_request(
            provider=normalized_provider,
            base_url=_vision_http_endpoint(normalized_provider, base_url),
            api_key=await _resolve_vision_api_key(normalized_provider, api_key),
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            schema=schema,
            timeout=timeout,
            max_tokens=await _resolve_ocr_output_budget(model),
        )
        _reject_openai_incomplete_response(data)
        return _extract_openai_compatible_response_text(data)

    if normalized_provider == "anthropic":
        return await _request_anthropic_vision_response(
            base_url=_vision_http_endpoint(normalized_provider, base_url),
            api_key=await _resolve_vision_api_key(normalized_provider, api_key),
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            schema=schema,
            timeout=timeout,
            max_tokens=await _resolve_ocr_output_budget(model),
        )

    if normalized_provider == "claude_code":
        return await _request_claude_code_vision_response(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            schema=schema,
            timeout=timeout,
            max_tokens=await _resolve_ocr_output_budget(model),
        )

    if normalized_provider == "openai_codex":
        return await _request_codex_vision_response(
            base_url=_vision_http_endpoint(normalized_provider, base_url),
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            schema=schema,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported vision OCR provider: {normalized_provider}")


async def _retry_vision_request(operation: Any, timeout: float) -> str:
    deadline = time.monotonic() + max(0.1, timeout)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403} or exc.response.status_code not in {408, 409, 429, 500, 502, 503, 504, 529}:
                raise
            delay = _retry_delay(exc.response, attempt)
            last_error = exc
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
            delay = min(0.5 * (2**attempt), 4.0)
            last_error = exc
        if attempt == 2 or time.monotonic() + delay >= deadline:
            assert last_error is not None
            raise last_error
        await asyncio.sleep(delay)
    raise RuntimeError("Vision OCR retry loop exhausted.")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    try:
        return min(max(float(retry_after), 0.0), 10.0)
    except ValueError:
        return min(0.5 * (2**attempt), 4.0)


def _reject_ollama_incomplete_response(data: Any) -> None:
    if not isinstance(data, dict) or data.get("done") is False or data.get("done_reason") in {"length", "error", "load"}:
        raise RuntimeError("Ollama OCR response was incomplete.")


def _reject_openai_incomplete_response(data: Any) -> None:
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(first, dict) or first.get("finish_reason") in {"length", "content_filter"}:
        raise RuntimeError("Vision OCR response was incomplete or refused.")


async def _vision_ocr_limit() -> int:
    try:
        settings = await get_app_settings()
        return max(1, int(settings.get("ocr_concurrency_limit", DEFAULT_OCR_CONCURRENCY_LIMIT)))
    except Exception:
        return DEFAULT_OCR_CONCURRENCY_LIMIT


async def _get_vision_ocr_limiter() -> _VisionOcrLimiter:
    global _vision_ocr_limiter
    if _vision_ocr_limiter is None:
        _vision_ocr_limiter = _VisionOcrLimiter(await _vision_ocr_limit())
    return _vision_ocr_limiter


async def _resolve_vision_api_key(provider: str, supplied_key: str | None) -> str | None:
    if supplied_key:
        return supplied_key
    settings = await get_app_settings()
    return resolve_provider_api_key(settings, provider, "llm")


def _response_text_from_anthropic_payload(payload: dict[str, Any]) -> str:
    if payload.get("stop_reason") in {"max_tokens", "refusal"}:
        raise RuntimeError("Anthropic OCR response was incomplete or refused.")
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError("Anthropic OCR response was malformed.")
    text = [block["text"] for block in blocks if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]
    if not text:
        raise RuntimeError("Anthropic OCR response did not contain text.")
    return "\n".join(text)


async def _request_anthropic_vision_response(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    image_b64: str,
    image_mime_type: str,
    schema: dict[str, Any] | None,
    timeout: float,
    max_tokens: int = OCR_FALLBACK_OUTPUT_BUDGET,
) -> str:
    if not api_key:
        raise RuntimeError("Anthropic API key is not configured.")
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": image_mime_type, "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if schema is not None:
        payload["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/messages", json=payload, headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
        )
        if schema is not None and response.status_code == 400 and "output_config" in response.text.lower():
            payload.pop("output_config")
            response = await client.post(
                f"{base_url.rstrip('/')}/v1/messages", json=payload, headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
            )
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Anthropic OCR response was malformed.")
    return _response_text_from_anthropic_payload(data)


async def _request_claude_code_vision_response(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    image_mime_type: str,
    schema: dict[str, Any] | None,
    timeout: float,
    max_tokens: int = OCR_FALLBACK_OUTPUT_BUDGET,
) -> str:
    client = ClaudeCodeChatModel(model=model, timeout=timeout, max_tokens=max_tokens)
    if schema is not None:
        client = client.with_ocr_json_schema(schema)
    response = await client.ainvoke(
        [
            HumanMessage(
                content=[{"type": "text", "text": prompt}, {"type": "image", "source": {"type": "base64", "media_type": image_mime_type, "data": image_b64}}]
            )
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("Claude Code OCR response was malformed.")
    return response.content


async def _request_codex_vision_response(
    *, base_url: str, model: str, prompt: str, image_b64: str, image_mime_type: str, schema: dict[str, Any] | None, timeout: float
) -> str:
    token = await ensure_openai_codex_token_fresh()
    if not token:
        raise RuntimeError("OpenAI Codex is not authenticated.")
    settings = await get_app_settings()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": f"data:{image_mime_type};base64,{image_b64}"}]
    payload: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "instructions": "Return only the requested OCR result.",
        "store": False,
        "stream": True,
    }
    if schema is not None:
        payload["text"] = {"format": {"type": "json_schema", "name": "ragtime_vision_ocr_result", "schema": schema, "strict": True}}
    request = build_codex_responses_request(httpx.Request("POST", f"{base_url.rstrip('/')}/responses"), json.dumps(payload).encode())
    headers = {
        **dict(request.headers),
        **codex_response_headers(str(settings.get("openai_codex_account_id") or "")),
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", request.url, content=request.content, headers=headers) as response:
            response.raise_for_status()
            return await _collect_codex_sse_text(response)


async def _collect_codex_sse_text(response: httpx.Response) -> str:
    """Extract Responses API output text from its real SSE event stream."""
    text_parts: list[str] = []
    completed = False
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI Codex OCR emitted invalid SSE JSON.") from exc
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type == "response.completed":
            response_data = event.get("response")
            if not isinstance(response_data, dict) or response_data.get("status") != "completed":
                raise RuntimeError("OpenAI Codex OCR response was incomplete or refused.")
            completed = True
            if not text_parts:
                text_parts.extend(_codex_output_text(response_data))
        elif event_type in {"response.failed", "response.incomplete", "error"}:
            raise RuntimeError("OpenAI Codex OCR response was incomplete or refused.")
    output = "".join(text_parts).strip()
    if not completed or not output:
        raise RuntimeError("OpenAI Codex OCR response did not complete with text.")
    return output


def _codex_output_text(response: dict[str, Any]) -> list[str]:
    text: list[str] = []
    for output in response.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                text.append(content["text"])
    return text


def _parse_vision_response_json(
    response_text: str,
    *,
    log_parse_errors: bool = False,
) -> dict[str, Any] | None:
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        if log_parse_errors:
            logger.warning(
                "Failed to parse OCR JSON: %s. Raw response (%s chars): %r",
                exc,
                len(response_text),
                response_text[:500],
            )
        return None

    return result if isinstance(result, dict) else None


def _extract_vision_text_lines(result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in result.get("extracted_text", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if content:
            lines.append(str(content))
    return lines


def _vision_ocr_result_from_dict(result: dict[str, Any]) -> "VisionOcrResult":
    extracted_text = result.get("extracted_text")
    allowed_types = {
        "Title",
        "NarrativeText",
        "ListItem",
        "Table",
        "Code",
        "Header",
        "Footer",
        "FigureCaption",
        "Formula",
        "Handwritten",
        "Annotation",
        "Uncategorized",
    }
    if not isinstance(extracted_text, list) or any(
        not isinstance(block, dict) or not isinstance(block.get("type"), str) or not isinstance(block.get("content"), str) or block["type"] not in allowed_types
        for block in extracted_text
    ):
        raise ValueError("OCR response contains malformed extracted_text blocks.")
    return VisionOcrResult(
        extracted_text=extracted_text,
        image_type=result.get("image_type"),
        subject=result.get("subject"),
        description=result.get("description"),
        objects=result.get("objects"),
        colors=result.get("colors"),
        setting=result.get("setting"),
        tags=result.get("tags"),
    )


async def classify_image(
    image_content: bytes,
    base_url: str,
    model: str,
    timeout: float = 60.0,
) -> str:
    """
    Classify and describe an image when no text is found.

    Generates a semantic description of the image content for retrieval purposes.
    Useful for photos, diagrams, and other non-text images.

    Args:
        image_content: Preprocessed image bytes
        base_url: Ollama server base URL
        model: Vision model name
        timeout: Request timeout in seconds

    Returns:
        Image classification/description text formatted for retrieval
    """
    image_b64 = base64.b64encode(image_content).decode("utf-8")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": CLASSIFICATION_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "format": CLASSIFICATION_SCHEMA,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()

            # Parse structured JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse classification JSON, returning raw")
                return response_text

            # Format for retrieval indexing
            return "\n".join(format_classification_from_dict(result))

    except Exception as e:
        logger.warning(f"Image classification error with {model}: {e}")
        return ""


async def extract_text_with_vision(
    image_content: bytes,
    base_url: str | None,
    model: str,
    provider: str = "ollama",
    api_key: str | None = None,
    timeout: float = DEFAULT_VISION_OCR_TIMEOUT,
    prompt: Optional[str] = None,
    preprocess: bool = True,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    source_format: str | None = None,
    include_classification: bool = True,
) -> str:
    """
    Extract text from an image using a vision model.

    This performs semantic OCR - the model understands the image content
    and extracts text while preserving meaning and structure. When
    include_classification is True, also appends image classification/description
    which enables semantic search for image content (e.g., finding photos of birds).

    Args:
        image_content: Raw image bytes (PNG, JPEG, raw formats, etc.)
        base_url: Provider base URL (OpenAI may use https://api.openai.com/v1)
        model: Vision model name
        provider: OCR provider name
        api_key: Optional provider API key
        timeout: Request timeout in seconds (vision models can be slow)
        prompt: Optional custom prompt (defaults to text extraction prompt)
        preprocess: Whether to resize/optimize the image (default True)
        max_dimension: Max pixels on longest edge when preprocessing
        source_format: Source file extension hint (e.g., '.cr2' for raw)
        include_classification: Include image classification for semantic search (default True)

    Returns:
        Extracted text plus image classification (when enabled)
    """
    normalized_provider = normalize_provider_name(provider)
    uses_default_prompt = prompt is None
    processed_content, image_mime_type = _prepare_vision_image_content(
        image_content,
        provider=normalized_provider,
        preprocess=preprocess,
        max_dimension=max_dimension,
        source_format=source_format,
    )
    use_structured_output = False
    if prompt is None:
        use_structured_output = await _supports_structured_vision_output(
            normalized_provider,
            model,
            base_url,
        )

    prompt, schema = _resolve_vision_prompt_and_schema(
        prompt=prompt,
        include_classification=include_classification,
        use_structured_output=use_structured_output,
    )

    try:
        response_text = await _request_vision_response_text(
            provider=normalized_provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            schema=schema,
            processed_content=processed_content,
            image_mime_type=image_mime_type,
            timeout=timeout,
        )

        result = _parse_vision_response_json(
            response_text,
            log_parse_errors=schema is not None,
        )
        if result is None:
            if schema is not None:
                raise ValueError("Vision OCR provider returned malformed structured output.")
            return response_text

        if uses_default_prompt:
            _vision_ocr_result_from_dict(result)

        lines = _extract_vision_text_lines(result)

        if include_classification:
            lines.extend(format_classification_from_dict(result))

        return "\n".join(lines)

    except httpx.ConnectError:
        logger.warning(f"Cannot connect to {normalized_provider} at {base_url} for vision OCR")
        raise
    except httpx.TimeoutException:
        logger.warning(f"Vision OCR timeout ({timeout}s) for model {model}")
        raise
    except Exception as exc:
        logger.warning(f"Vision OCR error with {normalized_provider}/{model}: {exc}")
        raise


@dataclass
class VisionOcrResult:
    """Structured result from vision OCR with semantic element information."""

    # OCR extracted text elements with semantic types
    extracted_text: list[dict]  # [{type: str, content: str}, ...]
    # Image classification metadata
    image_type: str | None = None
    subject: str | None = None
    description: str | None = None
    objects: list[str] | None = None
    colors: list[str] | None = None
    setting: str | None = None
    tags: list[str] | None = None
    # Raw text fallback (when structured parsing fails)
    raw_text: str | None = None

    def _format_extracted_text(self) -> list[str]:
        return _extract_vision_text_lines({"extracted_text": self.extracted_text})

    def format_for_indexing(self) -> str:
        """Format result as plain text for indexing (no element type labels)."""
        lines = self._format_extracted_text()

        # Add classification metadata (with labels - these are meaningful)
        lines.extend(self._format_classification())

        return "\n".join(lines)

    def _format_classification(self) -> list[str]:
        """Format classification metadata using centralized helper."""
        return format_classification_metadata(
            image_type=self.image_type,
            subject=self.subject,
            description=self.description,
            objects=self.objects,
            colors=self.colors,
            setting=self.setting,
            tags=self.tags,
        )

    def get_semantic_segments(self) -> list[tuple[str, str]]:
        """
        Get content organized by semantic segments for intelligent chunking.

        Returns one tuple for every OCR block plus a separate classification
        tuple when classification metadata is present.
        """
        segments = []

        for item in self.extracted_text:
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            content = item.get("content")
            if not isinstance(block_type, str) or not isinstance(content, str) or not content:
                continue
            segments.append((block_type.lower(), content))

        # Combine classification as one semantic unit (always together)
        class_lines = self._format_classification()
        if class_lines:
            segments.append(("classification", "\n".join(class_lines)))

        return segments


async def extract_text_with_vision_structured(
    image_content: bytes,
    base_url: str | None,
    model: str,
    provider: str = "ollama",
    api_key: str | None = None,
    timeout: float = DEFAULT_VISION_OCR_TIMEOUT,
    preprocess: bool = True,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    source_format: str | None = None,
    include_classification: bool = True,
) -> VisionOcrResult:
    """
    Extract text from image with structured output for semantic chunking.

    Like extract_text_with_vision but returns VisionOcrResult with semantic
    structure preserved. This enables intelligent chunking that respects
    semantic boundaries (e.g., keeping classification metadata together).

    Args:
        image_content: Raw image bytes
        base_url: Provider base URL
        model: Vision model name
        provider: OCR provider name
        api_key: Optional provider API key
        timeout: Request timeout in seconds
        preprocess: Whether to resize/optimize the image
        max_dimension: Max pixels on longest edge when preprocessing
        source_format: Source file extension hint
        include_classification: Include image classification

    Returns:
        VisionOcrResult with structured semantic data
    """
    normalized_provider = normalize_provider_name(provider)
    processed_content, image_mime_type = _prepare_vision_image_content(
        image_content,
        provider=normalized_provider,
        preprocess=preprocess,
        max_dimension=max_dimension,
        source_format=source_format,
    )
    use_structured_output = await _supports_structured_vision_output(
        normalized_provider,
        model,
        base_url,
    )
    prompt, schema = _resolve_vision_prompt_and_schema(
        prompt=None,
        include_classification=include_classification,
        use_structured_output=use_structured_output,
    )

    try:
        response_text = await _request_vision_response_text(
            provider=normalized_provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            schema=schema,
            processed_content=processed_content,
            image_mime_type=image_mime_type,
            timeout=timeout,
        )

        result = _parse_vision_response_json(
            response_text,
            log_parse_errors=schema is not None,
        )
        if result is None:
            raise ValueError("Vision OCR provider returned malformed structured output.")

        return _vision_ocr_result_from_dict(result)

    except Exception as e:
        logger.warning(f"Vision OCR error: {e}")
        raise


async def validate_vision_model(
    model: str,
    base_url: str,
) -> tuple[bool, str]:
    """
    Validate that a model exists and supports vision.

    Checks the 'capabilities' array from Ollama's /api/show endpoint
    for "vision" capability.

    Args:
        model: Model name to validate
        base_url: Ollama server base URL

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        details = await get_model_details(model, base_url)
        if not details:
            return False, f"Model '{model}' not found on Ollama server"

        capabilities = extract_capabilities(details)
        if is_vision_capable(details, model):
            return (
                True,
                f"Model '{model}' supports vision (capabilities: {capabilities})",
            )
        else:
            return (
                False,
                f"Model '{model}' does not support vision (capabilities: {capabilities})",
            )

    except Exception as e:
        return False, f"Error validating model: {e}"
