"""Credential-free OCR provider target resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping
from urllib.parse import urlsplit

from ragtime.core.model_providers import (
    get_provider,
    normalize_provider_name,
    resolve_provider_base_url,
)

ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"
OPENAI_API_BASE_URL = "https://api.openai.com/v1"
OPENAI_CODEX_API_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OcrTarget:
    """Resolved non-secret OCR provider configuration."""

    provider: str
    model: str
    base_url: str | None

    def to_snapshot(self) -> dict[str, str | None]:
        """Return fields safe to persist with an indexing job."""
        return asdict(self)


class OcrTargetError(ValueError):
    """OCR target configuration cannot be resolved safely."""


def _setting_value(settings: Mapping[str, Any] | object, field: str, default: Any = None) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(field, default)
    return getattr(settings, field, default)


def _index_value(index_config: Mapping[str, Any] | object | None, field: str) -> tuple[bool, Any]:
    if index_config is None:
        return False, None
    if isinstance(index_config, Mapping):
        return field in index_config, index_config.get(field)
    fields_set = getattr(index_config, "model_fields_set", set())
    return field in fields_set, getattr(index_config, field, None)


def _normalized_model(value: Any) -> str | None:
    if isinstance(value, Enum):
        value = value.value
    model = str(value or "").strip()
    return model or None


def _normalized_provider(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return normalize_provider_name(value)


def _credential_free_endpoint(value: str | None) -> str | None:
    endpoint = str(value or "").strip()
    if not endpoint:
        return None
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OcrTargetError("OCR endpoint must not contain credentials, query parameters, or fragments")
    return endpoint.rstrip("/")


def resolve_ocr_base_url(settings: Mapping[str, Any] | object, provider: str | Enum | None) -> str | None:
    """Resolve a credential-free endpoint identity for an OCR provider."""
    if isinstance(provider, Enum):
        provider = provider.value
    provider = _normalized_provider(provider)
    if provider == "anthropic":
        return ANTHROPIC_API_BASE_URL
    if provider == "openai":
        return OPENAI_API_BASE_URL
    if provider == "openai_codex":
        return _credential_free_endpoint(_setting_value(settings, "openai_codex_base_url", OPENAI_CODEX_API_BASE_URL) or OPENAI_CODEX_API_BASE_URL)
    if provider == "openrouter":
        return OPENROUTER_API_BASE_URL
    if provider == "claude_code":
        return None
    # OCR used the LLM Ollama endpoint before OCR targets were introduced.
    # The shared LLM resolver retains its legacy embedding-endpoint fallback.
    return _credential_free_endpoint(resolve_provider_base_url(settings, provider, "llm"))


def _effective_ocr_mode(settings: Mapping[str, Any] | object, index_config: Mapping[str, Any] | object | None) -> str | None:
    if isinstance(index_config, Mapping):
        if "ocr_mode" in index_config:
            mode = index_config["ocr_mode"]
        elif isinstance(settings, Mapping) and "default_ocr_mode" not in settings:
            return None
        else:
            mode = _setting_value(settings, "default_ocr_mode")
    elif index_config is not None:
        mode = getattr(index_config, "ocr_mode", None)
    elif isinstance(settings, Mapping) and "default_ocr_mode" not in settings:
        return None
    else:
        mode = _setting_value(settings, "default_ocr_mode")
    normalized = str(getattr(mode, "value", mode) or "").strip().lower()
    return "vision" if normalized == "ollama" else normalized


def resolve_ocr_target(
    settings: Mapping[str, Any] | object,
    index_config: Mapping[str, Any] | object | None = None,
) -> OcrTarget | None:
    """Resolve the OCR target while enforcing provider/model inheritance rules."""
    mode = _effective_ocr_mode(settings, index_config)
    global_provider = _normalized_provider(_setting_value(settings, "default_ocr_provider"))
    if global_provider:
        global_descriptor = get_provider(global_provider)
        if global_descriptor is None or not global_descriptor.supports_vision_ocr:
            raise OcrTargetError(f"OCR provider '{global_provider}' does not support vision OCR")
    if mode != "vision":
        return None
    global_model = _normalized_model(_setting_value(settings, "default_ocr_vision_model"))
    has_provider_override, raw_provider = _index_value(index_config, "ocr_provider")
    has_model_override, raw_model = _index_value(index_config, "ocr_vision_model")

    override_provider = _normalized_provider(raw_provider) if has_provider_override else ""
    provider = override_provider or global_provider
    if not provider:
        if mode == "vision":
            raise OcrTargetError("OCR provider is required when OCR mode is vision")
        return None
    descriptor = get_provider(provider)
    if descriptor is None or not descriptor.supports_vision_ocr:
        raise OcrTargetError(f"OCR provider '{provider}' does not support vision OCR")

    override_model = _normalized_model(raw_model) if has_model_override else None
    provider_overridden = bool(override_provider and override_provider != global_provider)
    if provider_overridden and not override_model:
        raise OcrTargetError("ocr_vision_model is required when ocr_provider overrides the global provider")
    model = override_model or global_model
    if not model:
        raise OcrTargetError(f"ocr_vision_model is required for OCR provider '{provider}'")

    return OcrTarget(provider=provider, model=model, base_url=resolve_ocr_base_url(settings, provider))
