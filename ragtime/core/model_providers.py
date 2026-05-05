"""Shared model-provider metadata and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProviderRole = Literal["llm", "embedding"]


@dataclass(frozen=True)
class ProviderConnection:
    """Settings fields that describe one provider endpoint."""

    protocol_field: str
    host_field: str
    port_field: str
    base_url_field: str
    default_protocol: str
    default_host: str
    default_port: int
    default_base_url: str
    fallback_base_url_field: str | None = None
    fallback_default_base_url: str | None = None


@dataclass(frozen=True)
class ModelProvider:
    """Static metadata for a model provider."""

    name: str
    label: str
    aliases: tuple[str, ...] = ()
    llm_connection: ProviderConnection | None = None
    embedding_connection: ProviderConnection | None = None
    llm_api_key_field: str | None = None
    embedding_api_key_field: str | None = None
    supports_llm: bool = False
    supports_embeddings: bool = False
    openai_compatible_chat: bool = False
    openai_compatible_embeddings: bool = False
    local: bool = False


OLLAMA_EMBEDDING_CONNECTION = ProviderConnection(
    protocol_field="ollama_protocol",
    host_field="ollama_host",
    port_field="ollama_port",
    base_url_field="ollama_base_url",
    default_protocol="http",
    default_host="localhost",
    default_port=11434,
    default_base_url="http://localhost:11434",
)

OLLAMA_LLM_CONNECTION = ProviderConnection(
    protocol_field="llm_ollama_protocol",
    host_field="llm_ollama_host",
    port_field="llm_ollama_port",
    base_url_field="llm_ollama_base_url",
    default_protocol="http",
    default_host="localhost",
    default_port=11434,
    default_base_url="http://localhost:11434",
    fallback_base_url_field="ollama_base_url",
    fallback_default_base_url="http://localhost:11434",
)

LLAMA_CPP_EMBEDDING_CONNECTION = ProviderConnection(
    protocol_field="llama_cpp_protocol",
    host_field="llama_cpp_host",
    port_field="llama_cpp_port",
    base_url_field="llama_cpp_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=8081,
    default_base_url="http://host.docker.internal:8081",
)

LLAMA_CPP_LLM_CONNECTION = ProviderConnection(
    protocol_field="llm_llama_cpp_protocol",
    host_field="llm_llama_cpp_host",
    port_field="llm_llama_cpp_port",
    base_url_field="llm_llama_cpp_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=8080,
    default_base_url="http://host.docker.internal:8080",
)

LMSTUDIO_CONNECTION = ProviderConnection(
    protocol_field="lmstudio_protocol",
    host_field="lmstudio_host",
    port_field="lmstudio_port",
    base_url_field="lmstudio_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=1234,
    default_base_url="http://host.docker.internal:1234",
)

LMSTUDIO_LLM_CONNECTION = ProviderConnection(
    protocol_field="llm_lmstudio_protocol",
    host_field="llm_lmstudio_host",
    port_field="llm_lmstudio_port",
    base_url_field="llm_lmstudio_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=1234,
    default_base_url="http://host.docker.internal:1234",
)

OMLX_CONNECTION = ProviderConnection(
    protocol_field="omlx_protocol",
    host_field="omlx_host",
    port_field="omlx_port",
    base_url_field="omlx_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=8000,
    default_base_url="http://host.docker.internal:8000",
)

OMLX_LLM_CONNECTION = ProviderConnection(
    protocol_field="llm_omlx_protocol",
    host_field="llm_omlx_host",
    port_field="llm_omlx_port",
    base_url_field="llm_omlx_base_url",
    default_protocol="http",
    default_host="host.docker.internal",
    default_port=8000,
    default_base_url="http://host.docker.internal:8000",
)


MODEL_PROVIDERS: dict[str, ModelProvider] = {
    "openai": ModelProvider(
        name="openai",
        label="OpenAI",
        llm_api_key_field="openai_api_key",
        embedding_api_key_field="openai_api_key",
        supports_llm=True,
        supports_embeddings=True,
        openai_compatible_chat=True,
        openai_compatible_embeddings=True,
    ),
    "anthropic": ModelProvider(
        name="anthropic",
        label="Anthropic",
        llm_api_key_field="anthropic_api_key",
        supports_llm=True,
    ),
    "ollama": ModelProvider(
        name="ollama",
        label="Ollama",
        llm_connection=OLLAMA_LLM_CONNECTION,
        embedding_connection=OLLAMA_EMBEDDING_CONNECTION,
        supports_llm=True,
        supports_embeddings=True,
        local=True,
    ),
    "llama_cpp": ModelProvider(
        name="llama_cpp",
        label="llama.cpp",
        llm_connection=LLAMA_CPP_LLM_CONNECTION,
        embedding_connection=LLAMA_CPP_EMBEDDING_CONNECTION,
        supports_llm=True,
        supports_embeddings=True,
        openai_compatible_chat=True,
        openai_compatible_embeddings=True,
        local=True,
    ),
    "lmstudio": ModelProvider(
        name="lmstudio",
        label="LM Studio",
        llm_connection=LMSTUDIO_LLM_CONNECTION,
        embedding_connection=LMSTUDIO_CONNECTION,
        supports_llm=True,
        supports_embeddings=True,
        openai_compatible_chat=True,
        openai_compatible_embeddings=True,
        local=True,
    ),
    "omlx": ModelProvider(
        name="omlx",
        label="oMLX",
        llm_connection=OMLX_LLM_CONNECTION,
        embedding_connection=OMLX_CONNECTION,
        llm_api_key_field="omlx_api_key",
        embedding_api_key_field="omlx_api_key",
        supports_llm=True,
        supports_embeddings=True,
        openai_compatible_chat=True,
        openai_compatible_embeddings=True,
        local=True,
    ),
    "github_copilot": ModelProvider(
        name="github_copilot",
        label="GitHub Copilot",
        aliases=("github_models",),
        supports_llm=True,
        openai_compatible_chat=True,
    ),
}

PROVIDER_ALIASES = {
    alias: provider.name
    for provider in MODEL_PROVIDERS.values()
    for alias in provider.aliases
}

LLM_PROVIDER_NAMES = tuple(
    name for name, provider in MODEL_PROVIDERS.items() if provider.supports_llm
)
EMBEDDING_PROVIDER_NAMES = tuple(
    name for name, provider in MODEL_PROVIDERS.items() if provider.supports_embeddings
)
LOCAL_LLM_PROVIDER_NAMES = tuple(
    name
    for name, provider in MODEL_PROVIDERS.items()
    if provider.supports_llm and provider.local
)
LOCAL_EMBEDDING_PROVIDER_NAMES = tuple(
    name
    for name, provider in MODEL_PROVIDERS.items()
    if provider.supports_embeddings and provider.local
)


def normalize_provider_name(
    provider: str | None, *, model_id: str | None = None
) -> str:
    """Normalize provider identifiers and aliases to canonical app values."""
    raw = str(provider or "").strip()

    if "::" in raw:
        raw = raw.split("::", 1)[0].strip()

    if not raw and model_id:
        model_raw = str(model_id).strip()
        if "::" in model_raw:
            raw = model_raw.split("::", 1)[0].strip()

    normalized = raw.lower().replace("-", "_")
    return PROVIDER_ALIASES.get(normalized, normalized)


def providers_equivalent(selected: str | None, actual: str | None) -> bool:
    """Return whether two provider labels should be treated as compatible."""
    selected_norm = normalize_provider_name(selected)
    actual_norm = normalize_provider_name(actual)
    if selected_norm == actual_norm:
        return True
    return {selected_norm, actual_norm} <= {"openai", "github_copilot"}


def get_provider(provider: str | None) -> ModelProvider | None:
    """Return provider metadata for a canonical or aliased provider name."""
    return MODEL_PROVIDERS.get(normalize_provider_name(provider))


def get_provider_connection(
    provider: str | None,
    role: ProviderRole,
) -> ProviderConnection | None:
    """Return endpoint connection metadata for a provider role."""
    descriptor = get_provider(provider)
    if descriptor is None:
        return None
    if role == "llm":
        return descriptor.llm_connection
    return descriptor.embedding_connection


def _read_setting(settings: Any, field: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(field, default)
    return getattr(settings, field, default)


def resolve_provider_base_url(
    settings: Any,
    provider: str | None,
    role: ProviderRole,
    *,
    override: str | None = None,
) -> str:
    """Resolve the effective configured base URL for a provider endpoint."""
    if override:
        return str(override).strip()

    connection = get_provider_connection(provider, role)
    if connection is None:
        return ""

    base_url = str(
        _read_setting(settings, connection.base_url_field, connection.default_base_url)
        or ""
    ).strip()

    if connection.fallback_base_url_field and (
        not base_url or base_url == connection.default_base_url
    ):
        fallback = str(
            _read_setting(settings, connection.fallback_base_url_field, "") or ""
        ).strip()
        if fallback and fallback != (connection.fallback_default_base_url or ""):
            return fallback

    return base_url or connection.default_base_url


def build_base_url_from_parts(
    protocol: str | None,
    host: str | None,
    port: int | None,
    connection: ProviderConnection,
) -> str:
    """Build a base URL from editable connection fields and descriptor defaults."""
    resolved_protocol = protocol or connection.default_protocol
    resolved_host = host or connection.default_host
    resolved_port = port or connection.default_port
    return f"{resolved_protocol}://{resolved_host}:{resolved_port}"
