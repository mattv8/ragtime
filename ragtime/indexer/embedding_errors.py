"""Shared embedding failure classification and guarded execution."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import Any, NoReturn

import httpx
from langchain_core.embeddings import Embeddings

from ragtime.core.model_providers import normalize_provider_name

logger = logging.getLogger(__name__)


class EmbeddingFailureKind(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    PROVIDER = "provider"


_PROVIDER_DISPLAY_NAMES = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openai_codex": "OpenAI Codex",
    "openrouter": "OpenRouter",
    "llama_cpp": "llama.cpp",
    "lmstudio": "LM Studio",
    "omlx": "oMLX",
}


def provider_display_name(provider: str) -> str:
    key = normalize_provider_name(provider)
    return _PROVIDER_DISPLAY_NAMES.get(key, provider or "Embedding provider")


def format_embedding_failure(kind: EmbeddingFailureKind, provider: str, model: str) -> str:
    provider_name = provider_display_name(provider)
    if kind == EmbeddingFailureKind.CONNECTION:
        return f"Could not connect to the {provider_name} embedding server for model '{model}'. Verify the service is running and reachable."
    if kind == EmbeddingFailureKind.TIMEOUT:
        return f"The {provider_name} embedding server timed out while using model '{model}'. Verify the service is responsive and the model is available."
    if kind == EmbeddingFailureKind.CONFIGURATION:
        return f"The {provider_name} embedding configuration for model '{model}' is invalid or unauthorized. Verify the embedding provider settings and credentials."
    return f"The {provider_name} embedding request failed while using model '{model}'. Check the embedding provider and model configuration."


class EmbeddingOperationError(RuntimeError):
    def __init__(
        self,
        *,
        kind: EmbeddingFailureKind,
        provider: str,
        model: str,
        operation: str,
        endpoint: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.kind = kind
        self.provider = provider
        self.model = model
        self.operation = operation
        self.endpoint = endpoint
        self.user_message = format_embedding_failure(kind, provider, model)
        super().__init__(self.user_message)
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return self.user_message

    def __repr__(self) -> str:
        return (
            "EmbeddingOperationError("
            f"kind={self.kind!s}, provider={self.provider!r}, model={self.model!r}, operation={self.operation!r}, "
            f"message={self.user_message!r})"
        )


def iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def classify_embedding_exception(exc: Exception) -> EmbeddingFailureKind:
    for current in iter_exception_chain(exc):
        name = type(current).__name__
        text = str(current).lower()
        if isinstance(current, (asyncio.TimeoutError, httpx.ReadTimeout)) or name == "APITimeoutError":
            return EmbeddingFailureKind.TIMEOUT
        if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout)):
            return EmbeddingFailureKind.CONNECTION
        if name == "AuthenticationError":
            return EmbeddingFailureKind.CONFIGURATION
        if name in {"AuthorizationError", "UnauthorizedError"}:
            return EmbeddingFailureKind.CONFIGURATION
        if name == "ResponseError" and ("not found" in text or "model" in text):
            return EmbeddingFailureKind.CONFIGURATION
        if any(
            token in text
            for token in (
                "api key",
                "credential",
                "unauthorized",
                "authentication",
                "authorization",
                "invalid model",
                "missing model",
            )
        ):
            return EmbeddingFailureKind.CONFIGURATION
    return EmbeddingFailureKind.PROVIDER


def _setting_value(settings: Any, key: str) -> Any:
    if settings is None:
        return None
    value = getattr(settings, key, None)
    if value is not None:
        return value
    if isinstance(settings, dict):
        return settings.get(key)
    return None


def build_embedding_configuration_error(
    settings: Any,
    *,
    operation: str,
    cause: Exception | None = None,
    endpoint: str | None = None,
) -> EmbeddingOperationError:
    provider = normalize_provider_name(_setting_value(settings, "embedding_provider") or "ollama")
    model = str(_setting_value(settings, "embedding_model") or "nomic-embed-text")
    return EmbeddingOperationError(
        kind=EmbeddingFailureKind.CONFIGURATION,
        provider=provider,
        model=model,
        operation=operation,
        endpoint=endpoint,
        cause=cause,
    )


class GuardedEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        endpoint: str | None,
        query_embeddings: Embeddings,
        document_embeddings: Embeddings,
        query_timeout_seconds: float,
        document_timeout_seconds: float,
    ) -> None:
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self.query_embeddings = query_embeddings
        self.document_embeddings = document_embeddings
        self.query_timeout_seconds = query_timeout_seconds
        self.document_timeout_seconds = document_timeout_seconds

    def _raise_wrapped(self, *, operation: str, started_at: float, exc: Exception) -> NoReturn:
        error = EmbeddingOperationError(
            kind=classify_embedding_exception(exc),
            provider=self.provider,
            model=self.model,
            operation=operation,
            endpoint=self.endpoint,
            cause=exc,
        )
        duration_ms = (time.monotonic() - started_at) * 1000.0
        logger.error(
            "Embedding operation failed provider=%s model=%s operation=%s endpoint=%s duration_ms=%.2f original_exception=%r",
            self.provider,
            self.model,
            operation,
            self.endpoint,
            duration_ms,
            exc,
            exc_info=True,
        )
        raise error from exc

    def embed_query(self, text: str) -> list[float]:
        started_at = time.monotonic()
        try:
            return self.query_embeddings.embed_query(text)
        except Exception as exc:
            self._raise_wrapped(operation="query", started_at=started_at, exc=exc)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started_at = time.monotonic()
        try:
            return self.document_embeddings.embed_documents(texts)
        except Exception as exc:
            self._raise_wrapped(operation="documents", started_at=started_at, exc=exc)

    async def aembed_query(self, text: str) -> list[float]:
        started_at = time.monotonic()
        try:
            return await asyncio.wait_for(self.query_embeddings.aembed_query(text), timeout=self.query_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._raise_wrapped(operation="query", started_at=started_at, exc=exc)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        started_at = time.monotonic()
        try:
            return await asyncio.wait_for(self.document_embeddings.aembed_documents(texts), timeout=self.document_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._raise_wrapped(operation="documents", started_at=started_at, exc=exc)
