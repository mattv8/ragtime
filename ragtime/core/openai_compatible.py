"""Metadata discovery for one administrator-configured OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from numbers import Real
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ragtime.core import model_limits as models_dev

_DISCOVERY_TTL_SECONDS = 300
_DISCOVERY_CACHE_MAX_ENTRIES = 128
_EMBEDDING_PROBE_TIMEOUT_SECONDS = 10.0
_EMBEDDING_DISCOVERY_TIMEOUT_SECONDS = 30.0
_EMBEDDING_PROBE_CONCURRENCY = 2
_EMBEDDING_DISCOVERY_MAX_CANDIDATES = 20
_discovery_cache: dict[str, tuple[float, list["CompatibleModel"]]] = {}


class CompatibleProviderError(Exception):
    """Safe error suitable for returning from a generic provider endpoint."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ModelLimitOverride(BaseModel):
    """Administrator-confirmed limits for one exact model id."""

    model_config = ConfigDict(extra="forbid")

    context_limit: int | None = Field(default=None, strict=True, gt=0)
    max_output_tokens: int | None = Field(default=None, strict=True, gt=0)


class CompatibleModel(BaseModel):
    id: str
    name: str
    context_limit: int | None = None
    max_output_tokens: int | None = None
    context_limit_source: str | None = None
    output_limit_source: str | None = None
    tool_call_supported: bool | None = None
    supported_endpoints: list[str] = ["/chat/completions"]


class CompatibleEmbeddingModel(BaseModel):
    """One verified embedding model at the configured compatible endpoint."""

    id: str
    name: str
    dimensions: int
    supported_endpoints: list[str] = ["/embeddings"]


def normalize_base_url(value: str) -> str:
    """Validate a full API root without modifying its path prefix."""
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        parsed.port
    except ValueError as exc:
        raise CompatibleProviderError("invalid_base_url", "Enter an absolute HTTP or HTTPS API URL.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in parsed.hostname)
    ):
        raise CompatibleProviderError("invalid_base_url", "Enter an absolute HTTP or HTTPS API URL.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _first_positive(*values: object) -> int | None:
    for value in values:
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    return None


def _nested(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _live_limits(row: dict[str, object]) -> tuple[int | None, int | None]:
    limit = _nested(row, "limit")
    top_provider = _nested(row, "top_provider")
    context = _first_positive(
        row.get("context_length"),
        row.get("context_window"),
        row.get("max_model_len"),
        limit.get("context"),
        top_provider.get("context_length"),
    )
    output = _first_positive(
        row.get("max_output_tokens"),
        row.get("max_completion_tokens"),
        limit.get("output"),
        top_provider.get("max_output_tokens"),
        top_provider.get("max_completion_tokens"),
    )
    return context, output


def _tool_capability(row: dict[str, object]) -> bool | None:
    for container in (row, _nested(row, "capabilities")):
        for key in ("tool_call", "tool_calls", "supports_tools", "tool_calling"):
            value = container.get(key)
            if isinstance(value, bool):
                return value
    supported_parameters = row.get("supported_parameters", row.get("supportedParameters"))
    if isinstance(supported_parameters, list):
        return "tools" in {str(parameter).strip().lower() for parameter in supported_parameters}
    return None


def _is_explicitly_non_chat(row: dict[str, object]) -> bool:
    kind = str(row.get("type") or row.get("object") or "").strip().lower()
    if kind in {"embedding", "embeddings", "image", "audio", "moderation"}:
        return True
    endpoints = row.get("supported_endpoints", row.get("supportedEndpoints"))
    if isinstance(endpoints, list):
        normalized = {f"/{str(endpoint).strip().lstrip('/').rstrip('/')}" for endpoint in endpoints}
        return bool(normalized) and "/chat/completions" not in normalized
    modality_outputs = (
        _nested(row, "modalities").get("output"),
        _nested(row, "architecture").get("output_modalities"),
    )
    for output in modality_outputs:
        if isinstance(output, list):
            values = {str(item).strip().lower() for item in output}
            if values and values <= {"embedding", "embeddings", "image", "images"}:
                return True
    return False


def _catalog_index(snapshot: dict[str, object] | None) -> dict[str, dict[str, object]]:
    """Index one selected catalog by exact declared model ID."""
    if snapshot is None:
        return {}
    models = snapshot.get("models")
    if not isinstance(models, dict):
        return {}
    indexed: dict[str, dict[str, object]] = {}
    for map_key, candidate in models.items():
        if not isinstance(candidate, dict):
            continue
        model_id = str(candidate.get("id") or map_key).strip()
        if model_id and model_id not in indexed:
            indexed[model_id] = candidate
    return indexed


def _validated_overrides(value: dict | None) -> dict[str, ModelLimitOverride]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CompatibleProviderError("invalid_model_limits", "Model limit overrides must be an object.")
    try:
        return {str(model_id): ModelLimitOverride.model_validate(raw) for model_id, raw in value.items()}
    except Exception as exc:
        raise CompatibleProviderError("invalid_model_limits", "Model limit overrides must contain positive integers.") from exc


async def _fetch_live_models(base_url: str, api_key: str) -> list[dict[str, object]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise CompatibleProviderError("timeout", "The provider timed out while listing models.") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        code = "authentication" if status in {401, 403} else "rate_limited" if status == 429 else "unavailable"
        raise CompatibleProviderError(code, "The provider could not list models.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise CompatibleProviderError("unavailable", "The provider returned an invalid model catalog.") from exc
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CompatibleProviderError("invalid_catalog", "The provider returned an invalid model catalog.")
    return [row for row in rows if isinstance(row, dict)]


def _supports_embeddings(row: dict[str, object]) -> bool:
    """Return only explicit embedding capability declarations, never ID guesses."""
    kind = str(row.get("type") or row.get("object") or "").strip().lower()
    if kind in {"embedding", "embeddings"}:
        return True
    endpoints = row.get("supported_endpoints", row.get("supportedEndpoints"))
    if isinstance(endpoints, list):
        normalized = {f"/{str(endpoint).strip().lstrip('/').rstrip('/')}" for endpoint in endpoints}
        if "/embeddings" in normalized:
            return True
    for container in (row, _nested(row, "capabilities")):
        for key in ("embedding", "embeddings", "supports_embeddings"):
            if container.get(key) is True:
                return True
    modalities = (_nested(row, "modalities").get("output"), _nested(row, "architecture").get("output_modalities"))
    return any(isinstance(values, list) and any(str(value).strip().lower() in {"embedding", "embeddings"} for value in values) for values in modalities)


async def probe_embedding_dimension(base_url: str, model_id: str, api_key: str = "") -> int:
    """Probe one explicitly chosen model; a successful vector is the dimension authority."""
    normalized_url = normalize_base_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=_EMBEDDING_PROBE_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post(
                f"{normalized_url}/embeddings",
                headers=headers,
                json={"model": model_id, "input": "test"},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise CompatibleProviderError("timeout", "The provider timed out while generating a test embedding.") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        code = "authentication" if status in {401, 403} else "rate_limited" if status == 429 else "unavailable"
        raise CompatibleProviderError(code, "The provider could not generate a test embedding.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise CompatibleProviderError("unavailable", "The provider returned an invalid embedding response.") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    vector = data[0].get("embedding") if isinstance(data, list) and data and isinstance(data[0], dict) else None
    if (
        not isinstance(vector, list)
        or not vector
        or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in vector)
    ):
        raise CompatibleProviderError("invalid_embedding", "The provider returned an invalid embedding response.")
    return len(vector)


async def list_embedding_models(
    base_url: str,
    api_key: str = "",
    *,
    selected_model: str = "",
) -> list[CompatibleEmbeddingModel]:
    """Discover explicit embedding candidates and verify each with /embeddings."""
    normalized_url = normalize_base_url(base_url)
    selected = str(selected_model or "").strip()
    try:
        rows = await _fetch_live_models(normalized_url, str(api_key or ""))
    except CompatibleProviderError:
        if not selected:
            raise
        rows = []
    candidates_by_id: dict[str, str] = {}
    for row in rows:
        model_id = str(row.get("id") or "").strip()
        if model_id and _supports_embeddings(row):
            candidates_by_id.setdefault(model_id, str(row.get("display_name") or row.get("name") or model_id).strip() or model_id)
    if selected:
        candidates = [(selected, candidates_by_id.pop(selected, selected)), *candidates_by_id.items()]
    else:
        candidates = list(candidates_by_id.items())
    candidates = candidates[:_EMBEDDING_DISCOVERY_MAX_CANDIDATES]
    if not candidates:
        return []

    semaphore = asyncio.Semaphore(_EMBEDDING_PROBE_CONCURRENCY)

    async def _probe_candidate(model_id: str, name: str) -> tuple[CompatibleEmbeddingModel | None, CompatibleProviderError | None]:
        try:
            async with semaphore:
                dimensions = await probe_embedding_dimension(normalized_url, model_id, str(api_key or ""))
            return CompatibleEmbeddingModel(id=model_id, name=name, dimensions=dimensions), None
        except CompatibleProviderError as exc:
            return None, exc

    tasks = [asyncio.create_task(_probe_candidate(model_id, name)) for model_id, name in candidates]
    done: set[asyncio.Task[tuple[CompatibleEmbeddingModel | None, CompatibleProviderError | None]]] = set()
    try:
        done, _pending = await asyncio.wait(tasks, timeout=_EMBEDDING_DISCOVERY_TIMEOUT_SECONDS)
        outcomes = [task.result() for task in tasks if task in done]
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    result = [model for model, _error in outcomes if model is not None]
    errors = [error for _model, error in outcomes if error is not None]
    if len(done) != len(tasks):
        errors.append(CompatibleProviderError("timeout", "The provider timed out while discovering embedding models."))
    if not result and errors:
        priority = {"authentication": 0, "rate_limited": 1, "unavailable": 2}
        raise min(errors, key=lambda error: priority.get(error.code, 3))
    return result


def _cache_key(base_url: str, api_key: str, catalog_provider: str, overrides: dict[str, ModelLimitOverride]) -> str:
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    serialized = json.dumps({key: value.model_dump() for key, value in sorted(overrides.items())}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{base_url}\0{fingerprint}\0{catalog_provider}\0{serialized}".encode()).hexdigest()


async def list_models(
    base_url: str,
    api_key: str = "",
    *,
    catalog_provider: str = "",
    model_limits: dict | None = None,
    force_refresh: bool = False,
) -> list[CompatibleModel]:
    """Discover models at one full API root with explicit metadata provenance."""
    normalized_url = normalize_base_url(base_url)
    overrides = _validated_overrides(model_limits)
    selected_catalog = str(catalog_provider or "").strip()
    key = _cache_key(normalized_url, str(api_key or ""), selected_catalog, overrides)
    now = time.monotonic()
    cached = _discovery_cache.get(key)
    if cached and not force_refresh and now - cached[0] < _DISCOVERY_TTL_SECONDS:
        return [model.model_copy(deep=True) for model in cached[1]]

    rows = await _fetch_live_models(normalized_url, str(api_key or ""))
    snapshot = await models_dev.get_models_dev_provider_snapshot(selected_catalog) if selected_catalog else None
    catalog_index = _catalog_index(snapshot)
    result: list[CompatibleModel] = []
    seen_ids: set[str] = set()
    explicitly_non_chat_ids: set[str] = set()
    for row in rows:
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        if _is_explicitly_non_chat(row):
            explicitly_non_chat_ids.add(model_id)
            continue
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        context, output = _live_limits(row)
        context_source = "provider" if context is not None else None
        output_source = "provider" if output is not None else None
        catalog = catalog_index.get(model_id)
        if catalog is not None:
            catalog_context, catalog_output = _live_limits(catalog)
            if context is None and catalog_context is not None:
                context, context_source = catalog_context, f"models.dev:{selected_catalog}"
            if output is None and catalog_output is not None:
                output, output_source = catalog_output, f"models.dev:{selected_catalog}"
        override = overrides.get(model_id)
        if override and override.context_limit is not None:
            context, context_source = override.context_limit, "configured"
        if override and override.max_output_tokens is not None:
            output, output_source = override.max_output_tokens, "configured"
        result.append(
            CompatibleModel(
                id=model_id,
                name=str(row.get("display_name") or row.get("name") or model_id).strip() or model_id,
                context_limit=context,
                max_output_tokens=output,
                context_limit_source=context_source,
                output_limit_source=output_source,
                tool_call_supported=_tool_capability(row),
            )
        )
    # A positive, administrator-confirmed context limit makes a manually
    # entered exact ID usable even when a successful /models response omits it.
    # Do not revive rows the endpoint explicitly classified as non-chat.
    for model_id, override in overrides.items():
        if model_id in seen_ids or model_id in explicitly_non_chat_ids or override.context_limit is None:
            continue
        result.append(
            CompatibleModel(
                id=model_id,
                name=model_id,
                context_limit=override.context_limit,
                max_output_tokens=override.max_output_tokens,
                context_limit_source="configured",
                output_limit_source="configured" if override.max_output_tokens is not None else None,
            )
        )
    if len(_discovery_cache) >= _DISCOVERY_CACHE_MAX_ENTRIES:
        oldest = min(_discovery_cache, key=lambda item: _discovery_cache[item][0])
        _discovery_cache.pop(oldest, None)
    _discovery_cache[key] = (now, [model.model_copy(deep=True) for model in result])
    return result


def _setting(settings: object, name: str, default: object = "") -> object:
    return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)


async def get_model(settings: object, model_id: str) -> CompatibleModel:
    """Resolve one exact configured compatible model without legacy fallbacks."""
    requested = str(model_id or "")
    configured_limits = _setting(settings, "openai_compatible_model_limits", {})
    if configured_limits is not None and not isinstance(configured_limits, dict):
        raise CompatibleProviderError("invalid_model_limits", "Model limit overrides must be an object.")
    models = await list_models(
        str(_setting(settings, "openai_compatible_base_url") or ""),
        str(_setting(settings, "openai_compatible_api_key") or ""),
        catalog_provider=str(_setting(settings, "openai_compatible_catalog_provider") or ""),
        model_limits=configured_limits,
    )
    for model in models:
        if model.id == requested:
            return model
    raise CompatibleProviderError("model_not_found", "The configured provider did not list this model.")


async def list_catalog_providers() -> list[dict[str, str]]:
    """List models.dev providers available for an explicit catalog reference."""
    snapshots = await models_dev.get_models_dev_provider_snapshots()
    providers: list[dict[str, str]] = []
    for provider_id, payload in snapshots.items():
        api = payload.get("api")
        providers.append(
            {
                "id": provider_id,
                "name": str(payload.get("name") or provider_id),
                "api": str(api) if isinstance(api, str) else "",
            }
        )
    return sorted(providers, key=lambda provider: provider["name"].casefold())
