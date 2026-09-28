from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from ragtime.core import model_limits
from ragtime.core import openai_compatible as compatible


@pytest.fixture(autouse=True)
def clear_compatible_cache() -> None:
    compatible._discovery_cache.clear()
    model_limits._models_dev_provider_snapshots.clear()


def test_normalize_base_url_preserves_path_and_rejects_unsafe_values() -> None:
    assert compatible.normalize_base_url(" https://example.test/api/v1/ ") == "https://example.test/api/v1"
    for value in (
        "",
        "example.test/v1",
        "https://user@example.test",
        "https://example.test/?x=1",
        "https://exa mple.test",
        "https://[::1",
        "ftp://example.test",
    ):
        with pytest.raises(compatible.CompatibleProviderError) as error:
            compatible.normalize_base_url(value)
        assert error.value.code == "invalid_base_url"


def test_override_requires_positive_integral_values() -> None:
    assert compatible.ModelLimitOverride(context_limit=1024).context_limit == 1024
    for value in (0, -1, 1.5, True, "100"):
        with pytest.raises(ValidationError):
            compatible.ModelLimitOverride.model_validate({"context_limit": value})


def test_sparse_live_catalog_uses_only_exact_selected_catalog_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_live(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "opaque-chat",
                "display_name": "Opaque chat",
                "context_length": 4096,
                "supported_parameters": ["temperature"],
            },
            {"id": "emb", "type": "embedding"},
        ]

    async def fetch_catalog(_provider: str) -> dict[str, object] | None:
        return {
            "id": "catalog-provider",
            "models": {
                "opaque-chat": {"id": "opaque-chat", "limit": {"context": 2048, "output": 512}},
                "opaque-chat-v2": {"id": "opaque-chat-v2", "limit": {"context": 9999}},
            },
        }

    monkeypatch.setattr(compatible, "_fetch_live_models", fetch_live)
    monkeypatch.setattr(model_limits, "get_models_dev_provider_snapshot", fetch_catalog)

    models = asyncio.run(
        compatible.list_models(
            "https://one.test/v1",
            catalog_provider="catalog-provider",
            model_limits={"opaque-chat": {"max_output_tokens": 700}},
        )
    )

    assert models == [
        compatible.CompatibleModel(
            id="opaque-chat",
            name="Opaque chat",
            context_limit=4096,
            max_output_tokens=700,
            context_limit_source="provider",
            output_limit_source="configured",
            tool_call_supported=False,
        )
    ]


def test_catalog_only_fills_unknown_live_values_and_cache_is_endpoint_and_key_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    async def fetch_live(base_url: str, api_key: str) -> list[dict[str, object]]:
        calls.append((base_url, api_key))
        return [{"id": "same-id"}]

    async def fetch_catalog(_provider: str) -> dict[str, object] | None:
        return {"models": {"same-id": {"id": "same-id", "limit": {"context": 1234, "output": 456}}}}

    monkeypatch.setattr(compatible, "_fetch_live_models", fetch_live)
    monkeypatch.setattr(model_limits, "get_models_dev_provider_snapshot", fetch_catalog)

    first = asyncio.run(compatible.list_models("https://one.test", "key-one", catalog_provider="chosen"))
    second = asyncio.run(compatible.list_models("https://two.test", "key-one", catalog_provider="chosen"))
    third = asyncio.run(compatible.list_models("https://one.test", "key-two", catalog_provider="chosen"))

    assert [model.context_limit for model in (first[0], second[0], third[0])] == [1234, 1234, 1234]
    assert first[0].context_limit_source == "models.dev:chosen"
    assert calls == [
        ("https://one.test", "key-one"),
        ("https://two.test", "key-one"),
        ("https://one.test", "key-two"),
    ]


def test_get_model_uses_settings_and_does_not_guess_unknown_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_live(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [{"id": "opaque"}]

    monkeypatch.setattr(compatible, "_fetch_live_models", fetch_live)
    result = asyncio.run(compatible.get_model(SimpleNamespace(openai_compatible_base_url="http://localhost:1234", openai_compatible_api_key=""), "opaque"))
    assert result.context_limit is None
    assert result.max_output_tokens is None


def test_rich_openrouter_metadata_and_explicit_non_chat_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_live(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "same-id",
                "context_length": 16_384,
                "top_provider": {"max_completion_tokens": 2048},
                "supported_parameters": ["tools", "temperature"],
            },
            {"id": "same-id-v2", "supportedEndpoints": ["chat/completions"]},
            {"id": "image-only", "architecture": {"output_modalities": ["image"]}},
            {"id": "embedding-only", "architecture": {"output_modalities": ["embedding"]}},
        ]

    monkeypatch.setattr(compatible, "_fetch_live_models", fetch_live)
    models = asyncio.run(compatible.list_models("https://one.test/v1"))

    assert [(model.id, model.context_limit, model.max_output_tokens, model.tool_call_supported) for model in models] == [
        ("same-id", 16_384, 2048, True),
        ("same-id-v2", None, None, None),
    ]


def test_manual_override_adds_only_unknown_chat_model_with_known_context(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_live(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [{"id": "listed-embedding", "type": "embedding"}, {"id": "duplicate"}, {"id": "duplicate"}]

    monkeypatch.setattr(compatible, "_fetch_live_models", fetch_live)
    models = asyncio.run(
        compatible.list_models(
            "https://one.test",
            model_limits={
                "manual-chat": {"context_limit": 16_384, "max_output_tokens": 1024},
                "listed-embedding": {"context_limit": 16_384},
                "no-context": {"max_output_tokens": 1024},
            },
        )
    )

    assert [(model.id, model.context_limit, model.context_limit_source) for model in models] == [
        ("duplicate", None, None),
        ("manual-chat", 16_384, "configured"),
    ]
    settings = SimpleNamespace(
        openai_compatible_base_url="https://one.test",
        openai_compatible_api_key="",
        openai_compatible_model_limits={"manual-chat": {"context_limit": 16_384}},
    )
    assert asyncio.run(compatible.get_model(settings, "manual-chat")).context_limit == 16_384


class _StubAsyncClient:
    def __init__(self, result: object, calls: list[tuple[str, dict[str, str]]], **kwargs: object) -> None:
        self.result = result
        self.calls = calls
        self.kwargs = kwargs

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        self.calls.append((url, headers))
        if isinstance(self.result, Exception):
            raise self.result
        assert isinstance(self.result, httpx.Response)
        return self.result


def test_live_fetch_uses_root_path_authorization_and_never_follows_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    clients: list[_StubAsyncClient] = []

    def client_factory(**kwargs: object) -> _StubAsyncClient:
        client = _StubAsyncClient(
            httpx.Response(200, json={"data": [{"id": "opaque"}]}, request=httpx.Request("GET", "https://provider.test/root/models")),
            calls,
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(compatible.httpx, "AsyncClient", client_factory)
    assert asyncio.run(compatible._fetch_live_models("https://provider.test/root", "private-key")) == [{"id": "opaque"}]
    assert calls == [("https://provider.test/root/models", {"Authorization": "Bearer private-key"})]
    assert clients[0].kwargs["follow_redirects"] is False


@pytest.mark.parametrize("status, exception, code", [(401, None, "authentication"), (429, None, "rate_limited"), (None, "timeout", "timeout")])
def test_live_fetch_returns_safe_structured_errors(monkeypatch: pytest.MonkeyPatch, status: int | None, exception: str | None, code: str) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    request = httpx.Request("GET", "https://provider.test/models")
    result: object = httpx.TimeoutException("private-key", request=request) if exception else httpx.Response(status or 401, request=request)
    monkeypatch.setattr(compatible.httpx, "AsyncClient", lambda **kwargs: _StubAsyncClient(result, calls, **kwargs))

    with pytest.raises(compatible.CompatibleProviderError) as error:
        asyncio.run(compatible._fetch_live_models("https://provider.test", "private-key"))
    assert error.value.code == code
    assert "private-key" not in str(error.value)
