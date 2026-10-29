import asyncio
import base64
import json
import unittest
from datetime import datetime, timedelta, timezone
from functools import partial
from types import SimpleNamespace
from typing import cast
from unittest import mock

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

import ragtime.core.model_limits as model_limits
from ragtime.core.model_providers import OPENAI_CODEX_RESPONSES_PAYLOAD_KEYS, get_provider
from ragtime.core.openai_codex_auth import (
    OPENAI_CODEX_RESPONSES_ENDPOINT,
    ensure_openai_codex_token_fresh,
    extract_openai_codex_account_id,
)
from ragtime.core.openai_codex_transport import build_codex_responses_request, codex_response_headers
from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.embedding_errors import EmbeddingOperationError
from ragtime.indexer.vector_utils import get_embeddings_model
from ragtime.rag.components import _build_codex_request, _OpenAICodexChatOpenAI
from ragtime.rag.responses_stream import astream_responses, stream_responses


def _jwt_with_payload(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.signature"


class _FakeJsonResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _CapturingAsyncClient:
    def __init__(
        self,
        captured: dict[str, object],
        url_keyword: str,
        payload: dict,
        *args: object,
        **kwargs: object,
    ) -> None:
        self._captured = captured
        self._url_keyword = url_keyword
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def get(self, url, params=None, headers=None):
        if self._url_keyword in str(url):
            self._captured["url"] = str(url)
            self._captured["params"] = params or {}
            self._captured["headers"] = headers or {}
        return _FakeJsonResponse(self._payload)


class FakeRepository:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    async def update_settings(self, updates: dict):
        self.updates.append(updates)
        return None

    async def rotate_openai_codex_tokens_if_refresh_token_matches(self, expected_refresh_token: str, updates: dict) -> bool:
        self.updates.append(updates)
        return True


class RefreshingRepository(FakeRepository):
    def __init__(self, settings: SimpleNamespace) -> None:
        super().__init__()
        self.settings = settings

    async def get_settings(self) -> SimpleNamespace:
        return self.settings

    async def rotate_openai_codex_tokens_if_refresh_token_matches(self, expected_refresh_token: str, updates: dict) -> bool:
        if self.settings.openai_codex_refresh_token != expected_refresh_token:
            return False
        self.updates.append(updates)
        for key, value in updates.items():
            setattr(self.settings, key, value)
        return True


class _SyncResponse:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    def __enter__(self):
        return iter(self._chunks)

    def __exit__(self, *args) -> None:
        return None


class _AsyncResponse:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _ResponsesModel:
    include_response_headers = False
    output_version = "responses/v1"

    def __init__(self, response: object, async_response: object) -> None:
        self.root_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

        async def create(**kwargs):
            return async_response

        self.root_async_client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def _get_request_payload(self, messages, stop=None, **kwargs):
        return {"input": messages, **kwargs}


def _function_call_added(output_index: int, name: str, call_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.added",
        output_index=output_index,
        item=SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=arguments, id=f"fc_{call_id}"),
    )


def _function_call_delta(output_index: int, delta: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.function_call_arguments.delta", output_index=output_index, delta=delta)


def _codex_adapter(response: object, async_response: object) -> _OpenAICodexChatOpenAI:
    adapter = _OpenAICodexChatOpenAI(
        model="gpt-5.6-sol",
        api_key=SecretStr("test-key"),
        use_responses_api=True,
        output_version="responses/v1",
    )
    adapter.root_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    async def create(**kwargs):
        return async_response

    adapter.root_async_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return adapter


class OpenAICodexProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_upstream_responses_stream_corrupts_interleaved_tool_aggregation(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select 1"}'),
            _function_call_delta(1, '{"query":"docs"}'),
        ]
        adapter = _codex_adapter(_SyncResponse(chunks), _AsyncResponse([]))

        with mock.patch.object(_OpenAICodexChatOpenAI, "_stream_responses", ChatOpenAI._stream_responses):
            messages = list(adapter.stream([HumanMessage(content="test")]))

        merged_message = cast(AIMessageChunk, sum(messages[1:], messages[0]))
        self.assertTrue(any(not call["name"] for call in merged_message.tool_calls))

    def test_codex_adapter_stream_aggregates_interleaved_tool_calls(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select 1"}'),
            _function_call_delta(1, '{"query":"docs"}'),
        ]
        adapter = _codex_adapter(_SyncResponse(chunks), _AsyncResponse([]))

        messages = list(adapter.stream([HumanMessage(content="test")]))

        merged_message = cast(AIMessageChunk, sum(messages[1:], messages[0]))
        self.assertEqual(
            [(call["name"], call["id"], call["args"]) for call in merged_message.tool_calls],
            [
                ("query_production_infoscan_database", "call_sql", {"query": "select 1"}),
                ("knowledge_search", "call_search", {"query": "docs"}),
            ],
        )
        self.assertEqual(merged_message.invalid_tool_calls, [])

    async def test_codex_adapter_astream_aggregates_interleaved_tool_calls(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select 1"}'),
            _function_call_delta(1, '{"query":"docs"}'),
        ]
        adapter = _codex_adapter(_SyncResponse([]), _AsyncResponse(chunks))

        with mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()):
            messages = [message async for message in adapter.astream([HumanMessage(content="test")])]

        merged_message = cast(AIMessageChunk, sum(messages[1:], messages[0]))
        self.assertEqual(
            [(call["name"], call["id"], call["args"]) for call in merged_message.tool_calls],
            [
                ("query_production_infoscan_database", "call_sql", {"query": "select 1"}),
                ("knowledge_search", "call_search", {"query": "docs"}),
            ],
        )
        self.assertEqual(merged_message.invalid_tool_calls, [])

    async def test_codex_adapter_astream_retries_expired_token_with_stable_tool_calls(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select 1"}'),
            _function_call_delta(1, '{"query":"docs"}'),
        ]
        adapter = _codex_adapter(_SyncResponse([]), _AsyncResponse([]))
        adapter.root_async_client = SimpleNamespace(
            responses=SimpleNamespace(create=mock.AsyncMock(side_effect=[RuntimeError("token expired"), _AsyncResponse(chunks)]))
        )

        with (
            mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()),
            mock.patch.object(adapter, "_refresh_expired_copilot_token", new=mock.AsyncMock(return_value=True)) as refresh,
        ):
            messages = [message async for message in adapter.astream([HumanMessage(content="test")])]

        refresh.assert_awaited_once()
        merged_message = cast(AIMessageChunk, sum(messages[1:], messages[0]))
        self.assertEqual(
            [(call["name"], call["id"], call["args"]) for call in merged_message.tool_calls],
            [
                ("query_production_infoscan_database", "call_sql", {"query": "select 1"}),
                ("knowledge_search", "call_search", {"query": "docs"}),
            ],
        )

    def test_responses_stream_keeps_parallel_tool_chunk_indices_stable(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select '),
            _function_call_delta(1, '{"query":"docs"}'),
            _function_call_delta(0, '1"}'),
        ]
        model = _ResponsesModel(_SyncResponse(chunks), _AsyncResponse([]))

        generations = list(stream_responses(model, [HumanMessage(content="test")]))
        tool_chunks = [tool for generation in generations for tool in cast(AIMessageChunk, generation.message).tool_call_chunks]

        self.assertEqual(
            [(chunk["name"], chunk["id"], chunk["args"], chunk["index"]) for chunk in tool_chunks if chunk.get("name")],
            [
                ("query_production_infoscan_database", "call_sql", "", 0),
                ("knowledge_search", "call_search", "", 1),
            ],
        )
        self.assertEqual(
            [(chunk["args"], chunk["index"]) for chunk in tool_chunks if not chunk.get("name")],
            [('{"query":"select ', 0), ('{"query":"docs"}', 1), ('1"}', 0)],
        )
        merged_message = cast(AIMessageChunk, sum((generation.message for generation in generations[1:]), generations[0].message))
        self.assertEqual(
            [(call["name"], call["id"], call["args"]) for call in merged_message.tool_calls],
            [
                ("query_production_infoscan_database", "call_sql", {"query": "select 1"}),
                ("knowledge_search", "call_search", {"query": "docs"}),
            ],
        )
        self.assertEqual(merged_message.invalid_tool_calls, [])

    def test_responses_stream_gives_unseen_text_done_a_fresh_content_index(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", '{"query":"select 1"}'),
            SimpleNamespace(type="response.output_item.added", output_index=1, item=SimpleNamespace(type="message", id="msg_1")),
            SimpleNamespace(type="response.output_text.done", output_index=1, content_index=0, item_id="msg_1"),
        ]
        model = _ResponsesModel(_SyncResponse(chunks), _AsyncResponse([]))

        generations = list(stream_responses(model, [HumanMessage(content="test")]))

        self.assertEqual(generations[-1].message.content, [{"id": "msg_1", "index": 1}])
        merged_message = cast(AIMessageChunk, sum((generation.message for generation in generations[1:]), generations[0].message))
        self.assertEqual(merged_message.tool_calls[0]["id"], "call_sql")
        assert isinstance(merged_message.content, list)
        text_block = merged_message.content[1]
        assert isinstance(text_block, dict)
        self.assertEqual(text_block["id"], "msg_1")

    async def test_async_responses_stream_keeps_parallel_tool_chunk_indices_stable(self) -> None:
        chunks = [
            _function_call_added(0, "query_production_infoscan_database", "call_sql", ""),
            _function_call_added(1, "knowledge_search", "call_search", ""),
            _function_call_delta(0, '{"query":"select 1"}'),
            _function_call_delta(1, '{"query":"docs"}'),
        ]
        model = _ResponsesModel(_SyncResponse([]), _AsyncResponse(chunks))

        generations = [generation async for generation in astream_responses(model, [HumanMessage(content="test")])]
        tool_chunks = [tool for generation in generations for tool in cast(AIMessageChunk, generation.message).tool_call_chunks]

        self.assertEqual(
            [(chunk["name"], chunk["id"], chunk["index"]) for chunk in tool_chunks if chunk.get("name")],
            [
                ("query_production_infoscan_database", "call_sql", 0),
                ("knowledge_search", "call_search", 1),
            ],
        )
        self.assertEqual(
            [(chunk["args"], chunk["index"]) for chunk in tool_chunks if not chunk.get("name")],
            [('{"query":"select 1"}', 0), ('{"query":"docs"}', 1)],
        )
        merged_message = cast(AIMessageChunk, sum((generation.message for generation in generations[1:]), generations[0].message))
        self.assertEqual(
            [(call["name"], call["id"], call["args"]) for call in merged_message.tool_calls],
            [
                ("query_production_infoscan_database", "call_sql", {"query": "select 1"}),
                ("knowledge_search", "call_search", {"query": "docs"}),
            ],
        )
        self.assertEqual(merged_message.invalid_tool_calls, [])

    def test_extract_account_id_prefers_chatgpt_claim(self) -> None:
        token = _jwt_with_payload(
            {
                "https://api.openai.com/auth.chatgpt_account_id": "acct-from-url-claim",
                "organizations": [{"id": "org-fallback"}],
            }
        )

        self.assertEqual(
            extract_openai_codex_account_id({"id_token": token}),
            "acct-from-url-claim",
        )

    def test_codex_provider_declares_responses_payload_keys(self) -> None:
        provider = get_provider("openai_codex")
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.responses_payload_keys, OPENAI_CODEX_RESPONSES_PAYLOAD_KEYS)

    def test_codex_request_rewrites_openai_endpoint(self) -> None:
        content = json.dumps({"input": "hello"}).encode("utf-8")
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/responses",
            headers={"Authorization": "Bearer token", "ChatGPT-Account-Id": "acct_123"},
            content=content,
        )

        rewritten = _build_codex_request(request, content)

        self.assertEqual(str(rewritten.url), OPENAI_CODEX_RESPONSES_ENDPOINT)
        self.assertEqual(rewritten.headers.get("Authorization"), "Bearer token")
        self.assertEqual(rewritten.headers.get("ChatGPT-Account-Id"), "acct_123")
        self.assertEqual(json.loads(rewritten.read()), {"input": "hello", "store": False})

    def test_reusable_codex_transport_builders_expose_safe_headers_and_payload(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses", content=b'{"input":"hello","temperature":0}')

        rewritten = build_codex_responses_request(request, request.read())

        self.assertEqual(str(rewritten.url), OPENAI_CODEX_RESPONSES_ENDPOINT)
        self.assertEqual(json.loads(rewritten.read()), {"input": "hello", "store": False})
        self.assertEqual(codex_response_headers(" acct_123 "), {"User-Agent": "ragtime", "ChatGPT-Account-Id": "acct_123"})

    def test_codex_request_normalizes_payload_to_supported_fields(self) -> None:
        content = json.dumps(
            {
                "input": "hello",
                "max_output_tokens": 1000,
                "temperature": 0,
                "store": True,
            }
        ).encode("utf-8")
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/responses",
            headers={"Content-Length": str(len(content))},
            content=content,
        )

        rewritten = _build_codex_request(request, content)

        rewritten_body = rewritten.read()
        self.assertEqual(json.loads(rewritten_body), {"input": "hello", "store": False})
        self.assertEqual(rewritten.headers.get("Content-Length"), str(len(rewritten_body)))

    def test_codex_request_strips_input_item_ids_when_store_false(self) -> None:
        content = json.dumps(
            {
                "input": [
                    {"id": "rs_070e90e825564696016a3e97aa47208193a7c3f33403ed19fa", "type": "reasoning", "summary": []},
                    {"id": "fc_123", "type": "function_call", "call_id": "call_123", "name": "lookup", "arguments": "{}"},
                ],
                "store": True,
            }
        ).encode("utf-8")
        request = httpx.Request("POST", "https://api.openai.com/v1/responses", content=content)

        rewritten = _build_codex_request(request, content)

        self.assertEqual(
            json.loads(rewritten.read()),
            {
                "input": [
                    {"type": "reasoning", "summary": []},
                    {"type": "function_call", "call_id": "call_123", "name": "lookup", "arguments": "{}"},
                ],
                "store": False,
            },
        )

    async def test_live_codex_model_catalog_registers_responses_endpoint(self) -> None:
        model_limits.invalidate_cache()
        try:
            payload = {
                "models": [
                    {
                        "slug": "gpt-5.5",
                        "context_window": 272000,
                        "input_modalities": ["text", "image"],
                        "support_verbosity": True,
                    },
                    {
                        "slug": "codex-auto-review",
                        "context_window": 272000,
                        "input_modalities": ["text", "image"],
                        "support_verbosity": True,
                    },
                ]
            }
            captured: dict[str, object] = {}

            settings = SimpleNamespace(openai_codex_account_id="acct_123")
            with (
                mock.patch(
                    "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                    new=mock.AsyncMock(return_value="codex-token"),
                ),
                mock.patch(
                    "ragtime.indexer.routes.httpx.AsyncClient",
                    partial(_CapturingAsyncClient, captured, "backend-api/codex/models", payload),
                ),
            ):
                result = await indexer_routes._fetch_openai_codex_models(settings)

            self.assertTrue(result.success)
            self.assertEqual(captured["url"], indexer_routes.OPENAI_CODEX_MODELS_ENDPOINT)
            self.assertEqual(captured["params"], {"client_version": indexer_routes.OPENAI_CODEX_MODELS_CLIENT_VERSION})
            self.assertEqual(captured["headers"], {"Authorization": "Bearer codex-token", "ChatGPT-Account-Id": "acct_123"})
            self.assertEqual(result.default_model, "gpt-5.5")
            self.assertIn("gpt-5.5", {model.id for model in result.models})
            self.assertIn("codex-auto-review", {model.id for model in result.models})
            codex_model = next(model for model in result.models if model.id == "gpt-5.5")
            self.assertEqual(codex_model.model_provider, "openai_codex")
            self.assertEqual(codex_model.host_provider_label, "OpenAI Codex")
            self.assertEqual(codex_model.model_provider_label, "OpenAI Codex")
            self.assertEqual(codex_model.supported_endpoints, ["/responses"])
            self.assertTrue(codex_model.reasoning_supported)
        finally:
            model_limits.invalidate_cache()

    async def test_live_codex_model_catalog_uses_maximum_context_override(self) -> None:
        payload = {
            "models": [
                {
                    "slug": "gpt-6-astra",
                    "context_window": 272_000,
                    "max_context_window": 872_000,
                },
                {
                    "slug": "gpt-6-default",
                    "context_window": 272_000,
                },
                {
                    "slug": "gpt-6-invalid-maximum",
                    "context_window": 272_000,
                    "max_context_window": "invalid",
                },
            ]
        }
        settings = SimpleNamespace(openai_codex_account_id="acct_123")

        async def load_fallback_metadata() -> None:
            model_limits.update_model_limit("gpt-6-astra", 272_000)
            model_limits.update_model_output_limit("gpt-6-astra", 128_000)

        with (
            mock.patch.dict(model_limits._model_limits_cache, clear=True),
            mock.patch.dict(model_limits._model_output_limits_cache, clear=True),
            mock.patch.object(model_limits, "_cache_loaded", True),
            mock.patch(
                "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                new=mock.AsyncMock(return_value="codex-token"),
            ),
            mock.patch(
                "ragtime.indexer.routes.ensure_model_metadata_loaded",
                new=mock.AsyncMock(side_effect=load_fallback_metadata),
            ) as ensure_metadata,
            mock.patch(
                "ragtime.indexer.routes.httpx.AsyncClient",
                partial(_CapturingAsyncClient, {}, "backend-api/codex/models", payload),
            ),
        ):
            result = await indexer_routes._fetch_openai_codex_models(settings)

            limits_by_id = {model.id: model.context_limit for model in result.models}
            output_limits_by_id = {model.id: model.max_output_tokens for model in result.models}
            self.assertEqual(
                limits_by_id,
                {
                    "gpt-6-astra": 872_000,
                    "gpt-6-default": 272_000,
                    "gpt-6-invalid-maximum": 272_000,
                },
            )
            self.assertEqual(output_limits_by_id["gpt-6-astra"], 128_000)
            self.assertEqual(model_limits._model_limits_cache["gpt-6-astra"], 872_000)
            self.assertEqual(model_limits._model_limits_cache["gpt-6-default"], 272_000)
            self.assertEqual(model_limits._model_limits_cache["gpt-6-invalid-maximum"], 272_000)
            self.assertEqual(model_limits._model_output_limits_cache["gpt-6-astra"], 128_000)
            ensure_metadata.assert_awaited_once()

    async def test_live_codex_model_catalog_compacts_gpt_56_variant_display_names(self) -> None:
        model_limits.invalidate_cache()
        try:
            payload = {
                "models": [
                    {
                        "slug": "gpt-5.6-terra",
                        "context_window": 272000,
                        "input_modalities": ["text", "image"],
                        "support_verbosity": True,
                    },
                    {
                        "slug": "gpt-5.6-luna",
                        "context_window": 272000,
                        "input_modalities": ["text", "image"],
                        "support_verbosity": True,
                    },
                    {
                        "slug": "gpt-5.6-sol",
                        "context_window": 272000,
                        "input_modalities": ["text", "image"],
                        "support_verbosity": True,
                    },
                ]
            }

            settings = SimpleNamespace(openai_codex_account_id="acct_123")
            with (
                mock.patch(
                    "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                    new=mock.AsyncMock(return_value="codex-token"),
                ),
                mock.patch(
                    "ragtime.indexer.routes.httpx.AsyncClient",
                    partial(_CapturingAsyncClient, {}, "backend-api/codex/models", payload),
                ),
            ):
                result = await indexer_routes._fetch_openai_codex_models(settings)

            self.assertTrue(result.success)
            label_by_id = {model.id: model.display_name for model in result.models}
            self.assertEqual(
                label_by_id,
                {
                    "gpt-5.6-terra": "5.6 Terra",
                    "gpt-5.6-luna": "5.6 Luna",
                    "gpt-5.6-sol": "5.6 Sol",
                },
            )

            for slug in ("gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol"):
                with self.subTest(slug=slug):
                    codex_model = next(model for model in result.models if model.id == slug)
                    self.assertEqual(codex_model.model_provider, "openai_codex")
                    self.assertEqual(codex_model.host_provider_label, "OpenAI Codex")
                    self.assertEqual(codex_model.model_provider_label, "OpenAI Codex")
                    self.assertEqual(codex_model.supported_endpoints, ["/responses"])
                    self.assertTrue(codex_model.reasoning_supported)
        finally:
            model_limits.invalidate_cache()

    async def test_codex_model_catalog_reports_live_fetch_failure(self) -> None:
        model_limits.invalidate_cache()
        try:
            settings = SimpleNamespace(openai_codex_account_id="")

            class _FailingClient:
                def __init__(self, *args, **kwargs) -> None:
                    raise httpx.HTTPError("boom")

            with (
                mock.patch(
                    "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                    new=mock.AsyncMock(return_value="codex-token"),
                ),
                mock.patch("ragtime.indexer.routes.httpx.AsyncClient", _FailingClient),
            ):
                result = await indexer_routes._fetch_openai_codex_models(settings)

            self.assertFalse(result.success)
            self.assertIn("Failed to fetch OpenAI Codex models", result.message)
            self.assertEqual(result.models, [])
        finally:
            model_limits.invalidate_cache()

    async def test_codex_embedding_catalog_probes_supported_models(self) -> None:
        captured_models: list[str] = []

        class _FakeResponse:
            status_code = 200

        class _FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args) -> None:
                return None

            async def post(self, url, headers=None, json=None):
                captured_models.append((json or {}).get("model", ""))
                return _FakeResponse()

        settings = SimpleNamespace(openai_codex_account_id="acct_123")
        with (
            mock.patch(
                "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                new=mock.AsyncMock(return_value="codex-token"),
            ),
            mock.patch("ragtime.indexer.routes.httpx.AsyncClient", _FakeClient),
        ):
            result = await indexer_routes._fetch_openai_codex_embedding_models(settings)

        self.assertTrue(result.success)
        self.assertEqual(captured_models, ["text-embedding-3-small", "text-embedding-3-large"])
        self.assertEqual(result.default_model, "text-embedding-3-small")
        self.assertEqual([(model.id, model.dimensions) for model in result.models], [("text-embedding-3-small", 1536), ("text-embedding-3-large", 3072)])

    async def test_claude_code_live_fetch_uses_oauth_models_endpoint(self) -> None:
        # Same mechanism as OpenAI/OpenRouter: query the provider's live
        # /v1/models endpoint with the configured credential.
        payload = {
            "data": [
                {"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
                {"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5"},
            ]
        }
        captured: dict[str, object] = {}

        model_limits.invalidate_cache()
        try:
            with (
                mock.patch(
                    "ragtime.indexer.routes.get_claude_code_oauth_token",
                    new=mock.AsyncMock(return_value="sk-ant-oat01-test"),
                ),
                mock.patch(
                    "ragtime.indexer.routes.httpx.AsyncClient",
                    partial(_CapturingAsyncClient, captured, "api.anthropic.com", payload),
                ),
            ):
                result = await indexer_routes._fetch_claude_code_models()

            self.assertTrue(result.success)
            self.assertTrue(str(captured["url"]).endswith("/v1/models"))
            headers = captured["headers"]
            assert isinstance(headers, dict)
            self.assertEqual(headers.get("Authorization"), "Bearer sk-ant-oat01-test")
            self.assertEqual(headers.get("anthropic-beta"), "oauth-2025-04-20")
            self.assertNotIn("x-api-key", headers)

            model_ids = {model.id for model in result.models}
            self.assertIn("claude-opus-4-8", model_ids)
            self.assertIn("claude-sonnet-4-6", model_ids)
            claude_model = next(model for model in result.models if model.id == "claude-opus-4-8")
            self.assertEqual(claude_model.model_provider, "claude_code")
            self.assertEqual(claude_model.host_provider_label, "Claude Code")
            self.assertEqual(claude_model.model_provider_label, "Claude Code")
        finally:
            model_limits.invalidate_cache()

    async def test_claude_code_requires_token_for_model_catalog(self) -> None:
        model_limits.invalidate_cache()
        try:
            with mock.patch(
                "ragtime.indexer.routes.get_claude_code_oauth_token",
                new=mock.AsyncMock(return_value=None),
            ):
                result = await indexer_routes._fetch_claude_code_models()

            self.assertFalse(result.success)
            self.assertIn("Claude Code is not authenticated", result.message)
            self.assertEqual(result.models, [])
        finally:
            model_limits.invalidate_cache()

    async def test_ensure_codex_token_fresh_persists_refreshed_tokens(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            openai_codex_account_id="",
        )
        repository = FakeRepository()
        id_token = _jwt_with_payload({"chatgpt_account_id": "acct_abc"})

        with mock.patch(
            "ragtime.core.openai_codex_auth.refresh_openai_codex_tokens",
            new=mock.AsyncMock(
                return_value={
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "expires_in": 3600,
                    "id_token": id_token,
                }
            ),
        ):
            token = await ensure_openai_codex_token_fresh(settings=settings, repository=repository)

        self.assertEqual(token, "fresh-access")
        self.assertEqual(len(repository.updates), 1)
        self.assertEqual(repository.updates[0]["openai_codex_access_token"], "fresh-access")
        self.assertEqual(repository.updates[0]["openai_codex_refresh_token"], "fresh-refresh")
        self.assertEqual(repository.updates[0]["openai_codex_account_id"], "acct_abc")

    async def test_ensure_codex_token_fresh_returns_empty_when_refresh_is_unauthorized(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        repository = FakeRepository()
        request = httpx.Request("POST", "https://auth.openai.com/oauth/token")
        error = httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))

        with mock.patch(
            "ragtime.core.openai_codex_auth.refresh_openai_codex_tokens",
            new=mock.AsyncMock(side_effect=error),
        ):
            token = await ensure_openai_codex_token_fresh(settings=settings, repository=repository)

        self.assertEqual(token, "")
        self.assertEqual(repository.updates, [])

    async def test_ensure_codex_token_fresh_keeps_valid_token_when_refresh_connection_fails(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="still-valid-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        repository = FakeRepository()

        with mock.patch(
            "ragtime.core.openai_codex_auth.refresh_openai_codex_tokens",
            new=mock.AsyncMock(side_effect=httpx.ConnectError("connection failed")),
        ):
            token = await ensure_openai_codex_token_fresh(settings=settings, repository=repository)

        self.assertEqual(token, "still-valid-access")
        self.assertEqual(repository.updates, [])

    async def test_concurrent_refresh_rotates_tokens_once_and_returns_the_saved_token(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="rotating-refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            openai_codex_account_id="acct_before",
        )
        repository = RefreshingRepository(settings)
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()

        async def refresh(refresh_token: str) -> dict[str, object]:
            self.assertEqual(refresh_token, "rotating-refresh-token")
            refresh_started.set()
            await release_refresh.wait()
            return {"access_token": "fresh-access", "refresh_token": "rotated-refresh", "expires_in": 3600}

        with mock.patch("ragtime.core.openai_codex_auth.refresh_openai_codex_tokens", side_effect=refresh) as refresh_tokens:
            first = asyncio.create_task(ensure_openai_codex_token_fresh(repository=repository))
            await refresh_started.wait()
            second = asyncio.create_task(ensure_openai_codex_token_fresh(repository=repository))
            release_refresh.set()
            tokens = await asyncio.gather(first, second)

        self.assertEqual(tokens, ["fresh-access", "fresh-access"])
        refresh_tokens.assert_awaited_once_with("rotating-refresh-token")
        self.assertEqual(len(repository.updates), 1)
        self.assertEqual(settings.openai_codex_refresh_token, "rotated-refresh")

    async def test_cancelled_waiter_does_not_interrupt_token_rotation(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            openai_codex_account_id="",
        )
        repository = RefreshingRepository(settings)
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()

        async def refresh(_: str) -> dict[str, object]:
            refresh_started.set()
            await release_refresh.wait()
            return {"access_token": "fresh-access", "refresh_token": "rotated-refresh"}

        with mock.patch("ragtime.core.openai_codex_auth.refresh_openai_codex_tokens", side_effect=refresh):
            cancelled_waiter = asyncio.create_task(ensure_openai_codex_token_fresh(repository=repository))
            await refresh_started.wait()
            cancelled_waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled_waiter
            winner = asyncio.create_task(ensure_openai_codex_token_fresh(repository=repository))
            release_refresh.set()
            self.assertEqual(await winner, "fresh-access")

        self.assertEqual(len(repository.updates), 1)
        self.assertEqual(settings.openai_codex_refresh_token, "rotated-refresh")

    async def test_failed_refresh_uses_a_token_rotated_by_another_process(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            openai_codex_account_id="",
        )
        repository = RefreshingRepository(settings)

        async def failed_refresh(_: str) -> dict[str, object]:
            settings.openai_codex_access_token = "other-process-token"
            settings.openai_codex_refresh_token = "other-process-refresh"
            settings.openai_codex_token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            raise httpx.ConnectError("connection failed")

        with mock.patch("ragtime.core.openai_codex_auth.refresh_openai_codex_tokens", side_effect=failed_refresh):
            self.assertEqual(await ensure_openai_codex_token_fresh(repository=repository), "other-process-token")

    async def test_stale_cas_rejection_returns_the_persisted_winning_token(self) -> None:
        settings = SimpleNamespace(
            openai_codex_access_token="expired-access",
            openai_codex_refresh_token="refresh-token",
            openai_codex_token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            openai_codex_account_id="",
        )
        repository = RefreshingRepository(settings)

        async def reject_stale_rotation(expected_refresh_token: str, updates: dict) -> bool:
            self.assertEqual(expected_refresh_token, "refresh-token")
            self.assertEqual(updates["openai_codex_access_token"], "unpersisted-token")
            settings.openai_codex_access_token = "winning-token"
            settings.openai_codex_refresh_token = "winning-refresh-token"
            settings.openai_codex_token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            return False

        repository.rotate_openai_codex_tokens_if_refresh_token_matches = reject_stale_rotation
        with mock.patch(
            "ragtime.core.openai_codex_auth.refresh_openai_codex_tokens",
            new=mock.AsyncMock(return_value={"access_token": "unpersisted-token", "refresh_token": "unpersisted-refresh-token"}),
        ):
            token = await ensure_openai_codex_token_fresh(repository=repository)

        self.assertEqual(token, "winning-token")
        self.assertEqual(repository.updates, [])

    async def test_codex_embedding_refresh_exception_returns_none_when_allowed(self) -> None:
        settings = SimpleNamespace(embedding_provider="openai_codex", embedding_model="text-embedding-3-small")

        with mock.patch(
            "ragtime.indexer.vector_utils.ensure_openai_codex_token_fresh",
            new=mock.AsyncMock(side_effect=RuntimeError("refresh failed")),
        ):
            model = await get_embeddings_model(settings, return_none_on_error=True)

        self.assertIsNone(model)

    async def test_codex_embedding_refresh_exception_raises_configuration_error(self) -> None:
        settings = SimpleNamespace(embedding_provider="openai_codex", embedding_model="text-embedding-3-small")

        with mock.patch(
            "ragtime.indexer.vector_utils.ensure_openai_codex_token_fresh",
            new=mock.AsyncMock(side_effect=RuntimeError("refresh failed")),
        ):
            with self.assertRaises(EmbeddingOperationError):
                await get_embeddings_model(settings)


if __name__ == "__main__":
    unittest.main()
