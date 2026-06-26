import base64
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import httpx

import ragtime.core.model_limits as model_limits
from ragtime.core.model_providers import OPENAI_CODEX_RESPONSES_PAYLOAD_KEYS, get_provider
from ragtime.core.openai_codex_auth import (
    OPENAI_CODEX_RESPONSES_ENDPOINT,
    ensure_openai_codex_token_fresh,
    extract_openai_codex_account_id,
)
from ragtime.indexer import routes as indexer_routes
from ragtime.rag.components import _build_codex_request


def _jwt_with_payload(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.signature"


class FakeRepository:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    async def update_settings(self, updates: dict):
        self.updates.append(updates)
        return None


class OpenAICodexProviderTests(unittest.IsolatedAsyncioTestCase):
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

            class _FakeResponse:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return payload

            class _FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args) -> None:
                    return None

                async def get(self, url, params=None, headers=None):
                    if "backend-api/codex/models" in str(url):
                        captured["url"] = str(url)
                        captured["params"] = params or {}
                        captured["headers"] = headers or {}
                    return _FakeResponse()

            settings = SimpleNamespace(openai_codex_account_id="acct_123")
            with (
                mock.patch(
                    "ragtime.indexer.routes.ensure_openai_codex_token_fresh",
                    new=mock.AsyncMock(return_value="codex-token"),
                ),
                mock.patch("ragtime.indexer.routes.httpx.AsyncClient", _FakeClient),
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

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return payload

        class _FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args) -> None:
                return None

            async def get(self, url, headers=None):
                # Only record the Anthropic models call; the shared client is also
                # used by model_limits' models.dev fetch in this environment.
                if "api.anthropic.com" in str(url):
                    captured["url"] = url
                    captured["headers"] = headers or {}
                return _FakeResponse()

        model_limits.invalidate_cache()
        try:
            with (
                mock.patch(
                    "ragtime.indexer.routes.get_claude_code_oauth_token",
                    new=mock.AsyncMock(return_value="sk-ant-oat01-test"),
                ),
                mock.patch("ragtime.indexer.routes.httpx.AsyncClient", _FakeClient),
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


if __name__ == "__main__":
    unittest.main()
