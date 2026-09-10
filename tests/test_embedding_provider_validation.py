import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

from ragtime.core import validation


class _FakeJsonResponse:
    """Fake HTTP response object."""

    def __init__(self, status_code: int, text: str = "", json_data: dict | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self) -> dict:
        return self._json_data


class _CapturingAsyncClient:
    """Async HTTP client that captures request details and returns a fake response."""

    def __init__(self, captured: dict[str, object], status_code: int, *args: object, **kwargs: object) -> None:
        self._captured = captured
        self._status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url: str, headers: dict | None = None, json: dict | None = None):
        self._captured["url"] = url
        self._captured["headers"] = headers or {}
        self._captured["json"] = json or {}
        return _FakeJsonResponse(self._status_code)


class EmbeddingProviderValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_codex_provider_validates_selected_embedding_model(self) -> None:
        """Test that Codex embedding validation dispatches correctly and captures request."""
        settings = SimpleNamespace(
            embedding_provider="openai_codex",
            embedding_model="text-embedding-3-small",
            openai_codex_account_id="acct_123",
        )
        captured: dict[str, object] = {}

        with (
            mock.patch(
                "ragtime.indexer.repository.repository.get_settings",
                new=mock.AsyncMock(return_value=settings),
            ),
            mock.patch(
                "ragtime.core.validation.ensure_openai_codex_token_fresh",
                new=mock.AsyncMock(return_value="codex-token"),
            ),
            mock.patch("ragtime.core.validation.httpx.AsyncClient", partial(_CapturingAsyncClient, captured, 200)),
        ):
            result = await validation.validate_embedding_provider()

        self.assertTrue(result.valid)
        self.assertEqual(captured["url"], "https://api.openai.com/v1/embeddings")
        self.assertEqual(
            captured["headers"],
            {
                "Authorization": "Bearer codex-token",
                "Content-Type": "application/json",
                "ChatGPT-Account-Id": "acct_123",
            },
        )
        self.assertEqual(captured["json"], {"model": "text-embedding-3-small", "input": "test"})

    async def test_codex_validation_rejects_missing_authentication(self) -> None:
        """Test that Codex validation fails when token is empty."""
        settings = SimpleNamespace(
            embedding_provider="openai_codex",
            embedding_model="text-embedding-3-small",
            openai_codex_account_id="acct_123",
        )

        with (
            mock.patch(
                "ragtime.indexer.repository.repository.get_settings",
                new=mock.AsyncMock(return_value=settings),
            ),
            mock.patch(
                "ragtime.core.validation.ensure_openai_codex_token_fresh",
                new=mock.AsyncMock(return_value=""),
            ),
        ):
            result = await validation.validate_embedding_provider()

        self.assertFalse(result.valid)
        self.assertEqual(result.error, "OpenAI Codex is not authenticated")

    async def test_codex_validation_reports_missing_model(self) -> None:
        """Test that Codex validation reports 404 model errors with helpful message."""
        settings = SimpleNamespace(
            embedding_provider="openai_codex",
            embedding_model="text-embedding-nonexistent",
            openai_codex_account_id="acct_123",
        )
        captured: dict[str, object] = {}

        with (
            mock.patch(
                "ragtime.indexer.repository.repository.get_settings",
                new=mock.AsyncMock(return_value=settings),
            ),
            mock.patch(
                "ragtime.core.validation.ensure_openai_codex_token_fresh",
                new=mock.AsyncMock(return_value="codex-token"),
            ),
            mock.patch("ragtime.core.validation.httpx.AsyncClient", partial(_CapturingAsyncClient, captured, 404)),
        ):
            result = await validation.validate_embedding_provider()

        self.assertFalse(result.valid)
        self.assertEqual(result.error, "OpenAI Codex model 'text-embedding-nonexistent' not found")
        self.assertIn("Check the model in Settings", result.details or "")


if __name__ == "__main__":
    unittest.main()
