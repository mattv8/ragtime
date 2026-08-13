import unittest
from unittest import mock

import httpx

from ragtime.core.ollama import warmup_embedding_model


class _FakeAsyncClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise self._exc


class OllamaEmbeddingWarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_embedding_model_logs_sanitized_connection_failure(self) -> None:
        with (
            mock.patch(
                "ragtime.core.ollama.httpx.AsyncClient",
                return_value=_FakeAsyncClient(httpx.ConnectError("socket secret detail")),
            ),
            self.assertLogs("ragtime.core.ollama", level="WARNING") as logs,
        ):
            result = await warmup_embedding_model("nomic-embed-text:latest", "http://private-embedding-host:11434")

        self.assertFalse(result)
        joined = "\n".join(logs.output)
        self.assertIn("Could not connect to the Ollama embedding server for model 'nomic-embed-text:latest'", joined)
        self.assertIn("private-embedding-host", joined)
        self.assertIn("socket secret detail", joined)

    async def test_warmup_embedding_model_logs_sanitized_timeout_failure(self) -> None:
        with (
            mock.patch(
                "ragtime.core.ollama.httpx.AsyncClient",
                return_value=_FakeAsyncClient(httpx.ReadTimeout("socket secret detail")),
            ),
            self.assertLogs("ragtime.core.ollama", level="WARNING") as logs,
        ):
            result = await warmup_embedding_model("nomic-embed-text:latest", "http://private-embedding-host:11434")

        self.assertFalse(result)
        joined = "\n".join(logs.output)
        self.assertIn("The Ollama embedding server timed out while using model 'nomic-embed-text:latest'", joined)
        self.assertIn("private-embedding-host", joined)
        self.assertIn("socket secret detail", joined)

    async def test_warmup_embedding_model_logs_sanitized_provider_failure(self) -> None:
        with (
            mock.patch(
                "ragtime.core.ollama.httpx.AsyncClient",
                return_value=_FakeAsyncClient(RuntimeError("socket secret detail")),
            ),
            self.assertLogs("ragtime.core.ollama", level="WARNING") as logs,
        ):
            result = await warmup_embedding_model("nomic-embed-text:latest", "http://private-embedding-host:11434")

        self.assertFalse(result)
        joined = "\n".join(logs.output)
        self.assertIn("The Ollama embedding request failed while using model 'nomic-embed-text:latest'", joined)
        self.assertIn("private-embedding-host", joined)
        self.assertIn("socket secret detail", joined)


if __name__ == "__main__":
    unittest.main()
