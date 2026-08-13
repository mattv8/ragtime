import asyncio
import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
import openai
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from ollama import ResponseError

from ragtime.indexer.embedding_errors import (
    EmbeddingFailureKind,
    EmbeddingOperationError,
    GuardedEmbeddings,
    build_embedding_configuration_error,
    classify_embedding_exception,
)
from ragtime.indexer.filesystem_service import FilesystemIndexerService
from ragtime.indexer.vector_utils import (
    EmbeddingBatchTimeoutError,
    _embed_documents_guarded,
    _uses_ollama_embeddings,
    embed_documents_subbatched,
    get_embeddings_model,
)


def _auth_error() -> openai.AuthenticationError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("bad auth", response=response, body=None)


def _timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("GET", "https://example.com")
    return openai.APITimeoutError(request=request)


class _SyncEmbeddings(Embeddings):
    def __init__(self, *, query_result=None, document_result=None, query_exc=None, document_exc=None):
        self.query_result = query_result if query_result is not None else [0.1, 0.2]
        self.document_result = document_result if document_result is not None else [[0.1, 0.2]]
        self.query_exc = query_exc
        self.document_exc = document_exc
        self.query_calls = []
        self.document_calls = []

    def embed_query(self, text):
        self.query_calls.append(text)
        if self.query_exc is not None:
            raise self.query_exc
        return self.query_result

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        if self.document_exc is not None:
            raise self.document_exc
        return self.document_result


class _AsyncEmbeddings(_SyncEmbeddings):
    def __init__(self, *, query_delay=0.0, document_delay=0.0, query_cancel=False, **kwargs):
        super().__init__(**kwargs)
        self.query_delay = query_delay
        self.document_delay = document_delay
        self.query_cancel = query_cancel

    async def aembed_query(self, text):
        self.query_calls.append(text)
        if self.query_cancel:
            raise asyncio.CancelledError()
        if self.query_delay:
            await asyncio.sleep(self.query_delay)
        if self.query_exc is not None:
            raise self.query_exc
        return self.query_result

    async def aembed_documents(self, texts):
        self.document_calls.append(list(texts))
        if self.document_delay:
            await asyncio.sleep(self.document_delay)
        if self.document_exc is not None:
            raise self.document_exc
        return self.document_result


class _FakeSemaphore:
    def __init__(self):
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class EmbeddingErrorContractTests(unittest.TestCase):
    def test_embedding_operation_error_sanitizes_user_visible_text(self):
        err = EmbeddingOperationError(
            kind=EmbeddingFailureKind.CONNECTION,
            provider="ollama",
            model="nomic-embed-text:latest",
            operation="query",
            endpoint="http://private-embedding-host:11434",
            cause=RuntimeError("socket secret detail"),
        )

        self.assertEqual(
            str(err),
            "Could not connect to the Ollama embedding server for model 'nomic-embed-text:latest'. Verify the service is running and reachable.",
        )
        self.assertNotIn("private-embedding-host", str(err))
        self.assertNotIn("socket secret detail", str(err))
        self.assertNotIn("private-embedding-host", repr(err))
        self.assertNotIn("socket secret detail", repr(err))

    def test_classify_embedding_exception_walks_causal_chain(self):
        wrapped = RuntimeError("wrapped")
        wrapped.__cause__ = httpx.ConnectError("boom")
        self.assertEqual(classify_embedding_exception(wrapped), EmbeddingFailureKind.CONNECTION)

    def test_classify_embedding_exception_distinguishes_failure_kinds(self):
        cases = [
            (httpx.ConnectError("boom"), EmbeddingFailureKind.CONNECTION),
            (httpx.ConnectTimeout("boom"), EmbeddingFailureKind.CONNECTION),
            (httpx.ReadTimeout("boom"), EmbeddingFailureKind.TIMEOUT),
            (asyncio.TimeoutError(), EmbeddingFailureKind.TIMEOUT),
            (_timeout_error(), EmbeddingFailureKind.TIMEOUT),
            (_auth_error(), EmbeddingFailureKind.CONFIGURATION),
            (ValueError("api key missing"), EmbeddingFailureKind.CONFIGURATION),
            (ResponseError("model 'missing' not found", 404), EmbeddingFailureKind.CONFIGURATION),
            (RuntimeError("boom"), EmbeddingFailureKind.PROVIDER),
        ]

        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(classify_embedding_exception(exc), expected)

    def test_classify_embedding_exception_does_not_treat_generic_author_prefix_as_configuration(self):
        self.assertEqual(
            classify_embedding_exception(RuntimeError("author metadata missing from response")),
            EmbeddingFailureKind.PROVIDER,
        )

    def test_build_embedding_configuration_error_normalizes_provider_and_defaults(self):
        error = build_embedding_configuration_error(
            {"embedding_provider": "OPENAI", "embedding_model": "text-embedding-3-large"},
            operation="query",
        )

        self.assertEqual(error.kind, EmbeddingFailureKind.CONFIGURATION)
        self.assertEqual(error.provider, "openai")
        self.assertEqual(
            str(error),
            "The OpenAI embedding configuration for model 'text-embedding-3-large' is invalid or unauthorized. Verify the embedding provider settings and credentials.",
        )

        defaulted = build_embedding_configuration_error(SimpleNamespace(), operation="configure")
        self.assertEqual(defaulted.provider, "ollama")
        self.assertEqual(defaulted.model, "nomic-embed-text")


class GuardedEmbeddingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_guarded_embeddings_routes_queries_and_documents_to_separate_clients(self):
        query_client = _AsyncEmbeddings(query_result=[0.9, 0.8])
        document_client = _AsyncEmbeddings(document_result=[[0.1, 0.2], [0.3, 0.4]])
        guarded = GuardedEmbeddings(
            provider="openai",
            model="text-embedding-3-small",
            endpoint="https://private-endpoint/v1",
            query_embeddings=query_client,
            document_embeddings=document_client,
            query_timeout_seconds=0.5,
            document_timeout_seconds=1.0,
        )

        self.assertEqual(guarded.embed_query("hello"), [0.9, 0.8])
        self.assertEqual(guarded.embed_documents(["a", "b"]), [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(await guarded.aembed_query("async hello"), [0.9, 0.8])
        self.assertEqual(await guarded.aembed_documents(["c"]), [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(query_client.query_calls, ["hello", "async hello"])
        self.assertEqual(document_client.document_calls, [["a", "b"], ["c"]])

    async def test_guarded_embeddings_wraps_sync_connection_failures(self):
        guarded = GuardedEmbeddings(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://private-embedding-host:11434",
            query_embeddings=_SyncEmbeddings(query_exc=httpx.ConnectError("socket secret detail")),
            document_embeddings=_SyncEmbeddings(),
            query_timeout_seconds=0.5,
            document_timeout_seconds=1.0,
        )

        with self.assertLogs("ragtime.indexer.embedding_errors", level="ERROR") as logs:
            with self.assertRaises(EmbeddingOperationError) as cm:
                guarded.embed_query("hello")

        self.assertEqual(cm.exception.kind, EmbeddingFailureKind.CONNECTION)
        self.assertIsInstance(cm.exception.__cause__, httpx.ConnectError)
        self.assertNotIn("private-embedding-host", str(cm.exception))
        self.assertNotIn("socket secret detail", str(cm.exception))
        self.assertIn("private-embedding-host", "\n".join(logs.output))
        self.assertIn("operation=query", "\n".join(logs.output))

    async def test_guarded_embeddings_uses_authoritative_async_timeouts(self):
        guarded = GuardedEmbeddings(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://private-embedding-host:11434",
            query_embeddings=_AsyncEmbeddings(query_delay=0.05),
            document_embeddings=_AsyncEmbeddings(document_delay=0.05),
            query_timeout_seconds=0.01,
            document_timeout_seconds=0.01,
        )

        with self.assertRaises(EmbeddingOperationError) as query_cm:
            await guarded.aembed_query("hello")
        with self.assertRaises(EmbeddingOperationError) as doc_cm:
            await guarded.aembed_documents(["a"])

        self.assertEqual(query_cm.exception.kind, EmbeddingFailureKind.TIMEOUT)
        self.assertEqual(doc_cm.exception.kind, EmbeddingFailureKind.TIMEOUT)

    async def test_guarded_embeddings_preserves_cancellation(self):
        guarded = GuardedEmbeddings(
            provider="openai",
            model="text-embedding-3-small",
            endpoint="https://private-endpoint/v1",
            query_embeddings=_AsyncEmbeddings(query_cancel=True),
            document_embeddings=_AsyncEmbeddings(),
            query_timeout_seconds=0.1,
            document_timeout_seconds=0.1,
        )

        with self.assertRaises(asyncio.CancelledError):
            await guarded.aembed_query("hello")


class EmbeddingFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_embeddings_model_wraps_ollama_clients_and_preserves_provider_metadata(self):
        document_client = _AsyncEmbeddings()
        query_client = _AsyncEmbeddings()
        constructed = []

        def _factory(**kwargs):
            constructed.append(kwargs)
            return document_client if len(constructed) == 1 else query_client

        settings = {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "ollama_base_url": "http://private-embedding-host:11434",
            "ollama_embedding_timeout_seconds": 120,
        }

        with mock.patch("langchain_ollama.OllamaEmbeddings", side_effect=_factory):
            model = await get_embeddings_model(settings)

        self.assertIsInstance(model, GuardedEmbeddings)
        self.assertEqual(model.provider, "ollama")
        self.assertEqual(model.query_timeout_seconds, 45.0)
        self.assertEqual(model.document_timeout_seconds, 120.0)
        self.assertEqual(len(constructed), 2)
        self.assertEqual(constructed[0]["base_url"], "http://private-embedding-host:11434")
        self.assertIn("sync_client_kwargs", constructed[0])
        self.assertIn("async_client_kwargs", constructed[0])
        self.assertTrue(_uses_ollama_embeddings(model))

    async def test_get_embeddings_model_wraps_openai_clients_with_zero_retries(self):
        document_client = _AsyncEmbeddings()
        query_client = _AsyncEmbeddings()
        constructed = []

        def _factory(**kwargs):
            constructed.append(kwargs)
            return document_client if len(constructed) == 1 else query_client

        settings = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "openai_api_key": "secret",
            "ollama_embedding_timeout_seconds": 90,
        }

        with mock.patch("langchain_openai.OpenAIEmbeddings", side_effect=_factory):
            model = await get_embeddings_model(settings)

        self.assertIsInstance(model, GuardedEmbeddings)
        self.assertEqual(model.query_timeout_seconds, 45.0)
        self.assertEqual(model.document_timeout_seconds, 90.0)
        self.assertEqual(constructed[0]["max_retries"], 0)
        self.assertEqual(constructed[1]["max_retries"], 0)
        self.assertIsInstance(constructed[0]["timeout"], httpx.Timeout)
        self.assertIsInstance(constructed[1]["timeout"], httpx.Timeout)

    async def test_get_embeddings_model_openai_timeout_populates_request_timeout(self):
        model = OpenAIEmbeddings(model="text-embedding-3-small", api_key="secret", timeout=httpx.Timeout(12.0, connect=3.0), max_retries=0)
        self.assertEqual(model.max_retries, 0)
        self.assertIsNotNone(model.request_timeout)

    async def test_get_embeddings_model_missing_credentials_returns_none_or_configuration_error(self):
        settings = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "openai_api_key": "",
        }

        self.assertIsNone(await get_embeddings_model(settings, return_none_on_error=True))
        with self.assertRaises(EmbeddingOperationError) as cm:
            await get_embeddings_model(settings)
        self.assertEqual(cm.exception.kind, EmbeddingFailureKind.CONFIGURATION)


class SubBatchGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_documents_guarded_uses_ollama_semaphore_with_guarded_wrapper(self):
        semaphore = _FakeSemaphore()
        embeddings = GuardedEmbeddings(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://private-embedding-host:11434",
            query_embeddings=_AsyncEmbeddings(),
            document_embeddings=_AsyncEmbeddings(document_result=[[0.1, 0.2]]),
            query_timeout_seconds=0.5,
            document_timeout_seconds=0.5,
        )

        with mock.patch(
            "ragtime.core.ollama_concurrency.get_ollama_embedding_semaphore",
            new=mock.AsyncMock(return_value=semaphore),
        ):
            result = await _embed_documents_guarded(embeddings, ["a"])

        self.assertEqual(result, [[0.1, 0.2]])
        self.assertEqual(semaphore.enter_count, 1)

    async def test_embed_documents_subbatched_retries_timeout_with_smaller_batches(self):
        embeddings = GuardedEmbeddings(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://private-embedding-host:11434",
            query_embeddings=_AsyncEmbeddings(),
            document_embeddings=_AsyncEmbeddings(document_exc=asyncio.TimeoutError()),
            query_timeout_seconds=0.5,
            document_timeout_seconds=0.001,
        )

        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaises(EmbeddingBatchTimeoutError):
                await embed_documents_subbatched(embeddings, ["a", "b"], sub_batch_size=2)

    async def test_embed_documents_subbatched_does_not_downsize_connection_failures(self):
        embeddings = GuardedEmbeddings(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://private-embedding-host:11434",
            query_embeddings=_AsyncEmbeddings(),
            document_embeddings=_AsyncEmbeddings(document_exc=httpx.ConnectError("socket secret detail")),
            query_timeout_seconds=0.5,
            document_timeout_seconds=0.5,
        )

        with mock.patch.object(embeddings.document_embeddings, "aembed_documents", wraps=embeddings.document_embeddings.aembed_documents) as call_mock:
            with self.assertRaises(EmbeddingOperationError) as cm:
                await embed_documents_subbatched(embeddings, ["a", "b"], sub_batch_size=2)

        self.assertEqual(cm.exception.kind, EmbeddingFailureKind.CONNECTION)
        self.assertEqual(call_mock.await_count, 1)

    async def test_filesystem_fallback_recovers_context_length_from_wrapped_typed_error(self):
        service = FilesystemIndexerService()

        class _RecoveryEmbeddings:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                self.calls.append(list(texts))
                if len(texts) > 1:
                    raise EmbeddingOperationError(
                        kind=EmbeddingFailureKind.PROVIDER,
                        provider="ollama",
                        model="nomic-embed-text",
                        operation="documents",
                        endpoint="http://private-embedding-host:11434",
                        cause=RuntimeError("input length exceeds maximum context length socket secret detail"),
                    )
                return [[0.1, 0.2]]

        embeddings = _RecoveryEmbeddings()
        result = await service._embed_chunks_with_fallback(embeddings, ["first", "second"])

        self.assertEqual(result, [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(embeddings.calls[0], ["first", "second"])
        self.assertEqual(embeddings.calls[1:], [["first"], ["second"]])

    async def test_filesystem_fallback_still_surfaces_sanitized_final_non_context_errors(self):
        service = FilesystemIndexerService()

        class _FailingEmbeddings:
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                raise EmbeddingOperationError(
                    kind=EmbeddingFailureKind.CONNECTION,
                    provider="ollama",
                    model="nomic-embed-text",
                    operation="documents",
                    endpoint="http://private-embedding-host:11434",
                    cause=RuntimeError("socket secret detail"),
                )

        with self.assertRaises(EmbeddingOperationError) as caught:
            await service._embed_chunks_with_fallback(_FailingEmbeddings(), ["first", "second"])

        self.assertNotIn("private-embedding-host", str(caught.exception))
        self.assertNotIn("socket secret detail", str(caught.exception))

    async def test_filesystem_fallback_reraises_non_context_truncation_failures_without_retrying_smaller_sizes(self):
        service = FilesystemIndexerService()

        class _FailingEmbeddings:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                self.calls.append(list(texts))
                if len(texts) > 1:
                    raise EmbeddingOperationError(
                        kind=EmbeddingFailureKind.PROVIDER,
                        provider="ollama",
                        model="nomic-embed-text",
                        operation="documents",
                        endpoint="http://private-embedding-host:11434",
                        cause=RuntimeError("input length exceeds maximum context length"),
                    )
                raise EmbeddingOperationError(
                    kind=EmbeddingFailureKind.CONNECTION,
                    provider="ollama",
                    model="nomic-embed-text",
                    operation="documents",
                    endpoint="http://private-embedding-host:11434",
                    cause=RuntimeError("socket secret detail"),
                )

        embeddings = _FailingEmbeddings()
        with self.assertRaises(EmbeddingOperationError) as caught:
            await service._embed_chunks_with_fallback(embeddings, ["first", "second"])

        self.assertEqual(caught.exception.kind, EmbeddingFailureKind.CONNECTION)
        self.assertEqual(embeddings.calls, [["first", "second"], ["first"]])


if __name__ == "__main__":
    unittest.main()
