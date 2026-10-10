import asyncio
import signal
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.content_protection import provider
from ragtime.content_protection.models import ClassifierConfig, ContentCategory, ContentProtectionConfig, ContentProtectionError


def _config() -> ContentProtectionConfig:
    config = ContentProtectionConfig(classifier=ClassifierConfig(backend="llm", llm_model="openai_compatible::guard"))
    category = ContentCategory(
        id="finance",
        name="Finance",
        description="Internal finance",
        denial_message="Finance information is not available for this audience.",
    )
    return config.model_copy(update={"categories": [category]})


class JevCompletionRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        provider._detection_cache.clear()

    async def test_generic_provider_exception_is_safe(self) -> None:
        client = SimpleNamespace(ainvoke=mock.AsyncMock(side_effect=RuntimeError("raw provider failure")))
        with (
            mock.patch(
                "ragtime.core.app_settings.get_app_settings",
                mock.AsyncMock(return_value={"openai_compatible_base_url": "https://one", "openai_compatible_api_key": "one"}),
            ),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(64_000, False))),
            mock.patch.object(provider, "_client_for", return_value=client),
        ):
            with provider.security_classification_context(), self.assertRaises(ContentProtectionError) as raised:
                await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})
        self.assertEqual(raised.exception.code, "classifier_unavailable")

    async def test_generic_endpoint_or_key_rotation_misses_detection_cache(self) -> None:
        client = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content='{"finance":0.1}')))
        settings = [
            {"openai_compatible_base_url": "https://one", "openai_compatible_api_key": "one"},
            {"openai_compatible_base_url": "https://two", "openai_compatible_api_key": "two"},
        ]
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(side_effect=settings)),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(64_000, False))),
            mock.patch.object(provider, "_client_for", return_value=client),
        ):
            with provider.security_classification_context():
                first = await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})
                second = await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})
        self.assertFalse(first["cache_hit"])
        self.assertFalse(second["cache_hit"])
        self.assertEqual(client.ainvoke.await_count, 2)

    async def test_generic_failure_cancels_sibling_chunk(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def invoke(_messages: object, **_kwargs: object) -> SimpleNamespace:
            if not started.is_set():
                started.set()
                raise RuntimeError("first failed")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return SimpleNamespace(content='{"finance":0.1}')

        client = SimpleNamespace(ainvoke=invoke)
        with (
            mock.patch(
                "ragtime.core.app_settings.get_app_settings",
                mock.AsyncMock(return_value={"openai_compatible_base_url": "https://one", "openai_compatible_api_key": "one"}),
            ),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(64_000, False))),
            mock.patch.object(provider, "_client_for", return_value=client),
        ):
            with provider.security_classification_context(), self.assertRaises(ContentProtectionError):
                await provider.detect(_config(), {"direction": "inbound", "candidate": "x" * 20_000})
        self.assertTrue(cancelled.is_set())

    def test_chunker_real_multibyte_quote_bounds_serialization_work(self) -> None:
        envelope: dict[str, object] = {"direction": "tool_result", "candidate": ['数"据'] * 3_000}
        questions = provider.build_jev_questions(_config(), envelope)
        real_json_bytes = provider._json_bytes

        def timed_out(_signum: int, _frame: object) -> None:
            raise TimeoutError("chunker did not finish within the watchdog")

        previous = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, 5)
        try:
            with mock.patch.object(provider, "_json_bytes", wraps=real_json_bytes) as encode:
                chunks = provider._chunk_states(envelope, questions, "jev-latest")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

        # Boundary search keeps full JSON serialization to O(log n) probes per chunk.
        self.assertLess(encode.call_count, 100)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), provider._MAX_CHUNKS)
        for chunk in chunks:
            body = provider._json_bytes({"model": "jev-latest", "state": chunk, "questions": questions})
            self.assertLessEqual(len(body), provider._CONSERVATIVE_BODY_BYTES)
        for current in chunks[:-1]:
            current_chunk = current["untrusted_content_chunk_utf8"]
            assert isinstance(current_chunk, str)
            self.assertGreater(len(current_chunk.encode("utf-8")), provider._OVERLAP_BYTES)
