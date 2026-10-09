import json
import signal
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from ragtime.content_protection import provider
from ragtime.content_protection.models import ContentProtectionError


def _config(*, transport: str = "typesafe", model: str = "jev-latest", backend: str = "jev") -> SimpleNamespace:
    return SimpleNamespace(
        classifier=SimpleNamespace(backend=backend, jev=SimpleNamespace(transport=transport, model=model), llm_model="openai::classifier"),
        categories=[
            SimpleNamespace(
                id="company_finance",
                name="Company finance",
                description="Nonpublic forecasts.",
                includes=["forecasts"],
                excludes=[],
                examples=[],
                system=False,
            )
        ],
    )


def _response(request: httpx.Request) -> httpx.Response:
    question_ids = set(json.loads(request.content)["questions"])
    return httpx.Response(
        200,
        json={
            "model": "jev-1.13.0",
            "answers": {question_id: {"type": "noul", "noul": 0.8} for question_id in question_ids},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )


class JevDetectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        provider._detection_cache.clear()

    async def test_typesafe_request_uses_systemone_and_pinned_direct_model(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _response(request)

        transport = httpx.MockTransport(handler)
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"typesafe_api_key": "key"})),
            mock.patch.object(provider, "_make_http_client", side_effect=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout)),
        ):
            with provider.security_classification_context():
                result = await provider.detect(_config(model="jev-1.13"), {"direction": "inbound", "candidate": {"text": "forecast"}, "surface": "chat"})

        self.assertEqual(requests[0].url, httpx.URL("https://api.typesafe.ai/v1/systemone"))
        self.assertEqual(requests[0].headers["authorization"], "Bearer key")
        self.assertEqual(json.loads(requests[0].content)["model"], "jev-1.13.0")
        self.assertEqual(result["probabilities"], {"company_finance": 0.8})
        self.assertEqual(result["transport"], "typesafe")

    async def test_auto_prefers_direct_but_explicit_transport_does_not_fallback(self) -> None:
        self.assertEqual(provider._credential(_config(transport="auto"), {"typesafe_api_key": "direct", "openrouter_api_key": "router"})[0], "typesafe")
        with self.assertRaises(ContentProtectionError):
            provider._credential(_config(transport="openrouter"), {"typesafe_api_key": "direct"})

    async def test_openrouter_uses_its_endpoint_and_known_pinned_mapping(self) -> None:
        transport = httpx.MockTransport(_response)
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openrouter_api_key": "router"})),
            mock.patch.object(provider, "_make_http_client", side_effect=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout)),
        ):
            with provider.security_classification_context():
                result = await provider.detect(_config(transport="openrouter", model="jev-1.13"), {"direction": "inbound", "candidate": "forecast"})

        self.assertEqual(result["transport"], "openrouter")
        self.assertEqual(provider._resolved_model("openrouter", "jev-1.13"), "typesafe/jev-1.13")

    async def test_duplicate_or_incomplete_answers_fail_closed(self) -> None:
        duplicate = '{"model":"jev-1","answers":{"company_finance":{"type":"noul","noul":0.2},"company_finance":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}'
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=duplicate))
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"typesafe_api_key": "key"})),
            mock.patch.object(provider, "_make_http_client", side_effect=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout)),
            self.assertRaises(ContentProtectionError) as raised,
        ):
            with provider.security_classification_context():
                await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})
        self.assertEqual(raised.exception.code, "classifier_invalid_response")

    async def test_retries_one_rate_limited_call_when_retry_after_fits_budget(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return _response(request)

        transport = httpx.MockTransport(handler)
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"typesafe_api_key": "key"})),
            mock.patch.object(provider, "_make_http_client", side_effect=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout)),
        ):
            with provider.security_classification_context():
                result = await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})

        self.assertEqual(calls, 2)
        self.assertEqual(result["probabilities"], {"company_finance": 0.8})

    def test_retry_delay_defaults_and_rejects_nonfinite_values(self) -> None:
        self.assertEqual(provider._retry_delay(None, 0), 0.25)
        self.assertIsNone(provider._retry_delay("NaN", 0))

    async def test_cache_reuses_detection_without_retaining_audience_constraints(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(request)

        transport = httpx.MockTransport(handler)
        config = _config()
        envelope = {"direction": "inbound", "candidate": "forecast", "audience_constraints": [{"group": "private"}]}
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"typesafe_api_key": "key"})),
            mock.patch.object(provider, "_make_http_client", side_effect=lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout)),
        ):
            with provider.security_classification_context():
                first = await provider.detect(config, envelope)
                second = await provider.detect(config, {**envelope, "audience_constraints": [{"group": "other"}]})

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(calls, 1)

    def test_chunker_keeps_framing_and_overlap(self) -> None:
        chunks = provider._chunk_states({"direction": "inbound", "surface": "tool", "candidate": "x" * 20_000, "supporting_context": "y"})

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["boundary_framing"], {"direction": "inbound", "surface": "tool"})
        self.assertEqual(
            chunks[0]["untrusted_content_chunk_utf8"][-provider._OVERLAP_BYTES :], chunks[1]["untrusted_content_chunk_utf8"][: provider._OVERLAP_BYTES]
        )

    def test_chunker_sizes_quote_dense_serialized_request(self) -> None:
        config = _config()
        envelope = {"direction": "inbound", "surface": "tool", "candidate": '"\\' * 6_000}
        questions = provider.build_jev_questions(config, envelope)

        chunks = provider._chunk_states(envelope, questions, "jev-latest")

        self.assertGreater(len(chunks), 1)
        provider.validate_detection_capacity(chunks, questions, "jev-latest")
        self.assertTrue(
            all(
                len(provider._json_bytes({"model": "jev-latest", "state": chunk, "questions": questions})) <= provider._CONSERVATIVE_BODY_BYTES
                for chunk in chunks
            )
        )

    def test_chunker_makes_utf8_progress_under_escaped_body_pressure(self) -> None:
        config = _config()
        envelope = {"direction": "tool_result", "candidate": ['数"据'] * 3_000}
        questions = provider.build_jev_questions(config, envelope)

        def timed_out(_signum: int, _frame: object) -> None:
            raise TimeoutError("chunker did not make progress")

        previous = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            chunks = provider._chunk_states(envelope, questions, "jev-latest")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

        self.assertLessEqual(len(chunks), provider._MAX_CHUNKS)
        for current, following in zip(chunks, chunks[1:]):
            self.assertGreater(
                len(current["untrusted_content_chunk_utf8"].encode("utf-8")),
                provider._OVERLAP_BYTES,
            )
            self.assertNotEqual(current["untrusted_content_chunk_utf8"], following["untrusted_content_chunk_utf8"])

    def test_openrouter_rejects_nonzero_patch_pin(self) -> None:
        with self.assertRaises(ContentProtectionError):
            provider._resolved_model("openrouter", "jev-1.13.2")

    async def test_detect_requires_security_classification_context(self) -> None:
        with self.assertRaises(ContentProtectionError) as raised:
            await provider.detect(_config(), {"direction": "inbound", "candidate": "forecast"})
        self.assertEqual(raised.exception.code, "classifier_unavailable")

    async def test_generic_llm_receives_trusted_category_definitions(self) -> None:
        client = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content='{"company_finance":0.4}')))
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openai_api_key": "key"})),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(64_000, False))),
            mock.patch.object(provider, "_client_for", return_value=client),
        ):
            with provider.security_classification_context():
                result = await provider.detect(_config(backend="llm"), {"direction": "inbound", "candidate": "forecast"})

        system_message = client.ainvoke.await_args.args[0][0].content
        self.assertIn("Nonpublic forecasts.", system_message)
        self.assertEqual(result["probabilities"], {"company_finance": 0.4})

    async def test_generic_llm_accepts_native_text_block_response(self) -> None:
        client = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": '{"company_finance":0.4}'}])))
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openai_api_key": "key"})),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(64_000, False))),
            mock.patch.object(provider, "_client_for", return_value=client),
        ):
            with provider.security_classification_context():
                result = await provider.detect(_config(backend="llm"), {"direction": "inbound", "candidate": "forecast"})

        self.assertEqual(result["probabilities"], {"company_finance": 0.4})
