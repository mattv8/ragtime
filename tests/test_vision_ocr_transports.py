import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from PIL import Image

from ragtime.core import vision_models


class VisionOcrTransportTests(unittest.TestCase):
    def test_native_schemas_are_strict_and_require_each_object_field(self) -> None:
        schema = cast(dict[str, Any], vision_models.OCR_WITH_CLASSIFICATION_SCHEMA)
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["properties"]["extracted_text"]["items"]["additionalProperties"])
        self.assertIn("objects", schema["required"])
        self.assertIn("setting", schema["required"])

    def test_preprocess_preserves_below_limit_png_bytes_and_mime(self) -> None:
        image = Image.new("RGBA", (690, 942), (1, 2, 3, 4))
        payload = vision_models.io.BytesIO()
        image.save(payload, format="PNG")
        original = payload.getvalue()

        processed, mime_type = vision_models.preprocess_image(original, source_format=".png")

        self.assertEqual(processed, original)
        self.assertEqual(mime_type, "image/png")

    def test_bmp_is_converted_to_supported_mime(self) -> None:
        image = Image.new("RGB", (12, 12), (1, 2, 3))
        payload = vision_models.io.BytesIO()
        image.save(payload, format="BMP")

        processed, mime_type = vision_models._prepare_vision_image_content(
            payload.getvalue(), provider="anthropic", preprocess=True, max_dimension=2048, source_format=".bmp"
        )

        self.assertEqual(mime_type, "image/jpeg")
        self.assertLessEqual(vision_models._base64_size(processed), vision_models.ANTHROPIC_MAX_ENCODED_IMAGE_BYTES)

    def test_structured_result_preserves_block_types_and_classification_separately(self) -> None:
        result = vision_models._vision_ocr_result_from_dict(
            {
                "extracted_text": [
                    {"type": "Title", "content": "Schedule"},
                    {"type": "Table", "content": "| A | B |\n| --- | --- |\n|  | 2 |"},
                ],
                "image_type": "document",
                "subject": "schedule",
                "description": "A schedule page.",
                "tags": ["schedule"],
            }
        )

        self.assertEqual(
            result.get_semantic_segments(),
            [
                ("title", "Schedule"),
                ("table", "| A | B |\n| --- | --- |\n|  | 2 |"),
                ("classification", "[ImageType] document\n[Subject] schedule\n[Description] A schedule page.\n[Tags] schedule"),
            ],
        )

    def test_structured_result_rejects_malformed_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "extracted_text"):
            vision_models._vision_ocr_result_from_dict({"extracted_text": [{"type": "Title"}]})


class _RecordingHttpClient:
    """Stands in for httpx.AsyncClient and records the single POST it receives."""

    def __init__(self, response_json: dict[str, Any], captured: dict[str, Any]) -> None:
        self._response_json = response_json
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        self._captured.update({"url": url, **kwargs})
        request = vision_models.httpx.Request("POST", url)
        return vision_models.httpx.Response(200, json=self._response_json, request=request)


ANTHROPIC_END_TURN = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]}
OPENAI_STOP = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
OLLAMA_DONE = {"response": "{}", "done": True, "done_reason": "stop"}


class VisionOcrOutputBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Every test mocks catalog metadata; nothing may reach models.dev or a provider.
        patcher = patch.object(vision_models, "get_output_limit", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        vision_models._ocr_output_budget_cache.clear()
        self.addCleanup(vision_models._ocr_output_budget_cache.clear)

    async def test_budget_lookup_is_memoized_per_model(self) -> None:
        with patch.object(vision_models, "get_output_limit", return_value=None) as limit:
            for _ in range(3):
                self.assertEqual(await vision_models._resolve_ocr_output_budget("unknown"), vision_models.OCR_FALLBACK_OUTPUT_BUDGET)
        self.assertEqual(limit.await_count, 1)

    async def test_expired_budget_is_looked_up_again(self) -> None:
        vision_models._ocr_output_budget_cache["model"] = (
            time.monotonic() - vision_models.OCR_OUTPUT_BUDGET_CACHE_TTL_SECONDS - 1,
            1,
        )
        with patch.object(vision_models, "get_output_limit", return_value=2048) as limit:
            self.assertEqual(await vision_models._resolve_ocr_output_budget("model"), 2048)
        limit.assert_awaited_once_with("model")

    async def _dispatch(self, provider: str, model: str, output_cap: int | None, client: object) -> str:
        base_url = "https://ollama.test" if provider == "ollama" else f"https://{provider}.test"
        with (
            patch.object(vision_models, "get_output_limit", return_value=output_cap),
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
        ):
            return await vision_models._request_vision_response_once(
                normalized_provider=provider,
                base_url=base_url,
                model=model,
                api_key="key",
                prompt="read",
                schema=None,
                image_b64="YWJj",
                image_mime_type="image/png",
                timeout=1,
            )

    async def test_budget_uses_known_small_cap(self) -> None:
        with patch.object(vision_models, "get_output_limit", return_value=1024):
            self.assertEqual(await vision_models._resolve_ocr_output_budget("small"), 1024)

    async def test_budget_is_bounded_by_ceiling_for_large_cap(self) -> None:
        with patch.object(vision_models, "get_output_limit", return_value=64000):
            self.assertEqual(await vision_models._resolve_ocr_output_budget("large"), vision_models.OCR_OUTPUT_BUDGET_CEILING)

    async def test_budget_falls_back_when_cap_unknown_or_lookup_fails(self) -> None:
        with patch.object(vision_models, "get_output_limit", return_value=None):
            self.assertEqual(await vision_models._resolve_ocr_output_budget("unknown"), vision_models.OCR_FALLBACK_OUTPUT_BUDGET)
        with patch.object(vision_models, "get_output_limit", side_effect=RuntimeError("catalog down")):
            self.assertEqual(await vision_models._resolve_ocr_output_budget("failing"), vision_models.OCR_FALLBACK_OUTPUT_BUDGET)

    async def test_anthropic_payload_uses_known_small_cap_and_unknown_fallback(self) -> None:
        captured: dict[str, Any] = {}
        await self._dispatch("anthropic", "claude-small", 1024, _RecordingHttpClient(ANTHROPIC_END_TURN, captured))
        self.assertEqual(captured["json"]["max_tokens"], 1024)

        captured = {}
        await self._dispatch("anthropic", "claude-unknown", None, _RecordingHttpClient(ANTHROPIC_END_TURN, captured))
        self.assertEqual(captured["json"]["max_tokens"], vision_models.OCR_FALLBACK_OUTPUT_BUDGET)

    async def test_anthropic_max_tokens_stop_still_raises(self) -> None:
        truncated = {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "{"}]}
        with self.assertRaisesRegex(RuntimeError, "incomplete or refused"):
            await self._dispatch("anthropic", "claude-small", 1024, _RecordingHttpClient(truncated, {}))

    async def test_ollama_payload_uses_budget_as_num_predict(self) -> None:
        captured: dict[str, Any] = {}
        await self._dispatch("ollama", "llava", 2048, _RecordingHttpClient(OLLAMA_DONE, captured))
        self.assertEqual(captured["json"]["options"]["num_predict"], 2048)

        captured = {}
        await self._dispatch("ollama", "llava-unknown", None, _RecordingHttpClient(OLLAMA_DONE, captured))
        self.assertEqual(captured["json"]["options"]["num_predict"], vision_models.OCR_FALLBACK_OUTPUT_BUDGET)

    async def test_ollama_length_stop_still_raises(self) -> None:
        truncated = {"response": "{", "done": True, "done_reason": "length"}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            await self._dispatch("ollama", "llava", 2048, _RecordingHttpClient(truncated, {}))

    async def test_chat_openai_payload_uses_max_tokens_with_temperature(self) -> None:
        captured: dict[str, Any] = {}
        await self._dispatch("openai", "gpt-4o", 3000, _RecordingHttpClient(OPENAI_STOP, captured))
        payload = captured["json"]
        self.assertEqual(payload["max_tokens"], 3000)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertNotIn("max_completion_tokens", payload)

    async def test_reasoning_openai_payload_uses_completion_tokens_without_temperature(self) -> None:
        for model in ("o3-mini", "gpt-5"):
            with self.subTest(model=model):
                captured: dict[str, Any] = {}
                await self._dispatch("openai", model, 3000, _RecordingHttpClient(OPENAI_STOP, captured))
                payload = captured["json"]
                self.assertEqual(payload["max_completion_tokens"], 3000)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn("temperature", payload)

    async def test_openai_finish_length_still_raises(self) -> None:
        truncated = {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
        with self.assertRaisesRegex(RuntimeError, "incomplete or refused"):
            await self._dispatch("openai", "gpt-4o", 1024, _RecordingHttpClient(truncated, {}))

    async def test_claude_code_adapter_receives_budget(self) -> None:
        captured: dict[str, Any] = {}

        class FakeClaudeCodeModel:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

            async def ainvoke(self, messages):
                return SimpleNamespace(content="{}")

        with (
            patch("ragtime.core.vision_models.ClaudeCodeChatModel", FakeClaudeCodeModel),
            patch.object(vision_models, "get_output_limit", return_value=1024),
        ):
            result = await vision_models._request_vision_response_once(
                normalized_provider="claude_code",
                base_url=None,
                model="claude-small",
                api_key=None,
                prompt="read",
                schema=None,
                image_b64="YWJj",
                image_mime_type="image/png",
                timeout=1,
            )

        self.assertEqual(result, "{}")
        self.assertEqual(captured["max_tokens"], 1024)

    async def test_codex_payload_never_sends_max_output_tokens(self) -> None:
        captured: dict[str, Any] = {}

        class FakeStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def raise_for_status(self) -> None:
                return None

            async def aiter_lines(self):
                yield 'data: {"type":"response.output_text.delta","delta":"{}"}'
                yield 'data: {"type":"response.completed","response":{"status":"completed"}}'

        class FakeCodexClient(_RecordingHttpClient):
            def stream(self, method, url, content, headers):
                captured.update({"url": url, "content": content})
                return FakeStream()

        with (
            patch("ragtime.core.vision_models.ensure_openai_codex_token_fresh", AsyncMock(return_value="token")),
            patch.object(vision_models, "get_app_settings", AsyncMock(return_value={})),
            patch.object(vision_models.httpx, "AsyncClient", return_value=FakeCodexClient({}, {})),
            patch.object(vision_models, "get_output_limit", return_value=1024) as limit,
        ):
            result = await vision_models._request_codex_vision_response(
                base_url="https://codex.test",
                model="gpt-5",
                prompt="read",
                image_b64="YWJj",
                image_mime_type="image/png",
                schema=None,
                timeout=1,
            )

        payload = json.loads(captured["content"])
        self.assertEqual(result, "{}")
        self.assertTrue(payload["stream"])
        self.assertNotIn("max_output_tokens", payload)
        limit.assert_not_called()


class VisionOcrTransportAsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Keep existing transport tests off the catalog; budget behavior is covered separately.
        patcher = patch.object(vision_models, "get_output_limit", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        vision_models._ocr_output_budget_cache.clear()
        self.addCleanup(vision_models._ocr_output_budget_cache.clear)

    async def test_public_structured_entrypoint_dispatches_anthropic_schema(self) -> None:
        captured: dict[str, Any] = {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                captured.update(kwargs)
                request = vision_models.httpx.Request("POST", url)
                return vision_models.httpx.Response(
                    200,
                    json={"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"extracted_text":[]}'}]},
                    request=request,
                )

        image = vision_models.io.BytesIO()
        Image.new("RGB", (1, 1)).save(image, format="PNG")
        with patch.object(vision_models.httpx, "AsyncClient", return_value=Client()):
            result = await vision_models.extract_text_with_vision_structured(
                image.getvalue(), "https://anthropic.test", "claude", provider="anthropic", api_key="key"
            )

        self.assertEqual(result.extracted_text, [])
        self.assertIn("output_config", captured["json"])

    async def test_native_providers_enable_schema_from_public_entrypoint(self) -> None:
        for provider in ("anthropic", "claude_code", "openai_codex"):
            self.assertTrue(await vision_models._supports_structured_vision_output(provider, "vision", "https://example.test"))

    async def test_codex_sse_requires_completed_response_and_collects_delta(self) -> None:
        request = vision_models.httpx.Request("POST", "https://codex.test/responses")
        response = vision_models.httpx.Response(
            200,
            content=(
                b'data: {"type":"response.output_text.delta","delta":"{\\"extracted_text\\":[]}"}\n'
                b'data: {"type":"response.completed","response":{"status":"completed"}}\n'
            ),
            request=request,
        )

        self.assertEqual(
            await vision_models._collect_codex_sse_text(response),
            '{"extracted_text":[]}',
        )

    async def test_retry_honors_retry_after_then_succeeds(self) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request = vision_models.httpx.Request("POST", "https://example.test")
                response = vision_models.httpx.Response(429, headers={"Retry-After": "0"}, request=request)
                raise vision_models.httpx.HTTPStatusError("rate limited", request=request, response=response)
            return "ok"

        self.assertEqual(await vision_models._retry_vision_request(operation, 1), "ok")
        self.assertEqual(attempts, 2)

    async def test_retry_retries_anthropic_overload_status(self) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request = vision_models.httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                response = vision_models.httpx.Response(529, request=request)
                raise vision_models.httpx.HTTPStatusError("overloaded", request=request, response=response)
            return "ok"

        self.assertEqual(await vision_models._retry_vision_request(operation, 1), "ok")
        self.assertEqual(attempts, 2)

    async def test_retry_propagates_cancellation(self) -> None:
        async def operation() -> str:
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await vision_models._retry_vision_request(operation, 1)

    async def test_shared_semaphore_limits_all_vision_requests(self) -> None:
        active = 0
        peak = 0

        async def request_once(**kwargs) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return '{"extracted_text": []}'

        original = vision_models._vision_ocr_limiter
        vision_models._vision_ocr_limiter = vision_models._VisionOcrLimiter(1)
        try:
            with (
                patch.object(vision_models, "_request_vision_response_once", request_once),
                patch.object(vision_models, "_vision_ocr_limit", return_value=1),
            ):
                await asyncio.gather(
                    vision_models._request_vision_response_text(
                        provider="openai",
                        base_url="https://example.test",
                        model="model",
                        api_key="key",
                        prompt="read",
                        schema=None,
                        processed_content=b"a",
                        image_mime_type="image/png",
                        timeout=1,
                    ),
                    vision_models._request_vision_response_text(
                        provider="anthropic",
                        base_url="https://example.test",
                        model="model",
                        api_key="key",
                        prompt="read",
                        schema=None,
                        processed_content=b"b",
                        image_mime_type="image/png",
                        timeout=1,
                    ),
                )
        finally:
            vision_models._vision_ocr_limiter = original

        self.assertEqual(peak, 1)

    async def test_queue_wait_does_not_consume_provider_deadline(self) -> None:
        original = vision_models._vision_ocr_limiter
        vision_models._vision_ocr_limiter = vision_models._VisionOcrLimiter(1)

        async def request_once(**kwargs) -> str:
            await asyncio.sleep(0.1)
            return '{"extracted_text": []}'

        try:
            with (
                patch.object(vision_models, "_request_vision_response_once", request_once),
                patch.object(vision_models, "_vision_ocr_limit", return_value=1),
            ):
                responses = await asyncio.gather(
                    *[
                        vision_models._request_vision_response_text(
                            provider="openai",
                            base_url="https://example.test",
                            model="model",
                            api_key="key",
                            prompt="read",
                            schema=None,
                            processed_content=b"a",
                            image_mime_type="image/png",
                            timeout=0.25,
                        )
                        for _ in range(5)
                    ]
                )
        finally:
            vision_models._vision_ocr_limiter = original

        self.assertEqual(responses, ['{"extracted_text": []}'] * 5)

    async def test_saved_openai_key_is_resolved_for_wire_request(self) -> None:
        captured: dict[str, object] = {}

        async def post_request(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"extracted_text":[]}'}}]}

        with (
            patch.object(vision_models, "_resolve_vision_api_key", return_value="saved-key"),
            patch.object(vision_models, "_post_openai_compatible_vision_request", post_request),
        ):
            await vision_models._request_vision_response_once(
                normalized_provider="openai",
                base_url="https://example.test",
                model="gpt-4o",
                api_key=None,
                prompt="read",
                schema=None,
                image_b64="YWJj",
                image_mime_type="image/png",
                timeout=1,
            )

        self.assertEqual(captured["api_key"], "saved-key")

    async def test_anthropic_wire_uses_image_block_and_schema(self) -> None:
        captured: dict[str, Any] = {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                captured.update({"url": url, **kwargs})
                request = vision_models.httpx.Request("POST", url)
                return vision_models.httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]}, request=request)

        with patch.object(vision_models.httpx, "AsyncClient", return_value=Client()):
            result = await vision_models._request_anthropic_vision_response(
                base_url="https://anthropic.test",
                api_key="key",
                model="claude",
                prompt="read",
                image_b64="YWJj",
                image_mime_type="image/png",
                schema={"type": "object"},
                timeout=1,
            )

        payload = captured["json"]
        self.assertEqual(result, "{}")
        self.assertEqual(captured["url"], "https://anthropic.test/v1/messages")
        self.assertEqual(payload["messages"][0]["content"][0]["source"]["media_type"], "image/png")
        self.assertEqual(payload["output_config"]["format"]["schema"], {"type": "object"})

    async def test_openai_compatible_wire_preserves_png_mime_and_schema(self) -> None:
        captured: dict[str, Any] = {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                captured.update({"url": url, **kwargs})
                request = vision_models.httpx.Request("POST", url)
                return vision_models.httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}, request=request)

        with patch.object(vision_models.httpx, "AsyncClient", return_value=Client()):
            data = await vision_models._post_openai_compatible_vision_request(
                provider="openrouter",
                base_url="https://router.test/api/v1",
                api_key="key",
                model="vision",
                prompt="read",
                image_b64="YWJj",
                image_mime_type="image/png",
                schema={"type": "object"},
                timeout=1,
            )

        payload = captured["json"]
        self.assertEqual(data["choices"][0]["message"]["content"], "{}")
        self.assertEqual(captured["url"], "https://router.test/api/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"][1]["image_url"]["url"], "data:image/png;base64,YWJj")
        self.assertEqual(payload["response_format"]["type"], "json_schema")


if __name__ == "__main__":
    unittest.main()
