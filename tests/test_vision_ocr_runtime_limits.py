import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from PIL import Image

from ragtime.core import vision_models
from ragtime.core.ocr_contract import _ocr_pipeline_fingerprint_from_vision_source


class VisionOcrRuntimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ollama_extraction_disables_thinking_and_bounds_context(self) -> None:
        captured: dict[str, Any] = {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                captured.update(kwargs)
                request = vision_models.httpx.Request("POST", url)
                return vision_models.httpx.Response(200, json={"response": "{}", "done": True}, request=request)

        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=Client()),
            patch.object(vision_models, "get_output_limit", return_value=2048),
        ):
            await vision_models._request_vision_response_once(
                normalized_provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt="read",
                schema={"type": "object"},
                image_b64="YWJj",
                image_mime_type="image/png",
                timeout=1,
            )

        payload = captured["json"]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertEqual(payload["format"], {"type": "object"})

    async def test_public_extraction_defaults_to_longer_timeout_but_preserves_explicit_timeout(self) -> None:
        image = vision_models.io.BytesIO()
        Image.new("RGB", (1, 1)).save(image, format="PNG")
        observed_timeouts: list[float] = []

        async def response_text(**kwargs: Any) -> str:
            observed_timeouts.append(kwargs["timeout"])
            return '{"extracted_text": []}'

        with (
            patch.object(vision_models, "_supports_structured_vision_output", AsyncMock(return_value=False)),
            patch.object(vision_models, "_request_vision_response_text", response_text),
        ):
            await vision_models.extract_text_with_vision(image.getvalue(), "https://ollama.test", "vision")
            await vision_models.extract_text_with_vision_structured(image.getvalue(), "https://ollama.test", "vision")
            await vision_models.extract_text_with_vision(image.getvalue(), "https://ollama.test", "vision", timeout=75.0)

        self.assertEqual(observed_timeouts, [180.0, 180.0, 75.0])

    async def test_retry_uses_remaining_caller_budget_after_one_minute(self) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request = vision_models.httpx.Request("POST", "https://ollama.test")
                response = vision_models.httpx.Response(429, headers={"Retry-After": "0"}, request=request)
                raise vision_models.httpx.HTTPStatusError("rate limited", request=request, response=response)
            return "ok"

        with (
            patch.object(vision_models, "time", SimpleNamespace(monotonic=iter([0.0, 61.0]).__next__)),
            patch.object(vision_models, "_retry_delay", return_value=61.0),
            patch.object(vision_models.asyncio, "sleep", AsyncMock()),
        ):
            self.assertEqual(await vision_models._retry_vision_request(operation, 180.0), "ok")
        self.assertEqual(attempts, 2)

    async def test_thinking_only_ollama_response_is_rejected_by_structured_extraction(self) -> None:
        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                request = vision_models.httpx.Request("POST", url)
                return vision_models.httpx.Response(
                    200,
                    json={"response": "", "thinking": '{"extracted_text": []}', "done": True},
                    request=request,
                )

        image = vision_models.io.BytesIO()
        Image.new("RGB", (1, 1)).save(image, format="PNG")
        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=Client()),
            patch.object(vision_models, "get_output_limit", return_value=1024),
            patch.object(vision_models, "_supports_structured_vision_output", AsyncMock(return_value=True)),
        ):
            with self.assertRaisesRegex(ValueError, "malformed structured output"):
                await vision_models.extract_text_with_vision_structured(image.getvalue(), "https://ollama.test", "vision")

    def test_fingerprint_tracks_ollama_extraction_context_and_thinking_semantics(self) -> None:
        source = (
            b"""OLLAMA_OCR_CONTEXT_LENGTH = 16384\nOLLAMA_OCR_THINK = False\ndef _extract_ollama_response_text(data):\n    return data.get("response", "")\n"""
        )
        changed_context = source.replace(b"16384", b"32768")
        changed_thinking = source.replace(b"False", b"True")
        changed_extractor = source.replace(b'get("response", "")', b'get("thinking", "")')

        self.assertNotEqual(_ocr_pipeline_fingerprint_from_vision_source(source), _ocr_pipeline_fingerprint_from_vision_source(changed_context))
        self.assertNotEqual(_ocr_pipeline_fingerprint_from_vision_source(source), _ocr_pipeline_fingerprint_from_vision_source(changed_thinking))
        self.assertNotEqual(_ocr_pipeline_fingerprint_from_vision_source(source), _ocr_pipeline_fingerprint_from_vision_source(changed_extractor))
