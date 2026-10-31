import asyncio
import io
import json
import unittest
from dataclasses import dataclass
from typing import Any, TypeAlias
from unittest.mock import AsyncMock, patch

from PIL import Image

from ragtime.core import vision_models

PAGE_TIMEOUT = 0.1


def _png(width: int = 20, height: int = 20) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def _structured() -> dict[str, Any]:
    return {
        "extracted_text": [
            {"type": "Title", "content": "Table 1"},
            {"type": "Table", "content": "| A | B |\n| --- | --- |\n| 1 | two |\n| Note |"},
            {"type": "NarrativeText", "content": "Preserved prose."},
        ]
    }


def _table() -> dict[str, Any]:
    return {
        "tables": [
            {
                "caption": "Table 1",
                "headers": ["A", "B"],
                "rows": [{"kind": "data", "cells": ["1", "two"]}],
                "footnotes": ["Note"],
            }
        ]
    }


def _ollama_response(response: str, *, done: bool = True, reason: str = "stop", thinking: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"done": done, "done_reason": reason, "response": response}
    if thinking is not None:
        payload["thinking"] = thinking
    return payload


def _rate_limited() -> vision_models.httpx.Response:
    request = vision_models.httpx.Request("POST", "https://ollama.test/api/generate")
    return vision_models.httpx.Response(429, headers={"Retry-After": "0"}, request=request)


@dataclass
class _WaitForPost:
    entered: asyncio.Event
    released: asyncio.Event


_OllamaResponse: TypeAlias = dict[str, Any] | vision_models.httpx.Response | _WaitForPost
_OllamaResponseSequence: TypeAlias = list[_OllamaResponse]


class _QueuedClient:
    def __init__(
        self,
        responses: _OllamaResponseSequence,
        payloads: list[dict[str, Any]],
        timeouts: list[float],
    ) -> None:
        self._responses = iter(responses)
        self._payloads = payloads
        self._timeouts = timeouts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url: str, **kwargs: Any) -> vision_models.httpx.Response:
        self._payloads.append(kwargs["json"])
        response = next(self._responses)
        if isinstance(response, _WaitForPost):
            response.entered.set()
            try:
                await response.released.wait()
            except asyncio.CancelledError:
                raise
        request = vision_models.httpx.Request("POST", url)
        if isinstance(response, vision_models.httpx.Response):
            return response
        return vision_models.httpx.Response(200, json=response, request=request)


class _TrackingLimiter(vision_models._VisionOcrLimiter):
    def __init__(self) -> None:
        super().__init__(1)
        self.acquires = 0
        self.releases = 0

    async def acquire(self, capacity: int) -> None:
        self.acquires += 1
        await super().acquire(capacity)

    async def release(self) -> None:
        self.releases += 1
        await super().release()


class VisionOcrRecoveryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def _request(self, client: _QueuedClient, limiter: _TrackingLimiter, *, timeout: float = PAGE_TIMEOUT) -> str:
        vision_models._ocr_output_budget_cache.clear()
        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
            patch.object(vision_models, "get_output_limit", AsyncMock(return_value=100)),
            patch.object(vision_models, "_get_vision_ocr_limiter", AsyncMock(return_value=limiter)),
            patch.object(vision_models, "_vision_ocr_limit", AsyncMock(return_value=1)),
        ):
            return await vision_models._request_vision_response_text(
                provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt=vision_models.OCR_ONLY_PROMPT,
                schema=vision_models.OCR_ONLY_SCHEMA,
                processed_content=_png(),
                image_mime_type="image/png",
                recovery_image_content=_png(),
                recovery_source_format=".png",
                timeout=timeout,
            )

    async def test_cancellation_releases_limiter_during_each_http_phase(self) -> None:
        cancellation_cases: tuple[tuple[str, _OllamaResponseSequence], ...] = (
            ("primary", [_WaitForPost(asyncio.Event(), asyncio.Event())]),
            ("structured", [_ollama_response("{", reason="length"), _WaitForPost(asyncio.Event(), asyncio.Event())]),
            ("table", [_ollama_response("{", reason="length"), _ollama_response(json.dumps(_structured())), _WaitForPost(asyncio.Event(), asyncio.Event())]),
        )
        for phase, responses in cancellation_cases:
            with self.subTest(phase=phase):
                payloads: list[dict[str, Any]] = []
                waiter = responses[-1]
                assert isinstance(waiter, _WaitForPost)
                client = _QueuedClient(responses, payloads, [])
                limiter = _TrackingLimiter()
                request = asyncio.create_task(self._request(client, limiter))
                await asyncio.wait_for(waiter.entered.wait(), timeout=PAGE_TIMEOUT)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
                self.assertEqual((limiter.acquires, limiter.releases, limiter.active), (1, 1, 0))

    async def test_phase_two_rate_limit_retries_without_repeating_original_request(self) -> None:
        payloads: list[dict[str, Any]] = []
        client = _QueuedClient(
            [_ollama_response("{", reason="length"), _rate_limited(), _ollama_response(json.dumps(_structured())), _ollama_response(json.dumps(_table()))],
            payloads,
            [],
        )

        result = await self._request(client, _TrackingLimiter())

        self.assertEqual(json.loads(result)["extracted_text"][-1]["content"], "Preserved prose.")
        self.assertEqual(len(payloads), 4)
        self.assertEqual(
            [payload["prompt"] for payload in payloads],
            [vision_models.OCR_ONLY_PROMPT, vision_models.OCR_ONLY_PROMPT, vision_models.OCR_ONLY_PROMPT, vision_models.OCR_TABLE_RECOVERY_PROMPT],
        )
        self.assertNotIn("repeat_penalty", payloads[0]["options"])
        self.assertTrue(all(payload["options"]["repeat_penalty"] == 1.05 for payload in payloads[1:]))

    async def test_phase_three_transient_retries_only_typed_table_request(self) -> None:
        payloads: list[dict[str, Any]] = []
        client = _QueuedClient(
            [_ollama_response("{", reason="length"), _ollama_response(json.dumps(_structured())), _rate_limited(), _ollama_response(json.dumps(_table()))],
            payloads,
            [],
        )

        await self._request(client, _TrackingLimiter())

        self.assertEqual(
            [payload["prompt"] for payload in payloads],
            [vision_models.OCR_ONLY_PROMPT, vision_models.OCR_ONLY_PROMPT, vision_models.OCR_TABLE_RECOVERY_PROMPT, vision_models.OCR_TABLE_RECOVERY_PROMPT],
        )

    async def test_unauthorized_primary_does_not_retry_or_recover(self) -> None:
        request = vision_models.httpx.Request("POST", "https://ollama.test/api/generate")
        payloads: list[dict[str, Any]] = []
        client = _QueuedClient([vision_models.httpx.Response(401, request=request)], payloads, [])

        with self.assertRaises(vision_models.httpx.HTTPStatusError):
            await self._request(client, _TrackingLimiter())

        self.assertEqual(len(payloads), 1)

    async def test_invalid_structured_recovery_responses_stop_before_table_phase(self) -> None:
        invalid_responses = (
            _ollama_response(json.dumps(_structured()), done=False),
            _ollama_response(json.dumps(_structured()), reason="length"),
            _ollama_response(json.dumps(_structured()), reason="error"),
            _ollama_response(json.dumps(_structured()), reason="load"),
            _ollama_response("[1]"),
            _ollama_response("{"),
            _ollama_response("", thinking=json.dumps(_structured())),
            _ollama_response('{"not_extracted_text": []}'),
        )
        for invalid in invalid_responses:
            with self.subTest(response=invalid):
                payloads: list[dict[str, Any]] = []
                client = _QueuedClient([_ollama_response("{", reason="length"), invalid], payloads, [])
                with self.assertRaises((ValueError, RuntimeError)):
                    await self._request(client, _TrackingLimiter())
                self.assertEqual(len(payloads), 2)

    async def test_invalid_typed_table_responses_stop_after_table_phase(self) -> None:
        invalid_responses = (
            _ollama_response(json.dumps(_table()), done=False),
            _ollama_response(json.dumps(_table()), reason="length"),
            _ollama_response(json.dumps(_table()), reason="error"),
            _ollama_response(json.dumps(_table()), reason="load"),
            _ollama_response("[1]"),
            _ollama_response("{"),
            _ollama_response("", thinking=json.dumps(_table())),
            _ollama_response('{"tables": []}'),
        )
        for invalid in invalid_responses:
            with self.subTest(response=invalid):
                payloads: list[dict[str, Any]] = []
                client = _QueuedClient([_ollama_response("{", reason="length"), _ollama_response(json.dumps(_structured())), invalid], payloads, [])
                with self.assertRaises((ValueError, RuntimeError)):
                    await self._request(client, _TrackingLimiter())
                self.assertEqual(len(payloads), 3)

    async def test_complete_original_response_is_returned_unchanged_once(self) -> None:
        payloads: list[dict[str, Any]] = []
        original = '{"extracted_text": [{"type": "Title", "content": "Original"}]}'
        client = _QueuedClient([_ollama_response(original)], payloads, [])

        self.assertEqual(await self._request(client, _TrackingLimiter()), original)
        self.assertEqual(len(payloads), 1)

    async def test_non_ollama_length_completion_never_invokes_recovery(self) -> None:
        request = vision_models.httpx.Request("POST", "https://openai.test/v1/chat/completions")

        class OpenAiClient:
            calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, _url: str, **_kwargs: Any) -> vision_models.httpx.Response:
                self.calls += 1
                return vision_models.httpx.Response(
                    200,
                    json={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
                    request=request,
                )

        client = OpenAiClient()
        limiter = _TrackingLimiter()
        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
            patch.object(vision_models, "get_output_limit", AsyncMock(return_value=100)),
            patch.object(vision_models, "_get_vision_ocr_limiter", AsyncMock(return_value=limiter)),
            patch.object(vision_models, "_vision_ocr_limit", AsyncMock(return_value=1)),
        ):
            with self.assertRaises(RuntimeError):
                await vision_models._request_vision_response_text(
                    provider="openai",
                    base_url="https://openai.test",
                    model="vision",
                    api_key="key",
                    prompt=vision_models.OCR_ONLY_PROMPT,
                    schema=vision_models.OCR_ONLY_SCHEMA,
                    processed_content=_png(),
                    image_mime_type="image/png",
                    timeout=PAGE_TIMEOUT,
                )
        self.assertEqual(client.calls, 1)
        self.assertEqual((limiter.acquires, limiter.releases, limiter.active), (1, 1, 0))

    async def test_page_deadline_cancels_unresponsive_typed_table_request_and_releases(self) -> None:
        payloads: list[dict[str, Any]] = []
        waiter = _WaitForPost(asyncio.Event(), asyncio.Event())
        client = _QueuedClient([_ollama_response("{", reason="length"), _ollama_response(json.dumps(_structured())), waiter], payloads, [])
        limiter = _TrackingLimiter()

        with self.assertRaises(TimeoutError):
            await self._request(client, limiter, timeout=0.01)

        self.assertTrue(waiter.entered.is_set())
        self.assertEqual((limiter.acquires, limiter.releases, limiter.active), (1, 1, 0))

    async def test_recovery_requests_use_decreasing_remaining_page_deadline(self) -> None:
        payloads: list[dict[str, Any]] = []
        timeouts: list[float] = []
        responses = [
            _ollama_response("{", reason="length"),
            _rate_limited(),
            _ollama_response(json.dumps(_structured())),
            _ollama_response(json.dumps(_table())),
        ]

        queued_responses = iter(responses)

        class RecordingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url: str, **kwargs: Any) -> vision_models.httpx.Response:
                payloads.append(kwargs["json"])
                response = next(queued_responses)
                request = vision_models.httpx.Request("POST", url)
                if isinstance(response, vision_models.httpx.Response):
                    return response
                return vision_models.httpx.Response(200, json=response, request=request)

        class RecordingFactory:
            def __call__(self, *, timeout: float):
                timeouts.append(timeout)
                return RecordingClient()

        limiter = _TrackingLimiter()
        vision_models._ocr_output_budget_cache.clear()
        with (
            patch.object(vision_models.httpx, "AsyncClient", new=RecordingFactory()),
            patch.object(vision_models, "get_output_limit", AsyncMock(return_value=100)),
            patch.object(vision_models, "_get_vision_ocr_limiter", AsyncMock(return_value=limiter)),
            patch.object(vision_models, "_vision_ocr_limit", AsyncMock(return_value=1)),
        ):
            await vision_models._request_vision_response_text(
                provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt=vision_models.OCR_ONLY_PROMPT,
                schema=vision_models.OCR_ONLY_SCHEMA,
                processed_content=_png(),
                image_mime_type="image/png",
                recovery_image_content=_png(),
                timeout=PAGE_TIMEOUT,
            )

        self.assertEqual(len(timeouts), 4)
        self.assertTrue(all(timeout < PAGE_TIMEOUT for timeout in timeouts))
        self.assertLess(timeouts[2], timeouts[0])
        self.assertLess(timeouts[3], timeouts[2])


class VisionOcrRecoveryImageBoundsTests(unittest.TestCase):
    def test_table_recovery_image_doubles_small_png_dimensions(self) -> None:
        content, mime_type = vision_models._ollama_ocr_table_image_content(_png(20, 15), ".png")

        self.assertEqual(mime_type, "image/png")
        self.assertEqual(Image.open(io.BytesIO(content)).size, (40, 30))

    def test_table_recovery_image_caps_oversized_aspect_ratio_at_2048(self) -> None:
        content, _mime_type = vision_models._ollama_ocr_table_image_content(_png(4096, 4), ".png")

        self.assertEqual(Image.open(io.BytesIO(content)).size, (2048, 2))

    def test_encoded_image_limit_rejects_before_any_provider_request(self) -> None:
        with patch.object(vision_models, "OPENAI_COMPATIBLE_MAX_ENCODED_IMAGE_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "exceeds the provider request limit"):
                vision_models._prepare_vision_image_content(_png(80, 80), provider="ollama", preprocess=True, max_dimension=2048, source_format=".png")


if __name__ == "__main__":
    unittest.main()
