import base64
import io
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from PIL import Image

from ragtime.core import vision_models
from ragtime.core.ocr_table_recovery import OllamaOcrIncompleteError, recover_ocr_table_blocks


def _png_image(width: int = 690, height: int = 942) -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(content, format="PNG")
    return content.getvalue()


def _structured_ocr_response() -> dict[str, Any]:
    return {
        "extracted_text": [
            {"type": "Title", "content": "Table 1"},
            {"type": "Table", "content": "| A | B |\n| --- | --- |\n| 1 | two |\n| Note |"},
            {"type": "NarrativeText", "content": "Preserved prose."},
        ]
    }


def _table_response() -> dict[str, Any]:
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


class _QueuedOllamaClient:
    def __init__(self, responses: list[dict[str, object]], payloads: list[dict[str, Any]]) -> None:
        self._responses = iter(responses)
        self._payloads = payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, **kwargs):
        self._payloads.append(kwargs["json"])
        request = vision_models.httpx.Request("POST", url)
        return vision_models.httpx.Response(200, json=next(self._responses), request=request)


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


class OllamaOcrIncompleteResponseTests(unittest.TestCase):
    def test_only_completed_length_envelope_is_recoverable(self) -> None:
        with self.assertRaises(OllamaOcrIncompleteError) as caught:
            vision_models._reject_ollama_incomplete_response(
                {
                    "done": True,
                    "done_reason": "length",
                    "eval_count": 4096,
                    "prompt_eval_count": 12,
                }
            )

        self.assertEqual(caught.exception.reason, "length")
        self.assertEqual(caught.exception.eval_count, 4096)
        self.assertEqual(caught.exception.prompt_eval_count, 12)

    def test_nonfinal_and_malformed_envelopes_remain_nonrecoverable(self) -> None:
        for payload, reason in (
            (None, "malformed"),
            ({"done": False, "done_reason": "length"}, "incomplete"),
            ({"done": True, "done_reason": "error"}, "error"),
            ({"done": True, "done_reason": "load"}, "load"),
            ({"done": "true", "done_reason": "length"}, "incomplete"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(OllamaOcrIncompleteError) as caught:
                    vision_models._reject_ollama_incomplete_response(payload)
                self.assertEqual(caught.exception.reason, reason)

    def test_completed_legacy_envelopes_are_accepted_without_recovery(self) -> None:
        for payload in (
            {"response": "{}"},
            {"done": True, "response": "{}"},
            {"done": True, "done_reason": "unload", "response": "{}"},
        ):
            with self.subTest(payload=payload):
                vision_models._reject_ollama_incomplete_response(payload)

    def test_length_requires_explicit_completed_envelope(self) -> None:
        for payload in (
            {"done_reason": "length", "response": "{"},
            {"done": None, "done_reason": "length", "response": "{"},
            {"done": False, "done_reason": "length", "response": "{"},
            {"done": "true", "done_reason": "length", "response": "{"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(OllamaOcrIncompleteError) as caught:
                    vision_models._reject_ollama_incomplete_response(payload)
                self.assertNotEqual(caught.exception.reason, "length")

    def test_recovery_phase_length_preserves_safe_counters(self) -> None:
        with self.assertRaises(OllamaOcrIncompleteError) as caught:
            vision_models._reject_ollama_incomplete_response(
                {"done": True, "done_reason": "length", "eval_count": 7, "prompt_eval_count": 3},
                required_reason="stop",
            )

        self.assertEqual((caught.exception.reason, caught.exception.eval_count, caught.exception.prompt_eval_count), ("length", 7, 3))

    def test_real_assembler_rejects_ambiguous_and_shifted_recovery_tables(self) -> None:
        multiple_tables = {"tables": [_table_response()["tables"][0], _table_response()["tables"][0]]}
        shifted_cells = _table_response()
        shifted_cells["tables"][0]["rows"][0]["cells"] = ["two", "1"]

        for table_payload in (multiple_tables, shifted_cells):
            with self.subTest(table_payload=table_payload):
                with self.assertRaisesRegex(ValueError, "Invalid recovered OCR table"):
                    recover_ocr_table_blocks(_structured_ocr_response(), table_payload, include_classification=False)


class OllamaOcrRecoveryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_structured_ocr_accepts_legacy_reply_without_done_key(self) -> None:
        payloads: list[dict[str, Any]] = []
        client = _QueuedOllamaClient([{"response": json.dumps(_structured_ocr_response())}], payloads)
        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
            patch.object(vision_models, "_supports_structured_vision_output", AsyncMock(return_value=True)),
        ):
            result = await vision_models.extract_text_with_vision_structured(_png_image(), "https://ollama.test", "vision")

        self.assertEqual(result.extracted_text[-1]["content"], "Preserved prose.")
        self.assertEqual(len(payloads), 1)

    async def test_recovery_freezes_the_primary_output_budget_for_all_phases(self) -> None:
        payloads: list[dict[str, Any]] = []
        client = _QueuedOllamaClient(
            [
                {"done": True, "done_reason": "length", "response": "{"},
                {"done": True, "done_reason": "stop", "response": json.dumps(_structured_ocr_response())},
                {"done": True, "done_reason": "stop", "response": json.dumps(_table_response())},
            ],
            payloads,
        )
        image = _png_image()
        budget = AsyncMock(side_effect=[1234, 8192])

        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
            patch.object(vision_models, "_resolve_ocr_output_budget", budget),
        ):
            await vision_models._request_vision_response_text(
                provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt=vision_models.OCR_ONLY_PROMPT,
                schema=vision_models.OCR_ONLY_SCHEMA,
                processed_content=image,
                image_mime_type="image/png",
                recovery_image_content=image,
                recovery_source_format=".png",
                timeout=1,
            )

        budget.assert_awaited_once_with("vision")
        self.assertEqual([payload["options"]["num_predict"] for payload in payloads], [1234, 1234, 1234])

    async def test_recovery_uses_real_assembler_and_bounded_2x_table_image(self) -> None:
        payloads: list[dict[str, Any]] = []
        client = _QueuedOllamaClient(
            [
                {"done": True, "done_reason": "length", "response": "{"},
                {"done": True, "done_reason": "stop", "response": json.dumps(_structured_ocr_response())},
                {"done": True, "done_reason": "stop", "response": json.dumps(_table_response())},
            ],
            payloads,
        )
        limiter = _TrackingLimiter()
        image = _png_image()
        vision_models._ocr_output_budget_cache.clear()

        with (
            patch.object(vision_models.httpx, "AsyncClient", return_value=client),
            patch.object(vision_models, "get_output_limit", return_value=1234),
            patch.object(vision_models, "_get_vision_ocr_limiter", return_value=limiter),
            patch.object(vision_models, "_vision_ocr_limit", return_value=1),
        ):
            result = await vision_models._request_vision_response_text(
                provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt=vision_models.OCR_ONLY_PROMPT,
                schema=vision_models.OCR_ONLY_SCHEMA,
                processed_content=image,
                image_mime_type="image/png",
                recovery_image_content=image,
                recovery_source_format=".png",
                timeout=1,
            )

        recovered = json.loads(result)
        self.assertEqual(recovered["extracted_text"][0]["content"], "Table 1")
        self.assertEqual(recovered["extracted_text"][-1]["content"], "Preserved prose.")
        self.assertEqual(len(payloads), 3)
        self.assertNotIn("repeat_penalty", payloads[0]["options"])
        for payload in payloads[1:]:
            self.assertEqual(payload["options"]["repeat_penalty"], 1.05)
            self.assertEqual(payload["options"]["num_ctx"], 16384)
            self.assertEqual(payload["options"]["num_predict"], 1234)
        table_image = Image.open(io.BytesIO(base64.b64decode(payloads[2]["images"][0])))
        self.assertEqual(table_image.size, (1380, 1884))
        self.assertEqual((limiter.acquires, limiter.releases, limiter.active), (1, 1, 0))

    async def test_both_public_entrypoints_recover_default_ocr_contracts(self) -> None:
        image = _png_image(20, 20)
        for entrypoint, include_classification in (
            (vision_models.extract_text_with_vision, True),
            (vision_models.extract_text_with_vision_structured, False),
        ):
            with self.subTest(entrypoint=entrypoint.__name__):
                structured = _structured_ocr_response()
                if include_classification:
                    structured.update(
                        {
                            "image_type": "document",
                            "subject": "table",
                            "description": "description",
                            "objects": [],
                            "setting": "page",
                            "tags": [],
                        }
                    )
                client = _QueuedOllamaClient(
                    [
                        {"done": True, "done_reason": "length", "response": "{"},
                        {"done": True, "done_reason": "stop", "response": json.dumps(structured)},
                        {"done": True, "done_reason": "stop", "response": json.dumps(_table_response())},
                    ],
                    [],
                )
                with (
                    patch.object(vision_models.httpx, "AsyncClient", return_value=client),
                    patch.object(vision_models, "get_output_limit", return_value=100),
                    patch.object(vision_models, "_supports_structured_vision_output", AsyncMock(return_value=True)),
                ):
                    result = await entrypoint(image, "https://ollama.test", "vision", include_classification=include_classification)
                indexed_text = result if isinstance(result, str) else result.format_for_indexing()
                self.assertIn("Preserved prose.", indexed_text)

    async def test_page_timeout_bounds_recovery_after_primary_length(self) -> None:
        async def primary(**_kwargs):
            raise OllamaOcrIncompleteError("length")

        async def stalled_recovery(**_kwargs):
            await vision_models.asyncio.sleep(0.02)
            return '{"extracted_text": []}'

        with (
            patch.object(vision_models, "_request_vision_response_once", primary),
            patch.object(vision_models, "_ollama_ocr_recover_response_text", stalled_recovery),
        ):
            with self.assertRaises(TimeoutError):
                await vision_models._request_vision_response_text(
                    provider="ollama",
                    base_url="https://ollama.test",
                    model="vision",
                    api_key=None,
                    prompt=vision_models.OCR_ONLY_PROMPT,
                    schema=vision_models.OCR_ONLY_SCHEMA,
                    processed_content=b"image",
                    image_mime_type="image/png",
                    timeout=0.001,
                )

    async def test_default_ollama_contract_runs_one_structured_recovery_before_table_repair(self) -> None:
        responses = iter(
            [
                {"done": True, "done_reason": "length", "response": "{"},
                {"done": True, "done_reason": "stop", "response": json.dumps(_structured_ocr_response())},
                {"done": True, "done_reason": "stop", "response": json.dumps(_table_response())},
            ]
        )

        post_once = AsyncMock(side_effect=lambda **_kwargs: next(responses))

        with (
            patch.object(vision_models, "_ollama_ocr_post_once", post_once) as request,
            patch.object(vision_models, "_ollama_ocr_table_image_content", return_value=(b"image", "image/png")),
        ):
            result = await vision_models._request_vision_response_text(
                provider="ollama",
                base_url="https://ollama.test",
                model="vision",
                api_key=None,
                prompt=vision_models.OCR_ONLY_PROMPT,
                schema=vision_models.OCR_ONLY_SCHEMA,
                processed_content=b"image",
                image_mime_type="image/png",
                timeout=1,
                recovery_image_content=b"image",
            )

        self.assertEqual(json.loads(result)["extracted_text"][-1]["content"], "Preserved prose.")
        self.assertEqual(request.await_count, 3)

    async def test_custom_prompt_never_recovers_length_completion(self) -> None:
        async def primary(**_kwargs):
            raise OllamaOcrIncompleteError("length")

        with patch.object(vision_models, "_request_vision_response_once", primary):
            with self.assertRaises(OllamaOcrIncompleteError):
                await vision_models._request_vision_response_text(
                    provider="ollama",
                    base_url="https://ollama.test",
                    model="vision",
                    api_key=None,
                    prompt="custom",
                    schema=None,
                    processed_content=b"image",
                    image_mime_type="image/png",
                    timeout=1,
                )


class OllamaOcrRecoveryImageTests(unittest.TestCase):
    def test_table_recovery_normalizes_unusual_image_modes_to_rgb(self) -> None:
        fixtures = (
            (Image.new("I;16", (10, 5)), "PNG"),
            (Image.new("CMYK", (10, 5), "white"), "JPEG"),
            (Image.new("P", (10, 5)), "PNG"),
            (Image.new("RGBA", (10, 5), (0, 0, 0, 0)), "PNG"),
        )
        for image, image_format in fixtures:
            with self.subTest(mode=image.mode):
                content = io.BytesIO()
                image.save(content, format=image_format)
                recovered, mime_type = vision_models._ollama_ocr_table_image_content(content.getvalue(), f".{image_format.lower()}")
                normalized = Image.open(io.BytesIO(recovered))
                self.assertEqual((mime_type, normalized.mode, normalized.size), ("image/png", "RGB", (20, 10)))
                if image.mode == "RGBA":
                    self.assertEqual(normalized.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
