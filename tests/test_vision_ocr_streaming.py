import json
import unittest

from ragtime.core import vision_models
from ragtime.core.ocr_table_recovery import OllamaOcrIncompleteError


class _StreamResponse:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames

    async def aiter_bytes(self):
        for frame in self.frames:
            yield frame


class VisionOcrStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_collector_joins_fragmented_response_deltas_and_requires_done(self) -> None:
        response = _StreamResponse(
            [
                b'{"response":"{\\"extracted_"}\n',
                b'{"response":"text\\":[]}"}\n{"done":true,"done_reason":"stop","eval_count":7}\n',
            ]
        )

        payload = await vision_models._collect_ollama_ndjson_response(response)

        self.assertEqual(payload["response"], '{"extracted_text":[]}')
        self.assertTrue(payload["done"])
        self.assertEqual(payload["eval_count"], 7)

    async def test_collector_rejects_eof_without_terminal_done_frame(self) -> None:
        with self.assertRaisesRegex(OllamaOcrIncompleteError, "reason=incomplete"):
            await vision_models._collect_ollama_ndjson_response(_StreamResponse([b'{"response":"{}"}\n']))

    async def test_collector_sanitizes_provider_error_as_fatal_ollama_error(self) -> None:
        with self.assertRaisesRegex(OllamaOcrIncompleteError, "reason=error"):
            await vision_models._collect_ollama_ndjson_response(_StreamResponse([b'{"error":"llama runner process has terminated"}\n']))

    async def test_detector_aborts_only_repeated_empty_table_rows(self) -> None:
        empty_rows = "".join("| | |\n" for _ in range(180))
        table = json.dumps(
            {"extracted_text": [{"type": "Table", "content": "| A | B |\n| --- | --- |\n" + empty_rows}]},
            separators=(",", ":"),
        )
        frames = [json.dumps({"response": table[:900]}).encode() + b"\n", json.dumps({"response": table[900:]}).encode() + b"\n"]

        with self.assertRaisesRegex(OllamaOcrIncompleteError, "reason=loop_detected"):
            await vision_models._collect_ollama_ndjson_response(_StreamResponse(frames))

    async def test_detector_does_not_abort_repeated_meaningful_rows_or_prose(self) -> None:
        repeated = "".join("| value | value |\n" for _ in range(170))
        payload = {"extracted_text": [{"type": "Table", "content": "| A | B |\n| --- | --- |\n" + repeated}]}
        response = _StreamResponse([json.dumps({"response": json.dumps(payload), "done": True, "done_reason": "stop"}).encode() + b"\n"])

        collected = await vision_models._collect_ollama_ndjson_response(response)

        self.assertEqual(collected["response"], json.dumps(payload))

    async def test_detector_resets_between_table_content_blocks(self) -> None:
        empty_rows = "".join("| | |\n" for _ in range(100))
        payload = {
            "extracted_text": [
                {"type": "Table", "content": "| A | B |\n| --- | --- |\n" + empty_rows},
                {"type": "Table", "content": empty_rows},
            ]
        }
        response = _StreamResponse([json.dumps({"response": json.dumps(payload), "done": True, "done_reason": "stop"}).encode() + b"\n"])

        collected = await vision_models._collect_ollama_ndjson_response(response)

        self.assertEqual(collected["response"], json.dumps(payload))

    def test_detector_discards_oversized_logical_line(self) -> None:
        detector = vision_models._OllamaTableLoopDetector()
        detector.feed('{"extracted_text":[{"type":"Table","content":"| A | B |\\n' + "x" * (vision_models.OLLAMA_TABLE_DETECTOR_LINE_MAX_BYTES + 1))

        self.assertEqual(detector._line, [])
