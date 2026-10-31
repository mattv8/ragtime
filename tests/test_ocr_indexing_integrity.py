"""Regression coverage for structured OCR ingestion integrity."""

from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

import httpx

from ragtime.core.ocr_contract import _ocr_pipeline_fingerprint_from_vision_source, ocr_pipeline_fingerprint
from ragtime.indexer.chunking import chunk_semantic_segments
from ragtime.indexer.document_parser import VisionOcrError, extract_image_structured_async, extract_text_from_file_async


def test_ocr_pipeline_fingerprint_is_stable_and_tracks_vision_source() -> None:
    assert ocr_pipeline_fingerprint() == ocr_pipeline_fingerprint()
    assert _ocr_pipeline_fingerprint_from_vision_source(b"first") != _ocr_pipeline_fingerprint_from_vision_source(b"second")


def test_ocr_pipeline_fingerprint_uses_only_ocr_semantics() -> None:
    source = b"""OCR_ONLY_PROMPT = "extract one"
def preprocess_image(data):
    return data
def unrelated_helper():
    return 1
"""
    unrelated_change = source.replace(b"return 1", b"return 2")
    prompt_change = source.replace(b"extract one", b"extract two")

    assert _ocr_pipeline_fingerprint_from_vision_source(source) == _ocr_pipeline_fingerprint_from_vision_source(unrelated_change)
    assert _ocr_pipeline_fingerprint_from_vision_source(source) != _ocr_pipeline_fingerprint_from_vision_source(prompt_change)


def test_table_segments_split_at_rows_and_repeat_header() -> None:
    table = "\n".join(
        [
            "| Item | Amount | Notes |",
            "| --- | --- | --- |",
            "| Alpha | 45 N | |",
            "| Beta | 50/ER30 | |",
            "| Gamma | 75/ER30 | |",
        ]
    )

    chunks = chunk_semantic_segments(
        [("table", table)],
        chunk_size=70,
        chunk_overlap=0,
        metadata={"source": "inspection.png", "page": 182},
    )

    assert len(chunks) == 3
    assert all(chunk.metadata["segment_type"] == "table" for chunk in chunks)
    assert all(chunk.metadata["page"] == 182 for chunk in chunks)
    assert all(chunk.page_content.startswith("| Item | Amount | Notes |\n| --- | --- | --- |\n") for chunk in chunks)
    assert "| Alpha | 45 N | |" in chunks[0].page_content
    assert "| Beta | 50/ER30 | |" in chunks[1].page_content
    assert "| Gamma | 75/ER30 | |" in chunks[2].page_content


class VisionOcrIngestionTests(IsolatedAsyncioTestCase):
    async def test_missing_vision_model_never_falls_back_to_tesseract(self) -> None:
        with mock.patch("ragtime.indexer.document_parser._extract_image_ocr") as tesseract:
            with self.assertRaisesRegex(VisionOcrError, "configured model"):
                await extract_text_from_file_async(Path("scan.png"), content=b"image", ocr_mode="vision")
        tesseract.assert_not_called()

    async def test_structured_vision_error_is_not_converted_to_none(self) -> None:
        with mock.patch(
            "ragtime.core.vision_models.extract_text_with_vision_structured",
            new=mock.AsyncMock(side_effect=RuntimeError("refused")),
        ):
            with self.assertRaisesRegex(VisionOcrError, "Structured vision OCR failed"):
                await extract_image_structured_async(Path("scan.png"), content=b"image", ocr_provider="openai", ocr_vision_model="gpt-4o")

    async def test_structured_http_failure_reports_provider_and_status_without_leaking_response(self) -> None:
        request = httpx.Request("POST", "http://ocr-user:hunter2@192.168.10.19:11434/api/generate")
        response = httpx.Response(500, request=request, content=b'{"error": "model crashed SECRET_OCR_TEXT"}')
        cause = httpx.HTTPStatusError("server error", request=request, response=response)

        message, raised = await self._structured_failure_message(cause)

        assert message.startswith("Structured vision OCR failed for source page-000275.png")
        assert "provider=ollama" in message
        assert "category=http_status" in message
        assert "status=500" in message
        assert raised.__cause__ is cause
        for leaked in ("hunter2", "ocr-user", "192.168.10.19", "api/generate", "SECRET_OCR_TEXT"):
            assert leaked not in message

    async def test_structured_value_error_reports_neutral_category_without_leaking_content(self) -> None:
        cause = ValueError("Vision OCR image could not be decoded. SECRET_RAW_OCR")

        message, raised = await self._structured_failure_message(cause)

        assert "provider=ollama" in message
        assert "category=extraction_failed" in message
        assert "error_type=ValueError" in message
        assert "category=malformed_response" not in message
        assert raised.__cause__ is cause
        assert "SECRET_RAW_OCR" not in message

    async def test_structured_timeout_reports_category_without_leaking_url_credentials(self) -> None:
        cause = httpx.ReadTimeout("read timed out for http://ocr-user:hunter2@192.168.10.19:11434/api/generate")

        message, raised = await self._structured_failure_message(cause)

        assert "category=timeout" in message
        assert raised.__cause__ is cause
        for leaked in ("hunter2", "ocr-user", "192.168.10.19"):
            assert leaked not in message

    async def test_structured_other_failure_reports_error_type_without_message(self) -> None:
        cause = httpx.ConnectError("connect failed for http://ocr-user:hunter2@192.168.10.19:11434/api/generate")

        message, raised = await self._structured_failure_message(cause)

        assert "category=extraction_failed" in message
        assert "error_type=ConnectError" in message
        assert raised.__cause__ is cause
        for leaked in ("hunter2", "ocr-user", "192.168.10.19"):
            assert leaked not in message

    async def _structured_failure_message(self, cause: BaseException) -> tuple[str, VisionOcrError]:
        with mock.patch(
            "ragtime.core.vision_models.extract_text_with_vision_structured",
            new=mock.AsyncMock(side_effect=cause),
        ):
            with self.assertRaises(VisionOcrError) as caught:
                await extract_image_structured_async(
                    Path("page-000275.png"),
                    content=b"image",
                    ocr_provider="ollama",
                    ocr_vision_model="llama3.2-vision",
                )
        return str(caught.exception), caught.exception

    async def test_structured_source_read_error_is_not_converted_to_none(self) -> None:
        missing = Path("missing-scan.png")

        with self.assertRaisesRegex(VisionOcrError, "could not read source"):
            await extract_image_structured_async(missing, ocr_provider="openai", ocr_vision_model="gpt-4o")

    async def test_structured_claude_code_target_does_not_require_a_base_url(self) -> None:
        expected = object()
        with mock.patch(
            "ragtime.core.vision_models.extract_text_with_vision_structured",
            new=mock.AsyncMock(return_value=expected),
        ) as extract:
            result = await extract_image_structured_async(
                Path("scan.png"),
                content=b"image",
                ocr_provider="claude_code",
                ocr_vision_model="claude-sonnet-4-5",
                vision_base_url=None,
            )

        assert result is expected
        extract.assert_awaited_once_with(
            image_content=b"image",
            base_url=None,
            model="claude-sonnet-4-5",
            provider="claude_code",
            api_key=None,
            source_format=".png",
            include_classification=True,
        )

    async def test_claude_code_target_does_not_require_a_base_url(self) -> None:
        with mock.patch(
            "ragtime.indexer.document_parser._extract_image_vision_ocr",
            new=mock.AsyncMock(return_value="vision text"),
        ) as extract:
            result = await extract_text_from_file_async(
                Path("scan.png"),
                content=b"image",
                ocr_mode="vision",
                ocr_provider="claude_code",
                ocr_vision_model="claude-sonnet-4-5",
                vision_base_url=None,
            )

        assert result == "vision text"
        extract.assert_awaited_once_with(
            b"image",
            None,
            "claude-sonnet-4-5",
            provider="claude_code",
            api_key=None,
            source_format=".png",
        )
