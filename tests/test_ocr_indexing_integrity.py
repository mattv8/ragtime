"""Regression coverage for structured OCR ingestion integrity."""

from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

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
