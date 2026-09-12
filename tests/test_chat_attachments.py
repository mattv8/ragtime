from __future__ import annotations

import base64
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from ragtime.indexer import chat_attachments
from ragtime.indexer.models import AppSettings, UpdateSettingsRequest


def _make_upload_file(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class ChatAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_chat_attachment_upload_rejects_image_files(self) -> None:
        upload = _make_upload_file("diagram.png", b"png-bytes", "image/png")

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)):
                with self.assertRaises(HTTPException) as exc_info:
                    await chat_attachments.store_chat_attachment_upload(
                        upload,
                        conversation_id="conv-1",
                        user_id="user-1",
                    )

        self.assertEqual(exc_info.exception.status_code, 400)

    async def test_preprocess_chat_attachment_content_parts_expands_uploaded_file(
        self,
    ) -> None:
        upload = _make_upload_file(
            "notes.txt",
            b"alpha\nbeta\ngamma",
            "text/plain",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)):
                metadata = await chat_attachments.store_chat_attachment_upload(
                    upload,
                    conversation_id="conv-1",
                    user_id="user-1",
                    workspace_id="workspace-1",
                )

                content = [
                    {"type": "text", "text": "Please review this file."},
                    {
                        "type": "file",
                        "attachment_id": metadata["attachment_id"],
                        "attachment_source": "chat_upload",
                        "filename": metadata["filename"],
                        "mime_type": metadata["mime_type"],
                    },
                ]

                fake_chunks = [
                    SimpleNamespace(page_content="first chunk", metadata={}),
                    SimpleNamespace(page_content="second chunk", metadata={}),
                ]

                with (
                    mock.patch.object(
                        chat_attachments,
                        "extract_text_from_file_async",
                        new=mock.AsyncMock(return_value="first chunk\nsecond chunk"),
                    ),
                    mock.patch.object(
                        chat_attachments,
                        "chunk_documents_parallel",
                        new=mock.AsyncMock(return_value=fake_chunks),
                    ),
                    mock.patch.object(chat_attachments, "get_app_settings", new=mock.AsyncMock(return_value={})),
                ):
                    processed, stats = await chat_attachments.preprocess_chat_attachment_content_parts(
                        content,
                        conversation_id="conv-1",
                        user_id="user-1",
                        workspace_id="workspace-1",
                        model_id="openai/gpt-4.1",
                    )

        self.assertIsInstance(processed, list)
        self.assertEqual(processed[0]["type"], "text")
        self.assertEqual(processed[0]["text"], "Please review this file.")
        self.assertEqual(processed[1]["type"], "text")
        self.assertIn("--- Attached file: notes.txt ---", processed[1]["text"])
        self.assertIn("Chunk 1/2", processed[1]["text"])
        self.assertIn("first chunk", processed[1]["text"])
        self.assertIn("second chunk", processed[1]["text"])
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats["file_count"], 1)
        self.assertEqual(stats["included_chunk_count"], 2)

    async def test_attachment_budget_default_and_zero_retain_trailing_text_across_files(self) -> None:
        content = [
            {"type": "file", "attachment_id": "attachment-1", "attachment_source": "chat_upload", "filename": "first.txt"},
            {"type": "file", "attachment_id": "attachment-2", "attachment_source": "chat_upload", "filename": "second.txt"},
        ]
        chunks = [
            ["first " + ("content " * 3_000) + "FIRST_TRAILING"],
            ["second " + ("content " * 3_000) + "SECOND_TRAILING"],
        ]

        for app_settings in ({}, {"chat_attachment_token_budget": 0}):
            with (
                mock.patch.object(chat_attachments, "get_app_settings", new=mock.AsyncMock(return_value=app_settings)),
                mock.patch.object(chat_attachments, "resolve_chat_attachment", return_value=({}, Path("unused"))),
                mock.patch.object(chat_attachments, "_extract_attachment_chunks", new=mock.AsyncMock(side_effect=chunks)),
            ):
                processed, stats = await chat_attachments.preprocess_chat_attachment_content_parts(content)

            self.assertIn("FIRST_TRAILING", processed[0]["text"])
            self.assertIn("SECOND_TRAILING", processed[1]["text"])
            assert stats is not None
            self.assertGreater(stats["used_tokens"], 2_048)

    async def test_positive_attachment_budget_caps_files_and_exhaustion_does_not_become_unlimited(self) -> None:
        content = [
            {"type": "file", "attachment_id": "attachment-1", "attachment_source": "chat_upload", "filename": "first.txt"},
            {"type": "file", "attachment_id": "attachment-2", "attachment_source": "chat_upload", "filename": "second.txt"},
        ]
        chunks = [["first"], ["SECOND_TRAILING"]]
        budget = chat_attachments.count_tokens(chat_attachments._format_chunk_block("first.txt", "first", 1, 1))

        with (
            mock.patch.object(chat_attachments, "get_app_settings", new=mock.AsyncMock(return_value={"chat_attachment_token_budget": budget})),
            mock.patch.object(chat_attachments, "resolve_chat_attachment", return_value=({}, Path("unused"))),
            mock.patch.object(chat_attachments, "_extract_attachment_chunks", new=mock.AsyncMock(side_effect=chunks)),
        ):
            processed, stats = await chat_attachments.preprocess_chat_attachment_content_parts(content)

        self.assertNotIn("SECOND_TRAILING", processed[1]["text"])
        self.assertIn("budget was exhausted", processed[1]["text"])
        assert stats is not None
        self.assertGreater(stats["used_tokens"], 0)

    def test_attachment_budget_settings_reject_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings(chat_attachment_token_budget=-1)
        with self.assertRaises(ValueError):
            UpdateSettingsRequest(chat_attachment_token_budget=-1)

    async def test_extract_chat_image_context_uses_structured_ocr_chunks(self) -> None:
        payload = base64.b64encode(b"image-bytes").decode("ascii")
        part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{payload}"},
        }
        structured_result = SimpleNamespace(
            raw_text=None,
            extracted_text=[{"type": "Title", "content": "Quarterly revenue"}],
            get_semantic_segments=lambda: [
                ("ocr_text", "Quarterly revenue"),
                ("classification", "Image type: chart\nTags: finance"),
            ],
        )

        with (
            mock.patch.object(
                chat_attachments,
                "extract_image_structured_async",
                new=mock.AsyncMock(return_value=structured_result),
            ) as structured_mock,
            mock.patch.object(
                chat_attachments,
                "extract_text_from_file_async",
                new=mock.AsyncMock(return_value="fallback text"),
            ) as text_mock,
        ):
            text = await chat_attachments.extract_chat_image_context_from_part(
                part,
                app_settings={
                    "default_ocr_mode": "vision",
                    "default_ocr_provider": "openai",
                    "default_ocr_vision_model": "gpt-4o-mini",
                    "openai_api_key": "test-key",
                },
            )

        structured_mock.assert_awaited_once()
        text_mock.assert_not_called()
        self.assertIn("Attached image OCR chunk 1/2", text)
        self.assertIn("Quarterly revenue", text)
        self.assertIn("Attached image OCR chunk 2/2", text)
        self.assertIn("Tags: finance", text)

    async def test_cleanup_expired_chat_attachments_removes_old_directories(
        self,
    ) -> None:
        upload = _make_upload_file("report.txt", b"body", "text/plain")

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)):
                metadata = await chat_attachments.store_chat_attachment_upload(
                    upload,
                    conversation_id="conv-1",
                    user_id="user-1",
                )

                attachment_id = str(metadata["attachment_id"])
                metadata_path = chat_attachments._metadata_path(attachment_id)
                expired_metadata = dict(metadata)
                expired_metadata["expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                metadata_path.write_text(__import__("json").dumps(expired_metadata), "utf-8")

                removed = await chat_attachments.cleanup_expired_chat_attachments(now=datetime.now(timezone.utc))

                self.assertEqual(removed, 1)
                self.assertFalse(chat_attachments._attachment_dir(attachment_id).exists())


if __name__ == "__main__":
    unittest.main()
