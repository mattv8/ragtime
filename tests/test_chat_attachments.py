from __future__ import annotations

import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from ragtime.indexer import chat_attachments


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
            with mock.patch.object(
                chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)
            ):
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
            with mock.patch.object(
                chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)
            ):
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
                    mock.patch.object(
                        chat_attachments,
                        "get_context_limit",
                        new=mock.AsyncMock(return_value=8192),
                    ),
                ):
                    processed, stats = (
                        await chat_attachments.preprocess_chat_attachment_content_parts(
                            content,
                            conversation_id="conv-1",
                            user_id="user-1",
                            workspace_id="workspace-1",
                            model_id="openai/gpt-4.1",
                        )
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

    async def test_cleanup_expired_chat_attachments_removes_old_directories(
        self,
    ) -> None:
        upload = _make_upload_file("report.txt", b"body", "text/plain")

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                chat_attachments, "CHAT_ATTACHMENT_DIR", Path(temp_dir)
            ):
                metadata = await chat_attachments.store_chat_attachment_upload(
                    upload,
                    conversation_id="conv-1",
                    user_id="user-1",
                )

                attachment_id = str(metadata["attachment_id"])
                metadata_path = chat_attachments._metadata_path(attachment_id)
                expired_metadata = dict(metadata)
                expired_metadata["expires_at"] = (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat()
                metadata_path.write_text(
                    __import__("json").dumps(expired_metadata), "utf-8"
                )

                removed = await chat_attachments.cleanup_expired_chat_attachments(
                    now=datetime.now(timezone.utc)
                )

                self.assertEqual(removed, 1)
                self.assertFalse(
                    chat_attachments._attachment_dir(attachment_id).exists()
                )


if __name__ == "__main__":
    unittest.main()
