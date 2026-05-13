import unittest
from datetime import datetime
from datetime import timezone
from unittest.mock import patch

from fastapi import HTTPException

from ragtime.chat_runtime.service import ChatRuntimeService
from ragtime.chat_runtime.service import _ChatDiagSession


class ChatRuntimeWebSearchPdfTests(unittest.IsolatedAsyncioTestCase):
    def test_detects_likely_pdf_results(self):
        self.assertTrue(
            ChatRuntimeService._is_likely_pdf_result(
                {
                    "title": "[PDF] Example paper",
                    "url": "https://example.com/article",
                    "snippet": "Search result",
                }
            )
        )
        self.assertTrue(
            ChatRuntimeService._is_likely_pdf_result(
                {
                    "title": "Example paper",
                    "url": "https://arxiv.org/pdf/1706.03762",
                    "snippet": "Search result",
                }
            )
        )
        self.assertFalse(
            ChatRuntimeService._is_likely_pdf_result(
                {
                    "title": "Example page",
                    "url": "https://example.com/article.html",
                    "snippet": "Search result",
                }
            )
        )

    async def test_attach_pdf_metadata_only_marks_likely_candidates(self):
        service = ChatRuntimeService()
        results = [
            {
                "title": "[PDF] First paper",
                "url": "https://example.com/first",
                "snippet": "",
            },
            {
                "title": "HTML page",
                "url": "https://example.com/page.html",
                "snippet": "",
            },
        ]

        attached = await service._attach_pdf_metadata(
            results,
            include_pdf_metadata=True,
        )

        self.assertEqual(attached, 1)
        self.assertEqual(results[0]["pdf"]["status"], "available")
        self.assertEqual(results[0]["pdf"]["read_tool"], "web_read_pdf")
        self.assertNotIn("text", results[0]["pdf"])
        self.assertNotIn("pdf", results[1])

    async def test_browse_url_retries_after_stale_runtime_session(self):
        service = ChatRuntimeService()
        service._sessions["conv"] = _ChatDiagSession(
            conversation_id="conv",
            workspace_id="chat-diag-conv",
            provider_session_id="old-session",
            last_used_at=datetime.now(timezone.utc).timestamp(),
        )
        calls: list[tuple[str, str]] = []

        async def fake_request(method, path, **kwargs):
            calls.append((method, path))
            if path == "/sessions/old-session/external-browse":
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Chat diagnostics runtime manager request failed (404): "
                        '{"detail":"Runtime session not found"}'
                    ),
                )
            if path == "/sessions/start":
                return {"provider_session_id": "new-session"}
            if path == "/sessions/new-session/external-browse":
                return {"ok": True, "url": "https://example.com"}
            raise AssertionError(f"unexpected request path: {path}")

        with patch.object(service, "_request", new=fake_request):
            result = await service.browse_url(
                conversation_id="conv",
                url="https://example.com",
            )

        self.assertEqual(result, {"ok": True, "url": "https://example.com"})
        self.assertEqual(
            calls,
            [
                ("POST", "/sessions/old-session/external-browse"),
                ("POST", "/sessions/start"),
                ("POST", "/sessions/new-session/external-browse"),
            ],
        )
        self.assertEqual(service._sessions["conv"].provider_session_id, "new-session")

    async def test_read_pdf_url_delegates_through_runtime_session_retry(self):
        service = ChatRuntimeService()
        service._sessions["conv"] = _ChatDiagSession(
            conversation_id="conv",
            workspace_id="chat-diag-conv",
            provider_session_id="old-session",
            last_used_at=datetime.now(timezone.utc).timestamp(),
        )
        calls: list[tuple[str, str, dict | None]] = []

        async def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs.get("json_payload")))
            if path == "/sessions/old-session/pdf-read":
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Chat diagnostics runtime manager request failed (404): "
                        '{"detail":"Runtime session not found"}'
                    ),
                )
            if path == "/sessions/start":
                return {"provider_session_id": "new-session"}
            if path == "/sessions/new-session/pdf-read":
                return {"status": "ok", "text": "needle"}
            raise AssertionError(f"unexpected request path: {path}")

        with patch.object(service, "_request", new=fake_request):
            result = await service.read_pdf_url(
                conversation_id="conv",
                url="https://example.com/paper.pdf",
                query="needle",
            )

        self.assertEqual(result, {"status": "ok", "text": "needle"})
        self.assertEqual(
            [(method, path) for method, path, _ in calls],
            [
                ("POST", "/sessions/old-session/pdf-read"),
                ("POST", "/sessions/start"),
                ("POST", "/sessions/new-session/pdf-read"),
            ],
        )
        self.assertEqual(calls[-1][2]["query"], "needle")
        self.assertEqual(service._sessions["conv"].provider_session_id, "new-session")


if __name__ == "__main__":
    unittest.main()
