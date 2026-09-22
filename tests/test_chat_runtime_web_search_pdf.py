import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from ragtime.chat_runtime.service import (
    _TAVILY_SEARCH_ENDPOINT,
    ChatRuntimeService,
    _ChatDiagSession,
)


class ChatRuntimeWebSearchPdfTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _search_client_factory(
        responses: list[dict[str, Any]],
        client_class: type[httpx.AsyncClient],
    ):
        def create_client(**kwargs: Any) -> httpx.AsyncClient:
            def handle(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=responses.pop(0), request=request)

            return client_class(transport=httpx.MockTransport(handle), **kwargs)

        return create_client

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

    async def test_attach_pdf_metadata_only_marks_likely_candidates(self) -> None:
        service = ChatRuntimeService()
        results: list[dict[str, Any]] = [
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

    async def test_searxng_search_preserves_result_preparation_and_provider_fields(self) -> None:
        service = ChatRuntimeService()
        responses: list[dict[str, Any]] = [
            {
                "results": [
                    {"title": "[PDF] First", "url": "https://example.com/first.pdf"},
                    {"title": "Missing URL"},
                    "malformed result",
                    {"title": "After malformed", "url": "https://example.com/after"},
                ],
                "answers": ["  SearXNG    answer  "],
                "search_time": 0.42,
                "suggestions": ["suggestion"],
                "unresponsive_engines": ["engine"],
            },
            {
                "results": [
                    {"title": "First", "url": "https://example.com/first.pdf"},
                    {"url": "https://example.com/missing-title"},
                    {"title": "Second", "url": "https://example.com/second"},
                    {"title": "Over cap", "url": "https://example.com/over-cap"},
                ]
            },
        ]
        client_class = httpx.AsyncClient
        with patch(
            "ragtime.chat_runtime.service.httpx.AsyncClient",
            side_effect=self._search_client_factory(responses, client_class),
        ):
            pdf_result = await service._search_web_searxng(
                query="pdf query",
                max_results=2,
                include_pdf_metadata=True,
            )
            capped_result = await service._search_web_searxng(
                query="cap query",
                max_results=2,
                include_pdf_metadata=False,
            )

        self.assertEqual([result["title"] for result in pdf_result["results"]], ["[PDF] First"])
        self.assertEqual(pdf_result["pdf_result_count"], 1)
        self.assertEqual(pdf_result["results"][0]["pdf"]["status"], "available")
        self.assertEqual(pdf_result["results"][0]["source_provider"], "searxng")
        self.assertEqual(pdf_result["answer"], "SearXNG answer")
        self.assertEqual(pdf_result["response_time"], 0.42)
        self.assertEqual(pdf_result["suggestions"], ["suggestion"])
        self.assertEqual(pdf_result["unresponsive_engines"], ["engine"])
        self.assertEqual(
            pdf_result["engine_url"],
            "http://searxng:8080/search?q=pdf+query&format=json&categories=general&language=en",
        )
        self.assertEqual([result["title"] for result in capped_result["results"]], ["First", "Second"])
        self.assertEqual(capped_result["pdf_result_count"], 0)
        self.assertNotIn("pdf", capped_result["results"][0])

    async def test_tavily_search_preserves_result_preparation_and_provider_fields(self) -> None:
        service = ChatRuntimeService()
        responses: list[dict[str, Any]] = [
            {
                "results": [
                    {"title": "[PDF] First", "url": "https://example.com/first.pdf"},
                    {"title": "Missing URL"},
                    "malformed result",
                    {"title": "After malformed", "url": "https://example.com/after"},
                ],
                "answer": "  Tavily answer  ",
                "response_time": 0.13,
                "request_id": " request-1 ",
            },
            {
                "results": [
                    {"title": "First", "url": "https://example.com/first.pdf"},
                    {"url": "https://example.com/missing-title"},
                    {"title": "Second", "url": "https://example.com/second"},
                    {"title": "Over cap", "url": "https://example.com/over-cap"},
                ]
            },
        ]
        client_class = httpx.AsyncClient
        with (
            patch.object(service, "_tavily_api_key", return_value="test-key"),
            patch(
                "ragtime.chat_runtime.service.httpx.AsyncClient",
                side_effect=self._search_client_factory(responses, client_class),
            ),
        ):
            pdf_result = await service._search_web_tavily(
                query="pdf query",
                max_results=2,
                include_pdf_metadata=True,
            )
            capped_result = await service._search_web_tavily(
                query="cap query",
                max_results=2,
                include_pdf_metadata=False,
            )

        self.assertEqual([result["title"] for result in pdf_result["results"]], ["[PDF] First"])
        self.assertEqual(pdf_result["pdf_result_count"], 1)
        self.assertEqual(pdf_result["results"][0]["pdf"]["status"], "available")
        self.assertEqual(pdf_result["results"][0]["source_provider"], "tavily")
        self.assertEqual(pdf_result["answer"], "Tavily answer")
        self.assertEqual(pdf_result["response_time"], 0.13)
        self.assertEqual(pdf_result["request_id"], "request-1")
        self.assertEqual(pdf_result["engine_url"], _TAVILY_SEARCH_ENDPOINT)
        self.assertEqual([result["title"] for result in capped_result["results"]], ["First", "Second"])
        self.assertEqual(capped_result["pdf_result_count"], 0)
        self.assertNotIn("pdf", capped_result["results"][0])

    async def test_browse_url_retries_after_stale_runtime_session(self) -> None:
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
                    detail=('Chat diagnostics runtime manager request failed (404): {"detail":"Runtime session not found"}'),
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

    async def test_read_pdf_url_delegates_through_runtime_session_retry(self) -> None:
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
                    detail=('Chat diagnostics runtime manager request failed (404): {"detail":"Runtime session not found"}'),
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
        payload = calls[-1][2]
        assert payload is not None
        self.assertEqual(payload["query"], "needle")
        self.assertEqual(service._sessions["conv"].provider_session_id, "new-session")


if __name__ == "__main__":
    unittest.main()
