import unittest
from unittest.mock import patch

import httpx
import pytest

models = pytest.importorskip("runtime.manager.models")
worker_service = pytest.importorskip("runtime.worker.service")
RuntimePdfReadRequest = models.RuntimePdfReadRequest
WorkerService = worker_service.WorkerService


def _patch_async_client(transport: httpx.MockTransport):
    real_async_client = httpx.AsyncClient
    return patch(
        "runtime.worker.service.httpx.AsyncClient",
        new=lambda *args, **kwargs: real_async_client(
            transport=transport,
            follow_redirects=kwargs.get("follow_redirects", False),
            timeout=kwargs.get("timeout"),
        ),
    )


class RuntimePdfReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_requested_range(self):
        service = WorkerService()

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.headers.get("accept", "").split(",", 1)[0],
                "application/pdf",
            )
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.4\nfake pdf bytes",
            )

        def fake_extract(content):
            self.assertEqual(content, b"%PDF-1.4\nfake pdf bytes")
            return "First page\n\n\nSecond page\n\nThird page"

        transport = httpx.MockTransport(handler)
        with _patch_async_client(transport), patch.object(
            service, "_extract_pdf_text", new=fake_extract
        ):
            result = await service.read_pdf(
                RuntimePdfReadRequest(
                    url="https://example.com/paper.pdf",
                    start_char=12,
                    max_chars=11,
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "Second page")
        self.assertEqual(result.text_start_char, 12)
        self.assertEqual(result.text_end_char, 23)
        self.assertEqual(result.text_length, len("First page\n\nSecond page\n\nThird page"))
        self.assertTrue(result.truncated)

    async def test_returns_query_snippets(self):
        service = WorkerService()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.4\nfake pdf bytes",
            )

        def fake_extract(content):
            return "Intro\n\nAttention heads attend to tokens.\n\nConclusion"

        transport = httpx.MockTransport(handler)
        with _patch_async_client(transport), patch.object(
            service, "_extract_pdf_text", new=fake_extract
        ):
            result = await service.read_pdf(
                RuntimePdfReadRequest(
                    url="https://example.com/paper.pdf",
                    query="attention heads",
                    max_chars=500,
                    max_matches=3,
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.query, "attention heads")
        self.assertEqual(result.match_count, 1)
        self.assertIn("Attention heads attend", result.text)
        self.assertEqual(result.matches[0].match_start_char, 7)

    async def test_rejects_non_pdf_response(self):
        service = WorkerService()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=b"<html><body>not a pdf</body></html>",
            )

        transport = httpx.MockTransport(handler)
        with _patch_async_client(transport):
            result = await service.read_pdf(
                RuntimePdfReadRequest(url="https://example.com/paper")
            )

        self.assertEqual(result.status, "not_pdf")
        self.assertEqual(result.content_type, "text/html; charset=utf-8")

    async def test_explains_upstream_http_denial(self):
        service = WorkerService()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                headers={"content-type": "text/html"},
                content=(
                    b"<html><body><h1>Access Denied</h1>"
                    b"You don't have permission to access this PDF.</body></html>"
                ),
            )

        transport = httpx.MockTransport(handler)
        with _patch_async_client(transport):
            result = await service.read_pdf(
                RuntimePdfReadRequest(url="https://example.com/paper.pdf")
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.failure_mode, "upstream_http_error")
        self.assertEqual(result.status_code, 403)
        self.assertIn("Access Denied", result.body_preview)
        self.assertIn("denied", result.error.lower())


if __name__ == "__main__":
    unittest.main()
