"""Omitted pages remain visible and are not retried within the same attempt."""

import asyncio
import base64
from functools import wraps
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.documents import Document

import ragtime.indexer.service as service
from ragtime.core import vision_models
from ragtime.indexer import document_parser
from ragtime.indexer.indexing_pipeline import BoundedIndexingPipeline
from ragtime.indexer.indexing_spool import IndexingSpool
from ragtime.indexer.models import IndexConfig, IndexJob, OcrMode, OcrProvider


def run_async(test):
    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


def test_failed_source_checkpoint_survives_reopen_without_duplicate_count(tmp_path: Path) -> None:
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fingerprint")
    spool.record_source_warning(1, "pages/page-129.png", "timeout")
    spool.record_source_warning(1, "pages/page-129.png", "timeout")
    spool.close()

    resumed = IndexingSpool.open_attempt(tmp_path, "job", "fingerprint")
    try:
        assert resumed.failed_source_ordinals() == {1}
        assert resumed.source_warning_summary() == {
            "count": 1,
            "items": [{"source": "pages/page-129.png", "category": "timeout"}],
        }
    finally:
        resumed.close()


def test_warning_samples_are_bounded_without_losing_total_or_checkpoint(tmp_path: Path) -> None:
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fingerprint")
    try:
        for ordinal in range(105):
            spool.record_source_warning(ordinal, f"pages/{ordinal}.png", "invalid_output")
        summary = spool.source_warning_summary()
        assert summary["count"] == 105
        assert len(summary["items"]) == 100
        assert summary["items"][0] == {"source": "pages/0.png", "category": "invalid_output"}
        assert summary["items"][-1] == {"source": "pages/99.png", "category": "invalid_output"}
        assert spool.failed_source_ordinals() == set(range(105))
    finally:
        spool.close()


def test_new_job_retries_sources_instead_of_inheriting_omissions(tmp_path: Path) -> None:
    old = IndexingSpool.open_attempt(tmp_path, "old-job", "fingerprint")
    old.record_source_warning(0, "page.png", "timeout")
    old.close()
    new = IndexingSpool.open_attempt(tmp_path, "new-job", "fingerprint")
    try:
        assert new.failed_source_ordinals() == set()
        assert new.source_warning_summary() == {"count": 0, "items": []}
    finally:
        new.close()


def test_warning_checkpoint_encodes_unsafe_filename_characters_and_bounds_long_paths(tmp_path: Path) -> None:
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fp")
    try:
        spool.record_source_warning(0, "pages/name:with\ncontrol.png", "timeout")
        spool.record_source_warning(1, "nested/" + "a" * 600 + ".png", "timeout")
        items = spool.source_warning_summary()["items"]
        assert items[0]["source"] == "pages/name%3Awith%0Acontrol.png"
        assert len(items[1]["source"]) <= 512
        assert "\n" not in items[0]["source"]
    finally:
        spool.close()


@run_async
async def test_page_failure_does_not_block_later_sources_or_repeat_on_resume(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    files = [source / "bad.png", source / "good.txt"]
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fp")
    visited: list[str] = []
    progress: list[int] = []

    async def load(path: Path) -> list[Document]:
        visited.append(path.name)
        if path.name == "bad.png":
            raise document_parser.VisionOcrPageError("Page timed out", category="timeout")
        return [Document(page_content="usable content")]

    async def loaded(count: int) -> None:
        progress.append(count)

    try:
        pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
        assert await pipeline.stage_documents(files, source, "index", load, loaded) == 1
        assert visited == ["bad.png", "good.txt"]
        assert progress == [1, 2]
        assert spool.source_warning_summary() == {"count": 1, "items": [{"source": "bad.png", "category": "timeout"}]}
        assert spool.summary()["document_count"] == 1
        assert await pipeline.stage_documents(files, source, "index", load, loaded) == 0
        assert visited == ["bad.png", "good.txt"]
        assert spool.source_warning_summary()["count"] == 1
    finally:
        spool.close()


@run_async
async def test_global_ocr_failure_still_aborts_without_page_warning(tmp_path: Path) -> None:
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fp")

    async def load(path: Path) -> list[Document]:
        raise document_parser.VisionOcrError("Missing configured model")

    try:
        pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
        with pytest.raises(document_parser.VisionOcrError, match="configured model"):
            await pipeline.stage_documents([tmp_path / "bad.png"], tmp_path, "index", load)
        assert spool.source_warning_summary()["count"] == 0
    finally:
        spool.close()


@run_async
async def test_all_omitted_pages_leave_no_publishable_document_records(tmp_path: Path) -> None:
    spool = IndexingSpool.open_attempt(tmp_path, "job", "fp")

    async def load(path: Path) -> list[Document]:
        raise document_parser.VisionOcrPageError("Unusable page", category="invalid_output")

    try:
        pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
        assert await pipeline.stage_documents([tmp_path / "bad.png"], tmp_path, "index", load) == 0
        assert spool.source_warning_summary()["count"] == 1
        assert spool.summary()["document_count"] == 0
    finally:
        spool.close()


@run_async
async def test_upload_all_page_failures_persist_warnings_on_resume_without_publishing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    page = root / "page.png"
    page.write_bytes(b"image")
    job = IndexJob(
        id="upload-job",
        name="existing-index",
        source_type="upload",
        config=IndexConfig(name="existing-index", ocr_mode=OcrMode.VISION, ocr_provider=OcrProvider.OLLAMA, ocr_vision_model="vision-model"),
    )
    indexer = service.IndexerService(index_base_path=str(tmp_path / "indexes"))
    persisted = []

    async def update(current):
        persisted.append(current.model_dump())

    class Permit:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    extract = AsyncMock(side_effect=document_parser.VisionOcrPageError("Page timeout", category="timeout"))
    publish = AsyncMock()
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    monkeypatch.setattr(service.repository, "update_job", update)
    monkeypatch.setattr(service.repository, "get_settings", AsyncMock(return_value={"ollama_base_url": "http://ollama.test:11434"}))
    monkeypatch.setattr(service, "collect_files_recursive", lambda *_a, **_kw: [(page, page.stat().st_size)])
    monkeypatch.setattr(service.resource_governor, "acquire", lambda _request: Permit())
    monkeypatch.setattr(service, "extract_image_structured_async", extract)
    monkeypatch.setattr(service, "prepare_faiss_artifact", publish)
    for _ in range(2):
        with pytest.raises(ValueError, match="No indexable documents"):
            await indexer._create_faiss_index(job, root)
    assert job.warnings.model_dump() == {"count": 1, "items": [{"source": "page.png", "category": "timeout"}]}
    assert persisted[-1]["warnings"]["count"] == 1
    extract.assert_awaited_once()
    publish.assert_not_awaited()


@run_async
async def test_provider_runner_error_stream_stays_fatal_after_a_successful_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    good, bad = source / "good.txt", source / "bad.png"
    good.write_text("Already loaded source")
    bad.write_bytes(base64.b64decode(vision_models.TINY_PNG_DATA_URL.split(",")[1]))
    job = IndexJob(
        id="runner-error",
        name="existing",
        source_type="upload",
        config=IndexConfig(name="existing", ocr_mode=OcrMode.VISION, ocr_provider=OcrProvider.OLLAMA, ocr_vision_model="vision-model"),
    )

    class Permit:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b'{"error":"PRIVATE runner failed"}\n'))
    monkeypatch.setattr(vision_models.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    monkeypatch.setattr(vision_models, "_resolve_ocr_output_budget", AsyncMock(return_value=4096))
    monkeypatch.setattr(vision_models, "_supports_structured_vision_output", AsyncMock(return_value=True))
    monkeypatch.setattr(vision_models, "_get_vision_ocr_limiter", AsyncMock(return_value=vision_models._VisionOcrLimiter(1)))
    monkeypatch.setattr(vision_models, "_vision_ocr_limit", AsyncMock(return_value=1))
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    monkeypatch.setattr(service.repository, "update_job", AsyncMock())
    monkeypatch.setattr(service.repository, "get_settings", AsyncMock(return_value={"ollama_base_url": "http://ollama.test:11434"}))
    monkeypatch.setattr(service, "collect_files_recursive", lambda *_a, **_kw: [(path, path.stat().st_size) for path in (good, bad)])
    monkeypatch.setattr(service.resource_governor, "acquire", lambda _request: Permit())
    publish = AsyncMock()
    monkeypatch.setattr(service, "prepare_faiss_artifact", publish)
    indexer = service.IndexerService(index_base_path=str(tmp_path / "indexes"))
    with pytest.raises(document_parser.VisionOcrError) as error:
        await indexer._create_faiss_index(job, source)
    assert not isinstance(error.value, document_parser.VisionOcrPageError)
    assert "PRIVATE" not in str(error.value)
    assert job.warnings.count == 0
    publish.assert_not_awaited()
    attempt = next(path for path in (tmp_path / "indexing" / job.id).iterdir() if path.is_dir())
    spool = IndexingSpool.open_existing(attempt)
    try:
        assert spool.summary()["document_count"] == 1
        assert spool.source_warning_summary()["count"] == 0
    finally:
        spool.close()
