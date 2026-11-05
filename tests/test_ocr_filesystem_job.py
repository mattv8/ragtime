"""Filesystem OCR replacement jobs must preserve serving data until promotion."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from ragtime.core.ocr_target import OcrTarget
from ragtime.indexer.filesystem_service import FilesystemIndexerService, _FilesystemChunk
from ragtime.indexer.models import FilesystemConnectionConfig, FilesystemIndexJob, FilesystemIndexStatus, OcrMode, OcrProvider


class RecordingBackend:
    """In-memory backend that models a private replacement generation."""

    def __init__(self, fail_insert: bool = False) -> None:
        self.fail_insert = fail_insert
        self.stored: list[tuple[str, str, list[str], list[dict]]] = []
        self.stored_embeddings: list[tuple[str, list[list[float]]]] = []
        self.published = 0
        self.discarded: list[str] = []

    async def delete_file_embeddings(self, index_name: str, file_path: str) -> int:
        return 0

    async def store_embeddings(self, index_name, file_path, chunks, embeddings, metadata, chunk_metadata=None):
        if self.fail_insert:
            raise RuntimeError("insert failed")
        self.stored.append((index_name, file_path, list(chunks), list(chunk_metadata or [])))
        self.stored_embeddings.append((file_path, list(embeddings)))
        return len(chunks)

    async def finalize_index(self, index_name: str) -> None:
        return None

    async def publish_staged_index(self, staging_name: str, index_name: str) -> None:
        self.published += 1

    async def discard_staged_index(self, staging_name: str) -> None:
        self.discarded.append(staging_name)


def run_async(test):
    """Run coroutine tests without requiring an optional pytest async plugin."""

    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


@pytest.fixture
def filesystem_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    service = FilesystemIndexerService()
    config = FilesystemConnectionConfig(base_path=str(tmp_path), index_name="serving", chunk_size=100, chunk_overlap=0)
    job = FilesystemIndexJob(id="job-1", tool_config_id="tool-1", index_name="serving")

    @asynccontextmanager
    async def mount(_config: FilesystemConnectionConfig, _tool_id: str | None) -> AsyncIterator[Path]:
        yield tmp_path

    monkeypatch.setattr(service, "_mount_filesystem", mount)
    monkeypatch.setattr(service, "_collect_files", lambda *_args: [tmp_path / "a.txt", tmp_path / "b.txt"])
    monkeypatch.setattr(service, "_update_job", AsyncMock())
    monkeypatch.setattr(service, "_upsert_file_metadata_batch", AsyncMock(return_value=2))
    monkeypatch.setattr(service, "_update_tool_config_last_indexed", AsyncMock())
    return service, job, config


def settings():
    return SimpleNamespace(
        embedding_dimension=None,
        embedding_config_hash=None,
        get_embedding_config_hash=lambda: "embedding-hash",
    )


@pytest.mark.parametrize("failure", ["extract", "embed", "insert"])
@run_async
async def test_full_process_failure_discards_stage_without_publishing(filesystem_job, failure):
    service, job, config = filesystem_job
    backend = RecordingBackend(fail_insert=failure == "insert")
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]
    service._load_and_chunk_file = AsyncMock(
        side_effect=RuntimeError("extract failed")
        if failure == "extract"
        else [[_FilesystemChunk("text", {"source": "a.txt"})], [_FilesystemChunk("text", {"source": "b.txt"})]]
    )  # type: ignore[method-assign]

    embed = AsyncMock(side_effect=RuntimeError("embed failed") if failure == "embed" else lambda _model, docs, **_kwargs: [[1.0] for _ in docs])
    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"embedding_provider": "ollama"})),
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
        patch("ragtime.indexer.filesystem_service.embed_documents_subbatched", embed),
    ):
        await service._process_index(job, config, full_reindex=True)

    assert job.status is FilesystemIndexStatus.FAILED
    assert backend.published == 0
    assert backend.discarded == ["__filesystem_staging__job-1"]
    repo.update_embedding_tracking.assert_not_awaited()


@run_async
async def test_full_process_rechunks_per_file_without_metadata_or_vector_misalignment(filesystem_job):
    service, job, config = filesystem_job
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]

    async def load(file_path, *_args, **_kwargs):
        if file_path.name == "a.txt":
            await asyncio.sleep(0.01)
            return [_FilesystemChunk("a", {"source": "/noncanonical/a.txt", "page": 1})]
        return [_FilesystemChunk("b", {"source": "/noncanonical/b.txt", "page": 2})]

    service._load_and_chunk_file = AsyncMock(side_effect=load)  # type: ignore[method-assign]

    def rechunk(documents, *_args):
        documents_by_text = {document.page_content: document for document in documents}
        a_document = documents_by_text["a"]
        return [
            documents_by_text["b"],
            Document(page_content="a-1", metadata=dict(a_document.metadata)),
            Document(page_content="a-2", metadata=dict(a_document.metadata)),
        ], 1

    async def embed(_model, docs, **_kwargs):
        vectors_by_text = {"a-1": [1.0], "a-2": [2.0], "b": [3.0]}
        return [vectors_by_text[text] for text in docs]

    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"embedding_provider": "ollama"})),
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
        patch("ragtime.indexer.filesystem_service.rechunk_documents_batch", rechunk),
        patch("ragtime.indexer.filesystem_service.embed_documents_subbatched", embed),
    ):
        await service._process_index(job, config, full_reindex=True)

    assert job.status is FilesystemIndexStatus.COMPLETED
    assert backend.published == 1
    assert [(path, chunks, metadata) for _, path, chunks, metadata in backend.stored] == [
        ("a.txt", ["a-1", "a-2"], [{"source": "a.txt", "page": 1}, {"source": "a.txt", "page": 1}]),
        ("b.txt", ["b"], [{"source": "b.txt", "page": 2}]),
    ]
    assert backend.stored_embeddings == [("a.txt", [[1.0], [2.0]]), ("b.txt", [[3.0]])]


@run_async
async def test_explicit_vision_target_is_frozen_once_and_cancel_discards_stage(filesystem_job):
    service, job, config = filesystem_job
    image = Path(config.base_path) / "scan.png"
    image.write_bytes(b"image")
    (Path(config.base_path) / "a.txt").unlink()
    (Path(config.base_path) / "b.txt").unlink()
    config = config.model_copy(update={"ocr_mode": OcrMode.VISION, "ocr_provider": OcrProvider.OPENAI, "ocr_vision_model": "vision-model"})
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    target = OcrTarget(provider="openai", model="vision-model", base_url="https://api.openai.com/v1")
    service._collect_files = lambda *_args: [image]  # type: ignore[method-assign]
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]

    async def cancelled_load(*_args, **kwargs):
        assert kwargs["ocr_target"] == target
        service._cancellation_flags[job.id] = True
        return [_FilesystemChunk("image", {"source": "scan.png"})]

    service._load_and_chunk_file = cancelled_load  # type: ignore[method-assign]
    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"default_ocr_mode": "disabled"})),
        patch("ragtime.indexer.filesystem_service.resolve_ocr_target", return_value=target) as resolve,
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
    ):
        await service._process_index(job, config, full_reindex=True)

    resolve.assert_called_once()
    assert job.status is FilesystemIndexStatus.CANCELLED
    assert backend.published == 0
    assert backend.discarded == ["__filesystem_staging__job-1"]


@run_async
async def test_blank_vision_image_is_requested_once_then_fails_safe_replacement(filesystem_job):
    service, job, config = filesystem_job
    image = Path(config.base_path) / "blank.png"
    image.write_bytes(b"image")
    (Path(config.base_path) / "a.txt").unlink()
    (Path(config.base_path) / "b.txt").unlink()
    config = config.model_copy(update={"ocr_mode": OcrMode.VISION, "ocr_provider": OcrProvider.OPENAI, "ocr_vision_model": "vision-model"})
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    target = OcrTarget(provider="openai", model="vision-model", base_url="https://api.openai.com/v1")
    blank_result = SimpleNamespace(raw_text="", extracted_text=[], get_semantic_segments=lambda: [])
    service._collect_files = lambda *_args: [image]  # type: ignore[method-assign]
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]

    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"embedding_provider": "ollama"})),
        patch("ragtime.indexer.filesystem_service.resolve_ocr_target", return_value=target),
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
        patch("ragtime.indexer.filesystem_service.extract_image_structured_async", AsyncMock(return_value=blank_result)) as extract,
    ):
        await service._process_index(job, config, full_reindex=True)

    extract.assert_awaited_once()
    assert job.status is FilesystemIndexStatus.FAILED
    assert backend.published == 0
    assert backend.discarded == ["__filesystem_staging__job-1"]


@run_async
async def test_image_semantic_chunks_keep_frozen_ocr_provenance(filesystem_job):
    service, job, config = filesystem_job
    image = Path(config.base_path) / "scan.png"
    image.write_bytes(b"image")
    (Path(config.base_path) / "a.txt").unlink()
    (Path(config.base_path) / "b.txt").unlink()
    config = config.model_copy(update={"ocr_mode": OcrMode.VISION, "ocr_provider": OcrProvider.OPENAI, "ocr_vision_model": "vision-model"})
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    target = OcrTarget(provider="openai", model="vision-model", base_url="https://api.openai.com/v1")
    result = SimpleNamespace(
        raw_text="",
        extracted_text=[object()],
        get_semantic_segments=lambda: [("NarrativeText", "visible text"), ("classification", "image description")],
    )
    service._collect_files = lambda *_args: [image]  # type: ignore[method-assign]
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]

    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"embedding_provider": "ollama", "chunking_use_tokens": True})),
        patch("ragtime.indexer.filesystem_service.resolve_ocr_target", return_value=target),
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
        patch("ragtime.indexer.filesystem_service.extract_image_structured_async", AsyncMock(return_value=result)),
        patch("ragtime.indexer.filesystem_service.embed_documents_subbatched", AsyncMock(return_value=[[1.0], [1.0]])),
    ):
        await service._process_index(job, config, full_reindex=True)

    metadata = backend.stored[0][3]
    assert all(item["ocr_provider"] == "openai" and item["ocr_model"] == "vision-model" for item in metadata)
    assert all(item["ocr_ingestion_version"] == 1 and item["ocr_pipeline_fingerprint"] for item in metadata)
    assert all("base_url" not in item for item in metadata)


CONTEXT_ERROR = "input length exceeds the maximum context length"


async def _chunks_for_each_file(file_path, *_args, **_kwargs):
    return [_FilesystemChunk(f"CONFIDENTIAL {file_path.name}", {"source": file_path.name})]


@contextmanager
def _exhausted_context_retries(backend, repo, embed):
    """Every embedding attempt hits the context limit, and rechunking never shrinks anything."""
    with (
        patch("ragtime.indexer.filesystem_service.get_app_settings", AsyncMock(return_value={"embedding_provider": "ollama"})),
        patch("ragtime.indexer.repository.IndexerRepository", return_value=repo),
        patch("ragtime.indexer.filesystem_service.get_backend", return_value=backend),
        patch("ragtime.core.embedding_models.get_embedding_model_context_limit", AsyncMock(return_value=100)),
        patch("ragtime.core.tokenization.count_tokens", return_value=10_000),
        patch("ragtime.indexer.filesystem_service.rechunk_documents_batch", lambda documents, *_args: (list(documents), 0)),
        patch("ragtime.indexer.filesystem_service.embed_documents_subbatched", embed),
    ):
        yield


@run_async
async def test_full_replacement_context_exhaustion_refuses_fallback_and_discards_stage(filesystem_job):
    service, job, config = filesystem_job
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]
    service._load_and_chunk_file = AsyncMock(side_effect=_chunks_for_each_file)  # type: ignore[method-assign]
    fallback = AsyncMock()
    service._embed_chunks_with_fallback = fallback  # type: ignore[method-assign]
    embed = AsyncMock(side_effect=RuntimeError(CONTEXT_ERROR))

    with _exhausted_context_retries(backend, repo, embed):
        await service._process_index(job, config, full_reindex=True)

    assert embed.await_count == 4  # initial attempt plus the three context-retry factors
    fallback.assert_not_awaited()
    assert job.status is FilesystemIndexStatus.FAILED
    assert "full replacement" in job.error_message
    assert "a.txt, b.txt" in job.error_message
    assert "existing index was preserved" in job.error_message.lower()
    assert "CONFIDENTIAL" not in job.error_message
    assert backend.published == 0
    assert backend.stored == []
    assert backend.discarded == ["__filesystem_staging__job-1"]
    repo.update_embedding_tracking.assert_not_awaited()


@pytest.mark.parametrize("case", ["vision", "table"])
@run_async
async def test_incremental_vision_or_table_context_exhaustion_refuses_fallback(filesystem_job, case):
    service, job, config = filesystem_job
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    target = OcrTarget(provider="openai", model="vision-model", base_url="https://api.openai.com/v1")
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]
    service._get_file_metadata_batch = AsyncMock(return_value={})  # type: ignore[method-assign]
    segment_metadata = {"segment_type": "table"} if case == "table" else {}

    async def load(file_path, *_args, **_kwargs):
        return [_FilesystemChunk(f"CONFIDENTIAL {file_path.name}", {"source": file_path.name, **segment_metadata})]

    service._load_and_chunk_file = AsyncMock(side_effect=load)  # type: ignore[method-assign]
    if case == "vision":
        config = config.model_copy(update={"ocr_mode": OcrMode.VISION, "ocr_provider": OcrProvider.OPENAI, "ocr_vision_model": "vision-model"})
    else:
        config = config.model_copy(update={"ocr_mode": OcrMode.DISABLED})
    fallback = AsyncMock()
    service._embed_chunks_with_fallback = fallback  # type: ignore[method-assign]
    embed = AsyncMock(side_effect=RuntimeError(CONTEXT_ERROR))

    with (
        patch("ragtime.indexer.filesystem_service.resolve_ocr_target", return_value=target),
        _exhausted_context_retries(backend, repo, embed),
    ):
        await service._process_index(job, config, full_reindex=False)

    fallback.assert_not_awaited()
    assert job.status is FilesystemIndexStatus.FAILED
    assert ("vision OCR output" if case == "vision" else "semantic table rows") in job.error_message
    assert "CONFIDENTIAL" not in job.error_message
    assert backend.stored == []
    assert backend.published == 0
    assert backend.discarded == []
    repo.update_embedding_tracking.assert_not_awaited()


@run_async
async def test_incremental_plain_context_exhaustion_keeps_legacy_fallback(filesystem_job):
    service, job, config = filesystem_job
    backend = RecordingBackend()
    repo = SimpleNamespace(get_settings=AsyncMock(return_value=settings()), update_embedding_tracking=AsyncMock())
    service._get_embeddings = AsyncMock(return_value=object())  # type: ignore[method-assign]
    service._get_file_metadata_batch = AsyncMock(return_value={})  # type: ignore[method-assign]
    service._load_and_chunk_file = AsyncMock(side_effect=_chunks_for_each_file)  # type: ignore[method-assign]

    async def legacy_fallback(_embeddings, chunks):
        return [[1.0] for _ in chunks]

    fallback = AsyncMock(side_effect=legacy_fallback)
    service._embed_chunks_with_fallback = fallback  # type: ignore[method-assign]
    embed = AsyncMock(side_effect=RuntimeError(CONTEXT_ERROR))

    with _exhausted_context_retries(backend, repo, embed):
        await service._process_index(job, config, full_reindex=False)

    fallback.assert_awaited_once()
    assert job.status is FilesystemIndexStatus.COMPLETED
    assert [file_path for _, file_path, _, _ in backend.stored] == ["a.txt", "b.txt"]
