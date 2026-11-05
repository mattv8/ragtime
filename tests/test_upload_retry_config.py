from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ragtime.core.ocr_target import OcrTarget
from ragtime.indexer import routes, service
from ragtime.indexer.models import IndexConfig, IndexJob, IndexStatus, OcrMode, OcrProvider, VectorStoreType
from ragtime.indexer.ocr_job_snapshot import OcrJobSnapshotError, create_ocr_job_snapshot
from ragtime.indexer.service import IndexerService


def _failed_upload_job(config: IndexConfig) -> IndexJob:
    return IndexJob(id="failed-job", name="retry-index", config=config, source_type="upload")


def test_retry_restores_vision_config_from_durable_snapshot_without_mutating_failed_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_config = IndexConfig(
        name="retry-index",
        description="Keep every configured field",
        file_patterns=["**/*.pdf"],
        exclude_patterns=["**/skip/**"],
        chunk_size=750,
        chunk_overlap=125,
        max_file_size_kb=2048,
        embedding_model="text-embedding-3-large",
        vector_store_type=VectorStoreType.PGVECTOR,
        ocr_mode=OcrMode.DISABLED,
        git_clone_timeout_minutes=12,
        git_history_depth=42,
        reindex_interval_hours=24,
        reindex_start_minute=300,
        reindex_timezone="America/Chicago",
    )
    failed_job = _failed_upload_job(failed_config)
    create_ocr_job_snapshot(
        tmp_path / "indexing",
        failed_job.id,
        OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
        "pipeline-hash",
    )
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))

    restored = asyncio.run(indexer.restore_retry_job_config(failed_job))

    assert restored.model_dump(mode="json") == {
        "name": "retry-index",
        "description": "Keep every configured field",
        "file_patterns": ["**/*.pdf"],
        "exclude_patterns": ["**/skip/**"],
        "chunk_size": 750,
        "chunk_overlap": 125,
        "max_file_size_kb": 2048,
        "embedding_model": "text-embedding-3-large",
        "vector_store_type": "pgvector",
        "ocr_mode": "vision",
        "ocr_provider": "openai",
        "ocr_vision_model": "gpt-4o-mini",
        "git_clone_timeout_minutes": 12,
        "git_history_depth": 42,
        "reindex_interval_hours": 24,
        "reindex_start_minute": 300,
        "reindex_timezone": "America/Chicago",
    }
    assert failed_job.config.ocr_mode is OcrMode.DISABLED
    assert failed_job.config.ocr_provider is None
    assert failed_job.config.ocr_vision_model is None


def test_direct_upload_retry_uses_restored_vision_config_and_keeps_archive_after_processing_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_job = _failed_upload_job(IndexConfig(name="retry-index", ocr_mode=OcrMode.DISABLED))
    create_ocr_job_snapshot(
        tmp_path / "indexing",
        failed_job.id,
        OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
        "pipeline-hash",
    )
    archive_dir = tmp_path / failed_job.id
    archive_dir.mkdir()
    archive_path = archive_dir / "source.zip"
    archive_path.write_bytes(b"archive")
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))
    monkeypatch.setattr(service.repository, "create_job", mock.AsyncMock())
    monkeypatch.setattr(indexer, "_create_optimistic_index_metadata", mock.AsyncMock())
    monkeypatch.setattr(indexer, "_process_upload", mock.AsyncMock(side_effect=RuntimeError("processing failed")))
    monkeypatch.setattr(service.uuid, "uuid4", lambda: SimpleNamespace(hex="new-job-id"))

    async def retry_and_wait_for_failure() -> IndexJob:
        new_job = await indexer.retry_upload_job(failed_job)
        with pytest.raises(RuntimeError, match="processing failed"):
            await indexer._processing_tasks[new_job.id]
        return new_job

    new_job = asyncio.run(retry_and_wait_for_failure())
    assert new_job.config.ocr_mode is OcrMode.VISION
    assert new_job.config.ocr_provider is OcrProvider.OPENAI
    assert new_job.config.ocr_vision_model == "gpt-4o-mini"
    assert (tmp_path / new_job.id / "source.zip").read_bytes() == b"archive"
    assert not archive_path.exists()


def test_retry_rejects_malformed_snapshot_before_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_job = _failed_upload_job(IndexConfig(name="retry-index", ocr_mode=OcrMode.DISABLED))
    snapshot_dir = tmp_path / "indexing" / failed_job.id
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "ocr-target.json").write_text("not json", encoding="utf-8")
    archive_dir = tmp_path / failed_job.id
    archive_dir.mkdir()
    archive_path = archive_dir / "source.zip"
    archive_path.write_bytes(b"archive")
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))
    monkeypatch.setattr(service.repository, "create_job", mock.AsyncMock())
    monkeypatch.setattr(indexer, "_create_optimistic_index_metadata", mock.AsyncMock())
    process_upload = mock.AsyncMock()
    monkeypatch.setattr(indexer, "_process_upload", process_upload)

    with pytest.raises(OcrJobSnapshotError):
        asyncio.run(indexer.retry_upload_job(failed_job))

    assert archive_path.is_file()
    process_upload.assert_not_awaited()


@pytest.mark.parametrize(
    ("provider", "model"),
    [(None, None), (OcrProvider.ANTHROPIC, "claude-3-5-sonnet")],
    ids=["current-global-defaults", "explicit-per-job-target"],
)
def test_retry_preserves_known_vision_config_over_old_snapshot(
    provider: OcrProvider | None, model: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = IndexConfig(name="retry-index", ocr_mode=OcrMode.VISION, ocr_provider=provider, ocr_vision_model=model)
    original_config = config.model_dump()
    failed_job = _failed_upload_job(config)
    create_ocr_job_snapshot(
        tmp_path / "indexing",
        failed_job.id,
        OcrTarget(provider="ollama", model="qwen3-vl:32b", base_url="http://old-ollama:11434"),
        "pipeline-hash",
    )
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))

    restored = asyncio.run(indexer.restore_retry_job_config(failed_job))

    assert restored.model_dump() == original_config
    assert restored is not failed_job.config
    assert failed_job.config.model_dump() == original_config


def test_retry_without_snapshot_preserves_known_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexConfig(
        name="retry-index",
        ocr_mode=OcrMode.VISION,
        ocr_provider=OcrProvider.ANTHROPIC,
        ocr_vision_model="claude-3-5-sonnet",
        vector_store_type=VectorStoreType.PGVECTOR,
    )
    failed_job = _failed_upload_job(config)
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))

    restored = asyncio.run(indexer.restore_retry_job_config(failed_job))

    assert restored == config
    assert restored is not config


@pytest.mark.parametrize("source_type", ["git", "upload"])
def test_route_restores_snapshot_config_before_validating_retry_target(source_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_job = IndexJob(
        id=f"failed-{source_type}",
        name="retry-index",
        status=IndexStatus.FAILED,
        config=IndexConfig(name="retry-index", ocr_mode=OcrMode.DISABLED),
        source_type=source_type,
        git_url="https://example.com/repo.git" if source_type == "git" else None,
    )
    create_ocr_job_snapshot(
        tmp_path / "indexing",
        failed_job.id,
        OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
        "pipeline-hash",
    )
    validated_configs: list[IndexConfig] = []

    async def validate(config: IndexConfig) -> None:
        validated_configs.append(config)

    async def create_git(*args, **kwargs) -> IndexJob:
        return failed_job.model_copy(update={"id": "new-git", "config": kwargs["config"]})

    async def retry_upload(job: IndexJob) -> IndexJob:
        return job.model_copy(update={"id": "new-upload"})

    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    monkeypatch.setattr(routes.repository, "get_job", mock.AsyncMock(return_value=failed_job))
    monkeypatch.setattr(routes, "_validate_vision_ocr_target", validate)
    monkeypatch.setattr(routes.indexer, "create_index_from_git", create_git)
    monkeypatch.setattr(routes.indexer, "retry_upload_job", retry_upload)

    asyncio.run(
        routes.retry_failed_job(
            failed_job.id,
            routes.RetryJobRequest(),
            _user=mock.sentinel.user,
            _=None,
        )
    )

    assert len(validated_configs) == 1
    assert validated_configs[0].ocr_mode is OcrMode.VISION
    assert validated_configs[0].ocr_provider is OcrProvider.OPENAI
    assert failed_job.config.ocr_mode is OcrMode.DISABLED
