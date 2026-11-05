from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest

from ragtime.core.ocr_target import OcrTarget
from ragtime.indexer import service
from ragtime.indexer.document_parser import VisionOcrError
from ragtime.indexer.models import IndexConfig, IndexJob, IndexStatus, OcrMode
from ragtime.indexer.ocr_job_snapshot import OcrJobSnapshotError, create_ocr_job_snapshot, load_ocr_job_snapshot
from ragtime.indexer.service import IndexerService


def test_snapshot_is_written_before_extraction_and_contains_no_secret(tmp_path: Path) -> None:
    target = OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    snapshot = create_ocr_job_snapshot(tmp_path, "job-1", target, "pipeline-hash")

    snapshot_path = tmp_path / "job-1" / "ocr-target.json"
    assert snapshot_path.is_file()
    assert snapshot.target == target
    assert "secret" not in snapshot_path.read_text(encoding="utf-8").lower()
    assert "api_key" not in snapshot_path.read_text(encoding="utf-8").lower()


def test_resume_uses_frozen_target_after_global_defaults_change(tmp_path: Path) -> None:
    original = OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    create_ocr_job_snapshot(tmp_path, "job-1", original, "pipeline-hash")

    restored = load_ocr_job_snapshot(tmp_path, "job-1")

    assert restored is not None
    assert restored.target == original
    assert restored.target.base_url == "https://api.openai.com/v1"
    assert restored.config_overrides == {
        "ocr_mode": "vision",
        "ocr_provider": "openai",
        "ocr_vision_model": "gpt-4o-mini",
    }


def test_new_job_uses_its_own_resolved_target(tmp_path: Path) -> None:
    create_ocr_job_snapshot(
        tmp_path,
        "old-job",
        OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
        "old-pipeline-hash",
    )
    create_ocr_job_snapshot(
        tmp_path,
        "new-job",
        OcrTarget(provider="anthropic", model="claude-sonnet", base_url="https://api.anthropic.com"),
        "new-pipeline-hash",
    )

    assert load_ocr_job_snapshot(tmp_path, "new-job").target.provider == "anthropic"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"target": {"provider": "openai"}}),
        json.dumps(
            {
                "target": {"provider": "openai", "model": "gpt-4o", "base_url": "https://user:token@example.com"},
                "config": {"ocr_mode": "vision", "ocr_provider": "openai", "ocr_vision_model": "gpt-4o"},
                "pipeline_fingerprint": "hash",
            }
        ),
    ],
)
def test_corrupt_snapshot_fails_instead_of_using_current_defaults(tmp_path: Path, payload: str) -> None:
    job_root = tmp_path / "job-1"
    job_root.mkdir()
    (job_root / "ocr-target.json").write_text(payload, encoding="utf-8")

    with pytest.raises(OcrJobSnapshotError):
        load_ocr_job_snapshot(tmp_path, "job-1")


def test_resume_restores_vision_mode_and_overrides_from_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_ocr_job_snapshot(
        tmp_path / "indexing",
        "job-1",
        OcrTarget(provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
        "pipeline-hash",
    )
    job = IndexJob(
        id="job-1",
        name="index",
        status=IndexStatus.PROCESSING,
        config=IndexConfig(name="index", ocr_mode=OcrMode.DISABLED),
        source_type="git",
        git_url="https://example.com/repo.git",
    )
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)

    with monkeypatch.context() as patches:
        patches.setattr(service.repository, "update_job", mock.AsyncMock())
        patches.setattr(indexer, "_process_git", mock.AsyncMock())
        patches.setattr(service.asyncio, "create_task", lambda coro: (coro.close(), object())[1])
        asyncio.run(indexer._resume_job(job))

    assert job.config.ocr_mode is OcrMode.VISION
    assert job.config.ocr_provider is not None
    assert job.config.ocr_provider.value == "openai"
    assert job.config.ocr_vision_model == "gpt-4o-mini"


def test_service_persists_snapshot_before_first_vision_extraction_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "scan.png"
    image.write_bytes(b"image")
    job = IndexJob(
        id="job-1",
        name="index",
        config=IndexConfig(name="index", ocr_mode=OcrMode.VISION),
        source_type="upload",
    )
    indexer = IndexerService(index_base_path=str(tmp_path / "indexes"))
    pipeline = mock.Mock()
    spool = mock.Mock()
    spool.is_stage_complete.return_value = False

    class Permit:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    async def stage_documents(paths, source_dir, index_name, load, progress):
        await load(next(iter(paths)))

    pipeline.stage_documents = mock.AsyncMock(side_effect=stage_documents)
    monkeypatch.setattr(service, "UPLOAD_TMP_DIR", tmp_path)
    monkeypatch.setattr(service.repository, "update_job", mock.AsyncMock())
    monkeypatch.setattr(
        service.repository,
        "get_settings",
        mock.AsyncMock(
            return_value={
                "default_ocr_mode": "vision",
                "default_ocr_provider": "openai",
                "default_ocr_vision_model": "gpt-4o-mini",
            }
        ),
    )
    monkeypatch.setattr(service, "collect_files_recursive", lambda *args, **kwargs: [(image, image.stat().st_size)])
    monkeypatch.setattr(service.ocr_contract, "ocr_pipeline_fingerprint", lambda: "pipeline-hash", raising=False)
    monkeypatch.setattr(service.resource_governor, "acquire", lambda request: Permit())
    monkeypatch.setattr(service.IndexingSpool, "open_attempt", lambda *args: spool)
    monkeypatch.setattr(service, "BoundedIndexingPipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(
        service,
        "extract_image_structured_async",
        mock.AsyncMock(side_effect=VisionOcrError("provider unavailable")),
    )

    with pytest.raises(VisionOcrError):
        asyncio.run(indexer._create_faiss_index(job, source))

    snapshot = tmp_path / "indexing" / job.id / "ocr-target.json"
    assert snapshot.is_file()
    assert "gpt-4o-mini" in snapshot.read_text(encoding="utf-8")
