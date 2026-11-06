from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pytest
from prisma.models import IndexConfig as PrismaIndexConfig
from prisma.models import IndexJob as PrismaIndexJob
from pydantic import TypeAdapter, ValidationError

from ragtime.indexer.filesystem_service import FilesystemIndexerService
from ragtime.indexer.job_warnings import IndexingWarning, IndexingWarnings, warnings_from_persisted
from ragtime.indexer.models import FilesystemIndexJob, IndexConfig, IndexJob
from ragtime.indexer.repository import IndexerRepository


def test_warning_summary_records_distinct_sources_and_bounds_samples() -> None:
    warnings = IndexingWarnings()

    warnings.record("invoices/page-1.png", "timeout")
    warnings.record("invoices/page-1.png", "incomplete_response")
    for number in range(100):
        warnings.record(f"invoices/page-{number + 2}.png", "invalid_output")

    assert warnings.count == 101
    assert len(warnings.items) == 100
    assert warnings.items[0] == IndexingWarning(source="invoices/page-1.png", category="incomplete_response")


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "/absolute/page.png", "category": "timeout"},
        {"source": "../outside/page.png", "category": "timeout"},
        {"source": "https://private.example/page.png", "category": "timeout"},
        {"source": "page.png", "category": "unsafe details"},
        {"source": "x" * 513, "category": "timeout"},
    ],
)
def test_warning_items_reject_unsafe_source_or_category(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        IndexingWarning.model_validate(payload)


def test_warning_hydration_rejects_malformed_persisted_json() -> None:
    with pytest.raises(ValidationError):
        warnings_from_persisted({"count": "not-a-number"})


def test_document_job_persists_and_hydrates_warnings() -> None:
    repository = IndexerRepository()
    job = IndexJob(id="job-1", name="index", config=IndexConfig(name="index"), source_type="upload")
    job.warnings.record("scans/page-1.png", "timeout")
    index_job = SimpleNamespace(create=mock.AsyncMock(), update=mock.AsyncMock())
    database = SimpleNamespace(indexjob=index_job)

    with mock.patch.object(repository, "_get_db", new=mock.AsyncMock(return_value=database)):
        asyncio.run(repository.create_job(job))
        asyncio.run(repository.update_job(job))

    assert index_job.create.await_args.kwargs["data"]["warnings"].data == job.warnings.model_dump(mode="json")
    assert index_job.update.await_args.kwargs["data"]["warnings"].data == job.warnings.model_dump(mode="json")

    persisted = TypeAdapter(PrismaIndexJob).validate_python(
        dict(
            id="job-1",
            name="index",
            status="pending",
            phase="preparing",
            sourceType="upload",
            sourcePath=None,
            gitUrl=None,
            gitBranch="main",
            gitToken=None,
            totalFiles=0,
            processedFiles=0,
            totalChunks=0,
            processedChunks=0,
            errorMessage=None,
            warnings=job.warnings.model_dump_json(),
            createdAt=datetime.now(timezone.utc),
            startedAt=None,
            completedAt=None,
            configId="config-1",
            config=PrismaIndexConfig(
                id="config-1",
                name="index",
                description="",
                filePatterns=["**/*"],
                excludePatterns=[],
                chunkSize=1000,
                chunkOverlap=200,
                embeddingModel="text-embedding-3-small",
                configSnapshot=None,
            ),
        )
    )

    assert repository._prisma_job_to_model(persisted).warnings == job.warnings
    legacy_persisted = PrismaIndexJob(**{**persisted.model_dump(), "warnings": None})
    assert repository._prisma_job_to_model(legacy_persisted).warnings == IndexingWarnings()


def test_filesystem_job_persists_and_hydrates_warnings() -> None:
    service = FilesystemIndexerService()
    job = FilesystemIndexJob(id="job-1", tool_config_id="tool-1", index_name="index")
    job.warnings.record("scans/page-1.png", "output_exhausted")
    filesystem_index_job = SimpleNamespace(create=mock.AsyncMock(), update=mock.AsyncMock())
    database = SimpleNamespace(filesystemindexjob=filesystem_index_job)

    with mock.patch("ragtime.indexer.filesystem_service.get_db", new=mock.AsyncMock(return_value=database)):
        asyncio.run(service._create_job(job))
        asyncio.run(service._update_job(job))

    assert filesystem_index_job.create.await_args.kwargs["data"]["warnings"].data == job.warnings.model_dump(mode="json")
    assert filesystem_index_job.update.await_args.kwargs["data"]["warnings"].data == job.warnings.model_dump(mode="json")
