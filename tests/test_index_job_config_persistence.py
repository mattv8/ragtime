from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
from prisma.models import IndexJob as PrismaIndexJob

from ragtime.indexer.models import IndexConfig, IndexJob, OcrMode, OcrProvider, VectorStoreType
from ragtime.indexer.repository import IndexerRepository


def complete_config() -> IndexConfig:
    return IndexConfig(
        name="engineering-docs",
        description="Internal engineering documentation",
        file_patterns=["docs/**/*.md", "src/**/*.py"],
        exclude_patterns=["**/.git/**", "**/generated/**"],
        chunk_size=1200,
        chunk_overlap=150,
        max_file_size_kb=4096,
        embedding_model="text-embedding-3-large",
        vector_store_type=VectorStoreType.PGVECTOR,
        ocr_mode=OcrMode.VISION,
        ocr_provider=OcrProvider.OLLAMA,
        ocr_vision_model="qwen3-vl:32b",
        git_clone_timeout_minutes=25,
        git_history_depth=50,
        reindex_interval_hours=24,
        reindex_start_minute=180,
        reindex_timezone="America/Chicago",
    )


def persisted_job(config: object) -> PrismaIndexJob:
    now = datetime.now(timezone.utc)
    return cast(
        PrismaIndexJob,
        SimpleNamespace(
            id="job-1",
            name="engineering-docs",
            status="pending",
            phase="preparing",
            sourceType="git",
            sourcePath=None,
            gitUrl="https://example.com/engineering.git",
            gitBranch="main",
            gitToken=None,
            totalFiles=0,
            processedFiles=0,
            totalChunks=0,
            processedChunks=0,
            errorMessage=None,
            createdAt=now,
            startedAt=None,
            completedAt=None,
            config=config,
        ),
    )


def test_job_config_round_trip_preserves_every_config_field() -> None:
    repository = IndexerRepository()
    config = complete_config()
    job = IndexJob(id="job-1", name=config.name, config=config, source_type="git")
    index_job = SimpleNamespace(create=mock.AsyncMock())
    database = SimpleNamespace(indexjob=index_job)

    with mock.patch.object(repository, "_get_db", new=mock.AsyncMock(return_value=database)):
        asyncio.run(repository.create_job(job))

    stored_config = index_job.create.await_args.kwargs["data"]["config"]["create"]
    restored = repository._prisma_job_to_model(
        persisted_job(
            SimpleNamespace(
                name=config.name,
                description=config.description,
                filePatterns=config.file_patterns,
                excludePatterns=config.exclude_patterns,
                chunkSize=config.chunk_size,
                chunkOverlap=config.chunk_overlap,
                embeddingModel=config.embedding_model,
                configSnapshot=stored_config["configSnapshot"].data,
            )
        )
    )

    assert restored.config == config


@pytest.mark.parametrize("include_snapshot", [False, True])
def test_legacy_config_snapshot_fallback_preserves_scalar_description(include_snapshot: bool) -> None:
    repository = IndexerRepository()
    legacy_config = SimpleNamespace(
        name="legacy-index",
        description="Legacy description",
        filePatterns=["**/*.txt"],
        excludePatterns=["**/tmp/**"],
        chunkSize=700,
        chunkOverlap=100,
        embeddingModel="legacy-embedding",
    )
    if include_snapshot:
        legacy_config.configSnapshot = None

    restored = repository._prisma_job_to_model(persisted_job(legacy_config))

    assert restored.config == IndexConfig(
        name="legacy-index",
        description="Legacy description",
        file_patterns=["**/*.txt"],
        exclude_patterns=["**/tmp/**"],
        chunk_size=700,
        chunk_overlap=100,
        embedding_model="legacy-embedding",
    )


def test_malformed_persisted_config_snapshot_is_rejected() -> None:
    repository = IndexerRepository()
    malformed_config = SimpleNamespace(
        name="index",
        description="description",
        filePatterns=["**/*"],
        excludePatterns=[],
        chunkSize=1000,
        chunkOverlap=200,
        embeddingModel="text-embedding-3-small",
        configSnapshot={"name": "index", "max_file_size_kb": "not-a-number"},
    )

    with pytest.raises(ValueError):
        repository._prisma_job_to_model(persisted_job(malformed_config))
