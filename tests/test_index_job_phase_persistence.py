import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from prisma.enums import IndexJobPhase as PrismaIndexJobPhase
from prisma.enums import IndexStatus as PrismaIndexStatus
from prisma.models import IndexConfig as PrismaIndexConfig
from prisma.models import IndexJob as PrismaIndexJob

from ragtime.indexer.models import IndexConfig, IndexJob, IndexJobPhase
from ragtime.indexer.repository import IndexerRepository


class IndexJobPhasePersistenceTests(unittest.IsolatedAsyncioTestCase):
    def _job(self, phase: IndexJobPhase) -> IndexJob:
        return IndexJob(
            id="job-1",
            name="ragtime",
            config=IndexConfig(name="ragtime"),
            source_type="git",
            phase=phase,
        )

    def _prisma_job(self, phase: PrismaIndexJobPhase) -> PrismaIndexJob:
        now = datetime.now(timezone.utc)
        return PrismaIndexJob(
            id="job-1",
            name="ragtime",
            status=PrismaIndexStatus.pending,
            phase=phase,
            sourceType="git",
            sourcePath=None,
            gitUrl="https://example.com/repo.git",
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
            configId="config-1",
            config=PrismaIndexConfig(
                id="config-1",
                name="ragtime",
                description="",
                filePatterns=["**/*"],
                excludePatterns=[],
                chunkSize=1000,
                chunkOverlap=200,
                embeddingModel="text-embedding-3-small",
            ),
        )

    async def test_create_job_persists_phase_enum(self) -> None:
        repo = IndexerRepository()
        job = self._job(IndexJobPhase.CHUNKING)
        fake_db = SimpleNamespace(indexjob=SimpleNamespace(create=mock.AsyncMock()))

        with mock.patch.object(repo, "_get_db", new=mock.AsyncMock(return_value=fake_db)):
            await repo.create_job(job)

        create_data = fake_db.indexjob.create.await_args.kwargs["data"]
        self.assertEqual(create_data["phase"].value, "chunking")

    async def test_update_job_persists_phase_enum(self) -> None:
        repo = IndexerRepository()
        job = self._job(IndexJobPhase.EMBEDDING)
        fake_db = SimpleNamespace(indexjob=SimpleNamespace(update=mock.AsyncMock()))

        with mock.patch.object(repo, "_get_db", new=mock.AsyncMock(return_value=fake_db)):
            await repo.update_job(job)

        update_data = fake_db.indexjob.update.await_args.kwargs["data"]
        self.assertEqual(update_data["phase"].value, "embedding")

    def test_prisma_job_round_trips_every_phase(self) -> None:
        repo = IndexerRepository()
        for phase in PrismaIndexJobPhase:
            model = repo._prisma_job_to_model(self._prisma_job(phase))
            self.assertEqual(model.phase, IndexJobPhase(phase.value))
