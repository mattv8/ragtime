import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from ragtime.indexer.models import IndexConfig, IndexJob, IndexJobPhase, IndexStatus
from ragtime.indexer.service import IndexerService


class IndexJobPhaseTransitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = IndexerService(index_base_path=self.temp_dir.name)
        self.config = IndexConfig(name="ragtime")

    def _job(self, **overrides: Any) -> IndexJob:
        job = IndexJob(
            id="job-1",
            name=self.config.name,
            config=self.config,
            source_type="git",
            git_url="https://example.com/repo.git",
            git_branch="main",
        )
        return job.model_copy(update=overrides)

    @staticmethod
    def _consume_background_task(coro, task):
        coro.close()
        return task

    @staticmethod
    def _collapse_adjacent(phases: list[IndexJobPhase]) -> list[IndexJobPhase]:
        return [phase for index, phase in enumerate(phases) if index == 0 or phases[index - 1] != phase]

    async def test_resume_resets_phase_to_preparing(self) -> None:
        job = self._job(status=IndexStatus.PROCESSING, phase=IndexJobPhase.EMBEDDING)
        fake_task = mock.Mock(name="git-processing-task")

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock()) as update_job,
            mock.patch.object(self.service, "_process_git", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.indexer.service.asyncio.create_task",
                side_effect=lambda coro: self._consume_background_task(coro, fake_task),
            ),
        ):
            await self.service._resume_job(job)

        self.assertEqual(job.phase, IndexJobPhase.PREPARING)
        update_job.assert_awaited_once_with(job)

    async def test_cancel_sets_cancelled_phase(self) -> None:
        job = self._job()
        self.service._active_jobs[job.id] = job

        with mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock()) as update_job:
            self.assertTrue(await self.service.cancel_job(job.id))

        self.assertEqual(job.status, IndexStatus.FAILED)
        self.assertEqual(job.phase, IndexJobPhase.CANCELLED)
        update_job.assert_awaited_once_with(job)

    async def test_failed_git_processing_sets_failed_phase(self) -> None:
        job = self._job()

        async def update_job(current_job):
            return current_job

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=update_job),
            mock.patch.object(self.service, "_clone_git_repo", new=mock.AsyncMock()),
            mock.patch.object(self.service, "_create_faiss_index", new=mock.AsyncMock(side_effect=RuntimeError("boom"))),
            mock.patch.object(self.service, "_cleanup_failed_index_metadata", new=mock.AsyncMock()),
            mock.patch.object(self.service, "_maybe_reinitialize_rag", new=mock.AsyncMock()),
            mock.patch("ragtime.indexer.service.pool_manager"),
        ):
            await self.service._process_git(job)

        self.assertEqual(job.phase, IndexJobPhase.FAILED)

    async def test_completed_git_processing_sets_completed_phase(self) -> None:
        job = self._job()

        async def update_job(current_job):
            return current_job

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=update_job),
            mock.patch.object(self.service, "_clone_git_repo", new=mock.AsyncMock()),
            mock.patch.object(self.service, "_create_faiss_index", new=mock.AsyncMock()),
            mock.patch.object(self.service, "_maybe_reinitialize_rag", new=mock.AsyncMock()),
            mock.patch("ragtime.indexer.service.pool_manager"),
        ):
            await self.service._process_git(job)

        self.assertEqual(job.phase, IndexJobPhase.COMPLETED)

    async def test_create_faiss_index_persists_scanning_loading_chunking_then_embedding(self) -> None:
        job = self._job(source_type="upload", source_path="archive.zip")
        source_dir = Path(self.temp_dir.name) / "source"
        source_dir.mkdir()
        source_file = source_dir / "notes.txt"
        source_file.write_text("hello world", encoding="utf-8")
        phases: list[IndexJobPhase] = []

        async def update_job(current_job):
            phases.append(current_job.phase)
            return current_job

        app_settings = SimpleNamespace(
            chunking_use_tokens=False,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            ollama_base_url=None,
        )

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock(side_effect=update_job)),
            mock.patch("ragtime.indexer.service.repository.get_settings", new=mock.AsyncMock(return_value=app_settings)),
            mock.patch("ragtime.indexer.service.collect_files_recursive", return_value=[(source_file, source_file.stat().st_size)]),
            mock.patch("ragtime.indexer.service.BoundedIndexingPipeline.stage_documents", new=mock.AsyncMock(return_value=1)),
            mock.patch("ragtime.indexer.service.BoundedIndexingPipeline.stage_chunks", new=mock.AsyncMock(return_value=1)),
            mock.patch.object(self.service, "_get_embeddings", new=mock.AsyncMock(side_effect=RuntimeError("embedding sentinel"))),
            mock.patch("ragtime.indexer.service.IndexingSpool.summary", return_value={"document_count": 1, "chunk_count": 1, "embedded_count": 0}),
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding sentinel"):
                await self.service._create_faiss_index(job, source_dir)

        self.assertEqual(
            self._collapse_adjacent(phases),
            [
                IndexJobPhase.SCANNING,
                IndexJobPhase.LOADING,
                IndexJobPhase.CHUNKING,
                IndexJobPhase.EMBEDDING,
            ],
        )

    async def test_clone_validation_persists_cloning_phase_before_git_url_error(self) -> None:
        job = self._job(git_url=None)
        phases: list[IndexJobPhase] = []

        async def update_job(current_job):
            phases.append(current_job.phase)
            return current_job

        with mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock(side_effect=update_job)):
            with self.assertRaisesRegex(ValueError, "Git URL is required"):
                await self.service._clone_git_repo(job, Path(self.temp_dir.name) / "repo")

        self.assertEqual(phases, [IndexJobPhase.CLONING])

    async def test_create_faiss_index_persists_finalizing_before_faiss_build(self) -> None:
        job = self._job(source_type="upload", source_path="archive.zip")
        source_dir = Path(self.temp_dir.name) / "source-finalizing"
        source_dir.mkdir()
        source_file = source_dir / "notes.txt"
        source_file.write_text("hello world", encoding="utf-8")
        phases: list[IndexJobPhase] = []

        async def update_job(current_job):
            phases.append(current_job.phase)
            return current_job

        app_settings = SimpleNamespace(
            chunking_use_tokens=False,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            ollama_base_url=None,
        )
        fake_embeddings = SimpleNamespace()
        artifact = SimpleNamespace(generation_path=source_dir, document_count=1, chunk_count=1, size_bytes=1)

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock(side_effect=update_job)),
            mock.patch("ragtime.indexer.service.repository.get_settings", new=mock.AsyncMock(return_value=app_settings)),
            mock.patch("ragtime.indexer.service.collect_files_recursive", return_value=[(source_file, source_file.stat().st_size)]),
            mock.patch("ragtime.indexer.service.BoundedIndexingPipeline.stage_documents", new=mock.AsyncMock(return_value=1)),
            mock.patch("ragtime.indexer.service.BoundedIndexingPipeline.stage_chunks", new=mock.AsyncMock(return_value=1)),
            mock.patch("ragtime.indexer.service.BoundedIndexingPipeline.stage_embeddings", new=mock.AsyncMock(return_value=1)),
            mock.patch("ragtime.indexer.service.IndexingSpool.summary", return_value={"document_count": 1, "chunk_count": 1, "embedded_count": 1}),
            mock.patch.object(self.service, "_get_embeddings", new=mock.AsyncMock(return_value=fake_embeddings)),
            mock.patch("ragtime.indexer.service.get_embedding_model_context_limit", new=mock.AsyncMock(return_value=8192)),
            mock.patch("ragtime.indexer.service.get_embedding_safety_margin", return_value=0.8),
            mock.patch("ragtime.indexer.service.prepare_faiss_artifact", new=mock.AsyncMock(side_effect=RuntimeError("finalizing sentinel"))),
        ):
            with self.assertRaisesRegex(RuntimeError, "finalizing sentinel"):
                await self.service._create_faiss_index(job, source_dir)

        self.assertEqual(self._collapse_adjacent(phases)[-1], IndexJobPhase.FINALIZING)


if __name__ == "__main__":
    unittest.main()
