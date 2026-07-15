import unittest
from io import BytesIO
from unittest import mock

from fastapi import UploadFile

import ragtime.indexer.routes as _ROUTES
from ragtime.indexer.models import CreateIndexRequest, IndexConfig, IndexJob, IndexJobPhase, IndexStatus


class GitIndexRouteResponseTests(unittest.IsolatedAsyncioTestCase):
    def _job(
        self,
        *,
        id: str = "job-123",
        name: str = "ragtime",
        config: IndexConfig | None = None,
        source_type: str = "git",
        phase: IndexJobPhase = IndexJobPhase.CHUNKING,
        total_files: int = 528,
        processed_files: int = 528,
        total_chunks: int = 1573,
        processed_chunks: int = 30,
        git_url: str | None = None,
        status: IndexStatus = IndexStatus.PENDING,
        clone_progress: float | None = None,
    ) -> IndexJob:
        return IndexJob(
            id=id,
            name=name,
            status=status,
            config=config or IndexConfig(name="ragtime"),
            source_type=source_type,
            git_url=git_url,
            total_files=total_files,
            processed_files=processed_files,
            total_chunks=total_chunks,
            processed_chunks=processed_chunks,
            phase=phase,
            clone_progress=clone_progress,
        )

    async def test_index_from_git_uses_stored_phase_and_processed_chunks(self) -> None:
        request = CreateIndexRequest(
            name="ragtime",
            git_url="https://example.com/repo.git",
            config=IndexConfig(name="ragtime"),
        )
        job = self._job(git_url=request.git_url, config=request.config)

        with mock.patch.object(_ROUTES.indexer, "create_index_from_git", new=mock.AsyncMock(return_value=job)):
            response = await _ROUTES.index_from_git(request, _user=mock.sentinel.user, _=None)

        self.assertEqual(response.phase, IndexJobPhase.CHUNKING)
        self.assertEqual(response.processed_chunks, 30)

    async def test_upload_response_copies_phase_clone_progress_and_processed_chunks(self) -> None:
        job = self._job(source_type="upload", phase=IndexJobPhase.LOADING, clone_progress=0.25)

        with mock.patch.object(_ROUTES.indexer, "create_index_from_upload", new=mock.AsyncMock(return_value=job)):
            response = await _ROUTES.upload_and_index(
                file=UploadFile(filename="repo.zip", file=BytesIO(b"archive-bytes")),
                name="ragtime",
                description="",
                file_patterns="",
                exclude_patterns="",
                chunk_size=1000,
                chunk_overlap=200,
                ocr_mode="disabled",
                ocr_provider=None,
                ocr_vision_model=None,
                vector_store_type="faiss",
                _user=mock.sentinel.user,
                _=None,
            )

        self.assertEqual(response.phase, IndexJobPhase.LOADING)
        self.assertEqual(response.clone_progress, 0.25)
        self.assertEqual(response.processed_chunks, 30)

    async def test_reindex_response_copies_phase_clone_progress_and_processed_chunks(self) -> None:
        job = self._job(phase=IndexJobPhase.EMBEDDING, clone_progress=0.5)
        metadata = mock.Mock(
            sourceType="git",
            source="https://example.com/repo.git",
            description="",
            configSnapshot={},
            gitBranch="main",
            gitToken=None,
        )

        with (
            mock.patch.object(_ROUTES.repository, "get_index_metadata", new=mock.AsyncMock(return_value=metadata)),
            mock.patch.object(_ROUTES.indexer, "create_index_from_git", new=mock.AsyncMock(return_value=job)),
        ):
            response = await _ROUTES.reindex_from_git("ragtime", _ROUTES.ReindexGitRequest(), _user=mock.sentinel.user, _=None)

        self.assertEqual(response.phase, IndexJobPhase.EMBEDDING)
        self.assertEqual(response.clone_progress, 0.5)
        self.assertEqual(response.processed_chunks, 30)

    async def test_retry_responses_copy_phase_from_new_job(self) -> None:
        failed_git_job = self._job(
            id="failed-git",
            git_url="https://example.com/repo.git",
            status=IndexStatus.FAILED,
        )
        failed_upload_job = self._job(
            id="failed-upload",
            source_type="upload",
            status=IndexStatus.FAILED,
        )
        new_git_job = self._job(id="new-git", phase=IndexJobPhase.CLONING, clone_progress=0.1)
        new_upload_job = self._job(id="new-upload", source_type="upload", phase=IndexJobPhase.FINALIZING)

        with (
            mock.patch.object(
                _ROUTES.repository,
                "get_job",
                new=mock.AsyncMock(side_effect=[failed_git_job, failed_upload_job]),
            ),
            mock.patch.object(_ROUTES.indexer, "create_index_from_git", new=mock.AsyncMock(return_value=new_git_job)),
            mock.patch.object(_ROUTES.indexer, "retry_upload_job", new=mock.AsyncMock(return_value=new_upload_job)),
        ):
            git_response = await _ROUTES.retry_failed_job(
                "failed-git",
                _ROUTES.RetryJobRequest(),
                _user=mock.sentinel.user,
                _=None,
            )
            upload_response = await _ROUTES.retry_failed_job(
                "failed-upload",
                _ROUTES.RetryJobRequest(),
                _user=mock.sentinel.user,
                _=None,
            )

        self.assertEqual(git_response.phase, IndexJobPhase.CLONING)
        self.assertEqual(git_response.clone_progress, 0.1)
        self.assertEqual(git_response.processed_chunks, 30)
        self.assertEqual(upload_response.phase, IndexJobPhase.FINALIZING)
        self.assertEqual(upload_response.processed_chunks, 30)
