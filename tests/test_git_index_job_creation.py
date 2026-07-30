import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.models import IndexConfig, IndexJob, OcrMode
from ragtime.indexer.repository import repository
from ragtime.indexer.service import IndexerService

URL = "https://example.com/repo.git"


class GitIndexJobCreationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = IndexerService(index_base_path=self.temp_dir.name)
        self.config = IndexConfig(name="git-index")

    def _job(self, *, id: str) -> IndexJob:
        return IndexJob(
            id=id,
            name=self.config.name,
            config=self.config,
            source_type="git",
            git_url=URL,
            git_branch="main",
        )

    @staticmethod
    def _consume_background_task(coro, task):
        coro.close()
        return task

    async def test_create_git_index_returns_existing_active_job(self) -> None:
        existing = self._job(id="existing")

        fake_task = mock.Mock(name="git-processing-task")

        with (
            mock.patch.object(repository, "get_active_job_for_index", new=mock.AsyncMock(return_value=existing)),
            mock.patch.object(repository, "create_job", new=mock.AsyncMock()) as create_job,
            mock.patch.object(self.service, "_create_optimistic_index_metadata", new=mock.AsyncMock()) as metadata,
            mock.patch(
                "ragtime.indexer.service.asyncio.create_task",
                side_effect=lambda coro: self._consume_background_task(coro, fake_task),
            ) as create_task,
        ):
            result = await self.service.create_index_from_git(URL, "main", existing.config)

        self.assertIs(result, existing)
        create_job.assert_not_awaited()
        metadata.assert_not_awaited()
        create_task.assert_not_called()

    async def test_try_create_git_index_returns_none_when_index_is_active(self) -> None:
        existing = self._job(id="existing")

        with (
            mock.patch.object(repository, "get_active_job_for_index", new=mock.AsyncMock(return_value=existing)),
            mock.patch.object(repository, "create_job", new=mock.AsyncMock()) as create_job,
        ):
            result = await self.service.try_create_index_from_git(URL, "main", self.config)

        self.assertIsNone(result)
        create_job.assert_not_awaited()

    async def test_create_git_index_still_reuses_active_job(self) -> None:
        existing = self._job(id="existing")

        with mock.patch.object(repository, "get_active_job_for_index", new=mock.AsyncMock(return_value=existing)):
            result = await self.service.create_index_from_git(URL, "main", self.config)

        self.assertIs(result, existing)

    async def test_create_git_index_concurrent_calls_share_one_active_job(self) -> None:
        first_lookup_started = asyncio.Event()
        release_first_lookup = asyncio.Event()
        created_job: IndexJob | None = None
        lookup_count = 0

        async def get_active_job_for_index(name: str) -> IndexJob | None:
            nonlocal lookup_count, created_job
            self.assertEqual(name, self.config.name)
            lookup_count += 1
            if lookup_count == 1:
                first_lookup_started.set()
                await release_first_lookup.wait()
                return None
            return created_job

        async def create_job(job: IndexJob) -> None:
            nonlocal created_job
            created_job = job

        fake_task = mock.Mock(name="git-processing-task")

        with (
            mock.patch.object(repository, "get_active_job_for_index", new=mock.AsyncMock(side_effect=get_active_job_for_index)),
            mock.patch.object(repository, "create_job", new=mock.AsyncMock(side_effect=create_job)) as create_job_mock,
            mock.patch.object(self.service, "_create_optimistic_index_metadata", new=mock.AsyncMock()) as metadata,
            mock.patch(
                "ragtime.indexer.service.asyncio.create_task",
                side_effect=lambda coro: self._consume_background_task(coro, fake_task),
            ) as create_task,
        ):
            loop = asyncio.get_running_loop()
            first_call = loop.create_task(self.service.create_index_from_git(URL, "main", self.config))
            await asyncio.wait_for(first_lookup_started.wait(), timeout=1)

            second_call = loop.create_task(self.service.create_index_from_git(URL, "main", self.config))
            await asyncio.sleep(0)

            release_first_lookup.set()
            first_result, second_result = await asyncio.gather(first_call, second_call)

        self.assertIsNotNone(created_job)
        self.assertIs(first_result, created_job)
        self.assertIs(second_result, created_job)
        self.assertEqual(lookup_count, 2)
        create_job_mock.assert_awaited_once()
        metadata.assert_awaited_once()
        create_task.assert_called_once()

    async def test_create_git_index_replaces_analyze_only_metadata_snapshot(self) -> None:
        final_config = IndexConfig(
            name="git-index",
            ocr_mode=OcrMode.VISION,
            git_history_depth=0,
        )
        placeholder_metadata = SimpleNamespace(
            description="",
            configSnapshot={
                "ocr_mode": "disabled",
                "git_history_depth": 1,
                "_analyze_only_git_token": True,
            },
            documentCount=0,
            chunkCount=0,
            sizeBytes=0,
        )

        fake_task = mock.Mock(name="git-processing-task")

        with (
            mock.patch.object(repository, "get_active_job_for_index", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(repository, "get_index_metadata", new=mock.AsyncMock(return_value=placeholder_metadata)),
            mock.patch.object(repository, "create_job", new=mock.AsyncMock()),
            mock.patch.object(repository, "upsert_index_metadata", new=mock.AsyncMock()) as upsert_metadata,
            mock.patch(
                "ragtime.indexer.service.asyncio.create_task",
                side_effect=lambda coro: self._consume_background_task(coro, fake_task),
            ),
        ):
            await self.service.create_index_from_git(URL, "main", final_config)

        upsert_metadata.assert_awaited_once()
        upsert_call = upsert_metadata.await_args
        assert upsert_call is not None
        config_snapshot = upsert_call.kwargs["config_snapshot"]
        self.assertEqual(config_snapshot["ocr_mode"], "vision")
        self.assertEqual(config_snapshot["git_history_depth"], 0)
        self.assertNotIn("_analyze_only_git_token", config_snapshot)

    async def test_create_optimistic_index_metadata_preserves_real_existing_snapshot(self) -> None:
        existing_metadata = SimpleNamespace(
            description="Keep me",
            configSnapshot={
                "ocr_mode": "disabled",
                "git_history_depth": 1,
                "reindex_interval_hours": 24,
            },
            documentCount=12,
            chunkCount=34,
            sizeBytes=56,
        )

        with (
            mock.patch.object(repository, "get_index_metadata", new=mock.AsyncMock(return_value=existing_metadata)),
            mock.patch.object(repository, "upsert_index_metadata", new=mock.AsyncMock()) as upsert_metadata,
        ):
            await self.service._create_optimistic_index_metadata(
                config=IndexConfig(
                    name="git-index",
                    ocr_mode=OcrMode.VISION,
                    git_history_depth=0,
                ),
                source_type="git",
                source=URL,
                git_branch="main",
            )

        upsert_metadata.assert_awaited_once()
        upsert_call = upsert_metadata.await_args
        assert upsert_call is not None
        self.assertEqual(
            upsert_call.kwargs["config_snapshot"],
            existing_metadata.configSnapshot,
        )


if __name__ == "__main__":
    unittest.main()
