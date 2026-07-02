import asyncio
import tempfile
import unittest
from unittest import mock

from ragtime.indexer.models import IndexConfig, IndexJob, IndexStatus
from ragtime.indexer.service import IndexerService


class IndexerServiceShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_marks_active_jobs_failed_and_awaits_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = IndexerService(index_base_path=temp_dir)
            job = IndexJob(
                id="job12345",
                name="shutdown-index",
                config=IndexConfig(name="shutdown-index"),
                source_type="git",
            )

            cancelled = asyncio.Event()

            async def wait_until_cancelled() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            task = asyncio.create_task(wait_until_cancelled())
            await asyncio.sleep(0)
            service._active_jobs[job.id] = job
            service._processing_tasks[job.id] = task

            with mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock()) as update_job:
                await service.shutdown()

            self.assertTrue(cancelled.is_set())
            self.assertTrue(task.done())
            self.assertEqual(job.status, IndexStatus.FAILED)
            self.assertIn("shutdown", job.error_message or "")
            self.assertIsNotNone(job.completed_at)
            self.assertEqual(service._active_jobs, {})
            self.assertEqual(service._processing_tasks, {})
            self.assertEqual(service._cancellation_flags, {})
            update_job.assert_awaited_once_with(job)


if __name__ == "__main__":
    unittest.main()
