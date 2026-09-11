import tempfile
import unittest
from unittest import mock

from ragtime.indexer import routes
from ragtime.indexer.service import IndexerService


class IndexDeleteCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_index_removes_orphaned_metadata_without_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = IndexerService(index_base_path=temp_dir)

            with mock.patch(
                "ragtime.indexer.service.repository.delete_index_metadata",
                new=mock.AsyncMock(return_value=True),
            ) as delete_metadata:
                result = await service.delete_index("orphan")

        self.assertTrue(result)
        delete_metadata.assert_awaited_once_with("orphan")

    async def test_delete_index_unloads_orphan_and_refreshes_runtime_state(self) -> None:
        with (
            mock.patch.object(routes.indexer, "delete_index", new=mock.AsyncMock(return_value=True)) as delete_index,
            mock.patch.object(routes.rag, "unload_index") as unload_index,
            mock.patch.object(routes.rag, "rebuild_agent", new=mock.AsyncMock()) as rebuild_agent,
            mock.patch.object(routes, "invalidate_settings_cache") as invalidate_cache,
            mock.patch.object(routes, "notify_tools_changed") as notify_tools_changed,
        ):
            result = await routes.delete_index("orphan")

        self.assertEqual(result, {"message": "Index 'orphan' deleted successfully"})
        delete_index.assert_awaited_once_with("orphan")
        unload_index.assert_called_once_with("orphan")
        rebuild_agent.assert_awaited_once()
        invalidate_cache.assert_called_once()
        notify_tools_changed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
