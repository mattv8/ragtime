import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core.datetimes import utc_now
from ragtime.indexer.service import IndexerService


class ActiveIndexListingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = IndexerService(index_base_path=self.temp_dir.name)

    def _metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="ragtime",
            path=str(self.temp_dir.name) + "/missing-index",
            documentCount=1255,
            chunkCount=5699,
            vectorStoreType="faiss",
            enabled=True,
            createdAt=utc_now(),
            lastModified=None,
            sizeBytes=0,
            description="",
            sourceType="git",
            source="https://example.com/repo.git",
            gitBranch="main",
            searchWeight=1.0,
            configSnapshot=None,
            gitToken=None,
            displayName=None,
        )

    async def test_list_indexes_retains_active_faiss_metadata_without_files(self) -> None:
        metadata = self._metadata()

        with (
            mock.patch("ragtime.indexer.service.repository.list_index_metadata", new=mock.AsyncMock(return_value=[metadata])),
            mock.patch("ragtime.indexer.service.repository.list_active_index_names", new=mock.AsyncMock(return_value={"ragtime"})),
        ):
            indexes = await self.service.list_indexes()

        self.assertEqual([index.name for index in indexes], ["ragtime"])
        self.assertEqual(indexes[0].document_count, 1255)
        self.assertEqual(indexes[0].chunk_count, 5699)

    async def test_list_indexes_skips_inactive_missing_faiss_metadata(self) -> None:
        metadata = self._metadata()

        with (
            mock.patch("ragtime.indexer.service.repository.list_index_metadata", new=mock.AsyncMock(return_value=[metadata])),
            mock.patch("ragtime.indexer.service.repository.list_active_index_names", new=mock.AsyncMock(return_value=set())),
        ):
            indexes = await self.service.list_indexes()

        self.assertEqual(indexes, [])

    async def test_list_indexes_propagates_active_lookup_failure(self) -> None:
        metadata = self._metadata()

        with (
            mock.patch("ragtime.indexer.service.repository.list_index_metadata", new=mock.AsyncMock(return_value=[metadata])),
            mock.patch(
                "ragtime.indexer.service.repository.list_active_index_names",
                new=mock.AsyncMock(side_effect=RuntimeError("database unavailable")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await self.service.list_indexes()


if __name__ == "__main__":
    unittest.main()
