import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer import routes
from ragtime.indexer.pdm_service import PdmIndexerService

COMPLETED_AT = datetime(2026, 11, 20, 20, 46, 33, tzinfo=timezone.utc)


def fake_db(job):
    find_first = mock.AsyncMock(return_value=job)
    return SimpleNamespace(pdmindexjob=SimpleNamespace(find_first=find_first)), find_first


class LastCompletedAtTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_the_newest_completed_job(self):
        db, find_first = fake_db(SimpleNamespace(completedAt=COMPLETED_AT))
        with mock.patch("ragtime.indexer.pdm_service.get_db", mock.AsyncMock(return_value=db)):
            result = await PdmIndexerService().get_last_completed_at("pdm-1")

        self.assertEqual(result, COMPLETED_AT)
        find_first.assert_awaited_once_with(
            where={"toolConfigId": "pdm-1", "status": "completed", "completedAt": {"not": None}},
            order={"completedAt": "desc"},
        )

    async def test_none_without_a_completed_job(self):
        db, _ = fake_db(None)
        with mock.patch("ragtime.indexer.pdm_service.get_db", mock.AsyncMock(return_value=db)):
            self.assertIsNone(await PdmIndexerService().get_last_completed_at("pdm-1"))


class PdmIndexStatsRouteTests(unittest.IsolatedAsyncioTestCase):
    async def stats(self, last_completed_at):
        indexer = mock.Mock(
            get_document_count=mock.AsyncMock(return_value=19932),
            get_embedding_count=mock.AsyncMock(return_value=19930),
            get_last_completed_at=mock.AsyncMock(return_value=last_completed_at),
        )
        with (
            mock.patch.object(routes.repository, "get_tool_config", mock.AsyncMock(return_value=SimpleNamespace(name="PDM Vault Dev"))),
            mock.patch.object(routes, "pdm_indexer", indexer),
        ):
            return await routes.get_pdm_index_stats("pdm-1", _user=mock.Mock())

    async def test_last_indexed_comes_from_the_last_completed_job(self):
        # A later running, failed or cancelled job must not hide the index the AI still uses.
        stats = await self.stats(COMPLETED_AT)

        self.assertEqual(stats.document_count, 19932)
        self.assertEqual(stats.embedding_count, 19930)
        self.assertEqual(stats.last_indexed_at, COMPLETED_AT.isoformat())

    async def test_never_indexed(self):
        self.assertIsNone((await self.stats(None)).last_indexed_at)


if __name__ == "__main__":
    unittest.main()
