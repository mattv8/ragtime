import unittest
from types import SimpleNamespace
from unittest import mock

from prisma.enums import IndexStatus as PrismaIndexStatus

from ragtime.indexer.repository import IndexerRepository


class ActiveIndexNamesTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_active_index_names_queries_distinct_pending_and_processing_rows(self) -> None:
        repo = IndexerRepository()
        fake_db = SimpleNamespace(
            indexjob=SimpleNamespace(
                find_many=mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(name="ragtime"),
                        SimpleNamespace(name="docs"),
                    ]
                )
            )
        )

        with mock.patch.object(repo, "_get_db", new=mock.AsyncMock(return_value=fake_db)):
            names = await repo.list_active_index_names()

        self.assertEqual(names, {"ragtime", "docs"})
        self.assertEqual(
            fake_db.indexjob.find_many.await_args.kwargs,
            {
                "where": {
                    "OR": [
                        {"status": PrismaIndexStatus.pending},
                        {"status": PrismaIndexStatus.processing},
                    ]
                },
                "distinct": ["name"],
            },
        )
