import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.models import SchemaIndexJob, SchemaIndexStatus, TableSchemaInfo
from ragtime.indexer.schema_service import SchemaIndexerService


class _FakeEmbeddings:
    def embed_documents(self, contents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in contents]


class SchemaIndexerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_schema_updates_last_indexed_timestamp(self) -> None:
        service = SchemaIndexerService()
        table = TableSchemaInfo(
            table_schema="public",
            table_name="widgets",
            full_name="public.widgets",
            columns=[{"name": "id", "type": "integer", "nullable": False}],
            primary_key=["id"],
        )
        schema_hash = service._compute_schema_hash([table])
        job = SchemaIndexJob(id="job-1", tool_config_id="tool-1", index_name="schema_tool_1")
        settings = SimpleNamespace(
            embedding_dimension=2,
            embedding_config_hash="cfg-1",
            ivfflat_lists=100,
            get_embedding_config_hash=lambda: "cfg-1",
        )

        with (
            mock.patch.object(service, "_update_job", new=mock.AsyncMock()),
            mock.patch.object(service, "test_connection", new=mock.AsyncMock(return_value=(True, None))),
            mock.patch.object(service, "_ensure_pgvector", new=mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.indexer.schema_service.repository.get_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch.object(service, "_get_embeddings", new=mock.AsyncMock(return_value=_FakeEmbeddings())),
            mock.patch.object(service, "_ensure_embedding_column", new=mock.AsyncMock()),
            mock.patch.object(service, "introspect_schema", new=mock.AsyncMock(return_value=[table])),
            mock.patch.object(service, "_update_schema_hash", new=mock.AsyncMock()) as update_schema_hash,
        ):
            await service._process_index(
                job,
                tool_type="postgres",
                connection_config={"schema_hash": schema_hash},
                full_reindex=False,
            )

        self.assertEqual(job.status, SchemaIndexStatus.COMPLETED)
        update_schema_hash.assert_awaited_once_with("tool-1", schema_hash)


if __name__ == "__main__":
    unittest.main()
