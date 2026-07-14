import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core.ssh import SSHResult
from ragtime.indexer.models import SchemaIndexJob, SchemaIndexStatus, TableSchemaInfo
from ragtime.indexer.schema_service import SchemaIndexerService


class _FakeEmbeddings:
    def embed_documents(self, contents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in contents]


class SchemaIndexerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_postgres_connection_uses_remote_docker_for_docker_ssh_config(self) -> None:
        service = SchemaIndexerService()
        remote_calls = []

        def fake_remote_execute(ssh_config, command, input_data=None):
            remote_calls.append((ssh_config, command, input_data))
            return SSHResult(stdout="1\n", stderr="", exit_code=0, success=True)

        with (
            mock.patch(
                "ragtime.indexer.schema_service.execute_docker_command_on_remote_host",
                side_effect=fake_remote_execute,
            ),
            mock.patch(
                "ragtime.indexer.schema_service.subprocess.run",
                side_effect=AssertionError("local docker should not be used for remote Docker configs"),
            ) as local_run,
        ):
            success, error = await service.test_connection(
                "postgres",
                {
                    "host": "stale-direct-host.example.com",
                    "container": "postgres-postgres-1",
                    "docker_ssh_enabled": True,
                    "docker_ssh_host": "remote.example.com",
                    "docker_ssh_user": "deploy",
                    "docker_ssh_password": "secret",
                },
            )

        self.assertTrue(success, error)
        self.assertIsNone(error)
        local_run.assert_not_called()
        self.assertEqual(len(remote_calls), 1)
        ssh_config, command, input_data = remote_calls[0]
        self.assertEqual(ssh_config.host, "remote.example.com")
        self.assertEqual(ssh_config.user, "deploy")
        self.assertIsNone(input_data)
        self.assertEqual(command[:4], ["docker", "exec", "-i", "postgres-postgres-1"])

    async def test_postgres_introspection_uses_remote_docker_for_docker_ssh_config(self) -> None:
        service = SchemaIndexerService()
        remote_calls = []

        def fake_remote_execute(ssh_config, command, input_data=None):
            remote_calls.append((ssh_config, command, input_data))
            inner_command = command[-1]
            if "information_schema.tables" in inner_command and "MATERIALIZED VIEW" not in inner_command:
                return SSHResult(stdout="public\twidgets\tBASE TABLE\t\\N\t1\n", stderr="", exit_code=0, success=True)
            if "MATERIALIZED VIEW" in inner_command:
                return SSHResult(stdout="", stderr="", exit_code=0, success=True)
            return SSHResult(stdout="[]\n", stderr="", exit_code=0, success=True)

        with (
            mock.patch(
                "ragtime.indexer.schema_service.execute_docker_command_on_remote_host",
                side_effect=fake_remote_execute,
            ),
            mock.patch(
                "ragtime.indexer.schema_service.subprocess.run",
                side_effect=AssertionError("local docker should not be used for remote Docker configs"),
            ) as local_run,
        ):
            tables = await service._introspect_postgres(
                {
                    "container": "postgres-postgres-1",
                    "docker_ssh_enabled": True,
                    "docker_ssh_host": "remote.example.com",
                    "docker_ssh_user": "deploy",
                    "docker_ssh_password": "secret",
                }
            )

        self.assertEqual(tables, [])
        local_run.assert_not_called()
        self.assertGreaterEqual(len(remote_calls), 3)
        self.assertTrue(all(call[1][:4] == ["docker", "exec", "-i", "postgres-postgres-1"] for call in remote_calls))

    async def test_connection_error_not_labeled_encryption_mismatch_when_tool_credentials_are_healthy(self) -> None:
        service = SchemaIndexerService()
        job = SchemaIndexJob(id="job-1", tool_config_id="tool-1", index_name="schema_tool_1")

        with (
            mock.patch.object(service, "_update_job", new=mock.AsyncMock()),
            mock.patch.object(
                service,
                "test_connection",
                new=mock.AsyncMock(return_value=(False, "Connection test failed: No such container: postgres-postgres-1")),
            ),
            mock.patch("ragtime.core.encryption.encryption_key_mismatch_detected", return_value=True),
            mock.patch(
                "ragtime.indexer.schema_service.repository.get_tool_config",
                new=mock.AsyncMock(return_value=SimpleNamespace(undecryptable_fields=[])),
            ),
        ):
            await service._process_index(job, "postgres", {}, full_reindex=False)

        self.assertEqual(job.status, SchemaIndexStatus.FAILED)
        self.assertIsNotNone(job.error_message)
        self.assertIn("No such container: postgres-postgres-1", job.error_message or "")
        self.assertNotIn("stored credentials could not be decrypted", job.error_message or "")

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
