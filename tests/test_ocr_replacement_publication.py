"""Replacement publication must keep serving filesystem indexes intact until promotion."""

import asyncio
import re
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.embeddings import Embeddings

from ragtime.indexer.faiss_artifacts import _build_faiss_generation
from ragtime.indexer.vector_backends import (
    FILESYSTEM_STAGING_PREFIX,
    FaissBackend,
    PgVectorBackend,
)
from ragtime.indexer.vector_utils import search_pgvector_embeddings


class _Transaction:
    def __init__(self) -> None:
        self.execute_raw = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _TransactionalStorage:
    """Small transactional model that rolls back mutation state on exceptions."""

    def __init__(self, rows: dict[str, list[str]], fail_after: int | None = None) -> None:
        self.rows = rows
        self.fail_after = fail_after
        self.executions = 0
        self._snapshot: dict[str, list[str]] | None = None

    def tx(self, **_kwargs):
        return self

    async def query_raw(self, _statement):
        return [{"has_rows": True}]

    async def __aenter__(self):
        self._snapshot = {table: list(names) for table, names in self.rows.items()}
        return self

    async def __aexit__(self, error_type, *_args):
        if error_type:
            assert self._snapshot is not None
            self.rows = self._snapshot
        return False

    async def execute_raw(self, statement):
        self.executions += 1
        if self.fail_after == self.executions:
            raise RuntimeError("transaction write failed")
        normalized = statement.replace("\n", " ")
        table_match = re.search(r"(filesystem_embeddings|filesystem_file_metadata)", normalized)
        assert table_match is not None
        table = table_match.group(1)
        operation = "DELETE" if "DELETE" in normalized else "UPDATE"
        target = re.findall(r"index_name = '([^']+)'", normalized)[-1]
        if operation == "DELETE":
            self.rows[table] = [name for name in self.rows[table] if name != target]
        else:
            replacement_match = re.search(r"SET index_name = '([^']+)'", statement)
            assert replacement_match is not None
            replacement = replacement_match.group(1)
            self.rows[table] = [replacement if name == target else name for name in self.rows[table]]


class _Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, _: str) -> list[float]:
        return [0.0, 1.0]


class ReplacementPublicationTests(IsolatedAsyncioTestCase):
    async def test_pgvector_publication_replaces_vectors_and_metadata_in_one_transaction(self) -> None:
        transaction = _Transaction()
        database = MagicMock()
        database.tx.return_value = transaction
        database.query_raw = AsyncMock(return_value=[{"has_rows": True}])

        with patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=database)):
            await PgVectorBackend().publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")

        database.tx.assert_called_once_with(max_wait=10000, timeout=60000)
        statements = [call.args[0] for call in transaction.execute_raw.await_args_list]
        self.assertEqual(
            statements,
            [
                "DELETE FROM filesystem_embeddings WHERE index_name = 'manuals'",
                f"UPDATE filesystem_embeddings SET index_name = 'manuals' WHERE index_name = '{FILESYSTEM_STAGING_PREFIX}job'",
                "DELETE FROM filesystem_file_metadata WHERE index_name = 'manuals'",
                f"UPDATE filesystem_file_metadata SET index_name = 'manuals' WHERE index_name = '{FILESYSTEM_STAGING_PREFIX}job'",
            ],
        )

    async def test_pgvector_publication_commits_vector_and_metadata_replacement_together(self) -> None:
        storage = _TransactionalStorage(
            {
                "filesystem_embeddings": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
                "filesystem_file_metadata": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
            }
        )
        with patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)):
            await PgVectorBackend().publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")

        self.assertEqual(storage.rows["filesystem_embeddings"], ["manuals"])
        self.assertEqual(storage.rows["filesystem_file_metadata"], ["manuals"])

    async def test_pgvector_publication_failure_rolls_back_vectors_and_metadata(self) -> None:
        original_rows = {
            "filesystem_embeddings": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
            "filesystem_file_metadata": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
        }
        storage = _TransactionalStorage({table: list(rows) for table, rows in original_rows.items()}, fail_after=3)
        with patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)):
            with self.assertRaisesRegex(RuntimeError, "transaction write failed"):
                await PgVectorBackend().publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")

        self.assertEqual(storage.rows, original_rows)

    async def test_unpublished_faiss_generation_is_not_in_searchable_indexes(self) -> None:
        backend = FaissBackend()
        backend._staged_indexes[f"{FILESYSTEM_STAGING_PREFIX}job"] = Path("staged-generation")

        self.assertEqual(backend.get_loaded_indexes(), [])

    async def test_failed_staging_cleanup_does_not_touch_serving_faiss_index(self) -> None:
        backend = FaissBackend()
        serving = object()
        backend._loaded_indexes["manuals"] = serving

        with patch.object(backend, "delete_index", new=AsyncMock(return_value=1)) as delete:
            await backend.discard_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job")

        self.assertIs(backend._loaded_indexes["manuals"], serving)
        delete.assert_not_awaited()

    async def test_pgvector_rows_keep_chunk_provenance_without_target_secrets(self) -> None:
        database = MagicMock()
        database.execute_raw = AsyncMock()

        with (
            patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=database)),
            patch.object(PgVectorBackend(), "_ensure_dimension", new=AsyncMock()),
        ):
            backend = PgVectorBackend()
            backend._ensure_dimension = AsyncMock()
            await backend.store_embeddings(
                "manuals",
                "scan.png",
                ["| Item | Amount |\n| 45 N | 50/ER30 |"],
                [[0.1, 0.2]],
                {"source": "scan.png"},
                [{"segment_type": "table", "chunk_index": 4, "ocr_provider": "openai", "ocr_model": "gpt-4o"}],
            )

        statement = database.execute_raw.await_args.args[0]
        self.assertIn('"segment_type": "table"', statement)
        self.assertIn('"chunk_index": 4', statement)
        self.assertIn('"ocr_provider": "openai"', statement)
        self.assertNotIn("base_url", statement)

    async def test_pgvector_global_search_excludes_staged_rows_and_rejects_staged_name(self) -> None:
        database = MagicMock()
        database.query_raw = AsyncMock(return_value=[])
        with patch("ragtime.indexer.vector_utils.get_db", new=AsyncMock(return_value=database)):
            await search_pgvector_embeddings("filesystem_embeddings", [0.1])
            await search_pgvector_embeddings("filesystem_embeddings", [0.1], index_name="__filesystem_stagingx")
            with self.assertRaises(ValueError):
                await search_pgvector_embeddings("filesystem_embeddings", [0.1], index_name=f"{FILESYSTEM_STAGING_PREFIX}job")

        query = database.query_raw.await_args.args[0]
        self.assertIn("LEFT(index_name, 22) <> '__filesystem_staging__'", query)
        self.assertIn("index_name = '__filesystem_stagingx'", query)

    async def test_faiss_publish_switches_pointer_without_removing_legacy_serving_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            backend = FaissBackend()
            serving_root = base_path / "manuals"
            serving_root.mkdir()
            (serving_root / "index.faiss").write_text("legacy")
            (serving_root / "index.pkl").write_text("legacy")
            generation = base_path / ".filesystem-artifacts" / "filesystem_staging__job" / ".generations" / ("a" * 32)
            generation.mkdir(parents=True)
            (generation / "index.faiss").write_text("new")
            (generation / "index.pkl").write_text("new")
            backend._staged_indexes[f"{FILESYSTEM_STAGING_PREFIX}job"] = generation
            storage = _TransactionalStorage(
                {
                    "filesystem_embeddings": [],
                    "filesystem_file_metadata": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
                }
            )

            with (
                patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path),
                patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)),
            ):
                await backend.publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")
                self.assertEqual(
                    backend._serving_index_path("manuals"),
                    base_path / ".filesystem-artifacts" / "manuals" / ".generations" / generation.name,
                )

            self.assertTrue((serving_root / "index.faiss").exists())
            self.assertEqual(storage.rows["filesystem_file_metadata"], ["manuals"])

    async def test_faiss_failed_pointer_switch_restores_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            backend = FaissBackend()
            artifact_root = base_path / ".filesystem-artifacts" / "manuals"
            old_generation = artifact_root / ".generations" / ("b" * 32)
            staged_generation = base_path / ".filesystem-artifacts" / "filesystem_staging__job" / ".generations" / ("c" * 32)
            for generation in (old_generation, staged_generation):
                generation.mkdir(parents=True)
                (generation / "index.faiss").write_text(generation.name)
                (generation / "index.pkl").write_text(generation.name)
            with patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path):
                backend._write_serving_generation(artifact_root, old_generation)
                backend._staged_indexes[f"{FILESYSTEM_STAGING_PREFIX}job"] = staged_generation
            storage = _TransactionalStorage(
                {
                    "filesystem_embeddings": [],
                    "filesystem_file_metadata": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
                }
            )
            with (
                patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path),
                patch.object(backend, "_write_serving_generation", side_effect=OSError("disk full")),
                patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)),
            ):
                with self.assertRaises(OSError):
                    await backend.publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")
                self.assertEqual(backend._serving_index_path("manuals"), old_generation)

    async def test_faiss_metadata_failure_restores_previous_generation_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            backend = FaissBackend()
            artifact_root = base_path / ".filesystem-artifacts" / "manuals"
            old_generation = artifact_root / ".generations" / ("d" * 32)
            staged_generation = base_path / ".filesystem-artifacts" / "filesystem_staging__job" / ".generations" / ("e" * 32)
            for generation in (old_generation, staged_generation):
                generation.mkdir(parents=True)
                (generation / "index.faiss").write_text(generation.name)
                (generation / "index.pkl").write_text(generation.name)
            storage = _TransactionalStorage(
                {
                    "filesystem_embeddings": [],
                    "filesystem_file_metadata": ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"],
                },
                fail_after=2,
            )
            with patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path):
                backend._write_serving_generation(artifact_root, old_generation)
                backend._staged_indexes[f"{FILESYSTEM_STAGING_PREFIX}job"] = staged_generation
            with (
                patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path),
                patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)),
            ):
                with self.assertRaisesRegex(RuntimeError, "transaction write failed"):
                    await backend.publish_staged_index(f"{FILESYSTEM_STAGING_PREFIX}job", "manuals")
                self.assertEqual(backend._serving_index_path("manuals"), old_generation)

            self.assertEqual(storage.rows["filesystem_file_metadata"], ["manuals", f"{FILESYSTEM_STAGING_PREFIX}job"])

    async def test_faiss_staged_publish_incremental_save_and_rename_survive_fresh_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            backend = FaissBackend()
            staging_name = f"{FILESYSTEM_STAGING_PREFIX}job"
            storage = _TransactionalStorage(
                {
                    "filesystem_embeddings": [],
                    "filesystem_file_metadata": [staging_name],
                }
            )

            async def build_fixture(_job_id: str, index_root: Path, spool_root: Path):
                return await asyncio.to_thread(_build_faiss_generation, index_root, spool_root, "l2", False)

            with (
                patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path),
                patch("ragtime.indexer.vector_backends.get_db", new=AsyncMock(return_value=storage)),
                patch("ragtime.indexer.vector_backends.prepare_faiss_artifact", new=build_fixture),
            ):
                await backend.store_embeddings(
                    staging_name,
                    "first.png",
                    ["first generation"],
                    [[0.1, 0.2]],
                    {"source": "first.png", "ocr_provider": "openai"},
                )
                await backend.finalize_index(staging_name)
                staged_generation = backend._staged_indexes[staging_name]
                self.assertIn("filesystem_staging__job", str(staged_generation))

                await backend.publish_staged_index(staging_name, "manuals")
                serving_generation = backend._serving_index_path("manuals")
                self.assertNotEqual(serving_generation, staged_generation)
                self.assertTrue((serving_generation / "index.faiss").is_file())
                self.assertTrue((serving_generation / "index.pkl").is_file())
                self.assertEqual(storage.rows["filesystem_file_metadata"], ["manuals"])

                fresh_backend = FaissBackend()
                self.assertTrue(await fresh_backend.load_index("manuals", _Embeddings()))
                first_document = next(iter(fresh_backend._loaded_indexes["manuals"].docstore._dict.values()))
                self.assertEqual(first_document.page_content, "first generation")
                self.assertEqual(first_document.metadata["ocr_provider"], "openai")

                await backend.store_embeddings(
                    "manuals",
                    "second.png",
                    ["incremental generation"],
                    [[0.3, 0.4]],
                    {"source": "second.png"},
                )
                await backend.finalize_index("manuals")
                self.assertNotEqual(backend._serving_index_path("manuals"), serving_generation)
                reloaded_backend = FaissBackend()
                self.assertTrue(await reloaded_backend.load_index("manuals", _Embeddings()))
                reloaded_document = next(iter(reloaded_backend._loaded_indexes["manuals"].docstore._dict.values()))
                self.assertEqual(reloaded_document.page_content, "incremental generation")

                self.assertTrue(await backend.rename_index("manuals", "renamed-manuals"))
                renamed_backend = FaissBackend()
                self.assertTrue(await renamed_backend.load_index("renamed-manuals", _Embeddings()))

    async def test_staging_finalize_failure_cleans_spool_and_artifact_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            backend = FaissBackend()
            staging_name = f"{FILESYSTEM_STAGING_PREFIX}job"
            with patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path):
                await backend.store_embeddings(staging_name, "page.png", ["text"], [[0.1, 0.2]], {})
                spool_root = backend._staging_spools[staging_name].root
                with patch(
                    "ragtime.indexer.vector_backends.prepare_faiss_artifact",
                    new=AsyncMock(side_effect=RuntimeError("build failed")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "build failed"):
                        await backend.finalize_index(staging_name)

            self.assertFalse(spool_root.exists())
            self.assertNotIn(staging_name, backend._staging_spools)

    def test_disk_index_discovery_ignores_legacy_staging_and_backup_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            for name in ("manuals", f"{FILESYSTEM_STAGING_PREFIX}job", "manuals.previous-old"):
                index_path = base_path / name
                index_path.mkdir()
                (index_path / "index.faiss").write_text("index")
            with patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", base_path):
                self.assertEqual(FaissBackend.list_disk_indexes(), ["manuals"])
