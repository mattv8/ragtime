"""
Vector Store Backends - Unified interface for FAISS and pgvector storage.

Provides a common abstraction for storing and searching embeddings
across different vector store backends, allowing filesystem indexes
to use either FAISS (in-memory) or pgvector (PostgreSQL).
"""

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.database import get_db
from ragtime.core.faiss_concurrency import FaissSearchBusyError, faiss_search_coordinator
from ragtime.core.logging import get_logger
from ragtime.indexer.faiss_artifacts import _fsync_directory, prepare_faiss_artifact
from ragtime.indexer.faiss_serialization import safe_load_faiss
from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput
from ragtime.indexer.models import VectorStoreType
from ragtime.indexer.vector_utils import (
    FAISS_SOURCE_METADATA_KEYS,
    FILESYSTEM_COLUMNS,
    FILESYSTEM_STAGING_PREFIX,
    ensure_embedding_column,
    ensure_pgvector_extension,
    get_embeddings_model,
    get_pgvector_table_size_bytes,
    search_pgvector_embeddings,
)

logger = get_logger(__name__)

# Base path for FAISS indexes (shared with document indexer - flat structure)
FAISS_INDEX_BASE_PATH = Path(settings.index_data_path)
STALE_FILESYSTEM_ARTIFACT_SECONDS = 24 * 60 * 60


def _first_metadata_value(metadata: Any, keys: tuple[str, ...]) -> str:
    """Return the first non-empty value across ``keys`` in a metadata mapping."""
    if not metadata:
        return ""
    if hasattr(metadata, "get"):
        for key in keys:
            value = metadata.get(key)
            if value:
                return str(value)
    return ""


def _chunk_row_metadata(metadata: Dict[str, Any], chunk_metadata: Dict[str, Any] | None, file_path: str) -> Dict[str, Any]:
    """Merge chunk semantics without letting a chunk change source identity."""
    merged = {**(chunk_metadata or {}), **metadata}
    merged["source"] = metadata.get("source", file_path)
    merged["file_path"] = metadata.get("file_path", file_path)
    return merged


async def _promote_staged_file_metadata(tx: Any, staging_index_name: str, index_name: str) -> None:
    """Replace serving filesystem file metadata within an existing transaction."""
    escaped_index_name = index_name.replace("'", "''")
    escaped_staging_name = staging_index_name.replace("'", "''")
    await tx.execute_raw(f"DELETE FROM filesystem_file_metadata WHERE index_name = '{escaped_index_name}'")
    await tx.execute_raw(f"UPDATE filesystem_file_metadata SET index_name = '{escaped_index_name}' WHERE index_name = '{escaped_staging_name}'")


class VectorStoreBackend(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def store_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Store embeddings for a file.

        Args:
            index_name: Name of the index
            file_path: Relative path of the source file
            chunks: List of text chunks
            embeddings: List of embedding vectors (same length as chunks)
            metadata: Additional metadata to store with each chunk

        Returns:
            Number of embeddings stored
        """
        pass

    @abstractmethod
    async def delete_file_embeddings(self, index_name: str, file_path: str) -> int:
        """
        Delete all embeddings for a specific file.

        Args:
            index_name: Name of the index
            file_path: Relative path of the source file

        Returns:
            Number of embeddings deleted
        """
        pass

    @abstractmethod
    async def delete_index(self, index_name: str) -> int:
        """
        Delete all embeddings for an entire index.

        Args:
            index_name: Name of the index to delete

        Returns:
            Number of embeddings deleted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        index_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query vector
            index_name: Optional index to restrict search to
            max_results: Maximum number of results

        Returns:
            List of results with content, file_path, similarity, etc.
        """
        pass

    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about an index.

        Args:
            index_name: Name of the index

        Returns:
            Dict with chunk_count, file_count, etc.
        """
        pass

    @abstractmethod
    async def finalize_index(self, index_name: str) -> None:
        """
        Finalize an index after batch insertion.

        For FAISS, this saves the index to disk.
        For pgvector, this may rebuild indexes if needed.
        """
        pass

    async def publish_staged_index(self, staging_index_name: str, index_name: str) -> None:
        """Publish a completed replacement staging index."""
        raise NotImplementedError("This vector backend does not support staged publication")

    async def discard_staged_index(self, staging_index_name: str) -> None:
        """Discard an unpublished replacement staging index."""
        raise NotImplementedError("This vector backend does not support staged publication")


class PgVectorBackend(VectorStoreBackend):
    """pgvector backend - stores embeddings in PostgreSQL."""

    def __init__(self) -> None:
        self._dimension_ensured: Dict[str, int] = {}

    async def _ensure_dimension(self, embedding_dim: int, *, preserve_existing: bool = False) -> None:
        """Ensure the embedding column has the correct dimension."""
        # Check if already ensured for this dimension
        if self._dimension_ensured.get("dim") == embedding_dim:
            return

        await ensure_pgvector_extension(logger_override=logger)

        if preserve_existing:
            db = await get_db()
            column = await db.query_raw(
                "SELECT a.atttypmod AS typmod FROM pg_attribute a "
                "JOIN pg_class c ON a.attrelid = c.oid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE c.relname = 'filesystem_embeddings' AND n.nspname = 'public' "
                "AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped"
            )
            current_dim = column[0].get("typmod") if column else None
            if isinstance(current_dim, int) and current_dim > 0 and current_dim != embedding_dim:
                populated = await db.query_raw(
                    "SELECT EXISTS(SELECT 1 FROM filesystem_embeddings "
                    f"WHERE embedding IS NOT NULL AND LEFT(index_name, {len(FILESYSTEM_STAGING_PREFIX)}) "
                    f"<> '{FILESYSTEM_STAGING_PREFIX}') AS populated"
                )
                if populated and populated[0].get("populated"):
                    raise RuntimeError(
                        "Filesystem replacement embedding dimension is incompatible with the populated pgvector column; "
                        "choose the existing embedding model or perform an explicit storage migration/rebuild."
                    )

        app_settings = await get_app_settings()
        index_lists = app_settings.get("ivfflat_lists", 100)

        await ensure_embedding_column(
            table_name="filesystem_embeddings",
            index_name="filesystem_embeddings_embedding_idx",
            embedding_dim=embedding_dim,
            index_lists=index_lists,
            logger_override=logger,
        )
        self._dimension_ensured["dim"] = embedding_dim

    async def store_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Store embeddings in pgvector table using batch inserts.

        Uses batch INSERT with multiple VALUES for better performance.
        Batches are sized to avoid exceeding PostgreSQL query limits.
        """
        if not embeddings:
            return 0

        # Ensure column dimension matches
        await self._ensure_dimension(len(embeddings[0]), preserve_existing=index_name.startswith(FILESYSTEM_STAGING_PREFIX))

        db = await get_db()
        escaped_file_path = file_path.replace("'", "''")

        # Batch insert for better performance
        # PostgreSQL has limits on query size, so we batch ~100 rows at a time
        batch_size = 100
        inserted = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            values_list = []

            for i in range(batch_start, batch_end):
                chunk = chunks[i]
                embedding = embeddings[i]
                row_metadata = _chunk_row_metadata(metadata, chunk_metadata[i] if chunk_metadata else None, file_path)
                metadata_json = json.dumps(row_metadata).replace("'", "''")
                embedding_id = str(uuid.uuid4())
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                escaped_chunk = chunk.replace("'", "''")

                values_list.append(
                    f"('{embedding_id}', '{index_name}', '{escaped_file_path}', {i}, "
                    f"'{escaped_chunk}', '{metadata_json}'::jsonb, "
                    f"'{embedding_str}'::vector, NOW())"
                )

            if values_list:
                values_sql = ",\n".join(values_list)
                await db.execute_raw(
                    f"""
                    INSERT INTO filesystem_embeddings
                    (id, index_name, file_path, chunk_index, content, metadata, embedding, created_at)
                    VALUES {values_sql}
                    """
                )
                inserted += len(values_list)

        return inserted

    async def delete_file_embeddings(self, index_name: str, file_path: str) -> int:
        """Delete embeddings for a specific file."""
        db = await get_db()
        result = await db.execute_raw(
            f"""
            DELETE FROM filesystem_embeddings
            WHERE index_name = '{index_name}' AND file_path = '{file_path.replace("'", "''")}'
        """
        )
        return result if isinstance(result, int) else 0

    async def delete_index(self, index_name: str) -> int:
        """Delete all embeddings for an index."""
        db = await get_db()
        result = await db.execute_raw(
            f"""
            DELETE FROM filesystem_embeddings WHERE index_name = '{index_name}'
        """
        )
        return result if isinstance(result, int) else 0

    async def search(
        self,
        query_embedding: List[float],
        index_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity via centralized search function."""
        return await search_pgvector_embeddings(
            table_name="filesystem_embeddings",
            query_embedding=query_embedding,
            index_name=index_name,
            max_results=max_results,
            columns=FILESYSTEM_COLUMNS,
            logger_override=logger,
        )

    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics from database."""
        db = await get_db()
        result = await db.query_raw(
            f"""
            SELECT
                COUNT(*) as chunk_count,
                COUNT(DISTINCT file_path) as file_count
            FROM filesystem_embeddings
            WHERE index_name = '{index_name}'
        """
        )

        if result:
            return {
                "chunk_count": result[0]["chunk_count"],
                "file_count": result[0]["file_count"],
                "vector_store_type": "pgvector",
            }
        return {"chunk_count": 0, "file_count": 0, "vector_store_type": "pgvector"}

    async def get_pgvector_table_size_bytes(self, index_name: str) -> int:
        """Calculate pgvector storage size for a specific index.

        Uses PostgreSQL's pg_column_size to estimate storage for this index's rows.
        """
        return await get_pgvector_table_size_bytes("filesystem_embeddings", index_name)

    async def finalize_index(self, index_name: str) -> None:
        """No-op for pgvector - data is already persisted."""
        pass

    async def publish_staged_index(self, staging_index_name: str, index_name: str) -> None:
        """Atomically replace an index with rows written under a private staging name."""
        db = await get_db()
        escaped_index_name = index_name.replace("'", "''")
        escaped_staging_name = staging_index_name.replace("'", "''")
        staged_rows = await db.query_raw(f"SELECT EXISTS(SELECT 1 FROM filesystem_embeddings WHERE index_name = '{escaped_staging_name}') AS has_rows")
        if isinstance(staged_rows, list) and (not staged_rows or not staged_rows[0].get("has_rows")):
            raise RuntimeError("Filesystem replacement contains no valid embeddings; the existing index was preserved")
        async with db.tx(max_wait=10000, timeout=60000) as tx:
            await tx.execute_raw(f"DELETE FROM filesystem_embeddings WHERE index_name = '{escaped_index_name}'")
            await tx.execute_raw(f"UPDATE filesystem_embeddings SET index_name = '{escaped_index_name}' WHERE index_name = '{escaped_staging_name}'")
            await _promote_staged_file_metadata(tx, staging_index_name, index_name)

    async def discard_staged_index(self, staging_index_name: str) -> None:
        """Remove unreachable rows left by an unsuccessful replacement."""
        db = await get_db()
        await db.filesystemembedding.delete_many(where={"indexName": staging_index_name})
        await db.filesystemfilemetadata.delete_many(where={"indexName": staging_index_name})


class FaissBackend(VectorStoreBackend):
    """FAISS backend - stores embeddings in memory and saves to disk."""

    def __init__(self) -> None:
        # In-memory storage during indexing: {index_name: {"texts": [], "metadatas": [], "embeddings": []}}
        self._pending: Dict[str, Dict[str, List]] = {}
        # Loaded FAISS indexes: {index_name: FAISS vectorstore}
        self._loaded_indexes: Dict[str, Any] = {}
        # Completed replacement generations are deliberately not searchable.
        self._staged_indexes: Dict[str, Path] = {}
        self._staging_spools: Dict[str, IndexingSpool] = {}
        self._cleanup_stale_filesystem_artifacts()

    @staticmethod
    def _is_filesystem_staging(index_name: str) -> bool:
        return index_name.startswith(FILESYSTEM_STAGING_PREFIX)

    @staticmethod
    def _filesystem_artifacts_path() -> Path:
        return FAISS_INDEX_BASE_PATH / ".filesystem-artifacts"

    def _artifact_root(self, index_name: str) -> Path:
        return self._filesystem_artifacts_path() / self._sanitize_index_name(index_name)

    def _cleanup_stale_filesystem_artifacts(self) -> None:
        """Bound cleanup to old interrupted private builds without touching active work."""
        artifacts = self._filesystem_artifacts_path()
        if not artifacts.is_dir():
            return
        removed = 0
        stale_before = time.time() - STALE_FILESYSTEM_ARTIFACT_SECONDS
        staging_artifact_prefix = self._sanitize_index_name(FILESYSTEM_STAGING_PREFIX)
        for stage_root in artifacts.iterdir():
            if removed >= 100:
                break
            if (
                stage_root.name.startswith(staging_artifact_prefix)
                and stage_root.is_dir()
                and not stage_root.is_symlink()
                and stage_root.stat().st_mtime < stale_before
            ):
                shutil.rmtree(stage_root, ignore_errors=True)
                removed += 1
        for building in artifacts.glob("*/.generations/.*.building"):
            if removed >= 100:
                break
            if building.is_dir() and not building.is_symlink() and building.stat().st_mtime < stale_before:
                shutil.rmtree(building, ignore_errors=True)
                removed += 1

    def _staging_spool(self, index_name: str) -> IndexingSpool:
        spool = self._staging_spools.get(index_name)
        if spool is None:
            spool = IndexingSpool.open_attempt(
                self._filesystem_artifacts_path() / "spools",
                index_name,
                index_name,
            )
            self._staging_spools[index_name] = spool
        return spool

    def _append_staged_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        chunk_metadata: Optional[List[Dict[str, Any]]],
    ) -> int:
        """Append one bounded filesystem batch to the durable FAISS spool."""
        spool = self._staging_spool(index_name)
        if len(chunks) != len(embeddings):
            raise ValueError("Filesystem chunks and embeddings must align")
        dimensions = len(embeddings[0])
        if not dimensions or any(len(vector) != dimensions for vector in embeddings):
            raise ValueError("Filesystem embeddings must have one non-zero dimension")
        task_dir = spool.root / "tasks" / uuid.uuid4().hex
        task_dir.mkdir(parents=True)
        records: list[SpoolRecord] = []
        manifest = task_dir / "chunks.jsonl"
        with manifest.open("w", encoding="utf-8") as lines:
            for offset, chunk in enumerate(chunks):
                record_metadata = _chunk_row_metadata(metadata, chunk_metadata[offset] if chunk_metadata else None, file_path)
                record_metadata.setdefault("chunk_index", offset)
                record_metadata["index_name"] = index_name
                record_id = IndexingSpool.stable_id(file_path, offset, offset, chunk)
                record_metadata["chunk_id"] = record_id
                text_path = task_dir / f"{record_id}.txt"
                text_path.write_text(chunk, encoding="utf-8")
                record = SpoolRecord(record_id, file_path, offset, str(text_path.relative_to(spool.root)), record_metadata, len(chunk.encode("utf-8")))
                records.append(record)
                lines.write(
                    json.dumps(
                        {
                            "record_id": record.record_id,
                            "source": record.source,
                            "ordinal": record.ordinal,
                            "text_path": record.text_path,
                            "metadata": record.metadata,
                            "text_bytes": record.text_bytes,
                        }
                    )
                    + "\n"
                )
        spool.accept_chunks(SpoolTaskOutput(str(manifest.relative_to(spool.root)), len(records), sum(record.text_bytes for record in records), 0))
        vectors_path = task_dir / "vectors.f32"
        import numpy as np

        np.asarray(embeddings, dtype="<f4").tofile(vectors_path)
        spool.accept_embeddings(EmbeddedBatch(tuple(records), str(vectors_path.relative_to(spool.root)), 0, len(records), dimensions))
        return len(records)

    def _sanitize_index_name(self, index_name: str) -> str:
        """Sanitize index name for safe filesystem usage."""
        # Replace any non-alphanumeric chars (except underscore/hyphen) with underscore
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", index_name).strip("_").lower()
        return safe_name or "default"

    def _get_index_path(self, index_name: str) -> Path:
        """Get the filesystem path for a FAISS index."""
        # Sanitize index name to prevent filesystem issues with spaces/special chars
        safe_name = self._sanitize_index_name(index_name)
        return FAISS_INDEX_BASE_PATH / safe_name

    def _serving_index_path(self, index_name: str) -> Path:
        """Resolve the current immutable filesystem generation or legacy root."""
        artifact_root = self._artifact_root(index_name)
        manifest = artifact_root / "current.json"
        if not manifest.is_file():
            return self._get_index_path(index_name)
        try:
            generation = str(json.loads(manifest.read_text(encoding="utf-8"))["generation"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Filesystem FAISS serving manifest is invalid for {index_name}") from exc
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise RuntimeError(f"Filesystem FAISS serving manifest has an invalid generation for {index_name}")
        path = artifact_root / ".generations" / generation
        if not (path / "index.faiss").is_file() or not (path / "index.pkl").is_file():
            raise RuntimeError(f"Filesystem FAISS serving generation is incomplete for {index_name}")
        return path

    def _write_serving_generation(self, artifact_root: Path, generation: Path) -> None:
        """Atomically point readers at a validated immutable generation."""
        generation_id = generation.name
        if not re.fullmatch(r"[0-9a-f]{32}", generation_id):
            raise ValueError("Filesystem FAISS generation has an invalid name")
        expected_generation = artifact_root / ".generations" / generation_id
        if generation.resolve() != expected_generation.resolve():
            raise ValueError("Filesystem FAISS generation is not stored under the serving index")
        if not (generation / "index.faiss").is_file() or not (generation / "index.pkl").is_file():
            raise RuntimeError("Filesystem FAISS generation is incomplete")
        artifact_root.mkdir(parents=True, exist_ok=True)
        temporary = artifact_root / f".current-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as pointer:
                json.dump({"generation": generation_id}, pointer)
                pointer.flush()
                os.fsync(pointer.fileno())
            os.replace(temporary, artifact_root / "current.json")
            _fsync_directory(artifact_root)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_serving_manifest(self, artifact_root: Path, previous_manifest: bytes | None) -> None:
        """Restore the prior pointer after a database transaction fails."""
        manifest = artifact_root / "current.json"
        if previous_manifest is None:
            manifest.unlink(missing_ok=True)
            _fsync_directory(artifact_root)
            return
        temporary = artifact_root / f".current-restore-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(previous_manifest)
            with temporary.open("rb") as pointer:
                os.fsync(pointer.fileno())
            os.replace(temporary, manifest)
            _fsync_directory(artifact_root)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_generation(generation: Path) -> None:
        """Require both FAISS files before a generation can be published."""
        if not generation.is_dir() or not (generation / "index.faiss").is_file() or not (generation / "index.pkl").is_file():
            raise RuntimeError("Filesystem FAISS generation is incomplete")

    def _move_staged_generation(self, artifact_root: Path, generation: Path) -> Path:
        """Move a private build beneath its serving index before publishing it."""
        self._validate_generation(generation)
        destination_parent = artifact_root / ".generations"
        destination = destination_parent / generation.name
        if destination.exists():
            raise RuntimeError("Filesystem FAISS generation already exists in the serving index")
        destination_parent.mkdir(parents=True, exist_ok=True)
        os.replace(generation, destination)
        _fsync_directory(generation.parent)
        _fsync_directory(destination_parent)
        self._validate_generation(destination)
        return destination

    def _save_incremental_generation(self, artifact_root: Path, faiss_db: FAISS) -> Path:
        """Persist an incremental update as a complete immutable generation."""
        generation_parent = artifact_root / ".generations"
        generation_id = uuid.uuid4().hex
        staging = generation_parent / f".{generation_id}.building"
        generation = generation_parent / generation_id
        generation_parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss_db.save_local(str(staging))
            self._validate_generation(staging)
            for artifact_file in staging.iterdir():
                if artifact_file.is_file():
                    with artifact_file.open("rb") as file_handle:
                        os.fsync(file_handle.fileno())
            _fsync_directory(staging)
            os.replace(staging, generation)
            _fsync_directory(generation_parent)
            return generation
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _retain_serving_generations(self, artifact_root: Path) -> None:
        """Keep the current generation and one rollback generation only."""
        try:
            current = str(json.loads((artifact_root / "current.json").read_text(encoding="utf-8"))["generation"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return
        generation_parent = artifact_root / ".generations"
        if not generation_parent.is_dir():
            return
        completed = [path for path in generation_parent.iterdir() if re.fullmatch(r"[0-9a-f]{32}", path.name) and path.is_dir()]
        completed.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        previous = [path.name for path in completed if path.name != current][:1]
        retained = {current, *previous}
        for generation in completed:
            if generation.name not in retained:
                shutil.rmtree(generation, ignore_errors=True)
        _fsync_directory(generation_parent)

    async def store_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Buffer embeddings for later FAISS index creation."""
        if not embeddings:
            return 0

        if self._is_filesystem_staging(index_name):
            return await asyncio.to_thread(
                self._append_staged_embeddings,
                index_name,
                file_path,
                chunks,
                embeddings,
                metadata,
                chunk_metadata,
            )

        if index_name not in self._pending:
            self._pending[index_name] = {"texts": [], "metadatas": [], "embeddings": []}

        pending = self._pending[index_name]
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            pending["texts"].append(chunk)
            pending["embeddings"].append(embedding)
            row_metadata = _chunk_row_metadata(metadata, chunk_metadata[i] if chunk_metadata else None, file_path)
            row_metadata.setdefault("chunk_index", i)
            row_metadata["index_name"] = index_name
            pending["metadatas"].append(row_metadata)

        return len(chunks)

    async def delete_file_embeddings(self, index_name: str, file_path: str) -> int:
        """
        Delete embeddings for a specific file.

        Note: For FAISS, this removes from pending buffer but cannot remove
        from an already-saved index. Full re-index is required for that.
        """
        if index_name not in self._pending:
            return 0

        pending = self._pending[index_name]
        original_count = len(pending["texts"])

        # Filter out entries for this file
        indices_to_keep = [i for i, m in enumerate(pending["metadatas"]) if m.get("file_path") != file_path]

        pending["texts"] = [pending["texts"][i] for i in indices_to_keep]
        pending["embeddings"] = [pending["embeddings"][i] for i in indices_to_keep]
        pending["metadatas"] = [pending["metadatas"][i] for i in indices_to_keep]

        return original_count - len(pending["texts"])

    async def delete_index(self, index_name: str) -> int:
        """Delete the entire FAISS index from disk and memory.

        Returns:
            Count of deleted items (from pending buffer) plus 1 if disk was deleted,
            or 0 if nothing was deleted.
        """
        if self._is_filesystem_staging(index_name):
            return await self._discard_filesystem_staging(index_name)

        deleted = 0

        # Clear from pending
        if index_name in self._pending:
            deleted = len(self._pending[index_name]["texts"])
            del self._pending[index_name]
            logger.info(f"Cleared {deleted} pending embeddings for FAISS index: {index_name}")

        # Clear from loaded indexes
        if index_name in self._loaded_indexes:
            del self._loaded_indexes[index_name]
            logger.info(f"Cleared loaded FAISS index from memory: {index_name}")

        self._staged_indexes.pop(index_name, None)

        # Delete from disk
        index_path = self._get_index_path(index_name)
        if index_path.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, index_path)
                logger.info(f"Deleted FAISS filesystem index from disk: {index_path}")
                # Count disk deletion as at least 1 to signal success to caller
                if deleted == 0:
                    deleted = 1
            except Exception as e:
                logger.error(f"Failed to delete FAISS index directory {index_path}: {e}")
        else:
            logger.debug(f"FAISS index path does not exist, nothing to delete: {index_path}")

        artifact_root = self._artifact_root(index_name)
        if artifact_root.exists():
            await asyncio.to_thread(shutil.rmtree, artifact_root)
            deleted = max(deleted, 1)

        return deleted

    async def rename_index(self, old_name: str, new_name: str) -> bool:
        """Rename a FAISS index on disk and update in-memory tracking."""
        old_path = self._get_index_path(old_name)
        new_path = self._get_index_path(new_name)
        old_artifact_root = self._artifact_root(old_name)
        new_artifact_root = self._artifact_root(new_name)

        if not old_path.exists() and not old_artifact_root.exists():
            logger.warning(f"FAISS index not found for rename: {old_name}")
            return False

        if new_path.exists() or new_artifact_root.exists():
            logger.error(f"FAISS index already exists at target: {new_name}")
            return False

        try:
            if old_path.exists():
                await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
            if old_artifact_root.exists():
                new_artifact_root.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.move, str(old_artifact_root), str(new_artifact_root))
            logger.info(f"Renamed FAISS index on disk: {old_name} -> {new_name}")

            # Update in-memory tracking
            if old_name in self._loaded_indexes:
                self._loaded_indexes[new_name] = self._loaded_indexes.pop(old_name)
                logger.info(f"Updated in-memory FAISS tracking: {old_name} -> {new_name}")

            # Update pending buffer if exists
            if old_name in self._pending:
                self._pending[new_name] = self._pending.pop(old_name)

            return True
        except Exception as e:
            logger.error(f"Failed to rename FAISS index: {e}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        index_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search loaded FAISS indexes."""
        if index_name and self._is_filesystem_staging(index_name):
            return []
        results = []

        indexes_to_search = {index_name: self._loaded_indexes[index_name]} if index_name and index_name in self._loaded_indexes else self._loaded_indexes

        for idx_name, faiss_db in indexes_to_search.items():
            try:
                docs_with_scores = await faiss_search_coordinator.run(
                    idx_name,
                    faiss_db.similarity_search_with_score_by_vector,
                    query_embedding,
                    k=max_results,
                )

                for doc, score in docs_with_scores:
                    # FAISS returns L2 distance; convert to similarity (lower is better)
                    # Cosine distance is between 0 and 2 for normalized vectors
                    similarity = 1 - (score / 2)  # Approximate conversion

                    results.append(
                        {
                            "id": doc.metadata.get("id", str(uuid.uuid4())),
                            "index_name": idx_name,
                            "file_path": doc.metadata.get("file_path", ""),
                            "chunk_index": doc.metadata.get("chunk_index", 0),
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity": similarity,
                        }
                    )
            except FaissSearchBusyError as e:
                logger.warning(f"FAISS search busy for index {idx_name}: {e}")
            except Exception as e:
                logger.warning(f"Error searching FAISS index {idx_name}: {e}")

        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about a FAISS index."""
        stats = {
            "embedding_count": 0,
            "chunk_count": 0,
            "file_count": 0,
            "vector_store_type": "faiss",
            "size_mb": None,
        }

        file_paths: set[str] = set()

        # Check pending buffer
        if index_name in self._pending:
            pending = self._pending[index_name]
            stats["chunk_count"] = len(pending["texts"])
            stats["embedding_count"] = len(pending["texts"])
            file_paths = {_first_metadata_value(m, FAISS_SOURCE_METADATA_KEYS) for m in pending["metadatas"]}
            file_paths.discard("")
            stats["file_count"] = len(file_paths)
            return stats

        # Check loaded index
        if index_name in self._loaded_indexes:
            faiss_db = self._loaded_indexes[index_name]
            try:
                # Get document count from FAISS
                stats["chunk_count"] = faiss_db.index.ntotal
                stats["embedding_count"] = faiss_db.index.ntotal
                # Count unique source files from docstore. Metadata key depends
                # on which indexer wrote the chunk: filesystem -> "file_path",
                # archive/git/document indexers -> "source".
                for doc_id in faiss_db.index_to_docstore_id.values():
                    doc = faiss_db.docstore.search(doc_id)
                    if doc and hasattr(doc, "metadata"):
                        source = _first_metadata_value(doc.metadata, FAISS_SOURCE_METADATA_KEYS)
                        if source:
                            file_paths.add(source)
                stats["file_count"] = len(file_paths)
            except Exception as e:
                logger.debug(f"Error getting FAISS stats for {index_name}: {e}")

        # Calculate file size on disk (blocking I/O, run in thread)
        try:
            index_path = self._serving_index_path(index_name)
        except RuntimeError:
            return stats

        def _calc_size() -> float | None:
            if not index_path.exists():
                return None
            total_size = 0
            faiss_file = index_path / "index.faiss"
            pkl_file = index_path / "index.pkl"
            if faiss_file.exists():
                total_size += faiss_file.stat().st_size
            if pkl_file.exists():
                total_size += pkl_file.stat().st_size
            return total_size / (1024 * 1024)

        size_mb = await asyncio.to_thread(_calc_size)
        if size_mb is not None:
            stats["size_mb"] = size_mb

        return stats

    async def finalize_index(self, index_name: str) -> None:
        """Build and save the FAISS index to disk."""
        if self._is_filesystem_staging(index_name):
            spool = self._staging_spools.get(index_name)
            if spool is None:
                raise RuntimeError(f"No filesystem staging spool for {index_name}")
            spool_root = spool.root
            try:
                spool.close()
                artifact = await prepare_faiss_artifact(index_name, self._artifact_root(index_name), spool_root)
                self._staged_indexes[index_name] = artifact.generation_path
            except BaseException:
                self._staged_indexes.pop(index_name, None)
                await asyncio.to_thread(shutil.rmtree, spool_root, ignore_errors=True)
                raise
            finally:
                self._staging_spools.pop(index_name, None)
            await asyncio.to_thread(shutil.rmtree, spool_root, ignore_errors=True)
            return
        if index_name not in self._pending:
            logger.warning(f"No pending data for FAISS index {index_name}")
            return

        pending = self._pending[index_name]
        if not pending["texts"]:
            logger.warning(f"No embeddings to save for FAISS index {index_name}")
            del self._pending[index_name]
            return

        logger.info(f"Finalizing FAISS index {index_name} with {len(pending['texts'])} chunks")

        # Create FAISS index from embeddings
        # We already have embeddings, so we use from_embeddings
        text_embeddings = list(zip(pending["texts"], pending["embeddings"]))

        # Get embedding model for FAISS (needed for later searches)
        app_settings = await get_app_settings()
        embeddings_model = await get_embeddings_model(app_settings, return_none_on_error=True, logger_override=logger)

        if not embeddings_model:
            raise RuntimeError("Could not get embedding model for FAISS index creation")

        # Create FAISS index
        faiss_db = await asyncio.to_thread(
            FAISS.from_embeddings,
            text_embeddings,
            embeddings_model,
            metadatas=pending["metadatas"],
        )

        artifact_root = self._artifact_root(index_name)
        if (artifact_root / "current.json").is_file():
            generation = await asyncio.to_thread(self._save_incremental_generation, artifact_root, faiss_db)
            await asyncio.to_thread(self._write_serving_generation, artifact_root, generation)
            await asyncio.to_thread(self._retain_serving_generations, artifact_root)
            logger.info("Saved FAISS index generation for %s", index_name)
        else:
            index_path = self._get_index_path(index_name)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(faiss_db.save_local, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

        self._loaded_indexes[index_name] = faiss_db

        # Clear pending buffer
        del self._pending[index_name]

    async def publish_staged_index(self, staging_index_name: str, index_name: str) -> None:
        """Promote a completed private FAISS generation without exposing it to search."""
        generation = self._staged_indexes.get(staging_index_name)
        if generation is None or not generation.is_dir():
            raise RuntimeError(f"Staged FAISS index does not exist: {staging_index_name}")
        artifact_root = self._artifact_root(index_name)
        manifest = artifact_root / "current.json"
        previous_manifest = manifest.read_bytes() if manifest.is_file() else None
        generation = await asyncio.to_thread(self._move_staged_generation, artifact_root, generation)
        self._staged_indexes[staging_index_name] = generation

        def publish() -> None:
            self._write_serving_generation(artifact_root, generation)

        try:
            db = await get_db()
            async with db.tx(max_wait=10000, timeout=60000) as tx:
                await faiss_search_coordinator.run(index_name, publish)
                await _promote_staged_file_metadata(tx, staging_index_name, index_name)
        except BaseException:
            await faiss_search_coordinator.run(index_name, self._restore_serving_manifest, artifact_root, previous_manifest)
            await asyncio.to_thread(shutil.rmtree, generation, ignore_errors=True)
            self._staged_indexes.pop(staging_index_name, None)
            raise
        self._staged_indexes.pop(staging_index_name, None)
        self._loaded_indexes.pop(index_name, None)
        await asyncio.to_thread(self._retain_serving_generations, artifact_root)

    async def discard_staged_index(self, staging_index_name: str) -> None:
        """Delete an unpublished FAISS generation."""
        await self._discard_filesystem_staging(staging_index_name)

    async def _discard_filesystem_staging(self, staging_index_name: str) -> int:
        """Delete only private staging data, never an active serving generation."""
        generation = self._staged_indexes.pop(staging_index_name, None)
        spool = self._staging_spools.pop(staging_index_name, None)
        deleted = 0
        if spool is not None:
            spool_root = spool.root
            spool.close()
            await asyncio.to_thread(shutil.rmtree, spool_root, ignore_errors=True)
            deleted = 1
        if generation is not None:
            await asyncio.to_thread(shutil.rmtree, generation, ignore_errors=True)
            deleted = 1
        return deleted

    async def load_index(self, index_name: str, embeddings_model: Any) -> bool:
        """Load a FAISS index from disk into memory."""
        try:
            index_path = self._serving_index_path(index_name)
        except RuntimeError as exc:
            logger.error("Cannot load filesystem FAISS index %s: %s", index_name, exc)
            return False

        if not index_path.exists():
            return False

        try:
            faiss_db = await asyncio.to_thread(safe_load_faiss, index_path, embeddings_model)
            self._loaded_indexes[index_name] = faiss_db
            logger.info(f"Loaded FAISS filesystem index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index {index_name}: {e}")
            return False

    def is_loaded(self, index_name: str) -> bool:
        """Check if an index is loaded in memory."""
        return index_name in self._loaded_indexes

    def get_loaded_indexes(self) -> List[str]:
        """Get list of loaded index names."""
        return list(self._loaded_indexes.keys())

    @classmethod
    def list_disk_indexes(cls) -> List[str]:
        """List all FAISS indexes saved on disk."""
        if not FAISS_INDEX_BASE_PATH.exists():
            return []

        indexes = []
        for path in FAISS_INDEX_BASE_PATH.iterdir():
            if path.name.startswith(".") or path.name.startswith(FILESYSTEM_STAGING_PREFIX) or ".previous-" in path.name:
                continue
            if path.is_dir() and (path / "index.faiss").exists():
                indexes.append(path.name)
        artifacts = FAISS_INDEX_BASE_PATH / ".filesystem-artifacts"
        if artifacts.is_dir():
            for artifact_root in artifacts.iterdir():
                if not artifact_root.is_dir() or artifact_root.name == "spools":
                    continue
                manifest = artifact_root / "current.json"
                try:
                    generation = str(json.loads(manifest.read_text(encoding="utf-8"))["generation"])
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                generation_path = artifact_root / ".generations" / generation
                if (generation_path / "index.faiss").is_file() and (generation_path / "index.pkl").is_file():
                    indexes.append(artifact_root.name)
        return sorted(set(indexes))


# Global backend instances (lazy initialization)
_pgvector_backend: Optional[PgVectorBackend] = None
_faiss_backend: Optional[FaissBackend] = None


def get_pgvector_backend() -> PgVectorBackend:
    """Get the pgvector backend instance."""
    global _pgvector_backend
    if _pgvector_backend is None:
        _pgvector_backend = PgVectorBackend()
    return _pgvector_backend


def get_faiss_backend() -> FaissBackend:
    """Get the FAISS backend instance."""
    global _faiss_backend
    if _faiss_backend is None:
        _faiss_backend = FaissBackend()
    return _faiss_backend


def get_backend(vector_store_type: VectorStoreType) -> VectorStoreBackend:
    """Get the appropriate backend for the given type."""
    if vector_store_type == VectorStoreType.FAISS:
        return get_faiss_backend()
    return get_pgvector_backend()
