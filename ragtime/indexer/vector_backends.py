"""
Vector Store Backends - Unified interface for FAISS and pgvector storage.

Provides a common abstraction for storing and searching embeddings
across different vector store backends, allowing filesystem indexes
to use either FAISS (in-memory) or pgvector (PostgreSQL).
"""

import asyncio
import json
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.indexer.models import VectorStoreType
from ragtime.indexer.vector_utils import (
    FILESYSTEM_COLUMNS,
    search_pgvector_embeddings,
)
from ragtime.indexer.vector_utils import (
    ensure_embedding_column,
    ensure_pgvector_extension,
)
from ragtime.indexer.vector_utils import get_embeddings_model
from ragtime.indexer.vector_utils import get_pgvector_table_size_bytes

logger = get_logger(__name__)

# Base path for FAISS indexes (shared with document indexer - flat structure)
FAISS_INDEX_BASE_PATH = Path(settings.index_data_path)


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


class PgVectorBackend(VectorStoreBackend):
    """pgvector backend - stores embeddings in PostgreSQL."""

    def __init__(self):
        self._dimension_ensured: Dict[str, int] = {}

    async def _ensure_dimension(self, embedding_dim: int) -> None:
        """Ensure the embedding column has the correct dimension."""
        # Check if already ensured for this dimension
        if self._dimension_ensured.get("dim") == embedding_dim:
            return

        await ensure_pgvector_extension(logger_override=logger)

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
    ) -> int:
        """Store embeddings in pgvector table using batch inserts.

        Uses batch INSERT with multiple VALUES for better performance.
        Batches are sized to avoid exceeding PostgreSQL query limits.
        """
        if not embeddings:
            return 0

        # Ensure column dimension matches
        await self._ensure_dimension(len(embeddings[0]))

        db = await get_db()
        metadata_json = json.dumps(metadata).replace("'", "''")
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


class FaissBackend(VectorStoreBackend):
    """FAISS backend - stores embeddings in memory and saves to disk."""

    def __init__(self):
        # In-memory storage during indexing: {index_name: {"texts": [], "metadatas": [], "embeddings": []}}
        self._pending: Dict[str, Dict[str, List]] = {}
        # Loaded FAISS indexes: {index_name: FAISS vectorstore}
        self._loaded_indexes: Dict[str, Any] = {}

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

    async def store_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> int:
        """Buffer embeddings for later FAISS index creation."""
        if not embeddings:
            return 0

        if index_name not in self._pending:
            self._pending[index_name] = {"texts": [], "metadatas": [], "embeddings": []}

        pending = self._pending[index_name]
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            pending["texts"].append(chunk)
            pending["embeddings"].append(embedding)
            pending["metadatas"].append(
                {
                    **metadata,
                    "file_path": file_path,
                    "chunk_index": i,
                    "index_name": index_name,
                }
            )

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
        indices_to_keep = [
            i
            for i, m in enumerate(pending["metadatas"])
            if m.get("file_path") != file_path
        ]

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
        deleted = 0

        # Clear from pending
        if index_name in self._pending:
            deleted = len(self._pending[index_name]["texts"])
            del self._pending[index_name]
            logger.info(
                f"Cleared {deleted} pending embeddings for FAISS index: {index_name}"
            )

        # Clear from loaded indexes
        if index_name in self._loaded_indexes:
            del self._loaded_indexes[index_name]
            logger.info(f"Cleared loaded FAISS index from memory: {index_name}")

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
                logger.error(
                    f"Failed to delete FAISS index directory {index_path}: {e}"
                )
        else:
            logger.debug(
                f"FAISS index path does not exist, nothing to delete: {index_path}"
            )

        return deleted

    async def rename_index(self, old_name: str, new_name: str) -> bool:
        """Rename a FAISS index on disk and update in-memory tracking."""
        old_path = self._get_index_path(old_name)
        new_path = self._get_index_path(new_name)

        if not old_path.exists():
            logger.warning(f"FAISS index not found for rename: {old_name}")
            return False

        if new_path.exists():
            logger.error(f"FAISS index already exists at target: {new_name}")
            return False

        try:
            # Move on disk
            await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
            logger.info(f"Renamed FAISS index on disk: {old_name} -> {new_name}")

            # Update in-memory tracking
            if old_name in self._loaded_indexes:
                self._loaded_indexes[new_name] = self._loaded_indexes.pop(old_name)
                logger.info(
                    f"Updated in-memory FAISS tracking: {old_name} -> {new_name}"
                )

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
        results = []

        indexes_to_search = (
            {index_name: self._loaded_indexes[index_name]}
            if index_name and index_name in self._loaded_indexes
            else self._loaded_indexes
        )

        for idx_name, faiss_db in indexes_to_search.items():
            try:
                # Use similarity_search_with_score for FAISS
                docs_with_scores = await asyncio.to_thread(
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

        # Check pending buffer
        if index_name in self._pending:
            pending = self._pending[index_name]
            stats["chunk_count"] = len(pending["texts"])
            stats["embedding_count"] = len(pending["texts"])
            file_paths = set(m.get("file_path") for m in pending["metadatas"])
            stats["file_count"] = len(file_paths)
            return stats

        # Check loaded index
        if index_name in self._loaded_indexes:
            faiss_db = self._loaded_indexes[index_name]
            try:
                # Get document count from FAISS
                stats["chunk_count"] = faiss_db.index.ntotal
                stats["embedding_count"] = faiss_db.index.ntotal
                # Count unique file paths from docstore
                file_paths = set()
                for doc_id in faiss_db.index_to_docstore_id.values():
                    doc = faiss_db.docstore.search(doc_id)
                    if doc and hasattr(doc, "metadata"):
                        file_paths.add(doc.metadata.get("file_path", ""))
                stats["file_count"] = len(file_paths)
            except Exception as e:
                logger.debug(f"Error getting FAISS stats for {index_name}: {e}")

        # Calculate file size on disk
        index_path = self._get_index_path(index_name)
        if index_path.exists():
            total_size = 0
            faiss_file = index_path / "index.faiss"
            pkl_file = index_path / "index.pkl"
            if faiss_file.exists():
                total_size += faiss_file.stat().st_size
            if pkl_file.exists():
                total_size += pkl_file.stat().st_size
            stats["size_mb"] = total_size / (1024 * 1024)

        return stats

    async def finalize_index(self, index_name: str) -> None:
        """Build and save the FAISS index to disk."""
        if index_name not in self._pending:
            logger.warning(f"No pending data for FAISS index {index_name}")
            return

        pending = self._pending[index_name]
        if not pending["texts"]:
            logger.warning(f"No embeddings to save for FAISS index {index_name}")
            del self._pending[index_name]
            return

        logger.info(
            f"Finalizing FAISS index {index_name} with {len(pending['texts'])} chunks"
        )

        # Create FAISS index from embeddings
        # We already have embeddings, so we use from_embeddings
        text_embeddings = list(zip(pending["texts"], pending["embeddings"]))

        # Get embedding model for FAISS (needed for later searches)
        app_settings = await get_app_settings()
        embeddings_model = await get_embeddings_model(
            app_settings, return_none_on_error=True, logger_override=logger
        )

        if not embeddings_model:
            raise RuntimeError("Could not get embedding model for FAISS index creation")

        # Create FAISS index
        faiss_db = await asyncio.to_thread(
            FAISS.from_embeddings,
            text_embeddings,
            embeddings_model,
            metadatas=pending["metadatas"],
        )

        # Save to disk
        index_path = self._get_index_path(index_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(faiss_db.save_local, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Store in loaded indexes
        self._loaded_indexes[index_name] = faiss_db

        # Clear pending buffer
        del self._pending[index_name]

    async def load_index(self, index_name: str, embeddings_model: Any) -> bool:
        """Load a FAISS index from disk into memory."""
        index_path = self._get_index_path(index_name)

        if not index_path.exists():
            return False

        try:
            faiss_db = await asyncio.to_thread(
                FAISS.load_local,
                str(index_path),
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
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
            if path.is_dir() and (path / "index.faiss").exists():
                indexes.append(path.name)
        return indexes


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
