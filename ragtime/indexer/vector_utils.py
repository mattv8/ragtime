"""Shared helpers for pgvector and embedding model setup used by indexers.

This module centralizes:
- pgvector extension and column management
- Embedding model factory (get_embeddings_model)
- pgvector similarity search (search_pgvector_embeddings)
- Embedding dimension warnings
"""

from typing import Any, Dict, List, Mapping, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

PGVECTOR_MAX_INDEX_DIM = 2000
_PGVECTOR_VALIDATED = False

# Default column sets for common embedding tables
# Base columns shared across all embedding tables
BASE_EMBEDDING_COLUMNS = [
    "id",
    "index_name",
    "content",
    "metadata",
]

FILESYSTEM_COLUMNS = [
    "id",
    "index_name",
    "file_path",
    "chunk_index",
    "content",
    "metadata",
]

PDM_COLUMNS = [
    "id",
    "index_name",
    "document_id",
    "document_type",
    "content",
    "part_number",
    "filename",
    "folder_path",
    "metadata",
]

# Schema columns - same structure as filesystem but with table-specific fields
SCHEMA_COLUMNS = [
    "id",
    "index_name",
    "table_name",
    "table_schema",
    "content",
    "metadata",
]


def _get_setting(settings: Any, key: str, default: Any = None) -> Any:
    """Retrieve a setting whether provided as mapping or attribute container."""
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)


async def ensure_pgvector_extension(logger_override=None) -> bool:
    """Ensure the pgvector extension exists (cached after first success)."""
    global _PGVECTOR_VALIDATED
    log = logger_override or logger

    if _PGVECTOR_VALIDATED:
        return True

    try:
        db = await get_db()
        result = await db.query_raw(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )

        if not result:
            log.info("pgvector extension not found, attempting to create...")
            await db.execute_raw("CREATE EXTENSION IF NOT EXISTS vector")

        _PGVECTOR_VALIDATED = True
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(
            "pgvector extension is unavailable. Ensure the database user can run CREATE EXTENSION vector: %s",
            exc,
        )
        return False


async def ensure_embedding_column(
    table_name: str,
    index_name: str,
    embedding_dim: int,
    *,
    logger_override=None,
    schema: str = "public",
    index_lists: int = 100,
    max_index_dim: int = PGVECTOR_MAX_INDEX_DIM,
) -> bool:
    """Ensure embedding column and index exist with the requested dimension."""
    log = logger_override or logger

    try:
        db = await get_db()
        result = await db.query_raw(
            f"""
            SELECT a.atttypmod AS typmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relname = '{table_name}'
              AND n.nspname = '{schema}'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
        """
        )

        current_dim: Optional[int] = None
        if result:
            typmod = result[0].get("typmod")
            if isinstance(typmod, int) and typmod > 0:
                current_dim = typmod

        dimension_changed = current_dim is not None and current_dim != embedding_dim

        if not result:
            log.info(
                f"Adding embedding column to {table_name} (vector({embedding_dim}))"
            )
            await db.execute_raw(
                f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding vector({embedding_dim})"
            )
        elif dimension_changed:
            log.warning(
                f"Embedding dimension changed for {table_name}: {current_dim} -> {embedding_dim}. "
                "Clearing existing embeddings and updating column."
            )
            # Must clear existing embeddings before changing dimension - vectors can't be cast
            await db.execute_raw(f"UPDATE {table_name} SET embedding = NULL")
            await db.execute_raw(f"DROP INDEX IF EXISTS {index_name}")
            await db.execute_raw(
                f"ALTER TABLE {table_name} ALTER COLUMN embedding TYPE vector({embedding_dim})"
            )

        if embedding_dim > max_index_dim:
            log.warning(
                f"Embedding dimension {embedding_dim} exceeds pgvector's {max_index_dim}-dim index limit. "
                "Using exact search; drop or reduce dimensions to enable IVFFlat indexing."
            )
            await db.execute_raw(f"DROP INDEX IF EXISTS {index_name}")
            return True

        index_info = await db.query_raw(
            f"""
            SELECT 1
            FROM pg_index i
            JOIN pg_class c ON i.indexrelid = c.oid
            WHERE c.relname = '{index_name}'
        """
        )

        if dimension_changed or not index_info:
            await db.execute_raw(f"DROP INDEX IF EXISTS {index_name}")
            log.info(
                f"Creating IVFFlat index for {table_name} embeddings with dimension {embedding_dim}"
            )
            await db.execute_raw(
                f"""
                CREATE INDEX {index_name}
                ON {table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {index_lists})
            """
            )

        return True

    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = f"Error ensuring embedding column for {table_name}: {exc}"
        log.error(error_msg)
        raise RuntimeError(error_msg) from exc


# =============================================================================
# Sub-batched Embedding (Event Loop Friendly)
# =============================================================================

# Maximum chunks per embedding API call to avoid blocking too long
EMBEDDING_SUB_BATCH_SIZE = 50


async def embed_documents_subbatched(
    embeddings,
    texts: List[str],
    *,
    sub_batch_size: int = EMBEDDING_SUB_BATCH_SIZE,
    logger_override=None,
) -> List[List[float]]:
    """Generate embeddings in sub-batches with event loop yields.

    This prevents long-running embedding calls from blocking the event loop,
    keeping the server responsive during large indexing jobs.

    Args:
        embeddings: The embedding model instance (OllamaEmbeddings, OpenAIEmbeddings, etc.)
        texts: List of text chunks to embed
        sub_batch_size: Number of chunks per API call (default 50)
        logger_override: Optional logger for sub-batch progress

    Returns:
        List of embedding vectors matching the input texts
    """
    import asyncio

    log = logger_override or logger

    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    total = len(texts)

    for batch_start in range(0, total, sub_batch_size):
        batch_end = min(batch_start + sub_batch_size, total)
        batch_texts = texts[batch_start:batch_end]

        # Generate embeddings for this sub-batch
        batch_embeddings = await asyncio.to_thread(
            embeddings.embed_documents, batch_texts
        )
        all_embeddings.extend(batch_embeddings)

        # Yield to event loop after each sub-batch to keep server responsive
        await asyncio.sleep(0)

        # Log progress for large batches
        if total > sub_batch_size * 2:
            log.debug(
                f"Embedded sub-batch {batch_start // sub_batch_size + 1}/"
                f"{(total + sub_batch_size - 1) // sub_batch_size} "
                f"({batch_end}/{total} chunks)"
            )

    return all_embeddings


async def get_embeddings_model(
    settings: Any,
    *,
    allow_missing_api_key: bool = False,
    return_none_on_error: bool = False,
    logger_override=None,
):
    """Build an embeddings model from settings (dict or settings object)."""
    log = logger_override or logger

    provider = (_get_setting(settings, "embedding_provider", "ollama") or "").lower()
    model = _get_setting(settings, "embedding_model", "nomic-embed-text")
    dimensions = _get_setting(settings, "embedding_dimensions")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        base_url = _get_setting(settings, "ollama_base_url", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = _get_setting(settings, "openai_api_key", "")
        if not api_key:
            message = "OpenAI embeddings selected but no API key configured in Settings"
            if allow_missing_api_key or return_none_on_error:
                log.warning(message)
                return None
            raise ValueError(message)

        kwargs = {"model": model, "api_key": api_key}
        if dimensions and str(model).startswith("text-embedding-3"):
            kwargs["dimensions"] = dimensions
            log.info(f"Using OpenAI embeddings with {dimensions} dimensions")
        return OpenAIEmbeddings(**kwargs)

    message = f"Unknown embedding provider: {provider}"
    if return_none_on_error:
        log.warning(message)
        return None
    raise ValueError(message)


# =============================================================================
# pgvector Similarity Search
# =============================================================================


async def search_pgvector_embeddings(
    table_name: str,
    query_embedding: List[float],
    *,
    index_name: Optional[str] = None,
    max_results: int = 10,
    columns: Optional[List[str]] = None,
    extra_where: Optional[str] = None,
    logger_override=None,
) -> List[Dict[str, Any]]:
    """
    Generic pgvector similarity search for any embeddings table.

    This is the canonical search implementation - all indexers should use this
    instead of duplicating the SQL pattern.

    Args:
        table_name: The embeddings table to search (e.g., 'filesystem_embeddings')
        query_embedding: The query vector
        index_name: Optional index name filter
        max_results: Maximum results to return
        columns: Columns to select (defaults vary by table, uses FILESYSTEM_COLUMNS)
        extra_where: Additional WHERE clause fragment (e.g., "AND document_type = 'SLDPRT'")
        logger_override: Optional logger to use

    Returns:
        List of dicts with selected columns plus 'similarity' score
    """
    log = logger_override or logger

    if columns is None:
        columns = FILESYSTEM_COLUMNS

    try:
        db = await get_db()

        # Build embedding vector string for SQL
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build WHERE clause components
        where_parts = []
        if index_name:
            # Escape single quotes to prevent SQL injection
            safe_index_name = index_name.replace("'", "''")
            where_parts.append(f"index_name = '{safe_index_name}'")
        if extra_where:
            where_parts.append(extra_where)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        # Build column list
        column_list = ", ".join(columns)

        # Execute similarity search using cosine distance
        query = f"""
            SELECT
                {column_list},
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM {table_name}
            {where_clause}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {max_results}
        """

        results = await db.query_raw(query)

        # Convert to list of dicts with proper types
        return [
            {
                **{col: row.get(col) for col in columns},
                "similarity": float(row.get("similarity", 0)),
            }
            for row in results
        ]

    except Exception as e:
        log.error(f"Error searching {table_name}: {e}")
        # Check if it's a pgvector not installed error
        if "vector" in str(e).lower() and "type" in str(e).lower():
            raise RuntimeError(
                "pgvector extension not installed. Run: CREATE EXTENSION IF NOT EXISTS vector;"
            ) from e
        raise


async def get_pgvector_table_size_bytes(
    table_name: str,
    index_name: str,
    logger_override=None,
) -> int:
    """
    Calculate the total size in bytes of rows in a pgvector table for a given index.

    Uses PostgreSQL's pg_column_size to calculate the actual storage size of the
    rows matching the index_name.

    Args:
        table_name: The pgvector table name (e.g., 'filesystem_embeddings')
        index_name: The index name to filter by
        logger_override: Optional logger to use

    Returns:
        Total size in bytes
    """
    log = logger_override or logger
    try:
        db = await get_db()
        safe_index_name = index_name.replace("'", "''")

        result = await db.query_raw(
            f"""
            SELECT COALESCE(SUM(pg_column_size(t.*)), 0)::bigint as total_bytes
            FROM {table_name} t
            WHERE index_name = '{safe_index_name}'
        """
        )

        if result and result[0]["total_bytes"]:
            return int(result[0]["total_bytes"])
        return 0
    except Exception as e:
        log.error(f"Error calculating size for {table_name} index {index_name}: {e}")
        return 0


# =============================================================================
# Embedding Dimension Helpers
# =============================================================================


def get_embedding_dimension_warning(
    dimension: Optional[int],
    provider: str = "",
    model: str = "",
    limit: int = PGVECTOR_MAX_INDEX_DIM,
) -> Optional[str]:
    """
    Check if embedding dimension exceeds pgvector's index limit.

    Args:
        dimension: Known embedding dimension (configured or tracked)
        provider: Embedding provider name (for heuristic defaults)
        model: Embedding model name (for heuristic defaults)
        limit: Maximum dimension for IVFFlat indexing (default 2000)

    Returns:
        Warning message if dimension exceeds limit, None otherwise
    """
    effective_dim = dimension

    # Heuristic defaults for OpenAI text-embedding-3 models when dimension not set
    if effective_dim is None and provider.lower() == "openai":
        if model.startswith("text-embedding-3-large"):
            effective_dim = 3072
        elif model.startswith("text-embedding-3-small"):
            effective_dim = 1536

    if effective_dim is not None and effective_dim > limit:
        return (
            f"Embedding dimension {effective_dim} exceeds pgvector's {limit}-dim index limit. "
            "Search will fall back to exact (non-indexed). "
            "Use a <=2000-dim model (e.g., set Embedding Dimensions to 1536 for OpenAI text-embedding-3-*) "
            "or choose a lower-dimension embedding model."
        )

    return None


async def append_embedding_dimension_warning(
    warnings: List[str],
    settings: Any = None,
    logger_override=None,
) -> None:
    """
    Append embedding dimension warning to warnings list if applicable.

    This is the canonical implementation - indexers should use this instead of
    duplicating the warning logic.

    Args:
        warnings: List to append warning to (modified in place)
        settings: Optional settings object (will load from repository if None)
        logger_override: Optional logger to use
    """
    log = logger_override or logger

    try:
        if settings is None:
            from ragtime.indexer.repository import IndexerRepository

            repo = IndexerRepository()
            settings = await repo.get_settings()

        provider = _get_setting(settings, "embedding_provider", "ollama")
        model = _get_setting(settings, "embedding_model", "nomic-embed-text")
        configured_dim = _get_setting(settings, "embedding_dimensions")
        tracked_dim = _get_setting(settings, "embedding_dimension")

        # Prefer explicit configuration, otherwise use tracked dimension
        dimension = configured_dim or tracked_dim

        warning = get_embedding_dimension_warning(dimension, provider, model)
        if warning:
            warnings.insert(0, warning)

    except Exception as e:
        log.debug(f"Could not check embedding dimension warning: {e}")
