"""Shared helpers for pgvector and embedding model setup used by indexers."""

from typing import Any, Mapping, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

PGVECTOR_MAX_INDEX_DIM = 2000
_PGVECTOR_VALIDATED = False


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
