"""Lightweight helpers for deriving FAISS docstore statistics."""

from collections.abc import Mapping
from typing import Any

# Document-archive and git indexers persist ``source``; the filesystem indexer
# persists ``file_path``. Both identify the source file represented by chunks.
FAISS_SOURCE_METADATA_KEYS: tuple[str, ...] = ("source", "file_path")


def _docstore_source(metadata: Any) -> str:
    """Return the first non-empty source-identifying value in metadata."""
    if not isinstance(metadata, Mapping):
        return ""
    for key in FAISS_SOURCE_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def count_faiss_docstore_stats(faiss_pickle_data: Any) -> tuple[int, int]:
    """Return unique source-document and total chunk counts for a FAISS payload."""
    docstore_dict: Mapping[Any, Any] | None = None
    idx_to_id: Mapping[Any, Any] | None = None

    if isinstance(faiss_pickle_data, tuple) and len(faiss_pickle_data) >= 2:
        docstore, idx_to_id_candidate = faiss_pickle_data[0], faiss_pickle_data[1]
        inner = getattr(docstore, "_dict", None)
        if isinstance(inner, Mapping):
            docstore_dict = inner
        if isinstance(idx_to_id_candidate, Mapping):
            idx_to_id = idx_to_id_candidate
    elif isinstance(faiss_pickle_data, Mapping):
        docstore_dict = faiss_pickle_data

    if docstore_dict is not None:
        chunk_count = len(docstore_dict)
        sources = {_docstore_source(getattr(doc, "metadata", None)) for doc in docstore_dict.values()}
        sources.discard("")
        return (len(sources), chunk_count) if sources else (chunk_count, chunk_count)

    if idx_to_id is not None:
        count = len(idx_to_id)
        return count, count

    return 0, 0
