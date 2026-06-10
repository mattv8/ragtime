"""Indexer module for creating FAISS indexes."""

from typing import TYPE_CHECKING

from ragtime.indexer.models import IndexConfig, IndexJob, IndexStatus

if TYPE_CHECKING:
    from ragtime.indexer.service import IndexerService

__all__ = ["IndexerService", "IndexJob", "IndexStatus", "IndexConfig"]


def __getattr__(name: str):
    if name == "IndexerService":
        from ragtime.indexer.service import IndexerService

        return IndexerService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
