"""Build immutable, disk-backed FAISS document-index artifacts.

This module deliberately prepares an artifact only.  Publishing its path in
``index_metadata`` and swapping the live retriever are separate operations so
an incomplete generation can never become authoritative.
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragtime.indexer.indexing_spool import IndexingSpool
from ragtime.indexer.memory_utils import estimate_index_memory
from ragtime.indexer.resource_governor import resource_governor
from ragtime.indexer.resource_workers import run_resource_task

_FINALIZATION_COLD_START_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PreparedFaissArtifact:
    generation_path: Path
    document_count: int
    chunk_count: int
    dimensions: int
    size_bytes: int


class _ArtifactEmbeddings(Embeddings):
    """Only used for the builder's required LangChain load round-trip."""

    def embed_documents(self, _: list[str]) -> list[list[float]]:
        raise RuntimeError("Artifact validation does not embed documents")

    def embed_query(self, _: str) -> list[float]:
        raise RuntimeError("Artifact validation does not embed queries")


def resolve_document_artifact_path(index_root: Path, metadata_path: str | None) -> Path:
    """Resolve a legacy root or completed generation without path escape fallback."""
    root = index_root.resolve()
    if not metadata_path:
        return root

    candidate = Path(metadata_path).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Document artifact path escapes index root: {candidate}") from exc

    # Only a legacy root or an exact child of .generations is serving-valid.
    if relative == Path("."):
        return root
    if len(relative.parts) != 2 or relative.parts[0] != ".generations":
        raise ValueError(f"Document artifact path is not a generation: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Referenced FAISS generation is missing: {candidate}")
    if not (candidate / "index.faiss").is_file() or not (candidate / "index.pkl").is_file():
        raise FileNotFoundError(f"Referenced FAISS generation is incomplete: {candidate}")
    return candidate


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        value = getattr(record, name, None)
        if value is not None:
            return value
    return default


def _spool_file(spool_root: Path, value: str) -> Path:
    """Resolve a spool-relative file while rejecting traversal and symlinks."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Spool paths must be attempt-relative")
    path = spool_root / relative
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(spool_root.resolve())
    except ValueError as exc:
        raise ValueError("Spool path escapes its attempt through a symlink") from exc
    except FileNotFoundError:
        raise FileNotFoundError(f"Spool file is missing: {relative}") from None
    if not resolved.is_file():
        raise ValueError(f"Spool path is not a file: {relative}")
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_document(spool_root: Path, record: Any) -> tuple[str, dict[str, Any], str]:
    metadata = dict(_value(record, "metadata", default={}) or {})
    text_path = _value(record, "text_path")
    if text_path is not None:
        text = _spool_file(spool_root, str(text_path)).read_text(encoding="utf-8")
    else:
        text = _value(record, "text", "page_content", "content")
    if text is None:
        raise ValueError("Spool embedding record has no text path")
    identifier = _value(record, "chunk_id", "record_id", "id") or metadata.get("chunk_id") or metadata.get("id")
    if not identifier:
        raise ValueError("Spool embedding record has no stable chunk id")
    return str(text), metadata, str(identifier)


def _read_vectors(path: Path, offset: int, rows: int, dimensions: int) -> np.ndarray:
    expected = rows * dimensions * np.dtype("<f4").itemsize
    with path.open("rb") as vector_file:
        vector_file.seek(offset)
        payload = vector_file.read(expected)
    if len(payload) != expected:
        raise ValueError(f"Truncated vector data in {path}")
    vectors = np.frombuffer(payload, dtype="<f4").reshape(rows, dimensions)
    if not np.isfinite(vectors).all():
        raise ValueError("Spool contains non-finite embedding values")
    return vectors


def _build_faiss_generation(
    index_root: Path,
    spool_root: Path,
    metric: str,
    normalize_l2: bool,
) -> PreparedFaissArtifact:
    """Blocking build entrypoint run by the supervised resource worker."""
    import faiss  # inline-import: keep; keep FAISS native loading in the supervised worker process.

    spool = IndexingSpool.open_existing(spool_root, readonly=True)
    try:
        summary = spool.summary()
        chunk_count = int(summary.get("chunk_count", 0))
        dimensions = int(summary.get("dimensions", 0))
        if chunk_count <= 0 or dimensions <= 0:
            raise ValueError("Cannot build a FAISS artifact without staged embeddings")

        # save_local writes a pickle and native index before its staging
        # directory can replace a generation. Reserve enough for the completed
        # pair plus the simultaneously-present staging output; this uses the
        # spool's common reserve policy rather than a second disk threshold.
        artifact_estimate = estimate_index_memory(
            chunk_count,
            dimensions,
            max(1, int(summary.get("text_bytes", 0)) // chunk_count),
        )
        predicted_artifact_bytes = int(artifact_estimate["steady_memory_bytes"])
        IndexingSpool.ensure_path_disk_space(index_root, predicted_artifact_bytes * 2)

        # The estimate is intentionally computed from the staged corpus before
        # any native allocation. Admission is owned by run_resource_task.
        estimate_index_memory(chunk_count, dimensions, max(1, int(summary.get("text_bytes", 0)) // chunk_count))
        if metric not in {"l2", "ip"}:
            raise ValueError(f"Unsupported FAISS metric: {metric}")
        index = faiss.IndexFlatIP(dimensions) if metric == "ip" else faiss.IndexFlatL2(dimensions)
        docstore = InMemoryDocstore({})
        index_to_docstore_id: dict[int, str] = {}

        row = 0
        for batch in spool.iter_embeddings(max_rows=256):
            records = tuple(batch.records)
            rows = int(batch.rows)
            batch_dimensions = int(batch.dimensions)
            if rows != len(records) or batch_dimensions != dimensions:
                raise ValueError("Spool embedding batch rows or dimensions do not match its manifest")
            vector_path = _spool_file(spool_root, batch.vectors_path)
            vectors = _read_vectors(vector_path, int(batch.vector_offset_bytes), rows, dimensions)
            if normalize_l2:
                # np.frombuffer() intentionally avoids a corpus copy but is
                # read-only; only normalized metrics need a bounded batch copy.
                vectors = vectors.copy()
                faiss.normalize_L2(vectors)
            index.add(vectors)
            for record in records:
                text, metadata, identifier = _record_document(spool_root, record)
                if identifier in docstore._dict:
                    raise ValueError(f"Duplicate stable chunk id in spool: {identifier}")
                docstore._dict[identifier] = Document(page_content=text, metadata=metadata)
                index_to_docstore_id[row] = identifier
                row += 1
        if row != chunk_count or index.ntotal != chunk_count:
            raise ValueError("FAISS/document mapping count does not match spool summary")

        generation_id = uuid.uuid4().hex
        generation_parent = index_root / ".generations"
        staging = generation_parent / f".{generation_id}.building"
        completed = generation_parent / generation_id
        generation_parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            store = FAISS(_ArtifactEmbeddings(), index, docstore, index_to_docstore_id, normalize_L2=normalize_l2)
            store.save_local(str(staging))
            for path in staging.iterdir():
                if path.is_file():
                    with path.open("rb") as artifact_file:
                        os.fsync(artifact_file.fileno())
            _fsync_directory(staging)
            # Do not retain the build graph while pickle deserialization builds
            # its own object graph for round-trip validation.
            del store, index, docstore, index_to_docstore_id
            gc.collect()
            # A standard LangChain reload catches mismatched pickle/index pairs
            # before the immutable directory is visible to metadata publication.
            loaded = FAISS.load_local(str(staging), _ArtifactEmbeddings(), allow_dangerous_deserialization=True)
            if loaded.index.ntotal != chunk_count or loaded.index.d != dimensions:
                raise ValueError("FAISS generation round-trip validation failed")
            del loaded
            gc.collect()
            os.replace(staging, completed)
            _fsync_directory(generation_parent)
        except BaseException:
            # Staging is never published, and removing it cannot affect an
            # already-published generation from a prior successful attempt.
            shutil.rmtree(staging, ignore_errors=True)
            raise

        size_bytes = sum(path.stat().st_size for path in completed.iterdir() if path.is_file())
        return PreparedFaissArtifact(
            generation_path=completed,
            document_count=int(summary.get("document_count", 0)),
            chunk_count=chunk_count,
            dimensions=dimensions,
            size_bytes=size_bytes,
        )
    finally:
        spool.close()


async def prepare_faiss_artifact(
    job_id: str,
    index_root: Path,
    spool_root: Path,
    *,
    metric: str = "l2",
    normalize_l2: bool = False,
) -> PreparedFaissArtifact:
    """Build and validate a generation; callers publish it transactionally."""

    def read_summary(attempt_root: Path) -> dict[str, int]:
        spool = IndexingSpool.open_existing(attempt_root, readonly=True)
        try:
            return spool.summary()
        finally:
            spool.close()

    # C has committed/checkpointed its spool before calling us. SQLite opening
    # and summary reads remain off the API event loop.
    summary = await asyncio.to_thread(read_summary, Path(spool_root))
    dimensions = int(summary.get("dimensions", 0))
    chunks = int(summary.get("chunk_count", 0))
    text_bytes = int(summary.get("text_bytes", 0))
    steady = max(
        _FINALIZATION_COLD_START_BYTES,
        estimate_index_memory(chunks, dimensions, max(1, text_bytes // max(chunks, 1)))["steady_memory_bytes"],
    )
    request = resource_governor.estimate_request(
        job_id=job_id,
        stage="finalizing",
        record_count=chunks,
        dimensions=dimensions,
        text_bytes=text_bytes,
        steady_bytes=steady,
        kind="document_job",
    )
    return await run_resource_task(
        request,
        _build_faiss_generation,
        (Path(index_root), Path(spool_root), metric, normalize_l2),
    )
