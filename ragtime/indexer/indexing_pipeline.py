"""Bounded document-index staging pipeline shared by upload and git jobs."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from ragtime.indexer.chunking import chunk_spooled_batch, is_context_length_error, rechunk_documents_batch
from ragtime.indexer.embedding_errors import iter_exception_chain
from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput
from ragtime.indexer.vector_utils import embed_documents_subbatched

MAX_SOURCE_TEXT_BYTES = 16 * 1024 * 1024
DEFAULT_BATCH_TEXT_BYTES = 4 * 1024 * 1024


def _append_jsonl(path: Path, record: SpoolRecord) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


class BoundedIndexingPipeline:
    def __init__(self, spool: IndexingSpool, *, job_id: str, cancelled: Callable[[], bool]) -> None:
        self.spool, self.job_id, self.cancelled = spool, job_id, cancelled

    async def stage_documents(
        self,
        files: Iterable[Path],
        source_root: Path,
        index_name: str,
        load: Callable[[Path], Awaitable[list[Document]]] | None = None,
        progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> int:
        completed_ordinals = await asyncio.to_thread(self.spool.document_ordinals)
        count = 0
        for ordinal, path in enumerate(files):
            if self.cancelled():
                raise __import__("asyncio").CancelledError()
            if ordinal in completed_ordinals:
                if progress:
                    await progress(ordinal + 1)
                continue
            task_dir = self.spool.root / "tasks" / f"load-{uuid.uuid4().hex}"
            text_dir = task_dir / "text"
            await asyncio.to_thread(text_dir.mkdir, parents=True)
            manifest = task_dir / "documents.jsonl"
            await asyncio.to_thread(manifest.touch)
            source_count = source_bytes = 0
            docs = await load(path) if load else await asyncio.to_thread(TextLoader(str(path), autodetect_encoding=True).load)
            for document_ordinal, doc in enumerate(docs):
                text = doc.page_content
                text_bytes = len(text.encode("utf-8"))
                if text_bytes > MAX_SOURCE_TEXT_BYTES:
                    raise ValueError(f"Extracted text for {path} exceeds {MAX_SOURCE_TEXT_BYTES} byte limit")
                source = str(path.relative_to(source_root))
                metadata = dict(doc.metadata)
                metadata.update({"source": source, "index_name": index_name})
                record_id = IndexingSpool.stable_id(source, ordinal, document_ordinal, text)
                text_path = text_dir / f"{ordinal:012d}-{document_ordinal:04d}.txt"
                await asyncio.to_thread(self.spool.ensure_disk_space, text_bytes)
                await asyncio.to_thread(text_path.write_text, text, encoding="utf-8")
                record = SpoolRecord(record_id, source, ordinal, str(text_path.relative_to(self.spool.root)), metadata, text_bytes)
                await asyncio.to_thread(_append_jsonl, manifest, record)
                count += 1
                source_count += 1
                source_bytes += text_bytes
            # Each source is a journal checkpoint. A restart therefore resumes
            # only unfinished sources instead of replaying one huge load stage.
            if source_count:
                await asyncio.to_thread(self.spool.accept_documents, SpoolTaskOutput(str(manifest.relative_to(self.spool.root)), source_count, source_bytes, 0))
            if progress:
                await progress(ordinal + 1)
        return count

    async def stage_extra_documents(self, documents: Iterable[Document]) -> int:
        """Append bounded auxiliary documents (for example git history) to loading."""
        task_dir = self.spool.root / "tasks" / f"extra-{uuid.uuid4().hex}"
        text_dir = task_dir / "text"
        await asyncio.to_thread(text_dir.mkdir, parents=True)
        manifest = task_dir / "documents.jsonl"
        await asyncio.to_thread(manifest.touch)
        count = total = 0
        for ordinal, doc in enumerate(documents):
            if self.cancelled():
                raise asyncio.CancelledError()
            text = doc.page_content
            text_bytes = len(text.encode("utf-8"))
            if text_bytes > MAX_SOURCE_TEXT_BYTES:
                raise ValueError(f"Extra document exceeds {MAX_SOURCE_TEXT_BYTES} byte limit")
            metadata = dict(doc.metadata)
            source = str(metadata.get("source", f"extra/{ordinal}"))
            record_id = IndexingSpool.stable_id(source, ordinal, 0, text)
            path = text_dir / f"{ordinal:012d}.txt"
            await asyncio.to_thread(self.spool.ensure_disk_space, text_bytes)
            await asyncio.to_thread(path.write_text, text, encoding="utf-8")
            record = SpoolRecord(record_id, source, ordinal, str(path.relative_to(self.spool.root)), metadata, text_bytes)
            await asyncio.to_thread(_append_jsonl, manifest, record)
            count += 1
            total += text_bytes
        if count:
            await asyncio.to_thread(self.spool.accept_documents, SpoolTaskOutput(str(manifest.relative_to(self.spool.root)), count, total, 0))
        return count

    async def stage_chunks(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        use_tokens: bool,
        max_documents: int = 10,
        progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> int:
        total = 0
        # The governor admits each task, so submit a small bounded wave.  Output
        # is committed in manifest order, not completion order.
        limit = 4
        position = 0
        while True:
            wave = []
            for _ in range(limit):
                batch, position = await asyncio.to_thread(
                    self.spool.batch_after, "documents", position, max_documents=max_documents, max_text_bytes=DEFAULT_BATCH_TEXT_BYTES
                )
                if batch is None:
                    break
                wave.append(batch)
            if not wave:
                break
            if self.cancelled():
                raise __import__("asyncio").CancelledError()
            tasks = [
                asyncio.create_task(
                    chunk_spooled_batch(self.job_id, self.spool.root, batch, chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_tokens=use_tokens)
                )
                for batch in wave
            ]
            try:
                outputs = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for batch, output in zip(wave, outputs):
                await asyncio.to_thread(self.spool.accept_chunks, output)
                total += output.record_count
                if progress:
                    await progress(len(batch.records), total)
        return total

    async def stage_embeddings(
        self,
        embeddings: Any,
        *,
        max_documents: int,
        max_text_bytes: int,
        resource_job_id: str | None = None,
        progress: Callable[[int], Awaitable[None]] | None = None,
        safe_token_limit: int | None = None,
        chunk_overlap: int = 0,
    ) -> int:
        committed = (await asyncio.to_thread(self.spool.summary))["embedded_count"]
        position = 0
        while True:
            batch, position = await asyncio.to_thread(
                self.spool.unembedded_chunk_batch_after, position, max_documents=max_documents, max_text_bytes=max_text_bytes
            )
            if batch is None:
                break
            if self.cancelled():
                raise __import__("asyncio").CancelledError()
            records = list(batch.records)
            texts = await asyncio.to_thread(lambda: [self.spool._file(record.text_path).read_text(encoding="utf-8") for record in records])
            if safe_token_limit:
                _replacements, rechunked = await asyncio.to_thread(
                    rechunk_documents_batch,
                    [Document(page_content=text, metadata=dict(record.metadata)) for record, text in zip(records, texts)],
                    safe_token_limit,
                    chunk_overlap,
                    is_cancelled=self.cancelled,
                )
                if rechunked:
                    grouped: list[list[Document]] = []
                    for record, text in zip(records, texts):
                        per_record, _ = await asyncio.to_thread(
                            rechunk_documents_batch,
                            [Document(page_content=text, metadata=dict(record.metadata))],
                            safe_token_limit,
                            chunk_overlap,
                            is_cancelled=self.cancelled,
                        )
                        grouped.append(per_record)
                    await asyncio.to_thread(self.spool.replace_chunks, tuple(records), grouped)
                    position = 0
                    continue
            context_factors = (1.0, 0.70, 0.50, 0.35)
            context_attempt = 0
            rate_attempt = 0
            replaced_for_context = False
            while True:
                try:
                    vectors = await embed_documents_subbatched(embeddings, texts, resource_job_id=resource_job_id or self.job_id)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if is_context_length_error(exc) and safe_token_limit and context_attempt < len(context_factors) - 1:
                        context_attempt += 1
                        limit = max(1, int(safe_token_limit * context_factors[context_attempt]))
                        replacements: list[list[Document]] = []
                        for record, text in zip(records, texts):
                            rechunked_documents, _count = await asyncio.to_thread(
                                rechunk_documents_batch,
                                [Document(page_content=text, metadata=dict(record.metadata))],
                                limit,
                                chunk_overlap,
                                is_cancelled=self.cancelled,
                            )
                            replacements.append(rechunked_documents)
                        # Re-chunking is committed before retrying, so retries and
                        # final artifacts share IDs, metadata, and text exactly.
                        await asyncio.to_thread(self.spool.replace_chunks, tuple(records), replacements)
                        position = 0  # Re-batch replacement chunks under the same caps.
                        replaced_for_context = True
                        break
                    if rate_attempt < 5 and _is_rate_limit_error(exc):
                        rate_attempt += 1
                        await asyncio.sleep(min(8.0, 0.25 * (2 ** (rate_attempt - 1))))
                        if self.cancelled():
                            raise asyncio.CancelledError()
                        continue
                    raise
            if replaced_for_context:
                # A context failure replaced the batch; fetch it again under the
                # ordinary document/byte limits before making a provider call.
                continue
            if len(vectors) != len(records) or not vectors:
                raise ValueError("Embedding provider returned a vector count that does not match staged chunks")
            dimensions = len(vectors[0])
            if dimensions <= 0 or any(len(vector) != dimensions or not all(math.isfinite(float(v)) for v in vector) for vector in vectors):
                raise ValueError("Embedding provider returned invalid or misaligned vectors")
            path = self.spool.root / "vectors" / f"{uuid.uuid4().hex}.f32"
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(self.spool.ensure_disk_space, len(records) * dimensions * 4)
            await asyncio.to_thread(np.asarray(vectors, dtype="<f4").tofile, path)
            await asyncio.to_thread(
                self.spool.accept_embeddings, EmbeddedBatch(tuple(records), str(path.relative_to(self.spool.root)), 0, len(records), dimensions)
            )
            committed += len(records)
            if progress:
                await progress(committed)
        return committed


def _is_rate_limit_error(exc: Exception) -> bool:
    """Recognize provider throttle responses without retrying unrelated errors."""
    for current in iter_exception_chain(exc):
        status = getattr(current, "status_code", None) or getattr(current, "http_status", None)
        if status == 429:
            return True
        text = str(current).lower()
        if "rate limit" in text or "rate_limit_exceeded" in text or "429" in text:
            return True
    return False
