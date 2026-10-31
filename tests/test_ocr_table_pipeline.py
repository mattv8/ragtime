"""End-to-end regression coverage for structured OCR table chunking."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, mock

import pytest
from langchain_core.documents import Document

from ragtime.core.tokenization import count_tokens
from ragtime.indexer import chunking
from ragtime.indexer.indexing_pipeline import BoundedIndexingPipeline
from ragtime.indexer.indexing_spool import IndexingSpool, SpoolRecord, SpoolTaskOutput
from ragtime.indexer.resource_workers import ResourceTaskError


def _table(rows: int = 3) -> str:
    return "\n".join(
        [
            "| Item | Amount | Notes |",
            "| --- | --- | --- |",
            *[f"| Item {index} | {index} N | |" for index in range(rows)],
        ]
    )


def _table_metadata() -> dict[str, object]:
    return {"source": "scan.png", "page": 182, "segment_type": "table", "chunker": "semantic_table_rows"}


def test_rechunk_documents_batch_repeats_header_and_preserves_metadata() -> None:
    table = _table()
    header_and_row_tokens = count_tokens("\n".join(table.splitlines()[:3]))
    chunks, rechunked = chunking.rechunk_documents_batch(
        [Document(page_content=table, metadata=_table_metadata())],
        safe_token_limit=header_and_row_tokens,
    )

    assert rechunked == 1
    assert len(chunks) == 3
    assert all(document.page_content.startswith("| Item | Amount | Notes |\n| --- | --- | --- |\n") for document in chunks)
    assert all(document.metadata["page"] == 182 for document in chunks)
    assert all(document.metadata["segment_type"] == "table" for document in chunks)
    assert all(count_tokens(document.page_content) <= header_and_row_tokens for document in chunks)


def test_semantic_table_chunking_uses_token_budget_when_requested() -> None:
    table = _table()
    limit = count_tokens("\n".join(table.splitlines()[:3]))

    chunks = chunking.chunk_semantic_segments(
        [("table", table)],
        chunk_size=limit,
        chunk_overlap=0,
        metadata=_table_metadata(),
        use_tokens=True,
    )

    assert len(chunks) == 3
    assert all(count_tokens(document.page_content) <= limit for document in chunks)


def test_indivisible_table_row_has_source_and_row_number() -> None:
    table = "\n".join(["| Item |", "| --- |", "| " + "word " * 100 + "|"])

    with pytest.raises(ValueError, match=r"row 1.*too large.*scan.png"):
        chunking.rechunk_documents_batch(
            [Document(page_content=table, metadata=_table_metadata())],
            safe_token_limit=10,
        )


@pytest.mark.parametrize(
    ("table", "expected_prefix"),
    [
        ("Quarterly results\n| Item | Amount |\n| --- | --- |\n| A | 1 |\n| B | 2 |", "Quarterly results\n| Item | Amount |\n| --- | --- |"),
        ("Item\tAmount\tNotes\nA\t1\t\nB\t2\t", "| Item | Amount | Notes |\n| --- | --- | --- |"),
        ("| Item | Amount |\n| --- | --- |", "| Item | Amount |\n| --- | --- |"),
    ],
)
def test_semantic_tables_normalize_caption_tsv_and_header_only(table: str, expected_prefix: str) -> None:
    chunks = chunking.chunk_semantic_segments([("table", table)], 10_000, 0, _table_metadata())

    assert len(chunks) == 1
    assert chunks[0].page_content.startswith(expected_prefix)
    assert chunks[0].metadata["segment_type"] == "table"


def test_unparseable_table_text_stays_atomic_or_fails_with_source() -> None:
    content = "not a table\nbut OCR labeled it as one"
    chunks = chunking.chunk_semantic_segments([("table", content)], 10_000, 0, _table_metadata())

    assert [chunk.page_content for chunk in chunks] == [content]
    assert chunks[0].metadata["segment_type"] == "table_text"
    assert chunks[0].metadata["table_format"] == "unparsed"
    staged, _ = chunking._chunk_document_batch_sync([(content, chunks[0].metadata)], 10_000, 0, False)
    assert staged == [(content, chunks[0].metadata)]
    with pytest.raises(ValueError, match=r"Unparsed OCR table text.*scan.png"):
        chunking.chunk_semantic_segments([("table", content)], 10, 0, _table_metadata())


class TablePipelineTests(IsolatedAsyncioTestCase):
    async def test_stage_chunks_preserves_table_rows_in_normal_worker_path(self) -> None:
        with TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            table = _table()
            path = spool.root / "table.txt"
            path.write_text(table, encoding="utf-8")
            record = SpoolRecord("table", "scan.png", 0, path.name, _table_metadata(), len(table.encode()))
            manifest = spool.root / "documents.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_documents(SpoolTaskOutput(manifest.name, 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            limit = count_tokens("\n".join(table.splitlines()[:3]))

            async def run_resource_task(_request, function, args):
                return function(*args)

            with mock.patch.object(chunking, "run_resource_task", side_effect=run_resource_task):
                assert await pipeline.stage_chunks(chunk_size=limit, chunk_overlap=0, use_tokens=True) == 3

            staged = [record for batch in spool.iter_chunks(max_documents=10, max_text_bytes=100_000) for record in batch.records]
            contents = [(spool.root / record.text_path).read_text(encoding="utf-8") for record in staged]
            assert all(content.startswith("| Item | Amount | Notes |\n| --- | --- | --- |\n") for content in contents)
            assert all("| |" in content for content in contents)
            spool.close()

    async def test_stage_chunks_preserves_rows_after_resource_worker_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            table = _table()
            path = spool.root / "table.txt"
            path.write_text(table, encoding="utf-8")
            record = SpoolRecord("table", "scan.png", 0, path.name, _table_metadata(), len(table.encode()))
            manifest = spool.root / "documents.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_documents(SpoolTaskOutput(manifest.name, 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            limit = count_tokens("\n".join(table.splitlines()[:3]))
            calls = 0

            async def run_resource_task(_request, function, args):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ResourceTaskError("worker crashed")
                return function(*args)

            with mock.patch.object(chunking, "run_resource_task", side_effect=run_resource_task):
                assert await pipeline.stage_chunks(chunk_size=limit, chunk_overlap=0, use_tokens=True) == 3

            staged = [record for batch in spool.iter_chunks(max_documents=10, max_text_bytes=100_000) for record in batch.records]
            contents = [(spool.root / record.text_path).read_text(encoding="utf-8") for record in staged]
            assert all(content.startswith("| Item | Amount | Notes |\n| --- | --- | --- |\n") for content in contents)
            assert all(record.metadata["segment_type"] == "table" for record in staged)
            spool.close()

    async def test_stage_chunks_normalizes_tsv_before_worker_chunking(self) -> None:
        with TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            table = "Item\tAmount\nA\t1\nB\t2"
            path = spool.root / "table.txt"
            path.write_text(table, encoding="utf-8")
            record = SpoolRecord("table", "scan.png", 0, path.name, _table_metadata(), len(table.encode()))
            manifest = spool.root / "documents.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_documents(SpoolTaskOutput(manifest.name, 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)

            async def run_resource_task(_request, function, args):
                return function(*args)

            with mock.patch.object(chunking, "run_resource_task", side_effect=run_resource_task):
                assert await pipeline.stage_chunks(chunk_size=100, chunk_overlap=0, use_tokens=False) == 1

            staged = [record for batch in spool.iter_chunks(max_documents=10, max_text_bytes=100_000) for record in batch.records]
            assert (spool.root / staged[0].text_path).read_text(encoding="utf-8") == "| Item | Amount |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
            spool.close()

    async def test_stage_chunks_propagates_indivisible_table_row_without_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            table = "\n".join(["| Item |", "| --- |", "| " + "word " * 100 + "|"])
            path = spool.root / "table.txt"
            path.write_text(table, encoding="utf-8")
            record = SpoolRecord("table", "scan.png", 0, path.name, _table_metadata(), len(table.encode()))
            manifest = spool.root / "documents.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_documents(SpoolTaskOutput(manifest.name, 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)

            async def run_resource_task(_request, function, args):
                try:
                    return function(*args)
                except ValueError as exc:
                    raise ResourceTaskError(f"worker ValueError: {exc}") from exc

            with mock.patch.object(chunking, "run_resource_task", side_effect=run_resource_task):
                with self.assertRaisesRegex(ValueError, r"row 1.*scan.png"):
                    await pipeline.stage_chunks(chunk_size=10, chunk_overlap=0, use_tokens=True)

            spool.close()

    async def test_embedding_context_retry_rechunks_table_at_rows(self) -> None:
        with TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            table = _table(rows=6)
            path = spool.root / "table.txt"
            path.write_text(table, encoding="utf-8")
            record = SpoolRecord("table", "scan.png", 0, path.name, _table_metadata(), len(table.encode()))
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_chunks(SpoolTaskOutput(manifest.name, 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            safe_limit = count_tokens(table) + 1
            embeddings = mock.AsyncMock(side_effect=[RuntimeError("maximum context length"), [[1.0], [1.0]]])

            with mock.patch("ragtime.indexer.indexing_pipeline.embed_documents_subbatched", embeddings):
                assert await pipeline.stage_embeddings(object(), max_documents=10, max_text_bytes=100_000, safe_token_limit=safe_limit) == 2

            staged = [record for batch in spool.iter_chunks(max_documents=10, max_text_bytes=100_000) for record in batch.records]
            assert len(staged) == 2
            assert all((spool.root / record.text_path).read_text().startswith("| Item | Amount | Notes |\n| --- | --- | --- |\n") for record in staged)
            spool.close()
