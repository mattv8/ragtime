import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document

from ragtime.indexer.indexing_pipeline import BoundedIndexingPipeline
from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput


class IndexingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_context_error_rechunks_and_replaces_staged_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fp")
            text = spool.root / "chunk.txt"
            text.write_text("very long text")
            record = SpoolRecord("chunk", "source", 0, "chunk.txt", {"source": "source"}, len("very long text"))
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text(__import__("json").dumps(record.__dict__) + "\n")
            spool.accept_chunks(SpoolTaskOutput("chunks.jsonl", 1, record.text_bytes, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            embeddings = mock.AsyncMock(side_effect=[RuntimeError("maximum context length"), [[1.0, 2.0]]])
            with mock.patch(
                "ragtime.indexer.indexing_pipeline.rechunk_documents_batch",
                side_effect=[
                    ([Document(page_content="very long text", metadata={"source": "source"})], 0),
                    ([Document(page_content="short", metadata={"source": "source"})], 1),
                    ([Document(page_content="short", metadata={"source": "source"})], 0),
                ],
            ):
                with mock.patch("ragtime.indexer.indexing_pipeline.embed_documents_subbatched", embeddings):
                    self.assertEqual(await pipeline.stage_embeddings(object(), max_documents=10, max_text_bytes=100, safe_token_limit=8), 1)
            staged = [r for batch in spool.iter_chunks(max_documents=10, max_text_bytes=100) for r in batch.records]
            self.assertNotEqual(staged[0].record_id, "chunk")
            self.assertEqual((spool.root / staged[0].text_path).read_text(), "short")
            spool.close()

    async def test_embedding_progress_includes_already_committed_rows_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fp")
            records = []
            for index, text in enumerate(("first", "second")):
                path = spool.root / f"{index}.txt"
                path.write_text(text)
                records.append(SpoolRecord(f"id-{index}", "source", index, path.name, {"source": "source"}, len(text)))
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text("\n".join(__import__("json").dumps(record.__dict__) for record in records))
            spool.accept_chunks(SpoolTaskOutput("chunks.jsonl", 2, sum(record.text_bytes for record in records), 0))
            vectors = spool.root / "vectors.f32"
            vectors.write_bytes(__import__("struct").pack("<2f", 1, 2))
            spool.accept_embeddings(EmbeddedBatch((records[0],), "vectors.f32", 0, 1, 2))
            progress = mock.AsyncMock()
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            with mock.patch("ragtime.indexer.indexing_pipeline.embed_documents_subbatched", new=mock.AsyncMock(return_value=[[3.0, 4.0]])):
                self.assertEqual(await pipeline.stage_embeddings(object(), max_documents=1, max_text_bytes=100, progress=progress), 2)
            progress.assert_awaited_once_with(2)
            spool.close()

    async def test_embedding_rejects_provider_vector_count_mismatch_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fp")
            text = spool.root / "chunk.txt"
            text.write_text("content")
            manifest = spool.root / "chunks.jsonl"
            record = SpoolRecord("chunk", "source", 0, "chunk.txt", {"source": "source"}, 7)
            manifest.write_text(__import__("json").dumps(record.__dict__) + "\n")
            spool.accept_chunks(SpoolTaskOutput("chunks.jsonl", 1, 7, 0))
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)
            with mock.patch("ragtime.indexer.indexing_pipeline.embed_documents_subbatched", new=mock.AsyncMock(return_value=[])):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    await pipeline.stage_embeddings(object(), max_documents=10, max_text_bytes=100)
            self.assertEqual(spool.summary()["chunk_count"], 1)
            self.assertEqual(spool.summary()["dimensions"], 0)
            spool.close()

    async def test_document_stage_orders_source_records_without_retaining_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            first, second = source / "a.txt", source / "b.txt"
            first.write_text("a")
            second.write_text("b")
            spool = IndexingSpool.open_attempt(root, "job", "fp")
            pipeline = BoundedIndexingPipeline(spool, job_id="job", cancelled=lambda: False)

            async def load(path):
                return [Document(page_content=path.read_text())]

            await pipeline.stage_documents([first, second], source, "index", load)
            self.assertEqual(
                [record.source for batch in spool.iter_documents(max_documents=10, max_text_bytes=100) for record in batch.records], ["a.txt", "b.txt"]
            )
            spool.close()
