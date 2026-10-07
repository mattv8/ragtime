import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_community.vectorstores import FAISS

from ragtime.indexer.faiss_artifacts import (
    _ArtifactEmbeddings,
    _build_faiss_generation,
    prepare_faiss_artifact,
    resolve_document_artifact_path,
)
from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput
from ragtime.indexer.memory_utils import FINALIZATION_WORKER_BASELINE_BYTES, estimate_faiss_finalization_memory
from ragtime.indexer.resource_governor import IndexingResourceGovernor


class FaissArtifactPublicationTests(unittest.TestCase):
    def test_resolver_accepts_legacy_and_completed_generation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "index"
            root.mkdir()
            generation = root / ".generations" / "one"
            generation.mkdir(parents=True)
            (generation / "index.faiss").touch()
            (generation / "index.pkl").touch()
            self.assertEqual(resolve_document_artifact_path(root, None), root.resolve())
            self.assertEqual(resolve_document_artifact_path(root, str(generation)), generation.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_document_artifact_path(root, str(root / ".generations" / "missing"))
            with self.assertRaises(ValueError):
                resolve_document_artifact_path(root, str(Path(directory) / "escape"))

    def test_builder_writes_standard_langchain_generation_from_offset_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "index"
            spool_parent = Path(directory) / "spools"
            spool = IndexingSpool.open_attempt(spool_parent, "job", "fingerprint")
            records = (
                SpoolRecord("chunk-a", "a", 0, "texts/a.txt", {"source": "a"}, 5),
                SpoolRecord("chunk-b", "b", 1, "texts/b.txt", {"source": "b"}, 6),
            )
            for record, text in zip(records, ("first", "second")):
                path = spool.root / record.text_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text("".join(json.dumps(record.__dict__) + "\n" for record in records), encoding="utf-8")
            output = SpoolTaskOutput("chunks.jsonl", 2, 11, 0)
            spool.accept_documents(output)
            spool.accept_chunks(output)
            vectors = spool.root / "vectors.bin"
            # Include a leading row so this verifies absolute byte offsets.
            vectors.write_bytes(struct.pack("<6f", 1, 1, 1, 2, 3, 4))
            spool.accept_embeddings(EmbeddedBatch(records, "vectors.bin", 8, 2, 2))
            spool_root = spool.root
            spool.close()
            artifact = _build_faiss_generation(root, spool_root, "l2", False)

            self.assertEqual(artifact.chunk_count, 2)
            self.assertTrue((artifact.generation_path / "index.faiss").is_file())
            loaded = FAISS.load_local(str(artifact.generation_path), _ArtifactEmbeddings(), allow_dangerous_deserialization=True)
            self.assertEqual(loaded.index.ntotal, 2)
            self.assertEqual(loaded.index_to_docstore_id[0], "chunk-a")
            estimate = estimate_faiss_finalization_memory(
                chunk_count=2,
                dimensions=2,
                text_bytes=11,
                metadata_bytes=sum(len(json.dumps(record.metadata, sort_keys=True).encode("utf-8")) for record in records),
                identifier_bytes=sum(len(record.record_id.encode("utf-8")) for record in records),
            )
            self.assertLessEqual(artifact.size_bytes, estimate["artifact_bytes"])

    def test_prepare_uses_finalization_envelope_without_legacy_sizing_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = IndexingSpool.open_attempt(root / "spools", "job", "fingerprint")
            record = SpoolRecord("chunk-id", "source", 0, "text.txt", {"source": "文"}, 1)
            (spool.root / record.text_path).write_text("x", encoding="utf-8")
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text(json.dumps(record.__dict__) + "\n", encoding="utf-8")
            spool.accept_chunks(SpoolTaskOutput(manifest.name, 1, 1, 0))
            spool_root = spool.root
            spool.close()
            governor = IndexingResourceGovernor()

            async def run() -> None:
                with patch.object(governor, "estimate_request", wraps=governor.estimate_request) as estimate_request:
                    with (
                        patch("ragtime.indexer.faiss_artifacts.resource_governor", governor),
                        patch("ragtime.indexer.faiss_artifacts.run_resource_task", return_value=None) as run_resource_task,
                    ):
                        await prepare_faiss_artifact("job", root / "index", spool_root)
                self.assertEqual(set(estimate_request.call_args.kwargs), {"job_id", "stage", "minimum_peak_bytes", "kind"})
                request = run_resource_task.call_args.args[0]
                self.assertGreater(request.estimated_peak_bytes, FINALIZATION_WORKER_BASELINE_BYTES)
                self.assertLess(request.estimated_peak_bytes, 966_367_641)

            asyncio.run(run())
