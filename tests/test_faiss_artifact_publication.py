import json
import struct
import tempfile
import unittest
from pathlib import Path

from langchain_community.vectorstores import FAISS

from ragtime.indexer.faiss_artifacts import (
    _ArtifactEmbeddings,
    _build_faiss_generation,
    resolve_document_artifact_path,
)
from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput


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
