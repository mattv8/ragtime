import struct
import tempfile
import unittest
from pathlib import Path

from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput


class IndexingSpoolTests(unittest.TestCase):
    def test_matching_fingerprint_reopens_attempt_and_persists_stage_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = IndexingSpool.open_attempt(parent, "job", "fingerprint")
            first.set_state("cursor", {"ordinal": 2})
            first.mark_stage_complete("loading")
            root = first.root
            first.close()

            resumed = IndexingSpool.open_attempt(parent, "job", "fingerprint")
            self.assertEqual(resumed.root, root)
            self.assertEqual(resumed.get_state("cursor"), {"ordinal": 2})
            self.assertTrue(resumed.is_stage_complete("loading"))
            resumed.close()

    def test_accepting_same_manifest_twice_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            text = spool.root / "text.txt"
            text.write_text("x")
            record = SpoolRecord("id", "source.txt", 0, "text.txt", {"source": "source.txt"}, 1)
            manifest = spool.root / "documents.jsonl"
            manifest.write_text(__import__("json").dumps(record.__dict__) + "\n")
            output = SpoolTaskOutput("documents.jsonl", 1, 1, 0)
            spool.accept_documents(output)
            spool.accept_documents(output)
            self.assertEqual(spool.summary()["document_count"], 1)
            spool.close()

    def test_duplicate_replay_accepts_identical_content_at_a_new_task_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = IndexingSpool.open_attempt(Path(directory), "job", "fingerprint")
            first_path = spool.root / "tasks/first/text.txt"
            first_path.parent.mkdir(parents=True)
            first_path.write_text("same")
            first = SpoolRecord("id", "source.txt", 0, "tasks/first/text.txt", {"source": "source.txt"}, 4)
            first_manifest = spool.root / "first.jsonl"
            first_manifest.write_text(__import__("json").dumps(first.__dict__) + "\n")
            spool.accept_documents(SpoolTaskOutput("first.jsonl", 1, 4, 0))
            replay_path = spool.root / "tasks/replay/text.txt"
            replay_path.parent.mkdir(parents=True)
            replay_path.write_text("same")
            replay = SpoolRecord("id", "source.txt", 0, "tasks/replay/text.txt", {"source": "source.txt"}, 4)
            replay_manifest = spool.root / "replay.jsonl"
            replay_manifest.write_text(__import__("json").dumps(replay.__dict__) + "\n")
            spool.accept_documents(SpoolTaskOutput("replay.jsonl", 1, 4, 0))
            self.assertEqual(spool.summary()["document_count"], 1)
            spool.close()

    def test_changed_fingerprint_creates_a_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = IndexingSpool.open_attempt(parent, "job", "first")
            root = first.root
            first.close()
            changed = IndexingSpool.open_attempt(parent, "job", "second")
            self.assertNotEqual(changed.root, root)
            changed.close()

    def test_embedding_slices_keep_absolute_offset_and_record_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            spool = IndexingSpool.open_attempt(parent, "job", "fingerprint")
            records = tuple(SpoolRecord(f"id-{i}", "source.txt", i, f"texts/{i}.txt", {"source": "source.txt"}, 1) for i in range(3))
            for record in records:
                path = spool.root / record.text_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x")
            manifest = spool.root / "chunks.jsonl"
            manifest.write_text("\n".join(__import__("json").dumps(record.__dict__) for record in records) + "\n")
            spool.accept_chunks(SpoolTaskOutput("chunks.jsonl", 3, 3, 0))
            vectors = spool.root / "vectors.bin"
            vectors.write_bytes(struct.pack("<6f", 1, 2, 3, 4, 5, 6))
            spool.accept_embeddings(EmbeddedBatch(records, "vectors.bin", 0, 3, 2))
            batches = list(spool.iter_embeddings(max_rows=2))
            self.assertEqual([batch.rows for batch in batches], [2, 1])
            self.assertEqual(batches[1].vector_offset_bytes, 16)
            self.assertEqual([r.record_id for r in batches[1].records], ["id-2"])
            spool.close()
