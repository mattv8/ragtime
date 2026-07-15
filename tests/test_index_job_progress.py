import unittest
from typing import Any

from ragtime.indexer.models import IndexConfig, IndexJob, IndexJobPhase


class IndexJobProgressTests(unittest.TestCase):
    def _job(self, **overrides: Any) -> IndexJob:
        job = IndexJob(
            id="job-1",
            name="ragtime",
            config=IndexConfig(name="ragtime"),
            source_type="git",
        )
        return job.model_copy(update=overrides)

    def test_chunking_phase_maps_document_progress_to_thirty_through_fifty(self) -> None:
        job = self._job(
            phase=IndexJobPhase.CHUNKING,
            total_files=528,
            processed_files=528,
            total_chunks=1573,
            processed_chunks=300,
        )

        self.assertEqual(job.phase, IndexJobPhase.CHUNKING)
        self.assertAlmostEqual(job.progress_percent, 30 + (300 / 1573) * 20)

    def test_embedding_starts_at_fifty_after_chunk_counter_reset(self) -> None:
        job = self._job(
            phase=IndexJobPhase.EMBEDDING,
            total_files=528,
            processed_files=528,
            total_chunks=4200,
            processed_chunks=0,
        )

        self.assertEqual(job.progress_percent, 50.0)

    def test_phase_progress_boundaries_are_authoritative(self) -> None:
        self.assertEqual(self._job(phase=IndexJobPhase.PREPARING).progress_percent, 0.0)
        self.assertEqual(self._job(phase=IndexJobPhase.CLONING).progress_percent, 0.0)
        self.assertEqual(self._job(phase=IndexJobPhase.SCANNING).progress_percent, 0.0)
        self.assertEqual(self._job(phase=IndexJobPhase.FINALIZING).progress_percent, 99.0)
        self.assertEqual(self._job(phase=IndexJobPhase.COMPLETED).progress_percent, 100.0)

    def test_error_message_does_not_change_phase(self) -> None:
        job = self._job(
            phase=IndexJobPhase.EMBEDDING,
            error_message="Chunking documents...",
            total_chunks=100,
            processed_chunks=20,
        )

        self.assertEqual(job.phase, IndexJobPhase.EMBEDDING)
        self.assertAlmostEqual(job.progress_percent, 59.8)
