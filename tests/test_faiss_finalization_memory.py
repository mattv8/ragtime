import unittest

from ragtime.indexer.memory_utils import estimate_faiss_finalization_memory


class FaissFinalizationMemoryTests(unittest.TestCase):
    def test_metadata_identifier_and_dimensions_growth_each_increase_the_envelope(self) -> None:
        baseline = estimate_faiss_finalization_memory(
            chunk_count=10,
            dimensions=8,
            text_bytes=100,
            metadata_bytes=0,
            identifier_bytes=0,
        )
        more_metadata = estimate_faiss_finalization_memory(
            chunk_count=10,
            dimensions=8,
            text_bytes=100,
            metadata_bytes=100,
            identifier_bytes=0,
        )
        more_identifiers = estimate_faiss_finalization_memory(
            chunk_count=10,
            dimensions=8,
            text_bytes=100,
            metadata_bytes=0,
            identifier_bytes=50,
        )
        more_dimensions = estimate_faiss_finalization_memory(
            chunk_count=10,
            dimensions=16,
            text_bytes=100,
            metadata_bytes=0,
            identifier_bytes=0,
        )

        self.assertGreater(more_metadata["peak_memory_bytes"], baseline["peak_memory_bytes"])
        self.assertGreater(more_identifiers["peak_memory_bytes"], baseline["peak_memory_bytes"])
        self.assertGreater(more_dimensions["peak_memory_bytes"], baseline["peak_memory_bytes"])
        self.assertGreater(more_metadata["artifact_bytes"], baseline["artifact_bytes"])
        self.assertGreater(more_identifiers["artifact_bytes"], baseline["artifact_bytes"])

    def test_baseline_is_additive_to_phase_peak(self) -> None:
        estimate = estimate_faiss_finalization_memory(
            chunk_count=1,
            dimensions=1,
            text_bytes=1,
            metadata_bytes=0,
            identifier_bytes=0,
        )

        self.assertGreater(estimate["peak_memory_bytes"], estimate["worker_baseline_bytes"])
        self.assertEqual(estimate["peak_memory_bytes"], max(estimate[name] for name in ("build_peak_bytes", "save_peak_bytes", "validation_peak_bytes")))

    def test_record_cost_and_negative_inputs_are_bounded(self) -> None:
        empty = estimate_faiss_finalization_memory(
            chunk_count=-1,
            dimensions=-1,
            text_bytes=-1,
            metadata_bytes=-1,
            identifier_bytes=-1,
        )
        records = estimate_faiss_finalization_memory(
            chunk_count=100,
            dimensions=0,
            text_bytes=0,
            metadata_bytes=0,
            identifier_bytes=0,
        )

        self.assertEqual(empty["artifact_bytes"], 0)
        self.assertGreater(records["peak_memory_bytes"], empty["peak_memory_bytes"])

    def test_production_shapes_fit_automatic_budget(self) -> None:
        automatic_budget = 31 * 1024**3
        for chunks, text_bytes, metadata_bytes in (
            (4954, 13862591, 1409024),
            (28285, 50688305, 8948503),
        ):
            estimate = estimate_faiss_finalization_memory(
                chunk_count=chunks,
                dimensions=768,
                text_bytes=text_bytes,
                metadata_bytes=metadata_bytes,
                identifier_bytes=chunks * 64,
            )
            self.assertLess(estimate["peak_memory_bytes"], automatic_budget)
