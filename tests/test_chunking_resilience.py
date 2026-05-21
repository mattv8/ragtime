import unittest
from unittest import mock

from ragtime.indexer import chunking
from ragtime.indexer.code_extraction import extract_metadata


class ChunkingResilienceTests(unittest.TestCase):
    def test_tsx_metadata_extraction_uses_supported_parser_input(self) -> None:
        text = """
import React from 'react';

interface Props {
  label: string;
}

function Example(props: Props) {
  return <div>{props.label}</div>;
}
"""

        imports, definitions = extract_metadata(text, ".tsx")

        self.assertIn("import React from 'react';", imports)
        self.assertTrue(any("function Example" in item for item in definitions))

    def test_worker_count_respects_configured_cap(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=100)
            # 32 cores // 4 = 8, capped at configured 4
            self.assertEqual(chunking._resolve_worker_count(32), 4)
            # 8 cores // 4 = 2 (below cap)
            self.assertEqual(chunking._resolve_worker_count(8), 2)
            # Tightening the cap reduces the resolved worker count.
            chunking.configure_chunking_pool(max_workers=2, max_batch_size=100)
            self.assertEqual(chunking._resolve_worker_count(32), 2)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_batch_size_respects_configured_cap(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=100)
            self.assertEqual(chunking._effective_batch_size(500), 100)
            self.assertEqual(chunking._effective_batch_size(50), 50)
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=25)
            self.assertEqual(chunking._effective_batch_size(500), 25)
            chunking.configure_chunking_pool(max_workers=4, max_batch_size=200)
            self.assertEqual(chunking._effective_batch_size(500), 200)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_configure_chunking_pool_clamps_to_safe_range(self) -> None:
        original_workers = chunking._configured_max_workers
        original_batch = chunking._configured_max_batch_size
        try:
            chunking.configure_chunking_pool(max_workers=999, max_batch_size=999_999)
            self.assertEqual(chunking._configured_max_workers, chunking._CHUNKING_WORKERS_HARD_CEILING)
            self.assertEqual(chunking._configured_max_batch_size, chunking._CHUNKING_BATCH_SIZE_HARD_CEILING)
            chunking.configure_chunking_pool(max_workers=0, max_batch_size=0)
            self.assertEqual(chunking._configured_max_workers, 1)
            self.assertEqual(chunking._configured_max_batch_size, 1)
        finally:
            chunking.configure_chunking_pool(
                max_workers=original_workers,
                max_batch_size=original_batch,
            )

    def test_recursive_fallback_marks_chunks(self) -> None:
        chunks, counts = chunking._chunk_document_batch_recursive_sync(
            [("alpha\n\nbeta\n\ngamma", {"source": "notes.txt"})],
            chunk_size=100,
            chunk_overlap=0,
            use_tokens=False,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1]["chunker"], "no_chunk_small")
        self.assertEqual(counts["chonkie_recursive_pool_fallback"], 1)

    def test_recursive_fallback_keeps_document_when_recursive_chunker_fails(self) -> None:
        with mock.patch.object(chunking, "_chunk_with_recursive", side_effect=RuntimeError("boom")):
            chunks, counts = chunking._chunk_document_batch_recursive_sync(
                [("alpha", {"source": "bad.txt"})],
                chunk_size=100,
                chunk_overlap=0,
                use_tokens=False,
            )

        self.assertEqual(chunks[0][0], "alpha")
        self.assertEqual(chunks[0][1]["chunker"], "whole_document_pool_fallback")
        self.assertEqual(counts["whole_document_pool_fallback"], 1)


if __name__ == "__main__":
    unittest.main()
