from __future__ import annotations

import unittest

from ragtime.indexer.faiss_docstore_stats import count_faiss_docstore_stats


class _Document:
    def __init__(self, metadata: object) -> None:
        self.metadata = metadata


class _Docstore:
    def __init__(self, documents: dict[str, _Document]) -> None:
        self._dict = documents


class FaissDocstoreStatsTests(unittest.TestCase):
    def test_counts_unique_sources_from_docstore_tuple(self) -> None:
        docstore = _Docstore(
            {
                "first": _Document({"source": "first.txt"}),
                "second": _Document({"source": "first.txt"}),
                "third": _Document({"file_path": "third.txt"}),
            }
        )

        self.assertEqual(count_faiss_docstore_stats((docstore, {0: "first", 1: "second", 2: "third"})), (2, 3))

    def test_prefers_source_to_file_path(self) -> None:
        docstore = _Docstore({"one": _Document({"source": "source.txt", "file_path": "other.txt"})})

        self.assertEqual(count_faiss_docstore_stats((docstore, {0: "one"})), (1, 1))

    def test_falls_back_to_chunk_count_when_docstore_sources_are_missing(self) -> None:
        docstore = _Docstore({"one": _Document({}), "two": _Document(None)})

        self.assertEqual(count_faiss_docstore_stats((docstore, {0: "one", 1: "two"})), (2, 2))

    def test_counts_legacy_mapping_payload(self) -> None:
        payload = {"one": _Document({"source": "one.txt"}), "two": _Document({"source": "two.txt"})}

        self.assertEqual(count_faiss_docstore_stats(payload), (2, 2))

    def test_falls_back_to_identifier_mapping_when_docstore_is_unavailable(self) -> None:
        self.assertEqual(count_faiss_docstore_stats((object(), {0: "one", 1: "two"})), (2, 2))
