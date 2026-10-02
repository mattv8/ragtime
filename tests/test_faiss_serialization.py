from __future__ import annotations

import copyreg
import pickle
import tempfile
import unittest
from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from ragtime.indexer.faiss_serialization import FaissSerializationError, count_faiss_metadata, load_faiss_metadata, safe_load_faiss
from ragtime.indexer.vector_utils import count_faiss_docstore_stats


def _payload(*, mapping: dict[int, str] | None = None, documents: dict[str, Document] | None = None, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
    documents = documents or {"one": Document(page_content="hello", metadata={"source": "test.txt"})}
    mapping = mapping or {0: "one"}
    return pickle.dumps((InMemoryDocstore(documents), mapping), protocol=protocol)


class _Reducer:
    def __reduce__(self):
        return (str, ("executed",))


class _DocumentReducer:
    def __reduce__(self):
        return (Document, (), {"page_content": "hello", "metadata": {}, "id": None})


class FaissSerializationTests(unittest.TestCase):
    def _load(self, payload: bytes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pkl"
            path.write_bytes(payload)
            return load_faiss_metadata(path)

    def test_reads_standard_langchain_exports_across_protocols(self) -> None:
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                docstore, mapping = self._load(_payload(protocol=protocol))
                self.assertIsInstance(docstore, InMemoryDocstore)
                self.assertEqual(docstore.search("one").page_content, "hello")
                self.assertEqual(mapping, {0: "one"})

    def test_rejects_reducers_and_unknown_globals(self) -> None:
        with self.assertRaises(FaissSerializationError):
            self._load(pickle.dumps((_Reducer(), {0: "one"})))

    def test_rejects_extension_opcodes(self) -> None:
        copyreg.add_extension(__name__, "_Reducer", 512)
        try:
            with self.assertRaises(FaissSerializationError):
                self._load(pickle.dumps(_Reducer(), protocol=2))
        finally:
            copyreg.remove_extension(__name__, "_Reducer", 512)

    def test_rejects_document_reduce_even_with_an_allowed_global(self) -> None:
        with self.assertRaises(FaissSerializationError):
            self._load(pickle.dumps(_DocumentReducer(), protocol=4))

    def test_rejects_persistent_ids(self) -> None:
        class PersistentPickler(pickle.Pickler):
            def persistent_id(self, obj):
                return "blocked" if obj == "marker" else None

        import io

        stream = io.BytesIO()
        PersistentPickler(stream, protocol=4).dump("marker")
        with self.assertRaises(FaissSerializationError):
            self._load(stream.getvalue())

    def test_rejects_malformed_documents_and_mappings(self) -> None:
        cases = [
            _payload(mapping={1: "one"}),
            _payload(mapping={0: "missing"}),
            _payload(documents={"one": Document(page_content="hello", metadata={}), "two": Document(page_content="two", metadata={})}),
            pickle.dumps(({"one": {"page_content": "hello", "metadata": {}}}, {0: "one"})),
        ]
        for payload in cases:
            with self.subTest(payload=payload[:10]):
                with self.assertRaises(FaissSerializationError):
                    self._load(payload)

    def test_safe_load_rebuilds_a_real_faiss_store(self) -> None:
        import faiss
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            index = faiss.IndexFlatL2(2)
            index.add(np.array([[1.0, 0.0]], dtype="float32"))
            (path / "index.faiss").write_bytes(faiss.serialize_index(index))
            (path / "index.pkl").write_bytes(_payload())
            store = safe_load_faiss(path, object())
            self.assertEqual(store.index.ntotal, 1)
            self.assertEqual(store.docstore.search("one").page_content, "hello")

    def test_safe_load_rejects_native_mapping_count_mismatch(self) -> None:
        import faiss
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            index = faiss.IndexFlatL2(2)
            index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
            faiss.write_index(index, str(path / "index.faiss"))
            (path / "index.pkl").write_bytes(_payload())
            with self.assertRaises(FaissSerializationError):
                safe_load_faiss(path, object())

    def test_rejects_metadata_beyond_the_nesting_limit(self) -> None:
        metadata: dict[str, object] = {}
        nested = metadata
        for _ in range(101):
            next_value: dict[str, object] = {}
            nested["nested"] = next_value
            nested = next_value
        with self.assertRaises(FaissSerializationError):
            self._load(_payload(documents={"one": Document(page_content="hello", metadata=metadata)}))

    def test_rejects_repeated_container_aliases(self) -> None:
        shared_dict = {"value": "shared"}
        shared_list = ["shared"]
        for metadata in (
            {"first": shared_dict, "second": shared_dict},
            {"first": shared_list, "second": shared_list},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(FaissSerializationError):
                    self._load(_payload(documents={"one": Document(page_content="hello", metadata=metadata)}))

    def test_rejects_compact_aliased_dag_and_cycles(self) -> None:
        node: dict[str, object] = {"leaf": "value"}
        for _ in range(10):
            node = {"left": node, "right": node}
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        for metadata in (node, cycle):
            with self.subTest(metadata=metadata):
                with self.assertRaises(FaissSerializationError):
                    self._load(_payload(documents={"one": Document(page_content="hello", metadata=metadata)}))

    def test_allows_equal_but_distinct_metadata_containers(self) -> None:
        metadata = {"first": {"value": "same"}, "second": {"value": "same"}}
        docstore, _mapping = self._load(_payload(documents={"one": Document(page_content="hello", metadata=metadata)}))
        self.assertEqual(docstore.search("one").metadata, metadata)

    def test_rejects_metadata_container_aliases_across_documents(self) -> None:
        shared_metadata = {"nested": ["shared"]}
        documents = {
            "one": Document(page_content="first", metadata={"shared": shared_metadata}),
            "two": Document(page_content="second", metadata={"shared": shared_metadata}),
        }
        with self.assertRaises(FaissSerializationError):
            self._load(_payload(documents=documents, mapping={0: "one", 1: "two"}))

    def test_allows_equal_but_distinct_nested_metadata_across_documents(self) -> None:
        first_metadata = {"nested": ["same"]}
        second_metadata = {"nested": ["same"]}
        documents = {
            "one": Document(page_content="first", metadata=first_metadata),
            "two": Document(page_content="second", metadata=second_metadata),
        }
        docstore, _mapping = self._load(_payload(documents=documents, mapping={0: "one", 1: "two"}))
        self.assertEqual(docstore.search("one").metadata, first_metadata)
        self.assertEqual(docstore.search("two").metadata, second_metadata)

    def test_count_uses_source_before_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pkl"
            path.write_bytes(_payload(documents={"one": Document(page_content="hello", metadata={"source": "source.txt", "file_path": "other.txt"})}))
            self.assertEqual(count_faiss_metadata(path), (1, 1))

    def test_pure_stats_helper_prefers_source_over_file_path(self) -> None:
        docstore = InMemoryDocstore({"one": Document(page_content="hello", metadata={"source": "source.txt", "file_path": "other.txt"})})
        self.assertEqual(count_faiss_docstore_stats((docstore, {0: "one"})), (1, 1))
