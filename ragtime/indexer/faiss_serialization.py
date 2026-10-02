"""Fail-closed readers for protocol>=2, JSON-metadata LangChain FAISS artifacts.

Unsupported custom pickle formats are rejected with an actionable error: users
must re-export a standard LangChain artifact or re-index the source content.
"""

from __future__ import annotations

import io
import math
import pickle
import pickletools
from pathlib import Path
from typing import Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ragtime.indexer.vector_utils import count_faiss_docstore_stats


class FaissSerializationError(ValueError):
    """Raised when a FAISS metadata artifact is not a supported safe export."""


class _InertDocument:
    """Unpickling target with no constructors or state hooks."""


class _InertInMemoryDocstore:
    """Unpickling target with no constructors or state hooks."""


_ALLOWED_GLOBALS = {
    ("langchain_community.docstore.in_memory", "InMemoryDocstore"): _InertInMemoryDocstore,
    ("langchain_core.documents.base", "Document"): _InertDocument,
    # Older LangChain versions exported this public module path.
    ("langchain_core.documents", "Document"): _InertDocument,
    ("langchain.docstore.in_memory", "InMemoryDocstore"): _InertInMemoryDocstore,
    ("langchain.schema.document", "Document"): _InertDocument,
}
# Protocol 2 normal Pydantic Document exports encode their internal field-set
# as this built-in global. It is discarded during reconstruction and cannot
# resolve user-defined code.
_ALLOWED_PRIMITIVE_GLOBALS = {("builtins", "set"): set, ("__builtin__", "set"): set}
_FORBIDDEN_OPCODES = {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
_MAX_JSON_NESTING = 100


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type[Any]:
        surrogate = _ALLOWED_GLOBALS.get((module, name)) or _ALLOWED_PRIMITIVE_GLOBALS.get((module, name))
        if surrogate is None:
            raise FaissSerializationError(f"Unsupported pickle global: {module}.{name}")
        return surrogate

    def persistent_load(self, pid: Any) -> Any:
        raise FaissSerializationError("Persistent pickle IDs are not supported")


def _reject_forbidden_opcodes(payload: bytes) -> None:
    try:
        for opcode, _argument, _position in pickletools.genops(payload):
            if opcode.name in _FORBIDDEN_OPCODES:
                raise FaissSerializationError(f"Unsupported pickle opcode: {opcode.name}")
    except FaissSerializationError:
        raise
    except Exception as exc:
        raise FaissSerializationError("Malformed pickle payload") from exc


def _is_json_value(value: Any, seen: set[int] | None = None, depth: int = 0) -> bool:
    """Validate JSON metadata with one linear traversal across the artifact.

    Container identities remain in ``seen`` after their first visit, so aliases
    and cycles are rejected instead of repeatedly expanding a shared graph in
    one document or across document metadata.
    """
    if depth > _MAX_JSON_NESTING:
        return False
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if seen is None:
        seen = set()
    if type(value) not in (dict, list):
        return False
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if type(value) is dict:
        return all(type(key) is str and _is_json_value(item, seen, depth + 1) for key, item in value.items())
    return all(_is_json_value(item, seen, depth + 1) for item in value)


def _document_state(document: _InertDocument) -> dict[str, Any]:
    state = vars(document)
    # Pydantic v2 Documents pickle their model state below ``__dict__``.
    nested = state.get("__dict__")
    if type(nested) is dict:
        return nested
    return state


def _rebuild_metadata(payload: Any) -> tuple[InMemoryDocstore, dict[int, str]]:
    if type(payload) is not tuple or len(payload) != 2:
        raise FaissSerializationError("FAISS metadata must be a docstore and ID mapping tuple")
    inert_docstore, mapping = payload
    if type(inert_docstore) is not _InertInMemoryDocstore or type(mapping) is not dict:
        raise FaissSerializationError("FAISS metadata has an unsupported shape")
    documents = vars(inert_docstore).get("_dict")
    if type(documents) is not dict:
        raise FaissSerializationError("FAISS docstore is malformed")
    if len(documents) != len(mapping):
        raise FaissSerializationError("FAISS docstore and mapping counts differ")

    rebuilt: dict[str, Document] = {}
    # Retain metadata container identities across the entire artifact so a
    # compact pickle cannot multiply validation work through cross-document
    # aliases.
    seen_metadata_containers: set[int] = set()
    for identifier, inert_document in documents.items():
        if type(identifier) is not str or type(inert_document) is not _InertDocument:
            raise FaissSerializationError("FAISS docstore contains unsupported entries")
        state = _document_state(inert_document)
        page_content = state.get("page_content")
        metadata = state.get("metadata")
        document_id = state.get("id")
        if type(page_content) is not str or type(metadata) is not dict or type(document_id) not in (str, type(None)):
            raise FaissSerializationError("FAISS document fields are malformed")
        if not _is_json_value(metadata, seen_metadata_containers):
            raise FaissSerializationError("FAISS document metadata must be JSON-like")
        rebuilt[identifier] = Document(page_content=page_content, metadata=metadata, id=document_id)

    rebuilt_mapping: dict[int, str] = {}
    identifiers: set[str] = set()
    for position, identifier in mapping.items():
        if type(position) is not int or type(identifier) is not str or identifier in identifiers or identifier not in rebuilt:
            raise FaissSerializationError("FAISS ID mapping is malformed")
        rebuilt_mapping[position] = identifier
        identifiers.add(identifier)
    if set(rebuilt_mapping) != set(range(len(rebuilt_mapping))) or identifiers != set(rebuilt):
        raise FaissSerializationError("FAISS ID mapping must be contiguous and cover the docstore")
    return InMemoryDocstore(rebuilt), rebuilt_mapping


def load_faiss_metadata(path: str | Path) -> tuple[InMemoryDocstore, dict[int, str]]:
    """Read a compatible LangChain metadata pickle without live unpickling."""
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise FaissSerializationError(f"Could not read FAISS metadata: {path}") from exc
    _reject_forbidden_opcodes(payload)
    try:
        data = _RestrictedUnpickler(io.BytesIO(payload)).load()
    except FaissSerializationError:
        raise
    except Exception as exc:
        raise FaissSerializationError("Unsupported or malformed FAISS metadata") from exc
    try:
        return _rebuild_metadata(data)
    except (RecursionError, MemoryError) as exc:
        raise FaissSerializationError("FAISS metadata exceeds safe validation limits") from exc


def safe_load_faiss(path: str | Path, embeddings: Any, **kwargs: Any) -> FAISS:
    """Load FAISS after safely rebuilding metadata from a trusted artifact.

    The native ``faiss.read_index`` parser runs in-process and is not
    sandboxed; safe metadata handling does not make untrusted native FAISS
    bytes safe to parse.
    """
    directory = Path(path)
    docstore, mapping = load_faiss_metadata(directory / "index.pkl")
    try:
        import faiss

        index = faiss.read_index(str(directory / "index.faiss"))
    except Exception as exc:
        raise FaissSerializationError("Could not read FAISS native index") from exc
    if index.ntotal != len(mapping):
        raise FaissSerializationError("FAISS native index and metadata counts differ")
    return FAISS(embeddings, index, docstore, mapping, **kwargs)


def count_faiss_metadata(path: str | Path) -> tuple[int, int]:
    """Return document and chunk counts from validated FAISS metadata."""
    docstore, mapping = load_faiss_metadata(path)
    return count_faiss_docstore_stats((docstore, mapping))
