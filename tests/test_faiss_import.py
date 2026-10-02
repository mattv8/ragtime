from __future__ import annotations

import contextlib
import io
import json
import pickle
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import faiss
import numpy as np

# Import actual LangChain classes for safe fixtures
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

import ragtime.indexer.routes as _routes_module
from ragtime.indexer.routes import import_faiss_index
from ragtime.indexer.service import IndexerService


def _create_real_faiss_bytes(chunk_count: int) -> bytes:
    """Create real FAISS IndexFlatL2 serialized bytes for testing.

    Uses actual faiss library to generate valid index data matching the
    chunk count. This ensures tests exercise the real FAISS deserialization
    path, not fake data.
    """
    dimension = 384  # Common embedding dimension in ragtime
    index = faiss.IndexFlatL2(dimension)

    # Add dummy vectors matching chunk count
    vectors = np.random.random((chunk_count, dimension)).astype("float32")
    index.add(vectors)

    # Serialize to bytes
    data = faiss.serialize_index(index)
    return bytes(data)


def _build_faiss_zip(
    name: str,
    *,
    description: str,
    chunks: int,
    native_chunks: int | None = None,
    native_bytes: bytes | None = None,
) -> bytes:
    """Build a zip that mirrors the structure produced by download_index.

    Uses actual LangChain InMemoryDocstore, Document, and real FAISS serialized
    bytes to ensure the safe loader is tested with genuine artifacts.
    """

    # Create actual LangChain Document objects
    documents = {f"doc-{i}": Document(page_content=f"chunk {i}", metadata={}) for i in range(chunks)}
    # Create actual InMemoryDocstore
    docstore = InMemoryDocstore(documents)

    # Tuple layout that FAISS.save_local produces:
    # (docstore, index_to_docstore_id)
    pkl_payload = (docstore, {i: f"doc-{i}" for i in range(chunks)})
    pkl_bytes = pickle.dumps(pkl_payload)

    metadata = {
        "name": name,
        "format_version": 1,
        "display_name": name,
        "description": description,
        "source_type": "upload",
        "source": "test-archive.tar.gz",
        "git_branch": None,
        "vector_store_type": "faiss",
        "ocr_mode": "disabled",
        "ocr_provider": None,
        "ocr_vision_model": None,
        "config_snapshot": {
            "file_patterns": ["**/*"],
            "exclude_patterns": [],
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_file_size_kb": 500,
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Use real FAISS serialized index bytes
        zf.writestr(
            f"{name}/index.faiss",
            native_bytes if native_bytes is not None else _create_real_faiss_bytes(native_chunks if native_chunks is not None else chunks),
        )
        zf.writestr(f"{name}/index.pkl", pkl_bytes)
        zf.writestr(f"{name}/metadata.json", json.dumps(metadata))
    return buf.getvalue()


def _build_metadata_zip(name: str) -> bytes:
    """Build a zip without the required FAISS files to exercise validation."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/readme.txt", "no faiss here")
    return buf.getvalue()


class _TempDir:
    """Wrapper that defers cleanup until release() is called.

    Avoids any auto-cleanup that may run when ``tempfile.TemporaryDirectory``
    goes out of scope during pytest teardown.
    """

    def __init__(self) -> None:
        self.path = Path(tempfile.mkdtemp(prefix="ragtime_faiss_import_test_"))

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "_TempDir":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def _make_upload(filename: str, data: bytes) -> Any:
    return SimpleNamespace(filename=filename, file=io.BytesIO(data))


def _make_admin() -> Any:
    return SimpleNamespace(id="admin", username="admin", role="admin")


class ImportFaissIndexTests(unittest.IsolatedAsyncioTestCase):
    def _make_service_and_patches(
        self,
        temp: _TempDir,
        *,
        load_return: bool | None = None,
        load_side_effect: Exception | None = None,
        load_error: str | None = None,
        metadata_return: Any = None,
    ) -> tuple[IndexerService, SimpleNamespace, SimpleNamespace]:
        service = IndexerService(index_base_path=str(temp.path))
        fake_repo = SimpleNamespace(
            get_index_metadata=AsyncMock(return_value=metadata_return),
            upsert_index_metadata=AsyncMock(),
        )
        fake_rag = SimpleNamespace(
            unload_index=Mock(),
            load_faiss_index_from_metadata=AsyncMock(return_value=load_return, side_effect=load_side_effect),
            loading_status={
                "index_details": ([{"name": "odev_proj", "error": load_error}] if load_error else []),
            },
        )
        return service, fake_repo, fake_rag

    @contextlib.contextmanager
    def _patch_routes(
        self,
        service: IndexerService,
        fake_repo: SimpleNamespace,
        fake_rag: SimpleNamespace,
    ):
        with (
            patch.object(_routes_module, "indexer", service),
            patch.object(_routes_module, "repository", fake_repo),
            patch.object(_routes_module, "rag", fake_rag),
        ):
            yield

    async def test_import_faiss_index_writes_files_and_metadata(self) -> None:
        description = "Custom description for odev_proj re-import"
        zip_bytes = _build_faiss_zip("odev_proj", description=description, chunks=7)
        upload = _make_upload("odev_proj_index.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp, load_return=True)
            with self._patch_routes(service, fake_repo, fake_rag):
                response = await import_faiss_index(
                    file=upload,
                    name=None,
                    description=None,
                    overwrite=False,
                    _user=_make_admin(),
                )

            # Files should be on disk at the indexer base path
            target = service.index_base_path / "odev_proj"
            self.assertTrue((target / "index.faiss").exists())
            self.assertTrue((target / "index.pkl").exists())
            # metadata.json is intentionally not re-extracted - it is the
            # transport envelope, not part of the FAISS data itself.
            self.assertFalse((target / "metadata.json").exists())

        # Response carries metadata restored from metadata.json
        self.assertEqual(response.name, "odev_proj")
        self.assertEqual(response.description, description)
        self.assertEqual(response.chunk_count, 7)
        self.assertEqual(response.source_type, "upload")
        self.assertTrue(response.loaded)
        self.assertIsNone(response.load_error)
        fake_rag.load_faiss_index_from_metadata.assert_awaited_once_with("odev_proj")

    async def test_import_faiss_index_reports_saved_but_unavailable_when_hot_load_returns_false(self) -> None:
        zip_bytes = _build_faiss_zip("odev_proj", description="", chunks=1)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(
                temp,
                load_return=False,
                load_error=(
                    "Embedding dimension mismatch: index has 768 dims, but current model produces 1024 dims. "
                    "Restore the original embedding model and dimensions, then retry loading, "
                    "or re-index with the current configuration."
                ),
            )
            with self._patch_routes(service, fake_repo, fake_rag):
                response = await import_faiss_index(
                    file=_make_upload("odev_proj.zip", zip_bytes),
                    name=None,
                    description=None,
                    overwrite=False,
                    _user=_make_admin(),
                )

        self.assertFalse(response.loaded)
        self.assertEqual(
            response.load_error,
            (
                "Embedding dimension mismatch: index has 768 dims, but current model produces 1024 dims. "
                "Restore the original embedding model and dimensions, then retry loading, "
                "or re-index with the current configuration."
            ),
        )
        self.assertIn("saved but unavailable for search", response.message)
        fake_rag.load_faiss_index_from_metadata.assert_awaited_once_with("odev_proj")

    async def test_import_faiss_index_reports_saved_but_unavailable_when_hot_load_raises(self) -> None:
        zip_bytes = _build_faiss_zip("odev_proj", description="", chunks=1)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(
                temp,
                load_side_effect=RuntimeError("loader failed"),
            )
            with self._patch_routes(service, fake_repo, fake_rag):
                response = await import_faiss_index(
                    file=_make_upload("odev_proj.zip", zip_bytes),
                    name=None,
                    description=None,
                    overwrite=False,
                    _user=_make_admin(),
                )

        self.assertFalse(response.loaded)
        self.assertTrue(response.load_error)
        self.assertIsNotNone(response.load_error)
        assert response.load_error is not None
        self.assertNotIn("loader failed", response.load_error)
        self.assertIn("saved but unavailable for search", response.message)
        fake_rag.load_faiss_index_from_metadata.assert_awaited_once_with("odev_proj")

    async def test_import_faiss_index_rejects_missing_faiss_files(self) -> None:
        from fastapi import HTTPException

        zip_bytes = _build_metadata_zip("broken")
        upload = _make_upload("broken.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            with self._patch_routes(service, fake_repo, fake_rag):
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("index.faiss", str(ctx.exception.detail))

    async def test_import_faiss_index_falls_back_to_filename_when_metadata_missing(self) -> None:
        """A zip without metadata.json still imports and derives a name
        from the zip filename."""

        docstore = InMemoryDocstore({})
        pkl_bytes = pickle.dumps((docstore, {}))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("myindex/index.faiss", _create_real_faiss_bytes(0))
            zf.writestr("myindex/index.pkl", pkl_bytes)
        zip_bytes = buf.getvalue()

        upload = _make_upload("myindex_index.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp, load_return=True)
            with self._patch_routes(service, fake_repo, fake_rag):
                response = await import_faiss_index(
                    file=upload,
                    name=None,
                    description=None,
                    overwrite=False,
                    _user=_make_admin(),
                )

        # Falls back to filename-derived name (sanitized)
        self.assertEqual(response.name, "myindex_index")
        # Empty description when metadata.json is absent
        self.assertEqual(response.description, "")

    async def test_import_faiss_index_rejects_conflicts_without_overwrite(self) -> None:
        from fastapi import HTTPException

        description = "Re-import test"
        zip_bytes = _build_faiss_zip("odev_proj", description=description, chunks=3)
        upload = _make_upload("odev_proj_index.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            (service.index_base_path / "odev_proj").mkdir()
            (service.index_base_path / "odev_proj" / "index.faiss").write_bytes(b"x")

            # Repository reports the index already has metadata
            fake_repo.get_index_metadata.return_value = SimpleNamespace(
                name="odev_proj",
                path=str(service.index_base_path / "odev_proj"),
            )

            with self._patch_routes(service, fake_repo, fake_rag):
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_overwrite_rejects_corrupt_native_artifact_without_touching_existing_index(self) -> None:
        from fastapi import HTTPException

        zip_bytes = _build_faiss_zip("odev_proj", description="", chunks=1, native_bytes=b"corrupt native index")
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(
                temp,
                metadata_return=SimpleNamespace(name="odev_proj"),
            )
            target = service.index_base_path / "odev_proj"
            target.mkdir()
            (target / "index.faiss").write_bytes(b"existing native bytes")
            (target / "index.pkl").write_bytes(b"existing metadata bytes")
            with self._patch_routes(service, fake_repo, fake_rag):
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(file=_make_upload("odev_proj.zip", zip_bytes), name=None, description=None, overwrite=True, _user=_make_admin())

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual((target / "index.faiss").read_bytes(), b"existing native bytes")
            self.assertEqual((target / "index.pkl").read_bytes(), b"existing metadata bytes")
            fake_rag.unload_index.assert_not_called()
            fake_repo.upsert_index_metadata.assert_not_awaited()
            fake_rag.load_faiss_index_from_metadata.assert_not_awaited()
            self.assertFalse(list(service.index_base_path.glob(".odev_proj.import-*")))

    async def test_overwrite_rejects_native_count_mismatch_without_touching_existing_index(self) -> None:
        from fastapi import HTTPException

        zip_bytes = _build_faiss_zip("odev_proj", description="", chunks=1, native_chunks=2)
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(
                temp,
                metadata_return=SimpleNamespace(name="odev_proj"),
            )
            target = service.index_base_path / "odev_proj"
            target.mkdir()
            (target / "index.faiss").write_bytes(b"existing native bytes")
            (target / "index.pkl").write_bytes(b"existing metadata bytes")
            with self._patch_routes(service, fake_repo, fake_rag):
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(file=_make_upload("odev_proj.zip", zip_bytes), name=None, description=None, overwrite=True, _user=_make_admin())

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual((target / "index.faiss").read_bytes(), b"existing native bytes")
            self.assertEqual((target / "index.pkl").read_bytes(), b"existing metadata bytes")
            fake_rag.unload_index.assert_not_called()
            fake_repo.upsert_index_metadata.assert_not_awaited()
            fake_rag.load_faiss_index_from_metadata.assert_not_awaited()
            self.assertFalse(list(service.index_base_path.glob(".odev_proj.import-*")))

    async def test_overwrite_publishes_valid_native_artifact_after_validation(self) -> None:
        zip_bytes = _build_faiss_zip("odev_proj", description="replacement", chunks=1)
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(
                temp,
                load_return=True,
                metadata_return=SimpleNamespace(name="odev_proj"),
            )
            target = service.index_base_path / "odev_proj"
            target.mkdir()
            (target / "index.faiss").write_bytes(b"existing native bytes")
            with self._patch_routes(service, fake_repo, fake_rag):
                response = await import_faiss_index(
                    file=_make_upload("odev_proj.zip", zip_bytes), name=None, description=None, overwrite=True, _user=_make_admin()
                )

            self.assertTrue(response.loaded)
            self.assertNotEqual((target / "index.faiss").read_bytes(), b"existing native bytes")
            fake_rag.unload_index.assert_called_once_with("odev_proj")
            fake_repo.upsert_index_metadata.assert_awaited_once()
            fake_rag.load_faiss_index_from_metadata.assert_awaited_once_with("odev_proj")


def _build_traversal_zip(name: str, *, evil_member: str) -> bytes:
    """Build a zip that contains a traversal member alongside valid FAISS files.

    The evil_member path is injected verbatim into the zip so the extraction
    code must reject it rather than writing outside the target directory.

    Uses actual InMemoryDocstore and real FAISS bytes.
    """
    docstore = InMemoryDocstore({})
    pkl_bytes = pickle.dumps((docstore, {}))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/index.faiss", _create_real_faiss_bytes(0))
        zf.writestr(f"{name}/index.pkl", pkl_bytes)
        # Inject the malicious member directly
        zf.writestr(evil_member, b"evil payload")
    return buf.getvalue()


# Malicious pickle regression test utilities.
# Track invocations of test-local gadget functions to verify they are NEVER executed.
_malicious_gadget_invocations: list[str] = []


def _record_gadget_invocation(gadget_name: str) -> None:
    """Marker function invoked by __reduce__ gadgets. Must never be called during safe import."""
    _malicious_gadget_invocations.append(gadget_name)


class _MaliciousReduceGadget:
    """A pickle gadget that records invocation via __reduce__.

    When unpickled, this gadget calls _record_gadget_invocation() to mark execution.
    This test-local function should never be called if the endpoint properly rejects
    malicious pickles before deserialization.
    """

    def __init__(self, gadget_id: str) -> None:
        self.gadget_id = gadget_id

    def __reduce__(self):
        # __reduce__ returns (callable, args) to be invoked during unpickling.
        # This gadget calls our test marker function.
        return (_record_gadget_invocation, (self.gadget_id,))


def _build_malicious_faiss_zip(name: str, *, description: str, chunks: int, gadget_id: str) -> bytes:
    """Build a zip with a malicious index.pkl containing a __reduce__ gadget.

    The pickle payload includes a _MaliciousReduceGadget that records invocation
    if unpickled. A safe endpoint should reject this zip with HTTP 400 before
    deserializing the pickle.

    For this malicious test, we construct the tuple manually with the gadget
    since we want to test that the safe loader rejects it before any unpickling.
    """
    # Create actual LangChain Document objects
    documents = {f"doc-{i}": Document(page_content=f"chunk {i}", metadata={}) for i in range(chunks)}
    # Create actual InMemoryDocstore
    docstore = InMemoryDocstore(documents)

    # Inject the gadget into the docstore tuple to trigger __reduce__ during unpickling
    gadget = _MaliciousReduceGadget(gadget_id)
    pkl_payload = (docstore, {i: f"doc-{i}" for i in range(chunks)}, gadget)
    pkl_bytes = pickle.dumps(pkl_payload)

    metadata = {
        "name": name,
        "format_version": 1,
        "display_name": name,
        "description": description,
        "source_type": "upload",
        "source": "malicious.tar.gz",
        "git_branch": None,
        "vector_store_type": "faiss",
        "ocr_mode": "disabled",
        "ocr_provider": None,
        "ocr_vision_model": None,
        "config_snapshot": {
            "file_patterns": ["**/*"],
            "exclude_patterns": [],
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_file_size_kb": 500,
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/index.faiss", _create_real_faiss_bytes(chunks))
        zf.writestr(f"{name}/index.pkl", pkl_bytes)
        zf.writestr(f"{name}/metadata.json", json.dumps(metadata))
    return buf.getvalue()


class MaliciousPickleRegressionTests(unittest.IsolatedAsyncioTestCase):
    """Verify that malicious index.pkl files are rejected before unpickling."""

    def setUp(self) -> None:
        """Reset invocation tracker before each test."""
        global _malicious_gadget_invocations
        _malicious_gadget_invocations = []

    def _make_service_and_patches(
        self,
        temp: _TempDir,
    ) -> tuple[IndexerService, SimpleNamespace, SimpleNamespace]:
        service = IndexerService(index_base_path=str(temp.path))
        fake_repo = SimpleNamespace(
            get_index_metadata=AsyncMock(return_value=None),
            upsert_index_metadata=AsyncMock(),
        )
        fake_rag = SimpleNamespace(
            unload_index=lambda _name: None,
            load_faiss_index_from_metadata=AsyncMock(return_value=True),
        )
        return service, fake_repo, fake_rag

    @contextlib.contextmanager
    def _patch_routes(
        self,
        service: IndexerService,
        fake_repo: SimpleNamespace,
        fake_rag: SimpleNamespace,
    ):
        with (
            patch.object(_routes_module, "indexer", service),
            patch.object(_routes_module, "repository", fake_repo),
            patch.object(_routes_module, "rag", fake_rag),
        ):
            yield

    async def test_malicious_pickle_with_reduce_gadget_is_rejected_before_unpickling(self) -> None:
        """Malicious index.pkl with __reduce__ gadget must be rejected before deserialization.

        The endpoint should reject the import with HTTP 400 and the gadget marker
        function must never execute.
        """
        from fastapi import HTTPException

        zip_bytes = _build_malicious_faiss_zip("malicious_idx", description="evil", chunks=2, gadget_id="reduce_gadget_1")
        upload = _make_upload("malicious_idx.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            with self._patch_routes(service, fake_repo, fake_rag):
                # The import endpoint should reject this zip before unpickling
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )

                # Must reject with HTTP 400, not allow partial extraction
                self.assertEqual(ctx.exception.status_code, 400)

        # Critical: the gadget marker must NOT have been invoked
        self.assertEqual(
            _malicious_gadget_invocations,
            [],
            "Malicious __reduce__ gadget must not be invoked; pickle should be rejected before unpickling",
        )

    async def test_malicious_pickle_does_not_publish_metadata_or_upsert(self) -> None:
        """Malicious index.pkl rejection must occur before repository upsert."""
        from fastapi import HTTPException

        zip_bytes = _build_malicious_faiss_zip("evil_upsert", description="test", chunks=1, gadget_id="reduce_gadget_2")
        upload = _make_upload("evil_upsert.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            with self._patch_routes(service, fake_repo, fake_rag):
                try:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )
                except HTTPException:
                    pass  # Expected

        # Metadata upsert must NOT have been called
        fake_repo.upsert_index_metadata.assert_not_called()

    async def test_malicious_pickle_does_not_trigger_hot_load(self) -> None:
        """Malicious index.pkl rejection must occur before hot-load attempt."""
        from fastapi import HTTPException

        zip_bytes = _build_malicious_faiss_zip("evil_load", description="test", chunks=1, gadget_id="reduce_gadget_3")
        upload = _make_upload("evil_load.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            with self._patch_routes(service, fake_repo, fake_rag):
                try:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )
                except HTTPException:
                    pass  # Expected

        # Hot-load must NOT have been attempted
        fake_rag.load_faiss_index_from_metadata.assert_not_awaited()

    async def test_malicious_pickle_with_overwrite_prevents_execution(self) -> None:
        """Malicious pickle is rejected even with overwrite=True before any unpickling."""
        from fastapi import HTTPException

        zip_bytes = _build_malicious_faiss_zip("evil_with_overwrite", description="evil", chunks=1, gadget_id="reduce_gadget_4")
        upload = _make_upload("evil_with_overwrite.zip", zip_bytes)

        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp)
            # Mock repository to report existing metadata (enabling overwrite path)
            fake_repo.get_index_metadata.return_value = SimpleNamespace(
                name="evil_with_overwrite",
                path=str(service.index_base_path / "evil_with_overwrite"),
            )

            with self._patch_routes(service, fake_repo, fake_rag):
                try:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=True,
                        _user=_make_admin(),
                    )
                except HTTPException:
                    pass  # Expected due to malicious pickle

        # Critical: the gadget marker must NOT have been invoked, even with overwrite=True
        self.assertEqual(
            _malicious_gadget_invocations,
            [],
            "Malicious __reduce__ gadget must not be invoked even with overwrite=True",
        )


class ZipSlipContainmentTests(unittest.IsolatedAsyncioTestCase):
    """Verify that _extract_zip rejects members that would escape target_path."""

    def _make_service_and_patches(self, temp_path: Path) -> tuple[IndexerService, SimpleNamespace, SimpleNamespace]:
        service = IndexerService(index_base_path=str(temp_path))
        fake_repo = SimpleNamespace(
            get_index_metadata=AsyncMock(return_value=None),
            upsert_index_metadata=AsyncMock(),
        )
        fake_rag = SimpleNamespace(
            unload_index=lambda _name: None,
            load_faiss_index_from_metadata=AsyncMock(return_value=True),
        )
        return service, fake_repo, fake_rag

    async def _run_import(self, service: IndexerService, fake_repo: SimpleNamespace, fake_rag: SimpleNamespace, zip_bytes: bytes, index_name: str) -> None:
        upload = _make_upload(f"{index_name}.zip", zip_bytes)
        with (
            patch.object(_routes_module, "indexer", service),
            patch.object(_routes_module, "repository", fake_repo),
            patch.object(_routes_module, "rag", fake_rag),
        ):
            await import_faiss_index(
                file=upload,
                name=index_name,
                description=None,
                overwrite=False,
                _user=_make_admin(),
            )

    async def test_dotdot_traversal_member_is_not_extracted(self) -> None:
        """A member with ../ components must not be written outside target_path."""
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp.path)
            sentinel = temp.path / "escaped.txt"
            zip_bytes = _build_traversal_zip("safe_idx", evil_member=f"safe_idx/../../../escaped.txt")

            await self._run_import(service, fake_repo, fake_rag, zip_bytes, "safe_idx")

            self.assertFalse(sentinel.exists(), "Traversal member must not be extracted outside target_path")

    async def test_absolute_path_member_is_not_extracted(self) -> None:
        """A member with an absolute path must not be written to that absolute location."""
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp.path)
            # Use a path inside the temp dir but expressed absolutely so it
            # would bypass the relative-path check if not caught.
            evil_target = temp.path / "abs_escaped.txt"
            zip_bytes = _build_traversal_zip("abs_idx", evil_member=str(evil_target))

            await self._run_import(service, fake_repo, fake_rag, zip_bytes, "abs_idx")

            self.assertFalse(evil_target.exists(), "Absolute-path member must not be extracted")

    async def test_unexpected_basename_is_not_extracted(self) -> None:
        """Members whose basename is not index.faiss or index.pkl must be skipped."""
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp.path)
            zip_bytes = _build_traversal_zip("extra_idx", evil_member="extra_idx/evil_script.sh")

            await self._run_import(service, fake_repo, fake_rag, zip_bytes, "extra_idx")

            target = service.index_base_path / "extra_idx"
            self.assertFalse((target / "evil_script.sh").exists(), "Unexpected basename must not be extracted")

    async def test_valid_members_still_extracted_despite_evil_sibling(self) -> None:
        """Presence of a traversal member must not prevent valid files from being extracted."""
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp.path)
            zip_bytes = _build_traversal_zip("good_idx", evil_member="good_idx/../../../escaped.txt")

            await self._run_import(service, fake_repo, fake_rag, zip_bytes, "good_idx")

            target = service.index_base_path / "good_idx"
            self.assertTrue((target / "index.faiss").exists(), "index.faiss must still be extracted")
            self.assertTrue((target / "index.pkl").exists(), "index.pkl must still be extracted")

    async def test_zip_extraction_respects_target_path_boundary(self) -> None:
        """Verify that zip extraction writes only to the target index directory.

        This regression test ensures that even with various zip member paths
        (traversal, unexpected basenames), the extraction respects the target
        directory boundary and the implementation doesn't accidentally write
        files elsewhere.
        """
        with _TempDir() as temp:
            service, fake_repo, fake_rag = self._make_service_and_patches(temp.path)
            # Build a zip with valid FAISS files and an evil traversal member
            zip_bytes = _build_traversal_zip("safe_idx", evil_member="safe_idx/../../../escaped_outside.txt")

            upload = _make_upload("safe_idx.zip", zip_bytes)

            with (
                patch.object(_routes_module, "indexer", service),
                patch.object(_routes_module, "repository", fake_repo),
                patch.object(_routes_module, "rag", fake_rag),
            ):
                await import_faiss_index(
                    file=upload,
                    name="safe_idx",
                    description=None,
                    overwrite=False,
                    _user=_make_admin(),
                )

            # The index should be extracted to the intended location
            target = service.index_base_path / "safe_idx"
            self.assertTrue((target / "index.faiss").exists(), "index.faiss must be extracted to target path")
            self.assertTrue((target / "index.pkl").exists(), "index.pkl must be extracted to target path")
            # Verify evil traversal member was not extracted outside target
            escaped_file = service.index_base_path / "escaped_outside.txt"
            self.assertFalse(escaped_file.exists(), "Traversal member must not be extracted outside target path")


if __name__ == "__main__":
    unittest.main()
