from __future__ import annotations

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
from unittest.mock import AsyncMock, patch

import ragtime.indexer.routes as _routes_module
from ragtime.indexer.routes import import_faiss_index
from ragtime.indexer.service import IndexerService


class _FakeInMemoryDocstore:
    """Minimal stand-in for langchain's InMemoryDocstore used by FAISS.

    The import endpoint only inspects ``_dict`` to count chunks, so a
    dict-like object is enough.
    """

    def __init__(self, items: dict[str, Any]) -> None:
        self._dict = items


def _build_faiss_zip(name: str, *, description: str, chunks: int) -> bytes:
    """Build a zip that mirrors the structure produced by download_index."""

    docstore = _FakeInMemoryDocstore({f"doc-{i}": SimpleNamespace(page_content=f"chunk {i}", metadata={}) for i in range(chunks)})
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
        zf.writestr(f"{name}/index.faiss", b"fake faiss data")
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
    async def test_import_faiss_index_writes_files_and_metadata(self) -> None:
        description = "Custom description for odev_proj re-import"
        zip_bytes = _build_faiss_zip("odev_proj", description=description, chunks=7)
        upload = _make_upload("odev_proj_index.zip", zip_bytes)

        with _TempDir() as temp:
            service = IndexerService(index_base_path=str(temp.path))

            fake_repo = SimpleNamespace(
                get_index_metadata=AsyncMock(return_value=None),
                upsert_index_metadata=AsyncMock(),
            )
            fake_rag = SimpleNamespace(
                unload_index=lambda _name: None,
                load_faiss_index_from_metadata=AsyncMock(return_value=True),
            )

            with (
                patch.object(_routes_module, "indexer", service),
                patch.object(_routes_module, "repository", fake_repo),
                patch.object(_routes_module, "rag", fake_rag),
            ):
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

    async def test_import_faiss_index_rejects_missing_faiss_files(self) -> None:
        from fastapi import HTTPException

        zip_bytes = _build_metadata_zip("broken")
        upload = _make_upload("broken.zip", zip_bytes)

        with _TempDir() as temp:
            service = IndexerService(index_base_path=str(temp.path))

            fake_repo = SimpleNamespace(
                get_index_metadata=AsyncMock(return_value=None),
                upsert_index_metadata=AsyncMock(),
            )
            fake_rag = SimpleNamespace(
                unload_index=lambda _name: None,
                load_faiss_index_from_metadata=AsyncMock(),
            )

            with (
                patch.object(_routes_module, "indexer", service),
                patch.object(_routes_module, "repository", fake_repo),
                patch.object(_routes_module, "rag", fake_rag),
            ):
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

        docstore = _FakeInMemoryDocstore({})
        pkl_bytes = pickle.dumps((docstore, {}))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("myindex/index.faiss", b"x")
            zf.writestr("myindex/index.pkl", pkl_bytes)
        zip_bytes = buf.getvalue()

        upload = _make_upload("myindex_index.zip", zip_bytes)

        with _TempDir() as temp:
            service = IndexerService(index_base_path=str(temp.path))

            fake_repo = SimpleNamespace(
                get_index_metadata=AsyncMock(return_value=None),
                upsert_index_metadata=AsyncMock(),
            )
            fake_rag = SimpleNamespace(
                unload_index=lambda _name: None,
                load_faiss_index_from_metadata=AsyncMock(return_value=True),
            )

            with (
                patch.object(_routes_module, "indexer", service),
                patch.object(_routes_module, "repository", fake_repo),
                patch.object(_routes_module, "rag", fake_rag),
            ):
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
            service = IndexerService(index_base_path=str(temp.path))
            (service.index_base_path / "odev_proj").mkdir()
            (service.index_base_path / "odev_proj" / "index.faiss").write_bytes(b"x")

            # Repository reports the index already has metadata
            fake_repo = SimpleNamespace(
                get_index_metadata=AsyncMock(
                    return_value=SimpleNamespace(
                        name="odev_proj",
                        path=str(service.index_base_path / "odev_proj"),
                    )
                ),
                upsert_index_metadata=AsyncMock(),
            )
            fake_rag = SimpleNamespace(
                unload_index=lambda _name: None,
                load_faiss_index_from_metadata=AsyncMock(),
            )

            with (
                patch.object(_routes_module, "indexer", service),
                patch.object(_routes_module, "repository", fake_repo),
                patch.object(_routes_module, "rag", fake_rag),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await import_faiss_index(
                        file=upload,
                        name=None,
                        description=None,
                        overwrite=False,
                        _user=_make_admin(),
                    )

        self.assertEqual(ctx.exception.status_code, 409)


def _build_traversal_zip(name: str, *, evil_member: str) -> bytes:
    """Build a zip that contains a traversal member alongside valid FAISS files.

    The evil_member path is injected verbatim into the zip so the extraction
    code must reject it rather than writing outside the target directory.
    """
    docstore = _FakeInMemoryDocstore({})
    pkl_bytes = pickle.dumps((docstore, {}))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/index.faiss", b"fake faiss data")
        zf.writestr(f"{name}/index.pkl", pkl_bytes)
        # Inject the malicious member directly
        zf.writestr(evil_member, b"evil payload")
    return buf.getvalue()


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


if __name__ == "__main__":
    unittest.main()
