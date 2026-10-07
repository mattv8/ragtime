from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

FORBIDDEN_MODULES = (
    "prisma",
    "ragtime.core.database",
    "ragtime.core.app_settings",
    "ragtime.indexer.vector_utils",
)


def _run_with_forbidden_dependencies_blocked(body: str) -> subprocess.CompletedProcess[str]:
    guard = f"""
        import importlib
        import importlib.abc
        import sys

        forbidden = {FORBIDDEN_MODULES!r}
        attempted = []

        class ForbiddenImportBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                    attempted.append(fullname)
                    raise ImportError(f"forbidden import attempted: {{fullname}}")
                return None

        sys.meta_path.insert(0, ForbiddenImportBlocker())
    """
    assert_no_forbidden_modules = """
        assert not attempted, f"forbidden imports attempted: {attempted}"
        loaded = set(sys.modules)
        assert not any(name == forbidden_name or name.startswith(forbidden_name + ".") for forbidden_name in forbidden for name in loaded), loaded
    """
    script = "\n".join((textwrap.dedent(guard), textwrap.dedent(body), textwrap.dedent(assert_no_forbidden_modules)))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


class FaissWorkerImportTests(unittest.TestCase):
    def test_serialization_import_avoids_application_dependencies(self) -> None:
        result = _run_with_forbidden_dependencies_blocked('importlib.import_module("ragtime.indexer.faiss_serialization")')

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_finalizer_import_avoids_application_dependencies(self) -> None:
        result = _run_with_forbidden_dependencies_blocked('importlib.import_module("ragtime.indexer.faiss_artifacts")')

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_guard_rejects_a_swallowed_forbidden_import(self) -> None:
        result = _run_with_forbidden_dependencies_blocked(
            """
            try:
                importlib.import_module("ragtime.core.database")
            except ImportError:
                pass
            """
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden imports attempted", result.stderr)

    def test_finalizer_builds_and_safely_reloads_an_artifact_in_a_fresh_process(self) -> None:
        result = _run_with_forbidden_dependencies_blocked(
            """
            import json
            import struct
            import tempfile
            from pathlib import Path

            from ragtime.indexer.faiss_artifacts import _build_faiss_generation
            from ragtime.indexer.indexing_spool import EmbeddedBatch, IndexingSpool, SpoolRecord, SpoolTaskOutput

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spool = IndexingSpool.open_attempt(root / "spools", "job", "fingerprint")
                record = SpoolRecord("chunk", "source", 0, "text.txt", {"source": "source"}, 5)
                (spool.root / record.text_path).write_text("hello", encoding="utf-8")
                manifest = spool.root / "chunks.jsonl"
                manifest.write_text(json.dumps(record.__dict__) + "\\n", encoding="utf-8")
                output = SpoolTaskOutput(manifest.name, 1, 5, 0)
                spool.accept_documents(output)
                spool.accept_chunks(output)
                vectors = spool.root / "vectors.bin"
                vectors.write_bytes(struct.pack("<2f", 1, 2))
                spool.accept_embeddings(EmbeddedBatch((record,), vectors.name, 0, 1, 2))
                spool_root = spool.root
                spool.close()
                artifact = _build_faiss_generation(root / "index", spool_root, "l2", False)
                assert (artifact.generation_path / "index.faiss").is_file()
                assert (artifact.generation_path / "index.pkl").is_file()
            """
        )

        self.assertEqual(result.returncode, 0, result.stderr)
