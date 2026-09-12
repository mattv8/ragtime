from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "fix_inline_imports.py"
_MARKED_SOURCE = """async def load_settings():
    from ragtime.core.app_settings import get_app_settings  # inline-import: keep

    return await get_app_settings()
"""
_UNMARKED_SOURCE = _MARKED_SOURCE.replace("  # inline-import: keep", "")
_ORIGINAL_DUPLICATE_TAIL_LINE_RE = re.compile(r"(?P<line>[^\n]*\S[^\n]*)(?:\n(?P=line)){1,2}\n?\Z")


def _load_inline_imports_module():
    spec = importlib.util.spec_from_file_location("fix_inline_imports_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load inline import helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collapse_duplicate_tail_lines = getattr(_load_inline_imports_module(), "collapse_duplicate_tail_lines")


def _original_collapse_duplicate_tail_lines(text: str) -> tuple[str, int]:
    match = _ORIGINAL_DUPLICATE_TAIL_LINE_RE.search(text)
    if not match:
        return text, 0
    block = match.group(0)
    line = match.group("line")
    duplicates_removed = len(block.rstrip("\n").split("\n")) - 1
    cleaned = text[: match.start()] + line
    if block.endswith("\n"):
        cleaned += "\n"
    return cleaned, duplicates_removed


class InlineImportSuppressionTests(unittest.TestCase):
    def _run_script(self, source: str, mode: str) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module_path = root / "sample.py"
            module_path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), str(root), mode],
                capture_output=True,
                text=True,
                check=False,
            )
            return result, module_path.read_text(encoding="utf-8")

    def test_marked_inline_import_is_kept_by_check_and_apply(self) -> None:
        for mode in ("--check", "--apply"):
            with self.subTest(mode=mode):
                result, rewritten = self._run_script(_MARKED_SOURCE, mode)

                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                self.assertEqual(rewritten, _MARKED_SOURCE)

    def test_unmarked_inline_import_still_fails_check(self) -> None:
        result, unchanged = self._run_script(_UNMARKED_SOURCE, "--check")

        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("Inline import check failed", result.stdout)
        self.assertEqual(unchanged, _UNMARKED_SOURCE)

    def test_inline_import_that_breaks_local_cycle_is_kept(self) -> None:
        source = """def load_b():
    from . import b

    return b
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "sample_package"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            module_path = package_dir / "a.py"
            module_path.write_text(source, encoding="utf-8")
            (package_dir / "b.py").write_text("from . import a\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), str(package_dir), "--check"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertEqual(module_path.read_text(encoding="utf-8"), source)


class CollapseDuplicateTailLinesTests(unittest.TestCase):
    def test_expected_tail_collapses_preserve_line_forms(self) -> None:
        cases = (
            ("one\none\n", "one\n", 1),
            ("one\none\none\n", "one\n", 2),
            ("one\none\none\none\n", "one\none\n", 2),
            ("one\none", "one", 1),
            ("  one\n  one\n", "  one\n", 1),
            ("prefixX\nX\nX\n", "prefixX\n", 2),
            ("one\n \n \n", "one\n \n \n", 0),
            ("one\ntwo\n", "one\ntwo\n", 0),
        )
        for source, expected, removed in cases:
            with self.subTest(source=source):
                self.assertEqual(collapse_duplicate_tail_lines(source), (expected, removed))

    def test_matches_original_regex_for_many_bounded_inputs(self) -> None:
        alphabet = ("X", " ", "\t", "\r", "\n")
        for length in range(7):
            for chars in product(alphabet, repeat=length):
                source = "".join(chars)
                with self.subTest(source=repr(source)):
                    self.assertEqual(
                        collapse_duplicate_tail_lines(source),
                        _original_collapse_duplicate_tail_lines(source),
                    )

    def test_long_nonduplicate_line_is_unchanged(self) -> None:
        source = "X" * 100_000
        self.assertEqual(collapse_duplicate_tail_lines(source), (source, 0))
