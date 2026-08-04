from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "fix_inline_imports.py"
_MARKED_SOURCE = """async def load_settings():
    from ragtime.core.app_settings import get_app_settings  # inline-import: keep

    return await get_app_settings()
"""
_UNMARKED_SOURCE = _MARKED_SOURCE.replace("  # inline-import: keep", "")


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
