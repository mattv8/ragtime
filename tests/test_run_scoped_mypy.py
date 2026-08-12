from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "run_scoped_mypy.py"


class RunScopedMypyCliTests(unittest.TestCase):
    def test_unset_scope_runs_default_targets(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run()

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                repo.recorded_args(),
                [
                    "--config-file",
                    "pyproject.toml",
                    "docker/scripts",
                    "ragtime",
                    "runtime",
                    "tests",
                ],
            )

    def test_all_scope_runs_default_targets(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope="all")

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                repo.recorded_args(),
                [
                    "--config-file",
                    "pyproject.toml",
                    "docker/scripts",
                    "ragtime",
                    "runtime",
                    "tests",
                ],
            )

    def test_none_scope_skips_mypy(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope="none")

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Skipping mypy", result.stdout)
            self.assertFalse(repo.record_file.exists())

    def test_files_scope_runs_exact_repo_relative_files(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope(["docker/scripts/helper.py", "tests/test_sample.py"]))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                repo.recorded_args(),
                [
                    "--config-file",
                    "pyproject.toml",
                    "docker/scripts/helper.py",
                    "tests/test_sample.py",
                ],
            )

    def test_empty_files_scope_skips_with_selector_message(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope([]))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("selector returned an empty Python file list", result.stdout)
            self.assertFalse(repo.record_file.exists())

    def test_malformed_base64_exits_nonzero(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope="files:not-base64!!")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid MYPY_SCOPE payload", result.stderr)

    def test_absolute_path_exits_nonzero(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope([str(repo.path / "tests/test_sample.py")]))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported MYPY_SCOPE path", result.stderr)

    def test_traversal_path_exits_nonzero(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope(["../outside.py"]))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported MYPY_SCOPE path", result.stderr)

    def test_missing_path_exits_nonzero(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope(["tests/missing.py"]))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MYPY_SCOPE path does not exist", result.stderr)

    def test_unknown_scope_exits_nonzero(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope="weird")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported MYPY_SCOPE", result.stderr)


class _RepoFixture:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temp_dir.name)
        self.record_file = self.path / "mypy-args.json"

    def __enter__(self) -> _RepoFixture:
        (self.path / "docker" / "scripts").mkdir(parents=True)
        (self.path / "ragtime").mkdir()
        (self.path / "runtime").mkdir()
        (self.path / "tests").mkdir()
        (self.path / "docker" / "scripts" / "helper.py").write_text("value = 1\n", encoding="utf-8")
        (self.path / "tests" / "test_sample.py").write_text("value = 1\n", encoding="utf-8")
        (self.path / "pyproject.toml").write_text("[tool.mypy]\n", encoding="utf-8")

        mypy_pkg = self.path / "mypy"
        mypy_pkg.mkdir()
        (mypy_pkg / "__init__.py").write_text("", encoding="utf-8")
        (mypy_pkg / "__main__.py").write_text(
            "import json\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['MYPY_RECORD_FILE']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._temp_dir.cleanup()

    def run(self, scope: str | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.path)
        env["MYPY_RECORD_FILE"] = str(self.record_file)
        if scope is None:
            env.pop("MYPY_SCOPE", None)
        else:
            env["MYPY_SCOPE"] = scope
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH)],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def recorded_args(self) -> list[str]:
        return json.loads(self.record_file.read_text(encoding="utf-8"))


def _encode_scope(paths: list[str]) -> str:
    payload = "".join(f"{path}\0" for path in paths).encode("utf-8")
    return f"files:{base64.b64encode(payload).decode('ascii')}"
