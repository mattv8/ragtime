from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "run_scoped_ruff.py"


class RunScopedRuffCliTests(unittest.TestCase):
    def test_unset_scope_checks_all_default_roots(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run()

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(repo.ruff_targets(), ["docker/scripts", "ragtime", "runtime", "tests"])

    def test_none_scope_skips_ruff(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope="none")

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Skipping Ruff", result.stdout)
            self.assertFalse(repo.record_file.exists())

    def test_files_scope_checks_exact_safe_python_files_with_end_of_options_marker(self) -> None:
        with _RepoFixture() as repo:
            result = repo.run(scope=_encode_scope(["docker/scripts/helper.py", "tests/name with spaces.py"]))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                repo.recorded_commands(),
                [
                    ["format", "--check", "--", "docker/scripts/helper.py", "tests/name with spaces.py"],
                    [
                        "check",
                        "--select",
                        "E9,F821,F822,F823,I",
                        "--",
                        "docker/scripts/helper.py",
                        "tests/name with spaces.py",
                    ],
                ],
            )

    def test_malformed_or_unsafe_scope_fails_closed(self) -> None:
        with _RepoFixture() as repo:
            malformed = repo.run(scope="files:not-base64!!")
            traversal = repo.run(scope=_encode_scope(["../outside.py"]))
            non_python = repo.run(scope=_encode_scope(["README.md"]))

            self.assertNotEqual(malformed.returncode, 0)
            self.assertIn("Invalid RUFF_SCOPE payload", malformed.stderr)
            self.assertNotEqual(traversal.returncode, 0)
            self.assertIn("Unsupported RUFF_SCOPE path", traversal.stderr)
            self.assertNotEqual(non_python.returncode, 0)
            self.assertIn("Unsupported RUFF_SCOPE path", non_python.stderr)

    def test_real_ruff_reports_format_and_lint_errors(self) -> None:
        with _RealRuffRepoFixture() as repo:
            repo.write_file("tests/bad.py", "import os\n\nvalue=missing_name\n")

            result = repo.run(_encode_scope(["tests/bad.py"]))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tests/bad.py", result.stdout)


class _RepoFixture:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temp_dir.name)
        self.record_file = self.path / "ruff-args.txt"

    def __enter__(self) -> _RepoFixture:
        for directory in ("docker/scripts", "ragtime", "runtime", "tests"):
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        (self.path / "docker/scripts/helper.py").write_text("value = 1\n", encoding="utf-8")
        (self.path / "tests/name with spaces.py").write_text("value = 1\n", encoding="utf-8")

        ruff_package = self.path / "ruff"
        ruff_package.mkdir()
        (ruff_package / "__init__.py").write_text("", encoding="utf-8")
        (ruff_package / "__main__.py").write_text(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['RUFF_RECORD_FILE']).open('a', encoding='utf-8') as record:\n"
            "    record.write('\\0'.join(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._temp_dir.cleanup()

    def run(self, scope: str | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.path)
        env["RUFF_RECORD_FILE"] = str(self.record_file)
        if scope is None:
            env.pop("RUFF_SCOPE", None)
        else:
            env["RUFF_SCOPE"] = scope
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH)],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def recorded_commands(self) -> list[list[str]]:
        return [line.split("\0") for line in self.record_file.read_text(encoding="utf-8").splitlines()]

    def ruff_targets(self) -> list[str]:
        return self.recorded_commands()[0][3:]


class _RealRuffRepoFixture:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temp_dir.name)

    def __enter__(self) -> _RealRuffRepoFixture:
        (self.path / "tests").mkdir()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._temp_dir.cleanup()

    def write_file(self, relative_path: str, content: str) -> None:
        (self.path / relative_path).write_text(content, encoding="utf-8")

    def run(self, scope: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["RUFF_SCOPE"] = scope
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH)],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


def _encode_scope(paths: list[str]) -> str:
    payload = "".join(f"{path}\0" for path in paths).encode("utf-8")
    return f"files:{base64.b64encode(payload).decode('ascii')}"
