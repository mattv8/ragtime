from __future__ import annotations

import base64
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "changed_lint_files.py"
_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _decode_scope(scope: str) -> list[str]:
    if scope == "none":
        return []
    if scope == "all":
        return ["__ALL__"]
    prefix = "files:"
    if not scope.startswith(prefix):
        raise AssertionError(f"Unexpected scope: {scope}")
    payload = base64.b64decode(scope[len(prefix) :])
    return payload.decode("utf-8").split("\0")[:-1]


class ChangedLintFilesCliTests(unittest.TestCase):
    maxDiff = None

    def test_explicit_range_filters_paths_sorts_and_omits_deleted_files(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/keep.py", "print('base')\n")
            repo.write_file("ragtime/frontend/keep.tsx", "export const keep = 1;\n")
            repo.write_file("ragtime/frontend/delete.ts", "export const doomed = true;\n")
            repo.commit("base")
            base_ref = repo.rev_parse("HEAD")

            repo.write_file("tests/z_test.py", "print('z')\n")
            repo.commit("python second")
            repo.write_file("docker/scripts/a_script.py", "print('a')\n")
            repo.write_file("runtime/data.pyi", "value: int\n")
            repo.write_file("ragtime/frontend/view.tsx", "export const view = 1;\n")
            repo.write_file("ragtime/frontend/a-view.ts", "export const a = 1;\n")
            repo.write_file("ragtime/frontend/ignore.css", "body {}\n")
            repo.write_file("docs/ignored.py", "print('ignore')\n")
            repo.delete_file("ragtime/frontend/delete.ts")
            head_ref = repo.commit("mixed changes")

            result = repo.run_script("--base-ref", base_ref, "--head-ref", head_ref)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            scopes = _parse_output(result.stdout)
            self.assertEqual(
                _decode_scope(scopes["mypy_scope"]),
                ["docker/scripts/a_script.py", "runtime/data.pyi", "tests/z_test.py"],
            )
            self.assertEqual(
                _decode_scope(scopes["eslint_scope"]),
                ["ragtime/frontend/a-view.ts", "ragtime/frontend/view.tsx"],
            )

    def test_all_zero_base_uses_empty_tree(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/new.py", "print('new')\n")
            repo.write_file("ragtime/frontend/new.ts", "export const value = 1;\n")
            head_ref = repo.commit("initial")

            result = repo.run_script("--base-ref", "0" * 40, "--head-ref", head_ref)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            scopes = _parse_output(result.stdout)
            self.assertEqual(_decode_scope(scopes["mypy_scope"]), ["ragtime/new.py"])
            self.assertEqual(_decode_scope(scopes["eslint_scope"]), ["ragtime/frontend/new.ts"])

            raw_paths = repo.git("diff", "--name-only", f"{_EMPTY_TREE}..{head_ref}").stdout.splitlines()
            self.assertIn("ragtime/new.py", raw_paths)

    def test_local_mode_uses_upstream_and_includes_local_changes(self) -> None:
        with _GitRepo.with_clone() as repo:
            repo.write_file("ragtime/base.py", "print('base')\n")
            repo.write_file("ragtime/frontend/base.ts", "export const base = 1;\n")
            repo.write_file("ragtime/ignore.txt", "ignore\n")
            repo.commit("base")
            repo.push("origin", "HEAD")

            repo.write_file("runtime/committed.pyi", "value: int\n")
            repo.commit("ahead of upstream")

            repo.write_file("docker/scripts/staged.py", "print('staged')\n")
            repo.git("add", "docker/scripts/staged.py")
            repo.write_file("tests/unstaged.py", "print('unstaged')\n")
            repo.write_file("ragtime/frontend/untracked.tsx", "export const untracked = 1;\n")
            repo.write_file("ragtime/frontend/ignored.css", "body {}\n")
            repo.write_file("docs/ignored.py", "print('ignored')\n")
            repo.write_file("runtime/deleted.py", "print('deleted')\n")
            repo.git("add", "runtime/deleted.py")
            repo.delete_file("runtime/deleted.py")

            result = repo.run_script("--local")

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            scopes = _parse_output(result.stdout)
            self.assertEqual(
                _decode_scope(scopes["mypy_scope"]),
                [
                    "docker/scripts/staged.py",
                    "runtime/committed.pyi",
                    "tests/unstaged.py",
                ],
            )
            self.assertEqual(
                _decode_scope(scopes["eslint_scope"]),
                ["ragtime/frontend/untracked.tsx"],
            )

    def test_local_mode_without_upstream_fails_clearly(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/local.py", "print('local')\n")
            repo.commit("initial")

            result = repo.run_script("--local")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("upstream", result.stderr.lower())

    def test_missing_base_or_head_ref_falls_back_to_all_scopes(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/file.py", "print('ok')\n")
            repo.commit("initial")

            missing_base = repo.run_script("--base-ref", "missing-base", "--head-ref", "HEAD")
            self.assertEqual(missing_base.returncode, 0, msg=missing_base.stderr)
            self.assertEqual(
                _parse_output(missing_base.stdout),
                {"mypy_scope": "all", "eslint_scope": "all"},
            )

            missing_head = repo.run_script("--base-ref", "HEAD", "--head-ref", "missing-head")
            self.assertEqual(missing_head.returncode, 0, msg=missing_head.stderr)
            self.assertEqual(
                _parse_output(missing_head.stdout),
                {"mypy_scope": "all", "eslint_scope": "all"},
            )

    def test_cli_requires_exactly_one_mode(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/file.py", "print('ok')\n")
            repo.commit("initial")

            missing_pair = repo.run_script("--base-ref", "HEAD")
            self.assertNotEqual(missing_pair.returncode, 0)

            mixed_modes = repo.run_script("--local", "--base-ref", "HEAD", "--head-ref", "HEAD")
            self.assertNotEqual(mixed_modes.returncode, 0)

    def test_root_matching_does_not_treat_prefix_siblings_as_in_scope(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("ragtime/file.py", "print('base')\n")
            base_ref = repo.commit("base")
            repo.write_file("ragtime-other/file.py", "print('ignored')\n")
            repo.write_file("ragtime/frontendish/file.ts", "export const ignored = true;\n")
            head_ref = repo.commit("prefix siblings")

            result = repo.run_script("--base-ref", base_ref, "--head-ref", head_ref)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                _parse_output(result.stdout),
                {"mypy_scope": "none", "eslint_scope": "none"},
            )

    def test_range_with_only_ignored_changes_emits_none_scopes(self) -> None:
        with _GitRepo() as repo:
            repo.write_file("README.md", "base\n")
            base_ref = repo.commit("base")
            repo.write_file("README.md", "changed\n")
            repo.write_file("notes.txt", "ignored\n")
            head_ref = repo.commit("docs only")

            result = repo.run_script("--base-ref", base_ref, "--head-ref", head_ref)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                _parse_output(result.stdout),
                {"mypy_scope": "none", "eslint_scope": "none"},
            )


class _GitRepo:
    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temp_dir.name)

    def __enter__(self) -> _GitRepo:
        self.git("init")
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._temp_dir.cleanup()

    @classmethod
    def with_clone(cls) -> _ClonedGitRepo:
        return _ClonedGitRepo()

    def write_file(self, relative_path: str, content: str) -> None:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def delete_file(self, relative_path: str) -> None:
        (self.path / relative_path).unlink()

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.rev_parse("HEAD")

    def rev_parse(self, ref: str) -> str:
        return self.git("rev-parse", ref).stdout.strip()

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        )

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), *args],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=False,
        )


class _ClonedGitRepo(_GitRepo):
    def __init__(self) -> None:
        super().__init__()
        self._origin_dir = tempfile.TemporaryDirectory()
        self._clone_dir = tempfile.TemporaryDirectory()
        self._origin_path = Path(self._origin_dir.name)

    def __enter__(self) -> _ClonedGitRepo:
        subprocess.run(
            ["git", "init", "--bare", str(self._origin_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "clone", str(self._origin_path), str(self.path)],
            capture_output=True,
            text=True,
            check=True,
        )
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        super().__exit__(exc_type, exc, tb)
        self._origin_dir.cleanup()
        self._clone_dir.cleanup()

    def push(self, remote: str, ref: str) -> None:
        self.git("push", "-u", remote, ref)


def _parse_output(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line]
    if len(lines) != 2:
        raise AssertionError(f"Expected two output lines, got {lines!r}")
    result: dict[str, str] = {}
    for line in lines:
        key, value = line.split("=", 1)
        result[key] = value
    return result
