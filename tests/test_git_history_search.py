import subprocess
import unittest
from pathlib import Path
from shutil import which
from unittest.mock import patch

from ragtime.tools.git_history import _search_commits


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _commit_file(repo: Path, relative_path: str, content: str, message: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", message)


async def _no_semantic_matches(*_args, **_kwargs):
    return []


class GitHistorySearchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if which("git") is None:
            raise unittest.SkipTest("git executable is required for git history tests")

    def _init_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")
        return repo

    async def test_multi_word_commit_search_does_not_require_exact_phrase(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            repo = self._init_repo(Path(directory))
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(repo, "src/flow.ts", "step flow\n", "Implement step flow v2 overhaul")

            with patch("ragtime.tools.git_history._search_commits_semantic", _no_semantic_matches):
                result = await _search_commits(
                    repo,
                    "infoscan2 step flow overhaul",
                    k=5,
                    index_name="infoscan",
                )

            self.assertIn("Implement step flow v2 overhaul", result)
            self.assertNotIn("No commits found", result)

    async def test_commit_search_matches_changed_file_names(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            repo = self._init_repo(Path(directory))
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(
                repo,
                "src/next_step/wo_step_update.ts",
                "export const nextStep = true;\n",
                "Refactor workflow routing",
            )

            with patch("ragtime.tools.git_history._search_commits_semantic", _no_semantic_matches):
                result = await _search_commits(
                    repo,
                    "step_flow v2 react code-backed next_step wo_step_update",
                    k=5,
                    index_name="infoscan",
                )

            self.assertIn("Refactor workflow routing", result)
            self.assertNotIn("No commits found", result)


if __name__ == "__main__":
    unittest.main()
