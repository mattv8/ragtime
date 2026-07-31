import subprocess
import unittest
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from unittest.mock import patch

from ragtime.core.faiss_concurrency import FaissSearchBusyError
from ragtime.tools.git_history import _search_commits, _search_commits_semantic


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

    async def test_semantic_commit_search_uses_faiss_search_coordinator_for_loaded_in_memory_index(self):
        fake_doc = SimpleNamespace(
            page_content="[Commit 1234567890ab] Add semantic search",
            metadata={
                "type": "git_commit",
                "commit_hash": "1234567890abcdef",
                "author": "Test User",
                "date": "2026-01-02",
            },
        )
        coordinator_run = unittest.mock.AsyncMock(return_value=[(fake_doc, 0.2)])
        fake_embeddings = SimpleNamespace(aembed_query=unittest.mock.AsyncMock(return_value=[0.1, 0.2]))
        fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score=object())})

        with (
            patch("ragtime.core.app_settings.get_app_settings", unittest.mock.AsyncMock(return_value={})),
            patch("ragtime.indexer.vector_utils.get_embeddings_model", unittest.mock.AsyncMock(return_value=fake_embeddings)),
            patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", unittest.mock.AsyncMock(return_value=[])),
            patch("ragtime.rag.components.rag", fake_rag),
            patch("ragtime.tools.git_history.faiss_search_coordinator", SimpleNamespace(run=coordinator_run), create=True),
        ):
            matches = await _search_commits_semantic("infoscan", "semantic search", 3)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["subject"], "Add semantic search")
        coordinator_run.assert_awaited_once()
        self.assertEqual(coordinator_run.await_args.args[:2], ("infoscan", fake_rag.faiss_dbs["infoscan"].similarity_search_with_score))
        self.assertEqual(coordinator_run.await_args.kwargs, {"k": 100})

    async def test_busy_semantic_faiss_search_falls_back_to_fuzzy_results(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            repo = self._init_repo(Path(directory))
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(repo, "src/search.ts", "semantic search\n", "Add semantic search")

            fake_embeddings = SimpleNamespace(aembed_query=unittest.mock.AsyncMock(return_value=[0.1, 0.2]))
            fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score=object())})

            with (
                patch("ragtime.core.app_settings.get_app_settings", unittest.mock.AsyncMock(return_value={})),
                patch("ragtime.indexer.vector_utils.get_embeddings_model", unittest.mock.AsyncMock(return_value=fake_embeddings)),
                patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", unittest.mock.AsyncMock(return_value=[])),
                patch("ragtime.rag.components.rag", fake_rag),
                patch(
                    "ragtime.tools.git_history.faiss_search_coordinator",
                    SimpleNamespace(run=unittest.mock.AsyncMock(side_effect=FaissSearchBusyError("busy"))),
                    create=True,
                ),
            ):
                result = await _search_commits(repo, "semantic search", k=5, index_name="infoscan")

        self.assertIn("Add semantic search", result)
        self.assertIn("fuzzy", result)


if __name__ == "__main__":
    unittest.main()
