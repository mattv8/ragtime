import subprocess
import unittest
from io import StringIO
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, patch

from ragtime.core.faiss_concurrency import FaissSearchBusyError
from ragtime.indexer.embedding_errors import EmbeddingFailureKind, EmbeddingOperationError
from ragtime.tools.git_history import _search_commits, _search_commits_semantic, search_git_history


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
        coordinator_run = AsyncMock(return_value=[(fake_doc, 0.2)])
        fake_embeddings = SimpleNamespace(aembed_query=AsyncMock(return_value=[0.1, 0.2]))
        similarity_search = object()
        fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score_by_vector=similarity_search)})

        with (
            patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
            patch("ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=fake_embeddings)),
            patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", AsyncMock(return_value=[])),
            patch("ragtime.rag.components.rag", fake_rag),
            patch("ragtime.tools.git_history.faiss_search_coordinator", SimpleNamespace(run=coordinator_run), create=True),
        ):
            matches = await _search_commits_semantic("infoscan", "semantic search", 3)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["subject"], "Add semantic search")
        coordinator_run.assert_awaited_once()
        await_args = coordinator_run.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.args[:2], ("infoscan", similarity_search))
        self.assertEqual(await_args.kwargs, {"k": 100})

    async def test_search_commits_semantic_accepts_precomputed_query_embedding(self) -> None:
        fake_doc = SimpleNamespace(
            page_content="[Commit 1234567890ab] Add semantic search",
            metadata={
                "type": "git_commit",
                "commit_hash": "1234567890abcdef",
                "author": "Test User",
                "date": "2026-01-02",
            },
        )
        coordinator_run = AsyncMock(return_value=[(fake_doc, 0.2)])
        fake_embeddings = SimpleNamespace(aembed_query=AsyncMock(return_value=[0.9, 0.8]))
        vector = [0.1, 0.2]
        similarity_search = object()
        fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score_by_vector=similarity_search)})

        with (
            patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
            patch("ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=fake_embeddings)),
            patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", AsyncMock(return_value=[])),
            patch("ragtime.rag.components.rag", fake_rag),
            patch("ragtime.tools.git_history.faiss_search_coordinator", SimpleNamespace(run=coordinator_run), create=True),
        ):
            matches = await _search_commits_semantic("infoscan", "semantic search", 3, query_embedding=vector)

        self.assertEqual(len(matches), 1)
        fake_embeddings.aembed_query.assert_not_awaited()
        await_args = coordinator_run.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.args[:2], ("infoscan", similarity_search))
        self.assertEqual(await_args.args[2], vector)

    async def test_search_commits_semantic_does_not_build_embeddings_client_when_query_vector_precomputed(self) -> None:
        fake_doc = SimpleNamespace(
            page_content="[Commit 1234567890ab] Add semantic search",
            metadata={
                "type": "git_commit",
                "commit_hash": "1234567890abcdef",
                "author": "Test User",
                "date": "2026-01-02",
            },
        )
        coordinator_run = AsyncMock(return_value=[(fake_doc, 0.2)])
        vector = [0.1, 0.2]
        similarity_search = object()
        fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score_by_vector=similarity_search)})

        with (
            patch("ragtime.indexer.vector_utils.get_embeddings_model", new=AsyncMock(side_effect=AssertionError("should not build embeddings client"))),
            patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", AsyncMock(return_value=[])),
            patch("ragtime.rag.components.rag", fake_rag),
            patch("ragtime.tools.git_history.faiss_search_coordinator", SimpleNamespace(run=coordinator_run), create=True),
        ):
            matches = await _search_commits_semantic("infoscan", "semantic search", 3, query_embedding=vector)

        self.assertEqual(len(matches), 1)

    async def test_search_git_history_falls_back_to_fuzzy_results_after_embedding_connection_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            indexes_dir = Path(directory) / "indexes"
            indexes_dir.mkdir(parents=True)
            repo = self._init_repo(indexes_dir)
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(repo, "src/search.ts", "semantic search\n", "Add semantic search")
            failure = EmbeddingOperationError(
                kind=EmbeddingFailureKind.CONNECTION,
                provider="ollama",
                model="nomic-embed-text:latest",
                operation="query",
                endpoint="http://private-host:11434",
                cause=RuntimeError("socket secret detail"),
            )
            log_stream = StringIO()
            handler = __import__("logging").StreamHandler(log_stream)
            logger_name = "ragtime.tools.git_history"

            with (
                patch("ragtime.tools.git_history._find_git_repos", AsyncMock(return_value=[("infoscan", repo)])),
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch(
                    "ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=SimpleNamespace(aembed_query=AsyncMock(side_effect=failure)))
                ),
            ):
                test_logger = __import__("logging").getLogger(logger_name)
                test_logger.addHandler(handler)
                try:
                    result = await search_git_history(action="search_commits", query="semantic search", index_name="infoscan", k=5)
                finally:
                    test_logger.removeHandler(handler)

        self.assertIn("Add semantic search", result)
        self.assertIn("fuzzy", result)
        self.assertNotIn("private-host", result)
        self.assertNotIn("socket secret detail", result)
        self.assertIn("Could not connect to the Ollama embedding server", log_stream.getvalue())

    async def test_search_git_history_falls_back_to_fuzzy_results_after_embedding_timeout_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            indexes_dir = Path(directory) / "indexes"
            indexes_dir.mkdir(parents=True)
            repo = self._init_repo(indexes_dir)
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(repo, "src/search.ts", "semantic search\n", "Add semantic search")
            failure = EmbeddingOperationError(
                kind=EmbeddingFailureKind.TIMEOUT,
                provider="ollama",
                model="nomic-embed-text:latest",
                operation="query",
                endpoint="http://private-host:11434",
                cause=RuntimeError("socket secret detail"),
            )

            with (
                patch("ragtime.tools.git_history._find_git_repos", AsyncMock(return_value=[("infoscan", repo)])),
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch(
                    "ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=SimpleNamespace(aembed_query=AsyncMock(side_effect=failure)))
                ),
            ):
                result = await search_git_history(action="search_commits", query="semantic search", index_name="infoscan", k=5)

        self.assertIn("Add semantic search", result)
        self.assertIn("fuzzy", result)
        self.assertNotIn("private-host", result)
        self.assertNotIn("socket secret detail", result)

    async def test_search_git_history_embeds_once_across_multiple_repositories(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a").mkdir()
            (base / "b").mkdir()
            (base / "c").mkdir()
            repo_a = self._init_repo(base / "a")
            repo_b = self._init_repo(base / "b")
            repo_c = self._init_repo(base / "c")
            for repo in (repo_a, repo_b, repo_c):
                _commit_file(repo, "README.md", "initial\n", "Initial commit")
                _commit_file(repo, "src/search.ts", "semantic search\n", f"Add semantic search {repo.name}")

            aembed_query = AsyncMock(return_value=[0.4, 0.5])
            similarity_a = object()
            similarity_b = object()
            similarity_c = object()
            fake_rag = SimpleNamespace(
                faiss_dbs={
                    "repo-a": SimpleNamespace(
                        similarity_search_with_score_by_vector=similarity_a,
                        embedding_function=SimpleNamespace(embed_query=mock.Mock(side_effect=AssertionError("should not re-embed"))),
                    ),
                    "repo-b": SimpleNamespace(
                        similarity_search_with_score_by_vector=similarity_b,
                        embedding_function=SimpleNamespace(embed_query=mock.Mock(side_effect=AssertionError("should not re-embed"))),
                    ),
                    "repo-c": SimpleNamespace(
                        similarity_search_with_score_by_vector=similarity_c,
                        embedding_function=SimpleNamespace(embed_query=mock.Mock(side_effect=AssertionError("should not re-embed"))),
                    ),
                }
            )

            async def _coordinator_result(index_name, _callable, _vector, *, k):
                fake_doc = SimpleNamespace(
                    page_content=f"[Commit 1234567890ab] Add semantic search {index_name}",
                    metadata={
                        "type": "git_commit",
                        "commit_hash": f"1234567890ab{index_name}",
                        "author": "Test User",
                        "date": "2026-01-02",
                    },
                )
                return [(fake_doc, 0.2)]

            with (
                patch("ragtime.tools.git_history._find_git_repos", AsyncMock(return_value=[("repo-a", repo_a), ("repo-b", repo_b), ("repo-c", repo_c)])),
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch("ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=SimpleNamespace(aembed_query=aembed_query))),
                patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", AsyncMock(return_value=[])),
                patch("ragtime.rag.components.rag", fake_rag),
                patch(
                    "ragtime.tools.git_history.faiss_search_coordinator", SimpleNamespace(run=AsyncMock(side_effect=_coordinator_result)), create=True
                ) as coordinator,
            ):
                result = await search_git_history(action="search_commits", query="semantic search", k=5)

        self.assertIn("=== repo-a ===", result)
        self.assertIn("=== repo-b ===", result)
        self.assertIn("=== repo-c ===", result)
        aembed_query.assert_awaited_once_with("semantic search")
        self.assertEqual(coordinator.run.await_count, 3)

    async def test_search_git_history_multi_repo_connection_failure_stays_fuzzy_only_after_first_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a").mkdir()
            (base / "b").mkdir()
            (base / "c").mkdir()
            repo_a = self._init_repo(base / "a")
            repo_b = self._init_repo(base / "b")
            repo_c = self._init_repo(base / "c")
            for repo in (repo_a, repo_b, repo_c):
                _commit_file(repo, "README.md", "initial\n", "Initial commit")
                _commit_file(repo, "src/search.ts", "semantic search\n", f"Add semantic search {repo.name}")

            failure = EmbeddingOperationError(
                kind=EmbeddingFailureKind.CONNECTION,
                provider="ollama",
                model="nomic-embed-text:latest",
                operation="query",
                endpoint="http://private-host:11434",
                cause=RuntimeError("socket secret detail"),
            )
            aembed_query = AsyncMock(side_effect=failure)
            embed_query_a = mock.Mock(side_effect=AssertionError("repo-a should not re-embed"))
            embed_query_b = mock.Mock(side_effect=AssertionError("repo-b should not re-embed"))
            embed_query_c = mock.Mock(side_effect=AssertionError("repo-c should not re-embed"))
            fake_rag = SimpleNamespace(
                faiss_dbs={
                    "repo-a": SimpleNamespace(similarity_search_with_score_by_vector=object(), embedding_function=SimpleNamespace(embed_query=embed_query_a)),
                    "repo-b": SimpleNamespace(similarity_search_with_score_by_vector=object(), embedding_function=SimpleNamespace(embed_query=embed_query_b)),
                    "repo-c": SimpleNamespace(similarity_search_with_score_by_vector=object(), embedding_function=SimpleNamespace(embed_query=embed_query_c)),
                }
            )

            with (
                patch("ragtime.tools.git_history._find_git_repos", AsyncMock(return_value=[("repo-a", repo_a), ("repo-b", repo_b), ("repo-c", repo_c)])),
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch("ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=SimpleNamespace(aembed_query=aembed_query))),
                patch("ragtime.indexer.vector_utils.search_pgvector_embeddings", AsyncMock(return_value=[])),
                patch("ragtime.rag.components.rag", fake_rag),
                patch(
                    "ragtime.tools.git_history.faiss_search_coordinator",
                    SimpleNamespace(run=AsyncMock(side_effect=AssertionError("semantic FAISS should not run after aggregate embed failure"))),
                    create=True,
                ),
            ):
                result = await search_git_history(action="search_commits", query="semantic search", k=5)

        self.assertIn("=== repo-a ===", result)
        self.assertIn("=== repo-b ===", result)
        self.assertIn("=== repo-c ===", result)
        self.assertIn("fuzzy", result)
        aembed_query.assert_awaited_once_with("semantic search")
        embed_query_a.assert_not_called()
        embed_query_b.assert_not_called()
        embed_query_c.assert_not_called()

    async def test_search_git_history_multi_repo_configuration_failure_stays_fuzzy_only_after_first_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a").mkdir()
            (base / "b").mkdir()
            repo_a = self._init_repo(base / "a")
            repo_b = self._init_repo(base / "b")
            for repo in (repo_a, repo_b):
                _commit_file(repo, "README.md", "initial\n", "Initial commit")
                _commit_file(repo, "src/search.ts", "semantic search\n", f"Add semantic search {repo.name}")

            with (
                patch("ragtime.tools.git_history._find_git_repos", AsyncMock(return_value=[("repo-a", repo_a), ("repo-b", repo_b)])),
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch(
                    "ragtime.indexer.vector_utils.get_embeddings_model",
                    AsyncMock(
                        side_effect=EmbeddingOperationError(
                            kind=EmbeddingFailureKind.CONFIGURATION,
                            provider="openai",
                            model="text-embedding-3-small",
                            operation="configure",
                            endpoint=None,
                        )
                    ),
                ),
            ):
                result = await search_git_history(action="search_commits", query="semantic search", k=5)

        self.assertIn("=== repo-a ===", result)
        self.assertIn("=== repo-b ===", result)
        self.assertIn("fuzzy", result)

    async def test_busy_semantic_faiss_search_falls_back_to_fuzzy_results(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            repo = self._init_repo(Path(directory))
            _commit_file(repo, "README.md", "initial\n", "Initial commit")
            _commit_file(repo, "src/search.ts", "semantic search\n", "Add semantic search")

            fake_embeddings = SimpleNamespace(aembed_query=AsyncMock(return_value=[0.1, 0.2]))
            fake_rag = SimpleNamespace(faiss_dbs={"infoscan": SimpleNamespace(similarity_search_with_score=object())})

            with (
                patch("ragtime.core.app_settings.get_app_settings", AsyncMock(return_value={})),
                patch("ragtime.indexer.vector_utils.get_embeddings_model", AsyncMock(return_value=fake_embeddings)),
                patch(
                    "ragtime.indexer.vector_utils.search_pgvector_embeddings",
                    AsyncMock(return_value=[]),
                ),
                patch("ragtime.rag.components.rag", fake_rag),
                patch(
                    "ragtime.tools.git_history.faiss_search_coordinator",
                    SimpleNamespace(run=AsyncMock(side_effect=FaissSearchBusyError("busy"))),
                    create=True,
                ),
            ):
                result = await _search_commits(repo, "semantic search", k=5, index_name="infoscan")

        self.assertIn("Add semantic search", result)
        self.assertIn("fuzzy", result)


if __name__ == "__main__":
    unittest.main()
