import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from unittest import mock

from ragtime.git_webhooks.models import GitWebhookDelivery, GitWebhookDeliveryStatus, GitWebhookTarget, GitWebhookTargetType
from ragtime.git_webhooks.repository import format_git_webhook_target_key
from ragtime.git_webhooks.service import GitWebhookService
from ragtime.indexer.models import IndexConfig, IndexJob, IndexStatus
from ragtime.indexer.service import IndexerService


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_no_cwd(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _configure_git_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")


def _commit_file(repo: Path, relative_path: str, content: str, message: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _push(repo: Path, branch: str = "main") -> None:
    _git(repo, "push", "origin", branch)


def _build_git_remote(tmpdir: str) -> tuple[Path, Path, Path, list[str]]:
    root = Path(tmpdir)
    remote = root / "remote.git"
    seed = root / "seed"
    persistent_checkout = root / ".git_repo"

    _git_no_cwd("init", "--bare", str(remote))
    _git_no_cwd("clone", str(remote), str(seed))
    _configure_git_identity(seed)
    _git(seed, "checkout", "-b", "main")

    commit_shas = [
        _commit_file(seed, "docs/readme.md", "alpha\n", "commit one"),
        _commit_file(seed, "docs/readme.md", "alpha\nbeta\n", "commit two"),
        _commit_file(seed, "docs/readme.md", "alpha\nbeta\ngamma\n", "commit three"),
        _commit_file(seed, "docs/readme.md", "alpha\nbeta\ngamma\ndelta\n", "commit four"),
    ]
    _push(seed)
    _git_no_cwd("-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    _git_no_cwd("clone", str(remote), str(persistent_checkout))

    return remote, seed, persistent_checkout, commit_shas


@dataclass
class _FakeCompletedJob:
    id: str = "job-1"
    status: str = IndexStatus.COMPLETED.value
    error_message: str | None = None


class _CaptureIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, IndexConfig, str | None]] = []

    async def try_create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> _FakeCompletedJob:
        self.calls.append((git_url, branch, config, git_token))
        return _FakeCompletedJob()


class _CaptureJobs:
    async def get_active_job_for_index(self, _name: str):
        return None

    async def get_job(self, _job_id: str) -> _FakeCompletedJob:
        return _FakeCompletedJob()


class _SingleDeliveryRepo:
    def __init__(self, target: GitWebhookTarget) -> None:
        self.target = target
        self.claimed = False
        self.linked_job_ids: list[tuple[str, str]] = []
        self.completed: list[tuple[str, GitWebhookDeliveryStatus, str, str | None]] = []

    async def resolve_target(self, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookTarget | None:
        if target_type is self.target.target_type and target_id == self.target.target_id:
            return self.target
        return None

    async def claim_latest_pending(self, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookDelivery | None:
        if self.claimed or target_type is not self.target.target_type or target_id != self.target.target_id:
            return None
        self.claimed = True
        return GitWebhookDelivery(
            id="delivery-1",
            target_type=target_type,
            target_id=target_id,
            provider_delivery_id="provider-1",
            event_name="push",
            branch=self.target.branch,
            head_commit="head-1",
            status=GitWebhookDeliveryStatus.PENDING,
            received_at=datetime.now(timezone.utc),
        )

    async def link_index_job(self, delivery_id: str, index_job_id: str) -> None:
        self.linked_job_ids.append((delivery_id, index_job_id))

    async def complete(
        self,
        delivery_id: str,
        *,
        status: GitWebhookDeliveryStatus,
        message: str,
        index_job_id: str | None = None,
    ) -> None:
        self.completed.append((delivery_id, status, message, index_job_id))

    async def defer_claim(self, _delivery_id: str) -> None:
        raise AssertionError("defer_claim should not be used in this contract test")


class _StopAfterDocuments(RuntimeError):
    pass


class GitWebhookIndexingContractTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if which("git") is None:
            raise unittest.SkipTest("git executable is required for git webhook indexing contract tests")

    async def test_fetch_git_updates_resets_persistent_checkout_to_latest_branch_tip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            remote, seed, repo_dir, commit_shas = _build_git_remote(tmpdir)
            initial_sha = _git(repo_dir, "rev-parse", "HEAD")
            self.assertEqual(initial_sha, commit_shas[-1])

            latest_remote_sha = _commit_file(seed, "docs/readme.md", "alpha\nbeta\ngamma\ndelta\nepsilon\n", "commit five")
            _push(seed)

            service = IndexerService(index_base_path=str(Path(tmpdir) / "indexes"))
            job = IndexJob(
                id="job-fetch",
                name="contract-index",
                config=IndexConfig(name="contract-index", git_history_depth=1),
                source_type="git",
                git_url=remote.as_uri(),
                git_branch="main",
            )

            with mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock()):
                await service._fetch_git_updates(job, repo_dir)

            self.assertNotEqual(initial_sha, latest_remote_sha)
            self.assertEqual(_git(repo_dir, "rev-parse", "HEAD"), latest_remote_sha)
            self.assertEqual(_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD"), "main")

    async def test_webhook_dispatch_preserves_git_history_depth_in_created_job_config(self) -> None:
        target = GitWebhookTarget(
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id="index-1",
            webhook_id="wh-index-1",
            secret="secret",
            provider="github",
            branch="main",
            created_at=datetime.now(timezone.utc),
            name="contract-index",
            key=format_git_webhook_target_key(GitWebhookTargetType.GIT_INDEX, "index-1"),
            source="file:///tmp/remote.git",
            git_token="token-1",
            config_snapshot={"git_history_depth": 3, "file_patterns": ["**/*"]},
        )
        repo = _SingleDeliveryRepo(target)
        indexer = _CaptureIndexer()
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=object(), index_jobs=_CaptureJobs(), poll_seconds=0.01)

        progressed = await service._drain_git_target(target.target_id)

        self.assertTrue(progressed)
        self.assertEqual(len(indexer.calls), 1)
        _, branch, config, git_token = indexer.calls[0]
        self.assertEqual(branch, "main")
        self.assertEqual(git_token, "token-1")
        self.assertEqual(config.git_history_depth, 3)

    async def test_git_history_depth_controls_added_commit_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, repo_dir, commit_shas = _build_git_remote(tmpdir)
            depth_one_docs = await self._capture_documents_for_depth(repo_dir, depth=1)
            depth_three_docs = await self._capture_documents_for_depth(repo_dir, depth=3)
            full_depth_docs = await self._capture_documents_for_depth(repo_dir, depth=0)

            depth_one_history = [doc for doc in depth_one_docs if doc.metadata.get("type") == "git_commit"]
            depth_three_history = [doc for doc in depth_three_docs if doc.metadata.get("type") == "git_commit"]
            full_depth_history = [doc for doc in full_depth_docs if doc.metadata.get("type") == "git_commit"]

            self.assertEqual(depth_one_history, [])
            self.assertEqual(len(depth_three_history), 3)
            self.assertEqual([doc.metadata.get("commit_hash") for doc in depth_three_history], list(reversed(commit_shas[-3:])))
            self.assertEqual(len(full_depth_history), len(commit_shas))
            self.assertEqual([doc.metadata.get("commit_hash") for doc in full_depth_history], list(reversed(commit_shas)))

    async def _capture_documents_for_depth(self, repo_dir: Path, *, depth: int):
        captured: dict[str, list] = {}
        service = IndexerService(index_base_path=str(repo_dir.parent / f"indexes-{depth}"))
        job = IndexJob(
            id=f"job-depth-{depth}",
            name=f"contract-index-{depth}",
            config=IndexConfig(name=f"contract-index-{depth}", git_history_depth=depth),
            source_type="git",
            git_url=repo_dir.parent.joinpath("remote.git").as_uri(),
            git_branch="main",
        )

        async def capture_chunk_documents(*, documents, **_kwargs):
            captured["documents"] = documents
            raise _StopAfterDocuments("captured documents before chunking")

        with (
            mock.patch("ragtime.indexer.service.repository.update_job", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.indexer.service.repository.get_settings",
                new=mock.AsyncMock(return_value=SimpleNamespace(chunking_use_tokens=False)),
            ),
            mock.patch("ragtime.indexer.service.chunk_documents_parallel", side_effect=capture_chunk_documents),
        ):
            with self.assertRaises(_StopAfterDocuments):
                await service._create_faiss_index(job, repo_dir)

        return captured["documents"]


if __name__ == "__main__":
    unittest.main()
