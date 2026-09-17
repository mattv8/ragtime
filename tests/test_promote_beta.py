import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

SCRIPT_PATH = Path(__file__).parents[1] / "docker/scripts/promote_beta.py"
SPEC = importlib.util.spec_from_file_location("promote_beta", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
promote_beta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote_beta)


class PromoteBetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.origin = self.root / "origin.git"
        self.checkout = self.root / "checkout"
        self._git(self.root, "init", "--bare", str(self.origin))
        self._git(self.root, "clone", str(self.origin), str(self.checkout))
        self._configure(self.checkout)
        (self.checkout / "README.md").write_text("initial\n")
        self._git(self.checkout, "add", "README.md")
        self._git(self.checkout, "commit", "-m", "initial")
        self._git(self.checkout, "branch", "-M", "main")
        self._git(self.checkout, "push", "-u", "origin", "main")
        self._git(self.checkout, "checkout", "-b", "beta")
        self._git(self.checkout, "push", "-u", "origin", "beta")
        self._commit_on_beta("candidate")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _git(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *arguments], cwd=directory, check=True, text=True, capture_output=True)

    def _configure(self, directory: Path) -> None:
        self._git(directory, "config", "user.name", "Test Committer")
        self._git(directory, "config", "user.email", "test@example.invalid")

    def _commit_on_beta(self, content: str) -> str:
        (self.checkout / "README.md").write_text(content + "\n")
        self._git(self.checkout, "add", "README.md")
        self._git(self.checkout, "commit", "-m", content)
        self._git(self.checkout, "push", "origin", "beta")
        return self._git(self.checkout, "rev-parse", "HEAD").stdout.strip()

    def _main(self, *arguments: str) -> int:
        return promote_beta.main(["--repo-root", str(self.checkout), *arguments])

    @staticmethod
    def _workflow_run(
        sha: str,
        *,
        branch: str = "beta",
        event: str = "push",
        status: str = "completed",
        conclusion: Optional[str] = "success",
    ) -> dict:
        return {
            "id": 1,
            "name": "Build and Push Container",
            "head_branch": branch,
            "head_sha": sha,
            "event": event,
            "status": status,
            "conclusion": conclusion,
        }

    @classmethod
    def _success_evidence(cls, _repo_slug: str, sha: str) -> dict:
        return {"total_count": 1, "workflow_runs": [cls._workflow_run(sha)]}

    def _origin_ref(self, branch: str) -> str:
        return self._git(self.origin, "rev-parse", "refs/heads/" + branch).stdout.strip()

    def test_happy_path_fast_forwards_main(self) -> None:
        candidate = self._origin_ref("beta")
        with (
            mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
            mock.patch.object(promote_beta, "query_workflow_runs", self._success_evidence),
        ):
            self.assertEqual(self._main(), 0)
        self.assertEqual(self._origin_ref("main"), candidate)

    def test_rejects_candidate_when_main_is_not_ancestor(self) -> None:
        self._git(self.checkout, "checkout", "main")
        (self.checkout / "main-only.txt").write_text("new main\n")
        self._git(self.checkout, "add", "main-only.txt")
        self._git(self.checkout, "commit", "-m", "advance main")
        self._git(self.checkout, "push", "origin", "main")
        with mock.patch.object(promote_beta, "query_workflow_runs", self._success_evidence):
            self.assertEqual(self._main(), 1)

    def test_rejects_sha_outside_beta_ancestry(self) -> None:
        self._git(self.checkout, "checkout", "main")
        (self.checkout / "not-beta.txt").write_text("not beta\n")
        self._git(self.checkout, "add", "not-beta.txt")
        self._git(self.checkout, "commit", "-m", "not beta")
        outside_sha = self._git(self.checkout, "rev-parse", "HEAD").stdout.strip()
        with mock.patch.object(promote_beta, "query_workflow_runs", self._success_evidence):
            self.assertEqual(self._main("--sha", outside_sha), 1)

    def test_non_force_push_rejects_concurrent_main_update(self) -> None:
        racer = self.root / "racer"

        def advance_main(repo_slug: str, sha: str) -> dict:
            self._git(self.root, "clone", str(self.origin), str(racer))
            self._configure(racer)
            self._git(racer, "checkout", "main")
            (racer / "race.txt").write_text("race\n")
            self._git(racer, "add", "race.txt")
            self._git(racer, "commit", "-m", "race")
            self._git(racer, "push", "origin", "main")
            return self._success_evidence(repo_slug, sha)

        with (
            mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
            mock.patch.object(promote_beta, "query_workflow_runs", advance_main),
        ):
            self.assertEqual(self._main(), 1)

    def test_evidence_requires_successful_completed_run(self) -> None:
        candidate = self._origin_ref("beta")
        self.assertFalse(promote_beta.has_successful_evidence({"workflow_runs": []}, candidate))
        self.assertFalse(promote_beta.has_successful_evidence({"workflow_runs": [{"conclusion": None}, {"conclusion": "failure"}]}, candidate))
        self.assertTrue(promote_beta.has_successful_evidence({"workflow_runs": [self._workflow_run(candidate)]}, candidate))

    def test_evidence_rejects_runs_not_for_successful_beta_push_candidate(self) -> None:
        candidate = self._origin_ref("beta")
        invalid_runs = (
            {"conclusion": "success"},
            self._workflow_run(candidate, branch="main"),
            self._workflow_run(candidate, event="workflow_dispatch"),
            self._workflow_run(candidate, event="workflow_dispatch", status="completed"),
            self._workflow_run("different-sha"),
        )
        for run in invalid_runs:
            with self.subTest(run=run):
                self.assertFalse(promote_beta.has_successful_evidence({"workflow_runs": [run]}, candidate))

    def test_commit_sha_uses_end_of_options_for_option_looking_revision(self) -> None:
        response = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="candidate-sha\n")
        with mock.patch.object(promote_beta, "run_git", return_value=response) as run_git:
            self.assertEqual(promote_beta._commit_sha(self.checkout, "--not-a-git-flag"), "candidate-sha")
        self.assertEqual(
            run_git.call_args.args,
            (
                self.checkout,
                ["rev-parse", "--verify", "--end-of-options", "--not-a-git-flag^{commit}"],
            ),
        )

    def test_workflow_query_filters_to_beta_push_runs(self) -> None:
        response = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout='{"total_count": 0, "workflow_runs": []}')
        with mock.patch.object(promote_beta.subprocess, "run", return_value=response) as run:
            promote_beta.query_workflow_runs("example/repository", "candidate-sha")
        self.assertEqual(
            run.call_args.args[0],
            [
                "gh",
                "api",
                "repos/example/repository/actions/workflows/build-container.yml/runs?head_sha=candidate-sha&branch=beta&event=push&per_page=100",
            ],
        )

    def test_rejects_missing_or_failed_evidence(self) -> None:
        main_before = self._origin_ref("main")
        payload: dict[str, Any]
        for payload in (
            {"workflow_runs": []},
            {
                "total_count": 1,
                "workflow_runs": [self._workflow_run(self._origin_ref("beta"), conclusion="failure")],
            },
        ):
            with (
                self.subTest(payload=payload),
                mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
                mock.patch.object(promote_beta, "query_workflow_runs", return_value=payload),
            ):
                self.assertEqual(self._main(), 1)
        self.assertEqual(self._origin_ref("main"), main_before)

    def test_rejects_in_progress_only_evidence(self) -> None:
        payload = {
            "total_count": 1,
            "workflow_runs": [self._workflow_run(self._origin_ref("beta"), status="in_progress", conclusion=None)],
        }
        with (
            mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
            mock.patch.object(promote_beta, "query_workflow_runs", return_value=payload),
        ):
            self.assertEqual(self._main(), 1)

    def test_skip_evidence_promotes_without_gh_query(self) -> None:
        with mock.patch.object(promote_beta, "query_workflow_runs") as query:
            self.assertEqual(self._main("--skip-evidence"), 0)
        query.assert_not_called()

    def test_dry_run_does_not_push(self) -> None:
        main_before = self._origin_ref("main")
        with (
            mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
            mock.patch.object(promote_beta, "query_workflow_runs", self._success_evidence),
        ):
            self.assertEqual(self._main("--dry-run"), 0)
        self.assertEqual(self._origin_ref("main"), main_before)

    def test_repo_slug_supports_ssh_and_https_remotes(self) -> None:
        self.assertEqual(
            promote_beta.repo_slug_from_remote("git@github.com:example/repository.git"),
            "example/repository",
        )
        self.assertEqual(
            promote_beta.repo_slug_from_remote("https://github.com/example/repository.git"),
            "example/repository",
        )

    def test_push_command_never_uses_force(self) -> None:
        command = promote_beta.build_push_command("abc123")
        self.assertEqual(command, ["git", "push", "origin", "abc123:refs/heads/main"])
        self.assertNotIn("--force", command)

    def test_shallow_clone_ancestry_checks_complete_history(self) -> None:
        # Create a shallow clone and verify ancestry checks work after history completion.
        shallow = self.root / "shallow"
        # Use file:// URL scheme to make --depth work with local clones.
        self._git(
            self.root,
            "clone",
            "--depth",
            "1",
            "--branch",
            "beta",
            "file://" + str(self.origin),
            str(shallow),
        )
        self._configure(shallow)
        # A normal single-branch clone fetches only beta by default.
        self.assertEqual(
            self._git(shallow, "config", "--get-all", "remote.origin.fetch").stdout.strip(),
            "+refs/heads/beta:refs/remotes/origin/beta",
        )
        self.assertNotEqual(
            subprocess.run(
                ["git", "rev-parse", "--verify", "origin/main"],
                cwd=shallow,
                text=True,
                capture_output=True,
            ).returncode,
            0,
        )
        # Verify the clone is shallow.
        is_shallow = self._git(shallow, "rev-parse", "--is-shallow-repository").stdout.strip()
        self.assertEqual(is_shallow, "true")
        # Call promotion logic with shallow checkout; it should complete history and succeed.
        with (
            mock.patch.object(promote_beta, "repo_slug_from_remote", return_value="example/repository"),
            mock.patch.object(promote_beta, "query_workflow_runs", self._success_evidence),
        ):
            code = promote_beta.main(["--repo-root", str(shallow), "--dry-run"])
        self.assertEqual(code, 0)
        # Verify shallow was unshallowed.
        is_shallow_after = self._git(shallow, "rev-parse", "--is-shallow-repository").stdout.strip()
        self.assertEqual(is_shallow_after, "false")


if __name__ == "__main__":
    unittest.main()
