from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VERSION_SPEC = importlib.util.spec_from_file_location("release_version", ROOT / "docker/scripts/release_version.py")
assert VERSION_SPEC is not None and VERSION_SPEC.loader is not None
RELEASE_VERSION = importlib.util.module_from_spec(VERSION_SPEC)
sys.modules["release_version"] = RELEASE_VERSION
VERSION_SPEC.loader.exec_module(RELEASE_VERSION)
PR_SPEC = importlib.util.spec_from_file_location("release_pr_under_test", ROOT / "docker/scripts/release_pr.py")
assert PR_SPEC is not None and PR_SPEC.loader is not None
RELEASE_PR = importlib.util.module_from_spec(PR_SPEC)
PR_SPEC.loader.exec_module(RELEASE_PR)
GhApi = RELEASE_PR.GhApi
PROMOTION_END = RELEASE_PR.PROMOTION_END
PROMOTION_START = RELEASE_PR.PROMOTION_START
_replace_preview = RELEASE_PR._replace_preview
coordinate_release = RELEASE_PR.coordinate_release
main = RELEASE_PR.main


class FakeApi:
    def __init__(self, repository: str) -> None:
        self.repository = repository
        self.prs: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.updated: list[tuple[int, str]] = []

    def list_open(self, base: str) -> list[dict[str, Any]]:
        return [pr for pr in self.prs if pr["base"]["ref"] == base]

    def create(self, title: str, head: str, base: str, body: str) -> dict[str, Any]:
        pr = {"number": len(self.prs) + 1, "body": body, "base": {"ref": base}, "head": {"ref": head, "repo": {"full_name": self.repository}}}
        self.prs.append(pr)
        self.created.append(pr)
        return pr

    def update(self, number: int, body: str) -> None:
        self.updated.append((number, body))
        next(pr for pr in self.prs if pr["number"] == number)["body"] = body


class ReleasePrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "release@example.test")
        self.git("config", "user.name", "Release Test")
        self.commit("initial")
        self.git("branch", "beta")
        self.git("remote", "add", "origin", str(self.repo))
        self.git("fetch", "origin", "main:refs/remotes/origin/main", "beta:refs/remotes/origin/beta")
        self.api = FakeApi("owner/ragtime")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=self.repo, text=True).strip()

    def commit(self, message: str) -> None:
        (self.repo / "content").write_text(message, encoding="utf-8")
        self.git("add", "content")
        self.git("commit", "-m", message)

    def beta_commit(self, message: str = "feat: beta change") -> None:
        self.git("checkout", "beta")
        self.commit(message)
        self.git("fetch", "origin", "beta:refs/remotes/origin/beta")

    def test_schedule_creates_once_and_skipped_weeks_reuse_same_pr(self) -> None:
        self.beta_commit()
        first = coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")
        second = coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")
        self.assertEqual(first["action"], "created")
        self.assertEqual(second["action"], "reused")
        self.assertEqual(len(self.api.created), 1)

    def test_default_bootstrap_preview_is_first_stable_release(self) -> None:
        self.beta_commit()
        coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")
        self.assertIn("v1.0.0", self.api.created[0]["body"])

    def test_zero_ahead_and_identical_tree_do_not_create(self) -> None:
        self.assertEqual(coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")["action"], "none")
        self.beta_commit("fix: temporary")
        self.git("checkout", "beta")
        self.git("checkout", "main", "--", "content")
        self.git("commit", "-am", "fix: restore tree")
        self.git("fetch", "origin", "beta:refs/remotes/origin/beta")
        self.assertEqual(coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")["action"], "none")

    def test_beta_refresh_preserves_maintainer_text_and_never_creates(self) -> None:
        self.beta_commit()
        self.api.create("custom", "beta", "main", "Maintainer notes\n\n" + PROMOTION_START + "\nold\n" + PROMOTION_END)
        result = coordinate_release(self.repo, "owner/ragtime", self.api, "beta-push")
        self.assertEqual(result["action"], "refreshed")
        self.assertIn("Maintainer notes", self.api.updated[0][1])
        self.assertEqual(len(self.api.created), 1)

    def test_preview_uses_label_ahead_count_and_beta_sha(self) -> None:
        self.git("tag", "v0.1.0")
        self.beta_commit()
        pr = self.api.create("custom", "beta", "main", "prefix\n\n" + PROMOTION_START + "\nold\n" + PROMOTION_END + "\n\nsuffix")
        pr["labels"] = [{"name": "release:major"}]
        coordinate_release(self.repo, "owner/ragtime", self.api, "beta-push")
        body = self.api.updated[0][1]
        self.assertIn("v1.0.0", body)
        self.assertIn("Beta commits ahead: `1`", body)
        self.assertIn(self.git("rev-parse", "origin/beta"), body)
        self.assertIn("prefix", body)
        self.assertIn("suffix", body)

    def test_conflicting_labels_and_malformed_markers_do_not_update(self) -> None:
        self.beta_commit()
        pr = self.api.create("custom", "beta", "main", "keep " + PROMOTION_START)
        pr["labels"] = [{"name": "release:major"}, {"name": "release:minor"}]
        with self.assertRaises(ValueError):
            coordinate_release(self.repo, "owner/ragtime", self.api, "beta-push")
        self.assertEqual(self.api.updated, [])
        pr["labels"] = []
        with self.assertRaises(ValueError):
            coordinate_release(self.repo, "owner/ragtime", self.api, "beta-push")
        self.assertEqual(self.api.updated, [])
        with self.assertRaises(ValueError):
            _replace_preview("orphan " + PROMOTION_END, "new")
        with self.assertRaises(ValueError):
            _replace_preview(PROMOTION_START + "x" + PROMOTION_END + PROMOTION_START + "y" + PROMOTION_END, "new")

    def test_fork_cannot_suppress_our_promotion(self) -> None:
        self.beta_commit()
        self.api.prs.append({"number": 99, "body": "fork", "base": {"ref": "main"}, "head": {"ref": "beta", "repo": {"full_name": "fork/ragtime"}}})
        self.assertEqual(coordinate_release(self.repo, "owner/ragtime", self.api, "schedule")["action"], "created")

    def test_api_failure_and_gh_payload_shapes_fail_closed(self) -> None:
        self.beta_commit()

        class FailingApi(FakeApi):
            def list_open(self, base: str) -> list[dict[str, Any]]:
                raise subprocess.CalledProcessError(1, ["gh", "api"])

        with self.assertRaises(subprocess.CalledProcessError):
            coordinate_release(self.repo, "owner/ragtime", FailingApi("owner/ragtime"), "schedule")

        valid: dict[str, Any] = {
            "number": 1,
            "body": None,
            "base": {"ref": "main"},
            "head": {"ref": "beta", "repo": {"full_name": "owner/ragtime"}},
            "labels": [],
        }
        gh_api = GhApi("owner/ragtime")
        with patch.object(gh_api, "_call", return_value=[[valid], [valid]]):
            self.assertEqual(len(gh_api.list_open("main")), 2)
        malformed_payloads: tuple[Any, ...] = ({}, [[{"number": "bad"}]])
        for payload in malformed_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                with patch.object(gh_api, "_call", return_value=payload):
                    gh_api.list_open("main")

    def test_missing_token_fails_before_api(self) -> None:
        previous = os.environ.pop("GH_TOKEN", None)
        try:
            self.assertEqual(main(["--event", "schedule", "--repository", "owner/ragtime", "--repo-root", str(self.repo)]), 1)
        finally:
            if previous is not None:
                os.environ["GH_TOKEN"] = previous

    def test_main_not_ancestor_creates_and_dedupes_reconciliation(self) -> None:
        self.git("checkout", "main")
        self.commit("fix: hotfix")
        self.git("fetch", "origin", "main:refs/remotes/origin/main")
        first = coordinate_release(self.repo, "owner/ragtime", self.api, "main-push")
        second = coordinate_release(self.repo, "owner/ragtime", self.api, "main-push")
        self.assertEqual(first["action"], "reconciliation")
        self.assertEqual(second["number"], first["number"])
        self.assertEqual(len(self.api.created), 1)

    def test_beta_push_with_divergence_does_not_create_reconciliation(self) -> None:
        self.git("checkout", "main")
        self.commit("fix: hotfix")
        self.git("fetch", "origin", "main:refs/remotes/origin/main")
        result = coordinate_release(self.repo, "owner/ragtime", self.api, "beta-push")
        self.assertEqual(result["action"], "none")
        self.assertEqual(self.api.created, [])
