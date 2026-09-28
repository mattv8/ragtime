from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_version_under_test", ROOT / "docker/scripts/release_version.py")
assert SPEC is not None and SPEC.loader is not None
RELEASE_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE_VERSION)
plan_release = RELEASE_VERSION.plan_release


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "release@example.test")
        self.git("config", "user.name", "Release Test")
        self.commit("initial")
        self.previous = Path.cwd()
        os.chdir(self.repo)

    def tearDown(self) -> None:
        os.chdir(self.previous)
        self.temp.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=self.repo, text=True).strip()

    def commit(self, message: str) -> None:
        (self.repo / "content").write_text(message, encoding="utf-8")
        self.git("add", "content")
        self.git("commit", "-m", message)

    def test_bootstrap_and_initial_validation(self) -> None:
        result = plan_release("HEAD", "0.3.0")
        self.assertEqual(result["tag"], "v0.3.0")
        self.assertEqual(result["previous_tag"], "")
        self.assertEqual(result["existing_tag"], "false")
        with self.assertRaises(ValueError):
            plan_release("HEAD", "1.2")

    def test_default_bootstrap_is_first_stable_release(self) -> None:
        self.assertEqual(plan_release("HEAD")["tag"], "v1.0.0")
        output = subprocess.check_output(
            ["python3", str(ROOT / "docker/scripts/release_version.py"), "--ref", "HEAD"],
            cwd=self.repo,
            text=True,
        )
        self.assertEqual(json.loads(output)["tag"], "v1.0.0")

    def test_infers_patch_minor_major_and_breaking_footer(self) -> None:
        self.git("tag", "v1.0.0")
        self.commit("fix: correct output")
        self.assertEqual(plan_release("HEAD")["tag"], "v1.0.1")
        self.git("tag", "v1.0.1")
        self.commit("feat(api): add endpoint")
        self.assertEqual(plan_release("HEAD")["tag"], "v1.1.0")
        self.git("tag", "v1.1.0")
        self.commit("refactor: wire format\n\nBREAKING CHANGE: changed field")
        self.assertEqual(plan_release("HEAD")["tag"], "v2.0.0")

    def test_scoped_and_unscoped_breaking_headers_and_footer_forms(self) -> None:
        for message in (
            "feat!: incompatible",
            "feat(api)!: incompatible",
            "fix(core)!: incompatible",
            "refactor(scope)!: incompatible",
            "docs: note\n\nBREAKING-CHANGE: incompatible",
        ):
            with self.subTest(message=message):
                self.git("tag", "v1.0.0")
                self.commit(message)
                self.assertEqual(plan_release("HEAD")["tag"], "v2.0.0")
                self.git("tag", "-d", "v1.0.0")
                self.git("reset", "--hard", "HEAD~1")

    def test_mixed_history_uses_highest_bump(self) -> None:
        self.git("tag", "v1.0.0")
        self.commit("fix: ordinary")
        self.commit("feat: capability")
        self.commit("fix(core)!: incompatible")
        self.assertEqual(plan_release("HEAD")["tag"], "v2.0.0")

    def test_uses_numeric_reachable_tag_order_and_ignores_prereleases(self) -> None:
        self.git("tag", "v1.9.0")
        self.git("tag", "v1.10.0")
        self.git("tag", "v9.0.0-beta.1")
        self.commit("chore: changed")
        result = plan_release("HEAD")
        self.assertEqual(result["previous_tag"], "v1.10.0")
        self.assertEqual(result["tag"], "v1.10.1")

    def test_tag_at_target_is_a_retry_with_preceding_baseline(self) -> None:
        self.git("tag", "v1.0.0")
        self.commit("fix: one")
        self.git("tag", "v1.0.1")
        result = plan_release("HEAD")
        self.assertEqual(result["tag"], "v1.0.1")
        self.assertEqual(result["previous_tag"], "v1.0.0")
        self.assertEqual(result["existing_tag"], "true")

    def test_cli_retry_contract_and_unrelated_tag(self) -> None:
        self.git("tag", "v1.0.0")
        self.git("checkout", "--orphan", "unrelated")
        self.commit("feat: unrelated")
        self.git("tag", "v9.0.0")
        self.git("checkout", "main")
        self.commit("fix: stable")
        self.git("tag", "v1.0.1")
        output = subprocess.check_output(["python3", str(ROOT / "docker/scripts/release_version.py"), "--ref", "HEAD"], cwd=self.repo, text=True)
        self.assertEqual(json.loads(output)["tag"], "v1.0.1")

    def test_override_and_invalid_input_fail_closed(self) -> None:
        self.git("tag", "v1.0.0")
        self.commit("docs: update")
        self.assertEqual(plan_release("HEAD", bump="minor")["tag"], "v1.1.0")
        with self.assertRaises(ValueError):
            plan_release("HEAD", bump="nope")
        with self.assertRaises(ValueError):
            plan_release("--bad")
