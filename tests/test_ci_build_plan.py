from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker/scripts/ci_build_plan.py"


def plan(*arguments: str) -> dict[str, str]:
    command = ["python3", str(SCRIPT), "--sha", "abcdef1" + "0" * 33, "--repository", "owner/ragtime", *arguments]
    return dict(line.split("=", 1) for line in subprocess.check_output(command, text=True).splitlines())


class CiBuildPlanTests(unittest.TestCase):
    def test_feature_push_from_ci_never_publishes(self) -> None:
        result = plan("--event", "push", "--ref-name", "feature/x", "--container-changed", "true", "--build-images", "false", "--build-legacy", "false")
        self.assertEqual(result["promote"], "false")

    def test_same_repository_pr_keeps_pr_tags(self) -> None:
        result = plan(
            "--event",
            "pull_request",
            "--ref-name",
            "main",
            "--pr-repository",
            "owner/ragtime",
            "--pr-number",
            "42",
            "--container-changed",
            "true",
            "--build-images",
            "true",
            "--build-legacy",
            "false",
        )
        self.assertEqual(result["app_tags"], "pr-42,pr-42-abcdef1")
        self.assertEqual(result["promote"], "true")

    def test_pr_environment_comes_from_the_base_branch(self) -> None:
        result = plan(
            "--event",
            "pull_request",
            "--ref-name",
            "123/merge",
            "--base-ref",
            "beta",
            "--pr-repository",
            "owner/ragtime",
            "--pr-number",
            "123",
            "--container-changed",
            "true",
            "--build-images",
            "true",
            "--build-legacy",
            "false",
        )
        self.assertEqual(result["environment"], "beta")

    def test_fork_and_unchanged_pr_do_not_publish(self) -> None:
        for repository, changed in (("fork/ragtime", "true"), ("owner/ragtime", "false")):
            with self.subTest(repository=repository, changed=changed):
                result = plan(
                    "--event",
                    "pull_request",
                    "--ref-name",
                    "main",
                    "--pr-repository",
                    repository,
                    "--pr-number",
                    "7",
                    "--container-changed",
                    changed,
                    "--build-images",
                    "true",
                    "--build-legacy",
                    "false",
                )
                self.assertEqual(result["promote"], "false")

    def test_manual_and_branch_tags_preserve_legacy_contract(self) -> None:
        result = plan("--event", "workflow_dispatch", "--ref-name", "beta", "--container-changed", "true", "--build-images", "true", "--build-legacy", "false")
        self.assertEqual(result["app_tags"], "beta,abcdef1,latest-beta")
        self.assertEqual(result["environment"], "beta")

    def test_release_refs_only_for_non_pr_events(self) -> None:
        result = plan(
            "--event", "workflow_dispatch", "--ref-name", "feature/x", "--container-changed", "true", "--build-images", "true", "--build-legacy", "false"
        )
        self.assertEqual(result["promote"], "false")

    def test_promotion_yaml_gate_fails_closed_for_non_success_statuses(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load((root / ".github/workflows/build-container.yml").read_text(encoding="utf-8"))
        expression = workflow["jobs"]["promote"]["if"]

        def evaluates(**values: str | bool) -> bool:
            rendered = expression
            for name, value in values.items():
                rendered = rendered.replace(name, repr(value))
            rendered = rendered.replace("&&", " and ").replace("||", " or ")
            rendered = re.sub(r"!(?!=)", " not ", rendered)
            return bool(eval(rendered.strip(), {"__builtins__": {}}, {}))

        baseline: dict[str, str | bool] = {
            "cancelled()": False,
            "needs.quality.result": "success",
            "needs.plan.result": "success",
            "needs.plan.outputs.promote": "true",
            "needs.plan.outputs.build_main": "true",
            "needs.candidate-main.result": "success",
            "needs.sbom.result": "success",
            "needs.plan.outputs.build_runtime": "true",
            "needs.candidate-runtime.result": "success",
            "needs.plan.outputs.build_legacy": "false",
            "needs.candidate-legacy.result": "skipped",
        }
        self.assertTrue(evaluates(**baseline))
        for field, value in (
            ("needs.quality.result", "failure"),
            ("needs.quality.result", "cancelled"),
            ("needs.quality.result", "skipped"),
            ("needs.candidate-main.result", "failure"),
            ("needs.sbom.result", "skipped"),
            ("needs.candidate-runtime.result", "cancelled"),
            ("cancelled()", True),
        ):
            with self.subTest(field=field, value=value):
                self.assertFalse(evaluates(**{**baseline, field: value}))
