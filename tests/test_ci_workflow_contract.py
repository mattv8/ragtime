"""Regression contracts for protected-branch CI status checks."""

import subprocess
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_workflow(name: str) -> dict[str, Any]:
    workflow = yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    if not isinstance(workflow, dict):
        raise TypeError(f"Workflow {name} must be a mapping")
    return workflow


def _run_wrapper_check(command: str, quality_result: str, aggregate_result: str) -> int:
    rendered = (
        command.replace("${{ needs.quality.result }}", quality_result)
        .replace("${{ needs.quality.outputs.backend_result }}", aggregate_result)
        .replace("${{ needs.quality.outputs.frontend_result }}", aggregate_result)
    )
    return subprocess.run(
        ["bash", "-e", "-c", rendered],
        check=False,
        capture_output=True,
        text=True,
    ).returncode


class CiWorkflowContractTests(unittest.TestCase):
    def test_required_status_wrappers_link_reusable_quality_results(self) -> None:
        ci_jobs = _load_workflow("ci.yml")["jobs"]
        quality_workflow = _load_workflow("quality.yml")
        quality_outputs = quality_workflow["on"]["workflow_call"]["outputs"]

        self.assertEqual(quality_outputs["backend_result"]["value"], "${{ jobs.backend.outputs.result }}")
        self.assertEqual(quality_outputs["frontend_result"]["value"], "${{ jobs.frontend.outputs.result }}")
        self.assertEqual(quality_workflow["jobs"]["backend"]["outputs"]["result"], "${{ steps.result.outputs.value }}")
        self.assertEqual(quality_workflow["jobs"]["frontend"]["outputs"]["result"], "${{ steps.result.outputs.value }}")

        for job_id, name, output in (
            ("backend-status", "Backend Quality and Tests", "backend_result"),
            ("frontend-status", "Frontend Build", "frontend_result"),
        ):
            job = ci_jobs[job_id]
            self.assertEqual(job["name"], name)
            self.assertIn("quality", job["needs"])
            self.assertIn("always()", job["if"])
            command = job["steps"][0]["run"]
            self.assertIn("needs.quality.result", command)
            self.assertIn(f"needs.quality.outputs.{output}", command)
            self.assertEqual(_run_wrapper_check(command, "success", "success"), 0)
            for quality_result, aggregate_result in (
                ("failure", ""),
                ("cancelled", ""),
                ("skipped", ""),
                ("success", "failure"),
                ("success", "cancelled"),
                ("success", "skipped"),
                ("success", ""),
            ):
                with self.subTest(job=name, quality=quality_result, aggregate=aggregate_result):
                    self.assertNotEqual(_run_wrapper_check(command, quality_result, aggregate_result), 0)


if __name__ == "__main__":
    unittest.main()
