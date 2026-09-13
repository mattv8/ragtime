"""Regression contracts for protected-branch CI status checks."""

import re
import subprocess
import unittest
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

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
        command.replace("${{ needs.pipeline.result }}", quality_result)
        .replace("${{ needs.pipeline.outputs.backend_result }}", aggregate_result)
        .replace("${{ needs.pipeline.outputs.frontend_result }}", aggregate_result)
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
        pipeline_workflow = _load_workflow("build-container.yml")
        pipeline_outputs = pipeline_workflow["on"]["workflow_call"]["outputs"]

        self.assertEqual(pipeline_outputs["backend_result"]["value"], "${{ jobs.quality.outputs.backend_result }}")
        self.assertEqual(pipeline_outputs["frontend_result"]["value"], "${{ jobs.quality.outputs.frontend_result }}")

        for job_id, name, output in (
            ("backend-status", "Backend Quality and Tests", "backend_result"),
            ("frontend-status", "Frontend Build", "frontend_result"),
        ):
            job = ci_jobs[job_id]
            self.assertEqual(job["name"], name)
            self.assertIn("pipeline", job["needs"])
            self.assertIn("always()", job["if"])
            command = job["steps"][0]["run"]
            self.assertIn("needs.pipeline.result", command)
            self.assertIn(f"needs.pipeline.outputs.{output}", command)
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

    def test_pipeline_graph_keeps_candidates_independent_of_quality_and_gates_promotion(self) -> None:
        jobs = _load_workflow("build-container.yml")["jobs"]
        for candidate in ("candidate-main", "candidate-runtime", "candidate-legacy"):
            self.assertNotIn("quality", jobs[candidate]["needs"])
            self.assertIn("push-by-digest=true", jobs[candidate]["steps"][-1]["with"]["outputs"])
        promotion = jobs["promote"]
        self.assertEqual(set(promotion["needs"]), {"quality", "plan", "candidate-main", "candidate-runtime", "candidate-legacy", "sbom"})
        self.assertIn("!cancelled()", promotion["if"])
        self.assertIn("needs.quality.result == 'success'", promotion["if"])
        stale_guard = next(step for step in promotion["steps"] if step.get("name") == "Reject stale branch ref")
        self.assertIn("git ls-remote", stale_guard["run"])
        self.assertIn("needs.sbom.result == 'success'", promotion["if"])
        self.assertEqual(promotion["steps"][-2]["if"], "github.event_name != 'pull_request'")

    def test_quality_uses_real_multiline_arguments_and_conditional_registry_login(self) -> None:
        quality = _load_workflow("quality.yml")["jobs"]
        for job_id in ("backend-fast", "backend-analysis", "frontend"):
            self.assertTrue(any(step.get("uses") == "docker/login-action@v3" and step.get("if") == "inputs.use_harbor" for step in quality[job_id]["steps"]))
        frontend_build = quality["frontend"]["steps"][-2]["with"]["build-args"]
        self.assertIn("\n", frontend_build)
        self.assertNotIn("\\n", frontend_build)

    def test_all_workflow_bash_scripts_are_syntactically_valid(self) -> None:
        """Catch missing loop/conditional terminators in workflow shell scripts."""
        workflow_dir = ROOT / ".github" / "workflows"
        workflows = list(workflow_dir.glob("*.yml"))
        self.assertGreater(len(workflows), 0, "No workflows found")

        for workflow_file in workflows:
            with self.subTest(workflow=workflow_file.name):
                workflow = _load_workflow(workflow_file.name)
                if not isinstance(workflow, dict) or "jobs" not in workflow:
                    continue

                jobs = workflow["jobs"]
                if not isinstance(jobs, dict):
                    continue

                for job_id, job in jobs.items():
                    if not isinstance(job, dict) or "steps" not in job:
                        continue

                    steps = job["steps"]
                    if not isinstance(steps, list):
                        continue

                    for step_idx, step in enumerate(steps):
                        if not isinstance(step, dict):
                            continue
                        shell = step.get("shell", "bash")
                        if not isinstance(shell, str) or shell.split()[0] not in {"bash", "sh"}:
                            continue

                        run_script = step.get("run")
                        if not isinstance(run_script, str):
                            continue

                        # Replace GitHub expressions with inert placeholders to avoid syntax issues.
                        # Non-greedy match: ${{ ... }} becomes ${_placeholder_}.
                        sanitized = re.sub(r"\$\{\{.*?\}\}", "${_placeholder_}", run_script, flags=re.DOTALL)

                        # Run bash -n (syntax check only) on the sanitized script.
                        result = subprocess.run(
                            ["bash", "-n"],
                            input=sanitized,
                            capture_output=True,
                            text=True,
                        )

                        self.assertEqual(
                            result.returncode,
                            0,
                            f"Bash syntax error in {workflow_file.name} job '{job_id}' step {step_idx}: {result.stderr}",
                        )


if __name__ == "__main__":
    unittest.main()
