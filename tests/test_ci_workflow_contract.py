"""Regression contracts for protected-branch CI status checks."""

import os
import re
import subprocess
import tempfile
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


def _run_workflow_script(script: str, environment: dict[str, str]) -> int:
    with tempfile.NamedTemporaryFile() as output:
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **environment, "GITHUB_OUTPUT": output.name},
        )
    return result.returncode


class CiWorkflowContractTests(unittest.TestCase):
    def test_ci_is_pull_request_only_and_gate_fails_closed(self) -> None:
        ci = _load_workflow("ci.yml")
        self.assertNotIn("push", ci["on"])
        ci_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("secrets: inherit", ci_text)

        gate = ci["jobs"]["gate"]
        self.assertEqual(gate["name"], "CI Gate")
        self.assertIn("always()", gate["if"])
        self.assertEqual(gate["needs"], ["bases", "quality"])
        gate_step = gate["steps"][0]
        script = gate_step["run"]
        self.assertNotIn("needs.", script)
        for name in (
            "BASES_RESULT",
            "QUALITY_RESULT",
            "BACKEND_RESULT",
            "FRONTEND_RESULT",
            "STORAGE_RESULT",
        ):
            self.assertIn(name, gate_step["env"])
            self.assertIn(f'"${name}"', script)

        success = {
            "BASES_RESULT": "success",
            "QUALITY_RESULT": "success",
            "BACKEND_RESULT": "success",
            "FRONTEND_RESULT": "success",
            "STORAGE_RESULT": "success",
        }
        self.assertEqual(_run_workflow_script(script, success), 0)
        for name in success:
            for result in ("failure", "cancelled", "skipped", ""):
                with self.subTest(output=name, result=result):
                    failed = {**success, name: result}
                    self.assertNotEqual(_run_workflow_script(script, failed), 0)

    def test_base_image_access_is_read_only_for_pr_and_ci_calls(self) -> None:
        base = _load_workflow("base-images.yml")
        access_step = base["jobs"]["access"]["steps"][0]
        script = access_step["run"]
        self.assertEqual(access_step["id"], "access")

        cases = (
            ("pull_request", "true", "owner/repo", "owner/repo", "false"),
            ("pull_request", "false", "owner/repo", "owner/repo", "false"),
            ("workflow_dispatch", "false", "", "owner/repo", "false"),
            ("workflow_dispatch", "", "", "owner/repo", "true"),
            ("push", "true", "", "owner/repo", "true"),
        )
        for event, requested, pr_repository, repository, expected in cases:
            with self.subTest(event=event, requested=requested):
                with tempfile.NamedTemporaryFile() as output:
                    result = subprocess.run(
                        ["bash", "-e", "-c", script],
                        check=False,
                        capture_output=True,
                        text=True,
                        env={
                            **os.environ,
                            "EVENT_NAME": event,
                            "REQUESTED": requested,
                            "PR_REPOSITORY": pr_repository,
                            "REPOSITORY": repository,
                            "GITHUB_OUTPUT": output.name,
                        },
                    )
                    output.seek(0)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"can_publish={expected}", output.read().decode())

        self.assertIn("can_publish == 'true'", base["jobs"]["build"]["if"])
        resolve_steps = base["jobs"]["resolve"]["steps"]
        read_only_resolve = next(step for step in resolve_steps if step.get("id") == "empty-images")
        self.assertIn("imagetools inspect", read_only_resolve["run"])

    def test_build_container_is_packaging_only(self) -> None:
        workflow = _load_workflow("build-container.yml")
        self.assertNotIn("pull_request", workflow["on"])
        self.assertNotIn("workflow_call", workflow["on"])
        self.assertNotIn("paths", workflow["on"]["push"])
        jobs = workflow["jobs"]
        self.assertNotIn("quality", jobs)
        promotion = jobs["promote"]
        self.assertNotIn("quality", promotion["needs"])
        self.assertIn("candidate-main", promotion["needs"])
        stale_guard = next(step for step in promotion["steps"] if step.get("name") == "Reject stale branch ref")
        self.assertIn("git ls-remote", stale_guard["run"])
        self.assertTrue(any("cosign" in step.get("uses", "") for step in promotion["steps"]))
        self.assertTrue(any(step.get("name") == "Sign promoted immutable digests" for step in promotion["steps"]))

    def test_quality_runs_frontend_and_storage_tests_with_read_only_caches(self) -> None:
        quality = _load_workflow("quality.yml")
        self.assertEqual(quality["on"]["workflow_call"]["outputs"]["storage_result"]["value"], "${{ jobs.storage.outputs.result }}")
        jobs = quality["jobs"]
        storage = jobs["storage"]
        self.assertEqual(storage["name"], "Storage Tests")
        storage_build = next(step for step in storage["steps"] if step.get("uses") == "docker/build-push-action@v5")
        self.assertEqual(storage_build["with"]["file"], "./docker/Dockerfile.storage")
        self.assertEqual(storage_build["with"]["target"], "storage-test")
        self.assertIn("MAVEN_SKIP_TESTS=1", storage_build["with"]["build-args"])
        self.assertIn("buildcache-storage-${{ inputs.cache_scope }}", storage_build["with"]["cache-from"])

        frontend_steps = jobs["frontend"]["steps"]
        eslint_index = next(index for index, step in enumerate(frontend_steps) if step.get("name") == "Frontend ESLint")
        vitest_index = next(index for index, step in enumerate(frontend_steps) if step.get("name") == "Frontend Vitest")
        build_index = next(index for index, step in enumerate(frontend_steps) if step.get("name") == "Build frontend")
        self.assertLess(eslint_index, vitest_index)
        self.assertLess(vitest_index, build_index)
        self.assertEqual(frontend_steps[vitest_index]["with"]["target"], "frontend-test")
        self.assertIn("FRONTEND_DEPS_IMAGE", frontend_steps[vitest_index]["with"]["build-args"])

        for job in jobs.values():
            for step in job.get("steps", []):
                if step.get("uses") == "docker/login-action@v3":
                    self.assertEqual(step.get("if"), "inputs.use_harbor")
                if step.get("uses") == "docker/build-push-action@v5":
                    cache_from = step.get("with", {}).get("cache-from")
                    if cache_from:
                        self.assertNotIn("inputs.use_harbor", cache_from)
                    cache_to = step.get("with", {}).get("cache-to")
                    if cache_to:
                        self.assertIn("inputs.use_harbor", cache_to)

    def test_backend_analysis_reads_the_published_python_ci_dependency_cache(self) -> None:
        base = _load_workflow("base-images.yml")
        publisher = next(entry for entry in base["jobs"]["build"]["strategy"]["matrix"]["include"] if entry["kind"] == "python-ci-deps")
        publisher_build = next(step for step in base["jobs"]["build"]["steps"] if step.get("uses") == "docker/build-push-action@v5")
        self.assertIn("buildcache-${{ matrix.kind }}", publisher_build["with"]["cache-to"])

        quality = _load_workflow("quality.yml")
        analysis_steps = quality["jobs"]["backend-analysis"]["steps"]
        analysis_build = next(step for step in analysis_steps if step.get("name") == "Build shared backend check image")
        cache_from = analysis_build["with"]["cache-from"]
        dependency_cache = f"type=registry,ref=hub.docker.visnovsky.us/library/ragtime-base:buildcache-{publisher['kind']}"
        self.assertIn(dependency_cache, cache_from)
        self.assertNotIn("inputs.use_harbor", cache_from)
        self.assertIn("inputs.use_harbor", analysis_build["with"]["cache-to"])
        login = next(step for step in analysis_steps if step.get("uses") == "docker/login-action@v3")
        self.assertEqual(login["if"], "inputs.use_harbor")

    def test_backend_analysis_removes_named_check_container_before_image(self) -> None:
        quality = _load_workflow("quality.yml")
        steps = quality["jobs"]["backend-analysis"]["steps"]
        check_steps = [step for step in steps if step.get("name") in {"Mypy type check", "Pyright problem check", "Duplicate code check", "Pytest suite"}]
        self.assertEqual(len(check_steps), 4)
        for step in check_steps:
            with self.subTest(check=step["name"]):
                self.assertEqual(
                    step["env"]["CHECK_CONTAINER"],
                    "${{ steps.buildx.outputs.builder_name }}-check",
                )
                self.assertIn('docker run --rm --name "$CHECK_CONTAINER"', step["run"])

        cleanup_index = next(index for index, step in enumerate(steps) if step.get("name") == "Remove backend analysis check container")
        image_cleanup_index = next(index for index, step in enumerate(steps) if step.get("name") == "Remove shared backend check image")
        cleanup = steps[cleanup_index]
        self.assertEqual(cleanup["if"], "always()")
        self.assertEqual(cleanup["env"]["BUILDER_NAME"], "${{ steps.buildx.outputs.builder_name }}")
        self.assertIn('[ -z "$BUILDER_NAME" ]', cleanup["run"])
        self.assertIn('CHECK_CONTAINER="${BUILDER_NAME}-check"', cleanup["run"])
        self.assertIn('docker container inspect "$CHECK_CONTAINER"', cleanup["run"])
        self.assertIn('docker rm -f "$CHECK_CONTAINER"', cleanup["run"])
        self.assertLess(cleanup_index, image_cleanup_index)

        image_cleanup = steps[image_cleanup_index]
        self.assertEqual(image_cleanup["env"]["IMAGE_TAG"], "${{ steps.buildx.outputs.image_tag }}")
        self.assertIn('[ -z "$IMAGE_TAG" ]', image_cleanup["run"])
        self.assertIn('docker image rm "$IMAGE_TAG"', image_cleanup["run"])

        with tempfile.TemporaryDirectory() as directory:
            docker_log = Path(directory) / "docker.log"
            fake_docker = Path(directory) / "docker"
            fake_docker.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
                'if [ "$1 $2" = "container inspect" ]; then exit "$DOCKER_INSPECT_STATUS"; fi\n'
                'if [ "$1 $2" = "rm -f" ]; then exit 0; fi\n'
                "exit 1\n",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            environment = {
                "PATH": f"{directory}:{os.environ['PATH']}",
                "DOCKER_LOG": str(docker_log),
            }
            for builder_name, inspect_status, expected_commands in (
                ("managed", "0", ["container inspect managed-check", "rm -f managed-check"]),
                ("managed", "1", ["container inspect managed-check"]),
                ("", "0", []),
            ):
                with self.subTest(builder_name=builder_name, inspect_status=inspect_status):
                    docker_log.write_text("", encoding="utf-8")
                    result = subprocess.run(
                        ["bash", "-e", "-c", cleanup["run"]],
                        check=False,
                        capture_output=True,
                        text=True,
                        env={
                            **os.environ,
                            **environment,
                            "BUILDER_NAME": builder_name,
                            "DOCKER_INSPECT_STATUS": inspect_status,
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(docker_log.read_text(encoding="utf-8").splitlines(), expected_commands)

    def test_managed_buildx_sites_have_scopes_and_builder_permissions_are_read_only(self) -> None:
        builder_sites = 0
        for workflow_name in ("base-images.yml", "quality.yml", "build-container.yml"):
            workflow = _load_workflow(workflow_name)
            self.assertEqual(workflow["permissions"]["actions"], "read")
            for job in workflow["jobs"].values():
                for step in job.get("steps", []):
                    if step.get("uses") == "./.github/actions/managed-buildx":
                        builder_sites += 1
                        self.assertIn("scope", step.get("with", {}))
        self.assertGreater(builder_sites, 0)

    def test_managed_buildx_preflight_rounds_buildkit_minimum_to_gib(self) -> None:
        action = yaml.load(
            (ROOT / ".github" / "actions" / "managed-buildx" / "action.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        minimum_gib = int(action["inputs"]["min-free-gib"]["default"])
        buildkit_config = (ROOT / "docker" / "buildkitd.ci.toml").read_text(encoding="utf-8")
        match = re.search(r'minFreeSpace = "(\d+)GB"', buildkit_config)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertGreater(minimum_gib * 1024**3, int(match.group(1)) * 1000**3)

    def test_workflows_contain_no_hardcoded_usernames(self) -> None:
        for workflow_file in (ROOT / ".github" / "workflows").glob("*.yml"):
            with self.subTest(workflow=workflow_file.name):
                self.assertNotIn("mattv8", workflow_file.read_text(encoding="utf-8"))

    def test_all_workflow_bash_scripts_are_syntactically_valid(self) -> None:
        """Catch missing loop/conditional terminators in workflow shell scripts."""
        for workflow_file in (ROOT / ".github" / "workflows").glob("*.yml"):
            workflow = _load_workflow(workflow_file.name)
            for job_id, job in workflow.get("jobs", {}).items():
                for step_idx, step in enumerate(job.get("steps", [])):
                    shell = step.get("shell", "bash")
                    if not isinstance(shell, str) or shell.split()[0] not in {"bash", "sh"}:
                        continue
                    run_script = step.get("run")
                    if not isinstance(run_script, str):
                        continue
                    sanitized = re.sub(r"\$\{\{.*?\}\}", "${_placeholder_}", run_script, flags=re.DOTALL)
                    result = subprocess.run(["bash", "-n"], input=sanitized, capture_output=True, text=True)
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"Bash syntax error in {workflow_file.name} job '{job_id}' step {step_idx}: {result.stderr}",
                    )


if __name__ == "__main__":
    unittest.main()
