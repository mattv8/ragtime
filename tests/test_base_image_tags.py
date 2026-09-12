from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "scripts" / "base_image_tags.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("base_image_tags", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load base image tag helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaseImageTagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative_path in {path for paths in self.module.INPUTS.values() for path in paths}:
            source = ROOT / relative_path
            destination = self.root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _tags(self) -> dict[str, str]:
        return self.module.compute_tags(self.root, "registry.example")

    def test_tags_are_fully_qualified_and_platform_specific(self) -> None:
        tags = self._tags()
        self.assertRegex(tags["frontend_tag"], r"^registry\.example/library/ragtime-base:frontend-[0-9a-f]{64}$")
        self.assertRegex(tags["python_ci_tag"], r"^registry\.example/library/ragtime-base:ci-[0-9a-f]{64}$")
        self.assertRegex(tags["production_tag"], r"^registry\.example/library/ragtime-base:production-[0-9a-f]{64}$")

    def test_application_source_change_does_not_change_any_tag(self) -> None:
        before = self._tags()
        application_file = self.root / "ragtime" / "main.py"
        application_file.parent.mkdir(parents=True, exist_ok=True)
        application_file.write_text("changed application source\n", encoding="utf-8")
        self.assertEqual(before, self._tags())

    def test_frontend_manifest_only_changes_frontend_and_production_tags(self) -> None:
        before = self._tags()
        with (self.root / "ragtime/frontend/package.json").open("a", encoding="utf-8") as file:
            file.write("\n")
        after = self._tags()
        self.assertNotEqual(before["frontend_tag"], after["frontend_tag"])
        self.assertEqual(before["python_ci_tag"], after["python_ci_tag"])
        self.assertNotEqual(before["production_tag"], after["production_tag"])

    def test_python_inputs_change_ci_and_production_as_applicable(self) -> None:
        before = self._tags()
        with (self.root / "pyproject.toml").open("a", encoding="utf-8") as file:
            file.write("\n")
        after = self._tags()
        self.assertEqual(before["frontend_tag"], after["frontend_tag"])
        self.assertNotEqual(before["python_ci_tag"], after["python_ci_tag"])
        self.assertNotEqual(before["production_tag"], after["production_tag"])

    def test_schema_changes_only_ci_tag(self) -> None:
        before = self._tags()
        with (self.root / "prisma/schema.prisma").open("a", encoding="utf-8") as file:
            file.write("\n")
        after = self._tags()
        self.assertEqual(before["frontend_tag"], after["frontend_tag"])
        self.assertNotEqual(before["python_ci_tag"], after["python_ci_tag"])
        self.assertEqual(before["production_tag"], after["production_tag"])

    def test_missing_required_input_fails(self) -> None:
        (self.root / "prisma/schema.prisma").unlink()
        with self.assertRaises(FileNotFoundError):
            self._tags()

    def test_dependency_export_stages_do_not_copy_application_source(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        for stage, forbidden in {
            "frontend-deps": ("COPY ragtime/frontend/ ./",),
            "python-ci-deps": ("COPY ragtime/ /ragtime/ragtime/", "COPY runtime/", "COPY tests/", "COPY .github/workflows/"),
            "production-deps": ("COPY ragtime/ /ragtime/ragtime/", "COPY runtime/", "COPY tests/", "COPY .github/workflows/"),
        }.items():
            body = dockerfile.split(f"AS {stage}", 1)[1].split("\nFROM ", 1)[0]
            for source_copy in forbidden:
                self.assertNotIn(source_copy, body, f"{stage} must not contain source: {source_copy}")

    def test_production_dependencies_start_clean_and_are_owned_by_ragtime_user(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        production_deps = dockerfile.split("FROM python:3.12-slim-trixie AS production-deps", 1)[1].split("\nFROM ", 1)[0]
        self.assertIn("COPY --from=python-builder /opt/venv /opt/venv", production_deps)
        self.assertNotIn("build-essential", production_deps)
        self.assertIn("useradd -r -g ragtime ragtime", production_deps)
        self.assertIn("COPY --chown=ragtime:ragtime ragtime/frontend/package.json", production_deps)
        self.assertIn("chown -R ragtime:ragtime /ragtime/ragtime/frontend", production_deps)

    def test_fast_ruff_stage_does_not_inherit_python_ci_dependencies(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim-trixie AS python-quality-base", dockerfile)
        quality_base = dockerfile.split("FROM python:3.12-slim-trixie AS python-quality-base", 1)[1].split("\nFROM ", 1)[0]
        self.assertIn('python -m pip install --no-cache-dir "$RUFF_REQUIREMENT"', quality_base)
        self.assertIn("COPY ragtime/ /ragtime/ragtime/", quality_base)
        self.assertNotIn("FROM python-ci-base", quality_base)

    def test_base_workflow_keeps_fork_and_refresh_behavior_explicit(self) -> None:
        workflow = (ROOT / ".github/workflows/base-images.yml").read_text(encoding="utf-8")
        self.assertIn('[ "$EVENT_NAME" = push ] || { [ "$EVENT_NAME" = pull_request ] && [ "$PR_REPOSITORY" = "$REPOSITORY" ]; }', workflow)
        self.assertIn("if: ${{ always() }}", workflow)
        self.assertIn("no-cache: ${{ inputs.refresh }}", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("^sha256:[0-9a-f]{64}$", workflow)

    def test_manifest_absence_condition_accepts_only_known_missing_errors(self) -> None:
        image = "hub.docker.visnovsky.us/library/ragtime-base:ci-base-absence-probe-20260912"
        condition = """\
output="$1"; image="$2"
if [ "$output" = "ERROR: $image: not found" ] || { printf '%s' "$output" | grep -Eqi '(manifest unknown|manifest not found|name unknown)' && ! printf '%s' "$output" | grep -Eqi '(unauthorized|authentication required|denied|forbidden|timeout|network|connection|tls)'; }; then
  printf missing
else
  printf fail
fi
"""

        def status(stderr: str) -> str:
            result = subprocess.run(
                ["bash", "-c", condition, "bash", stderr, image],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout

        self.assertEqual(status(f"ERROR: {image}: not found"), "missing")
        self.assertEqual(status("ERROR: failed to authorize: authentication required"), "fail")
        self.assertEqual(status("ERROR: manifest unknown: manifest unknown"), "missing")
        self.assertEqual(status("ERROR: network timeout while checking manifest unknown"), "fail")

    def test_base_access_condition_rejects_unknown_and_fork_events(self) -> None:
        condition = """\
requested="$1"; event_name="$2"; repository="$3"; pr_repository="$4"; can_publish=false
if [ "$event_name" = workflow_dispatch ]; then
  can_publish=true
elif [ "$requested" = true ] && { [ "$event_name" = push ] || { [ "$event_name" = pull_request ] && [ "$pr_repository" = "$repository" ]; }; }; then
  can_publish=true
fi
printf '%s' "$can_publish"
"""

        def access(requested: str, event_name: str, pr_repository: str = "owner/ragtime") -> str:
            result = subprocess.run(
                ["bash", "-c", condition, "bash", requested, event_name, "owner/ragtime", pr_repository],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout

        self.assertEqual(access("true", "workflow_dispatch"), "true")
        self.assertEqual(access("true", "push"), "true")
        self.assertEqual(access("true", "pull_request"), "true")
        self.assertEqual(access("true", "pull_request", "fork/ragtime"), "false")
        self.assertEqual(access("true", "pull_request_target"), "false")
        self.assertEqual(access("false", "push"), "false")
