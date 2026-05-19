from __future__ import annotations

import unittest
from pathlib import Path


def _repo_root_with_dockerfile() -> Path | None:
    for candidate in (Path(__file__).resolve().parents[1], Path.cwd()):
        if (candidate / "docker" / "Dockerfile").is_file():
            return candidate
    return None


def _dockerfile_stages(dockerfile: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    stage_parents: dict[str, str] = {}
    instructions_by_stage: dict[str, list[str]] = {}
    current_stage = ""
    current: list[str] = []

    for line in dockerfile.splitlines():
        if line.startswith("FROM "):
            parts = line.split()
            current_stage = parts[-1] if len(parts) >= 4 and parts[-2].upper() == "AS" else parts[1]
            stage_parents[current_stage] = parts[1]
            instructions_by_stage.setdefault(current_stage, [])
            current = []
            continue

        if line.startswith("RUN "):
            current = [line]
        elif current:
            current.append(line)
        else:
            continue

        if current and not line.rstrip().endswith("\\"):
            instructions_by_stage.setdefault(current_stage, []).append("\n".join(current))
            current = []

    return stage_parents, instructions_by_stage


def _stage_run_instructions(dockerfile: str) -> dict[str, list[str]]:
    return _dockerfile_stages(dockerfile)[1]


def _effective_stage_run_instructions(dockerfile: str, stage: str) -> list[str]:
    stage_parents, instructions_by_stage = _dockerfile_stages(dockerfile)
    instructions: list[str] = []
    lineage: list[str] = []
    current_stage = stage

    while current_stage in instructions_by_stage:
        lineage.append(current_stage)
        current_stage = stage_parents.get(current_stage, "")

    for inherited_stage in reversed(lineage):
        instructions.extend(instructions_by_stage[inherited_stage])

    return instructions


class DockerPrismaEngineTests(unittest.TestCase):
    def _read_dockerfile(self) -> str:
        root_dir = _repo_root_with_dockerfile()
        if root_dir is None:
            self.skipTest("docker/Dockerfile is not mounted in this test environment")
        return (root_dir / "docker" / "Dockerfile").read_text(encoding="utf-8")

    def test_production_prisma_generate_does_not_hide_engine_in_buildkit_cache(self):
        dockerfile = self._read_dockerfile()
        stage_instructions = _stage_run_instructions(dockerfile)
        generate_instructions = [instruction for instruction in stage_instructions["production"] if "python -m prisma generate" in instruction]

        self.assertEqual(len(generate_instructions), 1)
        for instruction in generate_instructions:
            self.assertNotIn("target=/root/.cache/prisma-python", instruction)

    def test_python_test_stage_installs_git_for_git_history_tests(self):
        dockerfile = self._read_dockerfile()
        apt_install_instructions = [
            instruction for instruction in _effective_stage_run_instructions(dockerfile, "python-test") if "apt-get install" in instruction
        ]

        self.assertTrue(any("git" in instruction.split() for instruction in apt_install_instructions))


if __name__ == "__main__":
    unittest.main()
