import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _stage_run_instructions(dockerfile: str) -> dict[str, list[str]]:
    instructions_by_stage: dict[str, list[str]] = {}
    current_stage = ""
    current: list[str] = []

    for line in dockerfile.splitlines():
        if line.startswith("FROM "):
            parts = line.split()
            current_stage = parts[-1] if len(parts) >= 4 and parts[-2].upper() == "AS" else parts[1]
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

    return instructions_by_stage


class DockerPrismaEngineTests(unittest.TestCase):
    def test_production_prisma_generate_does_not_hide_engine_in_buildkit_cache(self):
        dockerfile = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
        stage_instructions = _stage_run_instructions(dockerfile)
        generate_instructions = [
            instruction
            for instruction in stage_instructions["production"]
            if "python -m prisma generate" in instruction
        ]

        self.assertEqual(len(generate_instructions), 1)
        for instruction in generate_instructions:
            self.assertNotIn("target=/root/.cache/prisma-python", instruction)

    def test_python_test_stage_installs_git_for_git_history_tests(self):
        dockerfile = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
        stage_instructions = _stage_run_instructions(dockerfile)
        apt_install_instructions = [
            instruction
            for instruction in stage_instructions["python-test"]
            if "apt-get install" in instruction
        ]

        self.assertTrue(any("git" in instruction.split() for instruction in apt_install_instructions))


if __name__ == "__main__":
    unittest.main()
