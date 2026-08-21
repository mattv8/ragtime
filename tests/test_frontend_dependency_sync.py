from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"


def _extract_function(script_text: str, name: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}\(\) \{{.*?^\}}\s*$", re.MULTILINE | re.DOTALL)
    match = pattern.search(script_text)
    if match is None:
        raise AssertionError(f"Could not find function {name}() in entrypoint.sh")
    return match.group(0)


class FrontendDependencySyncTests(unittest.TestCase):
    maxDiff = None

    def _run_sync(self, *, stamp_contents: str | None, package_lock_contents: str) -> tuple[subprocess.CompletedProcess[str], str | None]:
        script_text = ENTRYPOINT.read_text(encoding="utf-8")
        shell = "\n".join(
            [
                "set -e",
                "log() { :; }",
                _extract_function(script_text, "frontend_lock_checksum"),
                _extract_function(script_text, "sync_frontend_dependencies"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frontend_dir = temp_path / "frontend"
            node_bin_dir = frontend_dir / "node_modules" / ".bin"
            node_bin_dir.mkdir(parents=True)
            (frontend_dir / "package-lock.json").write_text(package_lock_contents, encoding="utf-8")

            stamp_path = frontend_dir / "node_modules" / ".package-lock.sha256"
            if stamp_contents is not None:
                stamp_path.write_text(stamp_contents, encoding="utf-8")

            bin_dir = temp_path / "bin"
            bin_dir.mkdir()
            npm_log = temp_path / "npm-calls.log"
            npm_stub = textwrap.dedent(
                f"""\
                #!/bin/bash
                set -e
                printf 'npm %s\n' "$*" >> "{npm_log}"
                mkdir -p "{node_bin_dir}"
                """
            )
            npm_path = bin_dir / "npm"
            npm_path.write_text(npm_stub, encoding="utf-8")
            npm_path.chmod(0o755)

            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    shell + f'\nPATH="{bin_dir}:$PATH"\nsync_frontend_dependencies "{frontend_dir}"\n',
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            npm_calls = []
            if npm_log.exists():
                npm_calls = npm_log.read_text(encoding="utf-8").splitlines()
            result.npm_calls = npm_calls  # type: ignore[attr-defined]
            final_stamp_contents = None
            if stamp_path.exists():
                final_stamp_contents = stamp_path.read_text(encoding="utf-8")
            return result, final_stamp_contents

    def test_lock_checksum_change_reinstalls_frontend_dependencies(self) -> None:
        result, final_stamp_contents = self._run_sync(
            stamp_contents="stale-checksum\n",
            package_lock_contents='{"name":"ragtime","lockfileVersion":3}\n',
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(result.npm_calls, ["npm ci --loglevel=error"])  # type: ignore[attr-defined]
        self.assertIsNotNone(final_stamp_contents)
        self.assertNotEqual(final_stamp_contents, "stale-checksum\n")

    def test_matching_lock_checksum_skips_frontend_reinstall(self) -> None:
        package_lock_contents = '{"name":"ragtime","lockfileVersion":3,"packages":{}}\n'
        initial_result, current_checksum = self._run_sync(
            stamp_contents=None,
            package_lock_contents=package_lock_contents,
        )
        self.assertEqual(initial_result.returncode, 0, msg=initial_result.stdout + initial_result.stderr)
        self.assertIsNotNone(current_checksum)

        result, final_stamp_contents = self._run_sync(
            stamp_contents=current_checksum,
            package_lock_contents=package_lock_contents,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(result.npm_calls, [])  # type: ignore[attr-defined]
        self.assertEqual(final_stamp_contents, current_checksum)


if __name__ == "__main__":
    unittest.main()
