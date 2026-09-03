import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
EXTRACTOR = ROOT / "docker" / "scripts" / "install_deps_from_pyproject.py"


def _load_extractor_module():
    spec = importlib.util.spec_from_file_location("install_deps_from_pyproject", EXTRACTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load dependency extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyContainerDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = _load_extractor_module()
        self.pyproject_text = PYPROJECT.read_text(encoding="utf-8")
        self.dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")

    def test_legacy_group_pins_faiss_cpu_below_regressed_release(self) -> None:
        legacy_deps = self.extractor._extract_group(self.pyproject_text, "legacy")

        self.assertIn("faiss-cpu==1.13.0", legacy_deps)
        self.assertNotIn("chonkie==1.3.1", legacy_deps)
        self.assertNotIn("tree-sitter-language-pack==1.6.2", legacy_deps)

    def test_legacy_constraints_include_resolver_only_pins(self) -> None:
        legacy_constraints = self.extractor._extract_group(self.pyproject_text, "legacy-constraints")

        self.assertIn("faiss-cpu==1.13.0", legacy_constraints)
        self.assertIn("chonkie==1.3.1", legacy_constraints)
        self.assertIn("tree-sitter-language-pack==1.6.2", legacy_constraints)

    def test_app_group_keeps_faiss_available_for_non_legacy_builds(self) -> None:
        app_deps = self.extractor._extract_group(self.pyproject_text, "app")

        self.assertIn("faiss-cpu>=1.7.4,<2.0.0", app_deps)

    def test_legacy_build_constrains_app_install_before_force_reinstall(self) -> None:
        constraints_generation = "python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml legacy-constraints /tmp/constraints.legacy.txt"
        constrained_install = "pip install -c /tmp/constraints.legacy.txt -r /tmp/requirements.app.txt"
        force_reinstall = "pip install --force-reinstall --no-deps -r /tmp/requirements.legacy.txt"

        self.assertIn(constraints_generation, self.dockerfile_text)
        self.assertIn(constrained_install, self.dockerfile_text)
        self.assertIn(force_reinstall, self.dockerfile_text)
        self.assertLess(
            self.dockerfile_text.index(constrained_install),
            self.dockerfile_text.index(force_reinstall),
        )

    def test_legacy_build_asserts_faiss_pin_and_native_import(self) -> None:
        self.assertIn("version('faiss-cpu') == '1.13.0'", self.dockerfile_text)
        self.assertIn("import faiss", self.dockerfile_text)

    def test_production_stage_propagates_legacy_cpu_env(self) -> None:
        # The production stage must reset the ARG so the build-arg flows in,
        # and expose it to the entrypoint so the CPU preflight can branch.
        production_marker = "FROM python:3.12-slim-trixie AS production"
        self.assertIn(production_marker, self.dockerfile_text)
        production_stage = self.dockerfile_text.split(production_marker, 1)[1]
        self.assertIn("ARG LEGACY_CPU=0", production_stage)
        self.assertIn("RAGTIME_LEGACY_CPU=${LEGACY_CPU}", production_stage)

    def test_legacy_build_uses_pre_native_claude_cli(self) -> None:
        production_marker = "FROM python:3.12-slim-trixie AS production"
        production_stage = self.dockerfile_text.split(production_marker, 1)[1]

        self.assertIn('if [ "$LEGACY_CPU" = "1" ]; then', production_stage)
        self.assertIn("npm install -g @anthropic-ai/claude-code@2.1.112", production_stage)
        self.assertIn("npm install -g @anthropic-ai/claude-code@2.1.186", production_stage)
        self.assertIn("claude --version", production_stage)

    def _entrypoint_text(self) -> str:
        return (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    def _extract_cpu_preflight_function(self) -> str:
        lines = self._entrypoint_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("cpu_baseline_preflight()"))

        brace_depth = 0
        extracted: list[str] = []
        for line in lines[start:]:
            extracted.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth == 0:
                return "\n".join(extracted) + "\n"

        raise AssertionError("cpu_baseline_preflight function was not closed in docker/entrypoint.sh")

    def _cpuinfo_fixture(self, flags: str) -> str:
        return (
            "processor\t: 0\n"
            "vendor_id\t: GenuineIntel\n"
            f"flags\t\t: {flags}\n"
            "model name\t: Test CPU\n"
            "\n"
            "processor\t: 1\n"
            "vendor_id\t: GenuineIntel\n"
            f"flags\t\t: {flags}\n"
            "model name\t: Test CPU\n"
        )

    def _run_cpu_preflight(
        self,
        flags: str,
        *,
        legacy_cpu: Optional[str] = None,
        skip_preflight: Optional[str] = None,
        arch: str = "x86_64",
        write_cpuinfo: bool = True,
    ) -> Tuple[int, str]:
        wrapper = (
            f"#!/bin/bash\nlog() {{ printf '%s\\n' \"$*\"; }}\nuname() {{ echo {arch}; }}\n{self._extract_cpu_preflight_function()}cpu_baseline_preflight\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cpuinfo_path = tmpdir_path / "cpuinfo"
            wrapper_path = tmpdir_path / "cpu_preflight.sh"
            if write_cpuinfo:
                cpuinfo_path.write_text(self._cpuinfo_fixture(flags), encoding="utf-8")
            wrapper_path.write_text(wrapper, encoding="utf-8")

            env = os.environ.copy()
            env["RAGTIME_CPUINFO_PATH"] = str(cpuinfo_path)
            for name, value in {
                "RAGTIME_LEGACY_CPU": legacy_cpu,
                "RAGTIME_SKIP_CPU_PREFLIGHT": skip_preflight,
            }.items():
                if value is None:
                    env.pop(name, None)
                else:
                    env[name] = value

            result = subprocess.run(
                ["bash", str(wrapper_path)],
                check=False,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return result.returncode, result.stdout

    def test_entrypoint_cpu_preflight_accepts_linux_pni_x86_v2_flags(self) -> None:
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 pni ssse3 sse4_1 sse4_2 popcnt avx avx2"
        )

        self.assertEqual(returncode, 0, output)
        self.assertNotIn("missing", output.lower())

    def test_entrypoint_cpu_preflight_reports_missing_pni_on_non_legacy_hosts(self) -> None:
        returncode, output = self._run_cpu_preflight("fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2")

        self.assertNotEqual(returncode, 0)
        self.assertIn("pni", output)

    def test_entrypoint_cpu_preflight_allows_missing_pni_on_legacy_hosts(self) -> None:
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2",
            legacy_cpu="1",
        )

        self.assertEqual(returncode, 0, output)

    def test_entrypoint_cpu_preflight_keeps_whole_token_matching_for_pni(self) -> None:
        # Guards a future substring-matching rewrite; not the primary regression.
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ssse3 sse4_1 sse4_2 popcnt"
        )

        self.assertNotEqual(returncode, 0)
        self.assertIn("pni", output)

    def test_entrypoint_cpu_preflight_skip_flag_allows_incompatible_host(self) -> None:
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2",
            skip_preflight="1",
        )

        self.assertEqual(returncode, 0, output)
        self.assertIn("pni", output)

    def test_entrypoint_cpu_preflight_skips_on_non_x86_arch(self) -> None:
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2",
            arch="aarch64",
        )

        self.assertEqual(returncode, 0, output)
        self.assertNotIn("missing", output.lower())

    def test_entrypoint_cpu_preflight_skips_when_cpuinfo_unreadable(self) -> None:
        returncode, output = self._run_cpu_preflight(
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2",
            write_cpuinfo=False,
        )

        self.assertEqual(returncode, 0, output)
        self.assertNotIn("missing", output.lower())

    def test_entrypoint_still_includes_legacy_image_guidance(self) -> None:
        entrypoint = self._entrypoint_text()
        self.assertIn("cpu_baseline_preflight", entrypoint)
        self.assertIn("RAGTIME_LEGACY_CPU", entrypoint)
        self.assertIn("ragtime:legacy", entrypoint)


if __name__ == "__main__":
    unittest.main()
