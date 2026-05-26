import importlib.util
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
