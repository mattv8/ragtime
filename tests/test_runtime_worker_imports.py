from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RuntimeWorkerImportTests(unittest.TestCase):
    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_REPOSITORY_ROOT), environment.get("PYTHONPATH"))))
        return environment

    def _run_python(self, source: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=self._environment(),
        )

    def test_sandbox_launcher_module_invocation_still_works(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "runtime.worker.sandbox_launcher", "--help"],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=self._environment(),
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sandbox_launcher_help_does_not_load_worker_api_or_service(self) -> None:
        result = self._run_python(
            "import runpy\n"
            "import sys\n"
            "sys.argv = ['runtime.worker.sandbox_launcher', '--help']\n"
            "try:\n"
            "    runpy.run_module('runtime.worker.sandbox_launcher', run_name='__main__')\n"
            "except SystemExit as error:\n"
            "    assert error.code == 0\n"
            "else:\n"
            "    raise AssertionError('launcher help must exit')\n"
            "assert 'runtime.worker.api' not in sys.modules\n"
            "assert 'runtime.worker.service' not in sys.modules\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_manager_models_import_is_safe(self) -> None:
        result = self._run_python(
            "import runtime.manager.models\n"
            "import sys\n"
            "assert 'runtime.manager.api' not in sys.modules\n"
            "assert 'runtime.worker.api' not in sys.modules\n"
            "assert 'runtime.worker.service' not in sys.modules\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_worker_service_import_is_safe(self) -> None:
        result = self._run_python(
            "import runtime.worker.service\nimport sys\nassert 'runtime.manager.api' not in sys.modules\nassert 'runtime.worker.api' not in sys.modules\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manager_public_exports_are_lazy_and_preserve_identity(self) -> None:
        result = self._run_python(
            "import runtime.manager as manager\n"
            "import sys\n"
            "assert manager.__all__ == ['app', 'create_app']\n"
            "assert 'runtime.manager.api' not in sys.modules\n"
            "from runtime.manager.api import app, create_app\n"
            "assert manager.app is app\n"
            "assert manager.create_app is create_app\n"
            "try:\n"
            "    manager.not_an_export\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('unknown attributes must raise AttributeError')\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_utility_import_does_not_load_api_or_service(self) -> None:
        result = self._run_python(
            "import sys\n"
            "from runtime.worker import sandbox_launcher\n"
            "assert 'runtime.worker.api' not in sys.modules\n"
            "assert 'runtime.worker.service' not in sys.modules\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_public_exports_are_lazy_and_preserve_identity(self) -> None:
        result = self._run_python(
            "import runtime.worker as worker\n"
            "import sys\n"
            "assert worker.__all__ == ['app', 'create_app']\n"
            "assert 'runtime.worker.api' not in sys.modules\n"
            "assert 'runtime.worker.service' not in sys.modules\n"
            "from runtime.worker.api import app, create_app\n"
            "assert worker.app is app\n"
            "assert worker.create_app is create_app\n"
            "try:\n"
            "    worker.not_an_export\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('unknown attributes must raise AttributeError')\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
