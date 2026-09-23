import subprocess
import sys
import unittest


class RuntimePackageImportTests(unittest.TestCase):
    def _assert_child_succeeds(self, source: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_worker_service_can_be_imported_before_manager_models(self) -> None:
        self._assert_child_succeeds(
            "from runtime.worker.service import WorkerService; "
            "from runtime.manager.models import StartSessionRequest; "
            "assert WorkerService.__module__ == 'runtime.worker.service'; "
            "assert StartSessionRequest.__module__ == 'runtime.manager.models'"
        )

    def test_manager_models_can_be_imported_before_worker_service(self) -> None:
        self._assert_child_succeeds(
            "from runtime.manager.models import StartSessionRequest; "
            "from runtime.worker.service import WorkerService; "
            "assert StartSessionRequest.__module__ == 'runtime.manager.models'; "
            "assert WorkerService.__module__ == 'runtime.worker.service'"
        )

    def test_manager_public_exports_are_lazy_api_objects(self) -> None:
        self._assert_child_succeeds(
            "import sys; "
            "import runtime.manager as manager; "
            "assert 'runtime.manager.api' not in sys.modules; "
            "from runtime.manager import app, create_app; "
            "from runtime.manager.api import app as api_app, create_app as api_create_app; "
            "assert app is api_app; "
            "assert create_app is api_create_app; "
            "assert manager.app is api_app; "
            "assert manager.create_app is api_create_app"
        )


if __name__ == "__main__":
    unittest.main()
