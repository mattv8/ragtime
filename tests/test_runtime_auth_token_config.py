import importlib
import os
import unittest
from unittest import mock

from ragtime.config.settings import Settings


def _settings(env: dict[str, str]) -> Settings:
    with mock.patch.dict(os.environ, {"ENCRYPTION_KEY": "test-encryption-key", **env}, clear=True):
        return Settings(_env_file=None)


class RuntimeAuthTokenSettingsTests(unittest.TestCase):
    def test_shared_runtime_auth_token_is_used_directly(self) -> None:
        settings = _settings({"RUNTIME_AUTH_TOKEN": "shared-token"})

        self.assertEqual(settings.userspace_runtime_auth_token, "shared-token")

    def test_shared_runtime_auth_token_wins_over_stale_legacy_vars(self) -> None:
        settings = _settings(
            {
                "RUNTIME_AUTH_TOKEN": "shared-token",
                "RUNTIME_MANAGER_AUTH_TOKEN": "runtime-manager-token",
                "RUNTIME_WORKER_AUTH_TOKEN": "runtime-worker-token",
            }
        )

        self.assertEqual(settings.userspace_runtime_auth_token, "shared-token")

    def test_legacy_manager_token_bridges_when_shared_token_unset(self) -> None:
        settings = _settings({"RUNTIME_MANAGER_AUTH_TOKEN": "legacy-manager-token"})

        self.assertEqual(settings.userspace_runtime_auth_token, "legacy-manager-token")


class RuntimeAuthTokenWarningTests(unittest.TestCase):
    def test_no_warning_for_custom_shared_token(self) -> None:
        settings = _settings({"RUNTIME_AUTH_TOKEN": "a-strong-random-token"})

        self.assertFalse(settings.runtime_auth_token_warning())

    def test_warning_when_token_is_empty(self) -> None:
        settings = _settings({})

        self.assertTrue(settings.runtime_auth_token_warning())

    def test_warning_when_token_is_known_generic_default(self) -> None:
        for generic in ("runtime-auth-token", "runtime-manager-token", "runtime-worker-token"):
            settings = _settings({"RUNTIME_AUTH_TOKEN": generic})
            self.assertTrue(settings.runtime_auth_token_warning(), generic)

    def test_warning_when_token_resolved_via_legacy_bridge(self) -> None:
        settings = _settings({"RUNTIME_MANAGER_AUTH_TOKEN": "custom-legacy-token"})

        self.assertTrue(settings.runtime_auth_token_warning())

    def test_no_warning_in_debug_mode(self) -> None:
        settings = _settings({"DEBUG_MODE": "true", "RUNTIME_AUTH_TOKEN": "dev-runtime-auth-token"})

        self.assertFalse(settings.runtime_auth_token_warning())


class RuntimeAuthResolverTests(unittest.TestCase):
    def _reload_runtime_auth(self, env: dict[str, str]):
        try:
            import runtime.auth as runtime_auth
        except ModuleNotFoundError as exc:
            self.skipTest(f"runtime package is not available in this test environment: {exc}")
        with mock.patch.dict(os.environ, env, clear=True):
            return importlib.reload(runtime_auth)

    def test_shared_runtime_auth_token_is_used_directly(self) -> None:
        runtime_auth = self._reload_runtime_auth(
            {
                "RUNTIME_AUTH_TOKEN": "shared-token",
                "RUNTIME_MANAGER_AUTH_TOKEN": "runtime-manager-token",
                "RUNTIME_WORKER_AUTH_TOKEN": "runtime-worker-token",
            }
        )

        self.assertEqual(runtime_auth.get_runtime_auth_token(), "shared-token")

    def test_legacy_manager_token_bridges_when_shared_token_unset(self) -> None:
        runtime_auth = self._reload_runtime_auth({"RUNTIME_MANAGER_AUTH_TOKEN": "legacy-manager-token"})

        self.assertEqual(runtime_auth.get_runtime_auth_token(), "legacy-manager-token")

    def test_token_is_empty_when_nothing_is_configured(self) -> None:
        runtime_auth = self._reload_runtime_auth({})

        self.assertEqual(runtime_auth.get_runtime_auth_token(), "")
