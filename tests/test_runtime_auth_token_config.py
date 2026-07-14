import importlib
import os
import unittest
from unittest import mock

from ragtime.config.settings import Settings


def _settings(env: dict[str, str]) -> Settings:
    with (
        mock.patch.dict(os.environ, {"ENCRYPTION_KEY": "test-encryption-key", **env}, clear=True),
        mock.patch.dict(Settings.model_config, {"env_file": None}),
    ):
        return Settings()


class RuntimeAuthTokenSettingsTests(unittest.TestCase):
    def test_userspace_runtime_auth_token_resolution(self) -> None:
        cases = [
            ({"RUNTIME_AUTH_TOKEN": "shared-token"}, "shared-token"),
            (
                {
                    "RUNTIME_AUTH_TOKEN": "shared-token",
                    "RUNTIME_MANAGER_AUTH_TOKEN": "runtime-manager-token",
                    "RUNTIME_WORKER_AUTH_TOKEN": "runtime-worker-token",
                },
                "shared-token",
            ),
            ({"RUNTIME_MANAGER_AUTH_TOKEN": "legacy-manager-token"}, "legacy-manager-token"),
        ]

        for env, expected in cases:
            with self.subTest(env=env):
                settings = _settings(env)
                self.assertEqual(settings.userspace_runtime_auth_token, expected)


class RuntimeAuthTokenWarningTests(unittest.TestCase):
    def test_runtime_auth_token_warning_cases(self) -> None:
        cases = [
            ("custom shared token", {"RUNTIME_AUTH_TOKEN": "a-strong-random-token"}, False),
            ("empty token", {}, True),
            ("generic shared token", {"RUNTIME_AUTH_TOKEN": "runtime-auth-token"}, True),
            ("generic legacy manager token", {"RUNTIME_AUTH_TOKEN": "runtime-manager-token"}, True),
            ("generic legacy worker token", {"RUNTIME_AUTH_TOKEN": "runtime-worker-token"}, True),
            ("legacy bridge token", {"RUNTIME_MANAGER_AUTH_TOKEN": "custom-legacy-token"}, True),
            (
                "debug mode override",
                {"DEBUG_MODE": "true", "RUNTIME_AUTH_TOKEN": "dev-runtime-auth-token"},
                False,
            ),
        ]

        for label, env, expected in cases:
            with self.subTest(label=label, env=env):
                settings = _settings(env)
                self.assertEqual(settings.runtime_auth_token_warning(), expected)


class RuntimeAuthResolverTests(unittest.TestCase):
    def _reload_runtime_auth(self, env: dict[str, str]) -> str:
        try:
            import runtime.auth as runtime_auth
        except ModuleNotFoundError as exc:
            self.skipTest(f"runtime package is not available in this test environment: {exc}")
        with mock.patch.dict(os.environ, env, clear=True):
            runtime_auth = importlib.reload(runtime_auth)
            return runtime_auth.get_runtime_auth_token()

    def test_runtime_auth_token_resolution(self) -> None:
        cases = [
            (
                {
                    "RUNTIME_AUTH_TOKEN": "shared-token",
                    "RUNTIME_MANAGER_AUTH_TOKEN": "runtime-manager-token",
                    "RUNTIME_WORKER_AUTH_TOKEN": "runtime-worker-token",
                },
                "shared-token",
            ),
            ({"RUNTIME_MANAGER_AUTH_TOKEN": "legacy-manager-token"}, "legacy-manager-token"),
            ({}, ""),
        ]

        for env, expected in cases:
            with self.subTest(env=env):
                token = self._reload_runtime_auth(env)
                self.assertEqual(token, expected)
