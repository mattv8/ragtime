import unittest

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class RuntimeCredentialModeTests(unittest.TestCase):
    def test_finalization_replaces_all_caller_supplied_bridge_values(self) -> None:
        service = UserSpaceRuntimeService()
        env = service._finalize_workspace_env(
            "workspace",
            "session",
            {
                "EXISTING": "value",
                "RAGTIME_BRIDGE_URL": "https://caller.invalid",
                "RAGTIME_BRIDGE_TOKEN": "caller-token",
                "RAGTIME_BRIDGE_TOKEN_FILE": "/caller/token",
            },
        )
        self.assertEqual(env["EXISTING"], "value")
        self.assertIn("RAGTIME_BRIDGE_URL", env)
        self.assertNotIn("RAGTIME_BRIDGE_TOKEN", env)
        self.assertEqual(env["RAGTIME_BRIDGE_TOKEN_FILE"], "/run/.ragtime-bridge/token")
