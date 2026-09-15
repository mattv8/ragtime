import unittest

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class RuntimeCredentialModeTests(unittest.TestCase):
    def test_worker_file_finalization_does_not_expose_token(self) -> None:
        service = UserSpaceRuntimeService()
        env = service._finalize_workspace_env("workspace", "session", {"EXISTING": "value"}, bridge_credential_mode="worker_file")
        self.assertEqual(env["EXISTING"], "value")
        self.assertIn("RAGTIME_BRIDGE_URL", env)
        self.assertNotIn("RAGTIME_BRIDGE_TOKEN", env)

    def test_env_finalization_keeps_legacy_token(self) -> None:
        service = UserSpaceRuntimeService()
        env = service._finalize_workspace_env("workspace", "session", {}, bridge_credential_mode="env")
        self.assertIn("RAGTIME_BRIDGE_TOKEN", env)
