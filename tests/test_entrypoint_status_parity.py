import unittest

from ragtime.core.entrypoint_status import parse_entrypoint_content as parse_app_entrypoint_content
from runtime.core.shared import parse_entrypoint_content as parse_runtime_entrypoint_content


class EntrypointContentParserParityTests(unittest.TestCase):
    def _parse_with_both(self, content: str | None):
        return (parse_app_entrypoint_content(content), parse_runtime_entrypoint_content(content))

    def test_valid_content_normalizes_fields(self) -> None:
        for status in self._parse_with_both('{"command": " npm run dev ", "cwd": "app\\\\web", "framework": "VITE"}'):
            self.assertEqual(status.state, "valid")
            self.assertEqual(status.command, "npm run dev")
            self.assertEqual(status.cwd, "app/web")
            self.assertEqual(status.framework, "vite")
            self.assertTrue(status.framework_known)
            self.assertIsNone(status.error)

    def test_missing_command_is_invalid_with_normalized_metadata(self) -> None:
        for status in self._parse_with_both('{"cwd": " server ", "framework": "express"}'):
            self.assertEqual(status.state, "invalid")
            self.assertEqual(status.command, "")
            self.assertEqual(status.cwd, "server")
            self.assertEqual(status.framework, "express")
            self.assertTrue(status.framework_known)
            self.assertIn("has no command", status.error or "")

    def test_malformed_content_is_invalid(self) -> None:
        for status in self._parse_with_both("{"):
            self.assertEqual(status.state, "invalid")
            self.assertIn("Failed to parse", status.error or "")

    def test_unknown_framework_and_default_cwd_remain_valid(self) -> None:
        for status in self._parse_with_both('{"command": "serve", "framework": "bespoke"}'):
            self.assertEqual(status.state, "valid")
            self.assertEqual(status.cwd, ".")
            self.assertEqual(status.framework, "bespoke")
            self.assertFalse(status.framework_known)


if __name__ == "__main__":
    unittest.main()
