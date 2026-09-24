"""Focused tests for the deterministic external User Space instruction bundle."""

import unittest

from ragtime.userspace.instruction_bundle import build_instruction_bundle


class InstructionBundleTests(unittest.TestCase):
    def _context(self):
        return {
            "workspace": {
                "id": "ws-1",
                "name": "Sales",
                "sqlite_persistence_mode": "include",
                "caller_role": "owner",
            },
            "user": {"username": "ada", "display_name": "Ada Lovelace", "role": "owner"},
            "architecture": {
                "entrypoint_state": "valid",
                "framework": "node",
                "command": "node server.js",
                "cwd": ".",
                "file_count": 1,
                "key_files": ["server.js"],
            },
            "snapshot_summary": {"last_message": "Initial app"},
            "selected_tools": [
                {
                    "component_id": "tool-granted",
                    "name": "Sales DB",
                    "tool_type": "postgres",
                    "description": "Granted database",
                }
            ],
            "authorized_indexes": [
                {
                    "name": "Workspace code",
                    "source_type": "filesystem",
                    "document_count": 1,
                    "chunk_count": 2,
                }
            ],
            "authorized_build_credentials": [{"name": "deploy-key", "value": "AUTHORIZED-CREDENTIAL"}],
            "capabilities": {"tool_names": ["run_terminal_command"]},
            "authorized_resources": {
                "mounts": [
                    {
                        "workspace_relative_path": "data",
                        "target_path": "/workspace/data",
                        "source_name": "Sales DB export",
                        "source_path": "exports",
                        "sync_status": "ready",
                        "enabled": "true",
                    }
                ]
            },
        }

    def test_bundle_is_deterministic_and_uses_shared_prompt_builders(self):
        first = build_instruction_bundle(self._context())
        second = build_instruction_bundle(self._context())

        self.assertEqual(first, second)
        system = "\n".join(first["system_instructions"].values())
        self.assertIn("RAGTIME_BRIDGE_URL", system)
        self.assertIn("CURRENT USER", system)
        self.assertIn("Ada Lovelace", system)
        self.assertIn("SQLite local persistence", first["turn_instructions"])
        self.assertIn("AUTHORIZED-CREDENTIAL", str(first["capabilities"]))
        self.assertIn("Sales DB export", system)
        self.assertIn("`index_search`", first["system_instructions"]["authorized_indexes"])
        self.assertNotIn("`search_knowledge`", first["system_instructions"]["authorized_indexes"])

    def test_bundle_does_not_discover_or_include_unprovided_resources(self):
        context = self._context()
        context["unrelated_platform_secret"] = "DO-NOT-EXPORT"
        context["global_tools"] = [{"name": "Forbidden Tool"}]
        context["global_indexes"] = [{"name": "Forbidden Index"}]

        bundle = build_instruction_bundle(context)
        rendered = str(bundle)

        self.assertNotIn("DO-NOT-EXPORT", rendered)
        self.assertNotIn("Forbidden Tool", rendered)
        self.assertNotIn("Forbidden Index", rendered)
        self.assertIn("external harness responsibilities", rendered.lower())

    def test_bundle_preserves_missing_entrypoint_and_no_live_data_state(self):
        context = self._context()
        context["workspace"]["sqlite_persistence_mode"] = "exclude"
        context["architecture"] = {"entrypoint_state": "missing", "file_count": 0, "key_files": []}
        context["selected_tools"] = []

        bundle = build_instruction_bundle(context)
        system = bundle["system_instructions"]

        self.assertIn("No effective runtime entrypoint", system["entrypoint"])
        self.assertNotIn("SQLite local persistence", bundle["turn_instructions"])
        self.assertNotIn("live data execution results", system["workspace"])


if __name__ == "__main__":
    unittest.main()
