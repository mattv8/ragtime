"""Semantic canaries for externally delivered User Space guidance."""

import unittest

from ragtime.userspace.development_service import development_service
from ragtime.userspace.instruction_bundle import build_instruction_bundle
from ragtime.userspace.instruction_content import build_external_guidance_documents


class DevelopmentInstructionFidelityTests(unittest.TestCase):
    def test_static_guidance_has_stable_topics_without_workspace_facts(self) -> None:
        documents = build_external_guidance_documents()

        self.assertEqual(set(documents), {"workspace", "runtime", "live-data", "persistence", "identity", "ui", "storage"})
        self.assertIn("runtime-entrypoint.json", documents["runtime"])
        self.assertIn("context.components", documents["live-data"])
        self.assertNotIn("No effective runtime entrypoint", "\n".join(documents.values()))
        self.assertIn("ragtime-auth", documents["workspace"])
        self.assertIn("File tool workflow", documents["workspace"])
        self.assertIn("exec_start", documents["runtime"])
        self.assertIn("Finalization sequence", documents["workspace"])
        self.assertIn("Relay that reason and next step", documents["workspace"])
        self.assertIn("do not automatically retry", documents["workspace"])
        self.assertIn("Workspace environment variables", documents["workspace"])
        self.assertIn("Theme + CSS rules", documents["ui"])
        self.assertIn("RAGTIME_OBJECT_STORAGE_ENDPOINT", documents["storage"])
        self.assertNotIn("`run_terminal_command`", "\n".join(documents.values()))
        self.assertNotIn("search_knowledge", "\n".join(documents.values()))
        self.assertNotIn("{{path}}", "\n".join(documents.values()))
        self.assertNotIn("create_html_component", documents["ui"])

    def test_persistence_lanes_have_independent_applicability(self) -> None:
        persistence = build_external_guidance_documents()["persistence"]

        self.assertIn("declare SQLite persistence enabled", persistence)
        self.assertIn("selected live-data tools", persistence)
        self.assertNotIn("SQLite persistence or selected live-data tools", persistence)

    def test_bundle_preserves_default_static_and_safe_resource_facts(self) -> None:
        bundle = build_instruction_bundle(
            {
                "workspace": {"sqlite_persistence_mode": "exclude"},
                "architecture": {
                    "entrypoint_state": "valid",
                    "framework": "static",
                    "command": "python3 -m http.server",
                    "is_default_static": True,
                    "file_count": 1,
                    "key_files": ["index.html"],
                },
                "env_vars": [{"key": "SMOKE_ENV_CANARY", "has_value": True}],
                "authorized_resources": {"object_storage_buckets": [{"name": "default", "public_root": "/default/public", "private_root": "/default/private"}]},
            }
        )

        self.assertIn("No effective runtime entrypoint", bundle["system_instructions"]["entrypoint"])
        self.assertIn("SMOKE_ENV_CANARY(set)", bundle["turn_instructions"])
        self.assertIn("Relay that reason and next step", bundle["turn_instructions"])
        self.assertIn("/default/public", bundle["system_instructions"]["object_storage"])
        self.assertFalse(bundle["facts"]["entrypoint"]["is_default_static"] is False)
        self.assertEqual(bundle["facts"]["environment_variables"], [{"key": "SMOKE_ENV_CANARY", "has_value": True}])
        self.assertNotIn("entrypoint configured and valid", bundle["system_instructions"]["workspace"])

    def test_registry_exposes_describable_safe_contracts(self) -> None:
        operations = {item["name"]: item for item in development_service.list_operations()}

        self.assertIn("operation_describe", operations)
        self.assertIn("http_api_catalog_search", operations)
        self.assertEqual(operations["operation_describe"]["input_schema"]["required"], ["operation"])
        self.assertEqual(
            operations["http_api_catalog_search"]["input_schema"]["required"],
            ["component_id", "query"],
        )


if __name__ == "__main__":
    unittest.main()
