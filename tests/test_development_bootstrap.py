import asyncio
import json
import tomllib
import unittest
from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType, SimpleNamespace

from fastapi import HTTPException

from ragtime.userspace.development_bootstrap import (
    MAX_MCP_RESPONSE_BYTES,
    MAX_NATIVE_SKILL_BYTES,
    _catalog,
    build_bootstrap_artifacts,
    build_bootstrap_manifest,
    build_compact_context,
    content_hash,
    get_bootstrap_artifact,
    read_context_facts,
    read_document,
    read_operation_contract,
    read_resource_contract,
    read_resource_description,
    read_resources,
)


class DevelopmentBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guidance = {"workspace": "workspace guidance", "runtime": "runtime guidance"}
        self.manifest = build_bootstrap_manifest(origin="https://ragtime.test", workspace_id="workspace-1", scopes={"read"}, guidance=self.guidance)

    def test_manifest_has_hashed_native_assets_without_credentials(self) -> None:
        encoded = json.dumps(self.manifest)
        self.assertNotIn("rtdev_", encoded)
        self.assertRegex(self.manifest["credential_env_var"], r"^RAGTIME_DEVELOPMENT_TOKEN_[A-F0-9]{12}$")
        self.assertIn("skills/workspace/SKILL.md", {item["id"] for item in self.manifest["artifacts"]})
        self.assertLessEqual(
            len(
                get_bootstrap_artifact(
                    artifact_id="core/rules.md",
                    revision=self.manifest["guidance_revision"],
                    origin="https://ragtime.test",
                    workspace_id="workspace-1",
                    guidance=self.guidance,
                )["content"].encode()
            ),
            3 * 1024,
        )

    def test_catalog_materializes_mapping_capabilities(self) -> None:
        capabilities: Mapping[str, list[object]] = MappingProxyType({"authorized_tools": [], "authorized_indexes": []})
        catalog = _catalog({"capabilities": capabilities})
        self.assertIsInstance(catalog, dict)
        self.assertEqual(catalog, capabilities)

    def test_profiles_declare_native_paths_and_only_required_assets(self) -> None:
        artifacts = {item["id"]: item["path"] for item in self.manifest["artifacts"]}
        profiles = self.manifest["profiles"]
        self.assertEqual(profiles["opencode"]["skill_root"], ".opencode/skills")
        self.assertEqual(artifacts["opencode-skills/workspace/SKILL.md"], ".opencode/skills/ragtime-workspace/SKILL.md")
        self.assertEqual(profiles["codex"]["skill_root"], ".agents/skills")
        self.assertEqual(profiles["claude-code"]["skill_root"], ".claude/skills")
        for profile in profiles.values():
            self.assertIn("artifact_ids", profile)
            self.assertIn("destinations", profile)
            self.assertEqual(set(profile["artifact_ids"]), set(profile["destinations"]))
            self.assertTrue(all("*" not in artifact_id for artifact_id in profile["artifact_ids"]))
        self.assertNotIn("required_artifacts", json.dumps(profiles))

    def test_current_canonical_topics_are_complete_inline_native_skills(self) -> None:
        artifacts = build_bootstrap_artifacts(origin="https://ragtime.test", workspace_id="workspace-1")
        from ragtime.userspace.instruction_content import build_external_guidance_documents

        for topic, canonical_text in build_external_guidance_documents().items():
            for prefix in ("opencode-skills", "skills", "claude-skills"):
                with self.subTest(topic=topic, profile=prefix):
                    skill = artifacts[f"{prefix}/{topic}/SKILL.md"]["content"]
                    self.assertTrue(skill.startswith(f"---\nname: ragtime-{topic}\ndescription: "))
                    self.assertIn("\n---\n", skill)
                    self.assertIn(canonical_text, skill)
                    self.assertLessEqual(len(skill.encode("utf-8")), MAX_NATIVE_SKILL_BYTES)
                    self.assertFalse(any(key.startswith(f"{prefix}/{topic}/references/") for key in artifacts))

    def test_oversized_native_skill_references_reconstruct_exact_guidance(self) -> None:
        guidance = {"workspace": "λ" * 30000}
        artifacts = build_bootstrap_artifacts(origin="https://ragtime.test", workspace_id="workspace-1", guidance=guidance)
        for prefix in ("opencode-skills", "skills", "claude-skills"):
            with self.subTest(profile=prefix):
                skill = artifacts[f"{prefix}/workspace/SKILL.md"]["content"]
                self.assertLessEqual(len(skill.encode("utf-8")), MAX_NATIVE_SKILL_BYTES)
                reference_ids = sorted(key for key in artifacts if key.startswith(f"{prefix}/workspace/references/"))
                self.assertGreater(len(reference_ids), 1)
                self.assertEqual("".join(artifacts[key]["content"] for key in reference_ids), guidance["workspace"])
                for index, artifact_id in enumerate(reference_ids, start=1):
                    reference = artifacts[artifact_id]
                    self.assertIn(f"{index}. [Reference {index:03d}](references/{index:03d}.md)", skill)
                    expected_path = PurePosixPath(artifacts[f"{prefix}/workspace/SKILL.md"]["path"]).parent / "references" / f"{index:03d}.md"
                    self.assertEqual(str(expected_path), reference["path"])

    def test_core_binds_workspace_and_refresh_requirements(self) -> None:
        core = get_bootstrap_artifact(
            artifact_id="core/rules.md",
            revision=self.manifest["guidance_revision"],
            origin="https://ragtime.test",
            workspace_id="workspace-1",
            guidance=self.guidance,
        )["content"]
        self.assertIn("Workspace UUID: `workspace-1`", core)
        self.assertIn("operations inside the `workspace_development` wrapper", core)
        self.assertIn("start each new client turn", core)
        self.assertIn("after compaction", core)
        self.assertIn("Load `ragtime-workspace` first", core)
        self.assertIn("guidance hashes", core)

    def test_profile_config_artifacts_have_native_syntax(self) -> None:
        opencode = get_bootstrap_artifact(
            artifact_id="config/opencode.json",
            revision=self.manifest["guidance_revision"],
            origin="https://ragtime.test",
            workspace_id="workspace-1",
            guidance=self.guidance,
        )["content"]
        claude = get_bootstrap_artifact(
            artifact_id="config/claude.json",
            revision=self.manifest["guidance_revision"],
            origin="https://ragtime.test",
            workspace_id="workspace-1",
            guidance=self.guidance,
        )["content"]
        codex = get_bootstrap_artifact(
            artifact_id="config/codex.toml",
            revision=self.manifest["guidance_revision"],
            origin="https://ragtime.test",
            workspace_id="workspace-1",
            guidance=self.guidance,
        )["content"]
        self.assertIn("mcp", json.loads(opencode))
        self.assertIn("mcpServers", json.loads(claude))
        self.assertIn("mcp_servers", tomllib.loads(codex))

    def test_client_tool_prefixes_fit_provider_limits_and_isolate_origins(self) -> None:
        from ragtime.mcp.server import _development_tools_for_principal

        def server_key(origin: str) -> str:
            artifact = get_bootstrap_artifact(
                artifact_id="config/opencode.json",
                revision=None,
                origin=origin,
                workspace_id="workspace-1",
                guidance=self.guidance,
            )
            return next(iter(json.loads(artifact["content"])["mcp"]))

        key = server_key("https://ragtime.test")
        self.assertNotEqual(key, server_key("https://staging.ragtime.test"))
        tools = asyncio.run(_development_tools_for_principal(SimpleNamespace(scopes=frozenset({"read", "write", "exec"}))))
        for tool in tools:
            with self.subTest(tool=tool.name):
                self.assertLessEqual(len(f"{key}_{tool.name}"), 64)
                self.assertLessEqual(len(f"mcp__{key}__{tool.name}"), 64)

    def test_unknown_and_stale_artifacts_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as stale:
            get_bootstrap_artifact(
                artifact_id="core/rules.md", revision="old", origin="https://ragtime.test", workspace_id="workspace-1", guidance=self.guidance
            )
        self.assertEqual(stale.exception.status_code, 409)
        with self.assertRaises(HTTPException) as unknown:
            get_bootstrap_artifact(artifact_id="../../secret", revision=None, origin="https://ragtime.test", workspace_id="workspace-1", guidance=self.guidance)
        self.assertEqual(unknown.exception.status_code, 404)

    def test_guidance_is_paged_complete_and_hash_pinned(self) -> None:
        guidance = {"workspace": "λ" * 20000}
        first = read_document("workspace", None, guidance=guidance)
        result = first["text"]
        offset = first["next_offset"]
        while offset is not None:
            page = read_document("workspace", first["sha256"], guidance=guidance, offset=offset)
            result += page["text"]
            offset = page["next_offset"]
        self.assertEqual(result, guidance["workspace"])
        self.assertEqual(first["sha256"], content_hash(guidance["workspace"]))

    def test_continuations_require_the_target_sha_but_first_pages_do_not(self) -> None:
        guidance = {"workspace": "x" * 20000}
        first = read_document("workspace", None, guidance=guidance)
        self.assertIsNotNone(first["next_offset"])
        with self.assertRaises(HTTPException) as missing_pin:
            read_document("workspace", None, guidance=guidance, offset=first["next_offset"])
        self.assertEqual(missing_pin.exception.status_code, 422)
        continued = read_document("workspace", first["sha256"], guidance=guidance, offset=first["next_offset"])
        self.assertEqual(continued["offset"], first["next_offset"])
        # A changed live context revision does not invalidate this static target hash.
        compact = build_compact_context({"context_revision": "new-live-context", "workspace": {"id": "workspace-1"}}, guidance=guidance)
        self.assertNotEqual(compact["context_revision"], first["sha256"])
        self.assertEqual(read_document("workspace", first["sha256"], guidance=guidance, offset=first["next_offset"])["sha256"], first["sha256"])

    def test_context_references_include_ready_to_use_target_pins(self) -> None:
        full = {"context_revision": "live", "workspace": {"id": "workspace-1"}, "facts": {"key_files": ["app.py"]}}
        operations = [{"name": "file_read", "scope": "read", "input_schema": {}}]
        compact = build_compact_context(full, guidance=self.guidance, operations=operations)
        document = compact["guidance_documents"][0]
        self.assertEqual(document["read_arguments"]["workspace_id"], "workspace-1")
        self.assertEqual(document["read_arguments"]["revision"], document["sha256"])
        contract = compact["operation_contracts"][0]
        self.assertEqual(contract["read_arguments"]["revision"], contract["sha256"])
        self.assertLessEqual(len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode()), MAX_MCP_RESPONSE_BYTES)

    def test_compact_context_and_resources_do_not_truncate_large_descriptions(self) -> None:
        full = {"context_revision": "live", "capabilities": {"authorized_tools": [{"name": "large", "description": "x" * 70000}], "authorized_indexes": []}}
        compact = build_compact_context(full, guidance=self.guidance, operations=[])
        self.assertLessEqual(len(json.dumps(compact).encode()), MAX_MCP_RESPONSE_BYTES)
        page = read_resources(full)
        self.assertIsNone(page["items"][0]["description"])
        self.assertTrue(page["items"][0]["description_reference"]["complete"])
        description = read_resource_description(full, resource_id=page["items"][0]["resource_id"], revision=page["items"][0]["description_reference"]["sha256"])
        text = description["text"]
        while description["next_offset"] is not None:
            description = read_resource_description(
                full, resource_id=page["items"][0]["resource_id"], revision=description["sha256"], offset=description["next_offset"]
            )
            text += description["text"]
        self.assertEqual(text, "x" * 70000)

    def test_documents_and_large_operation_contracts_reconstruct_from_offsets(self) -> None:
        guidance = {"workspace": "λ" * 50000}
        first = read_document("workspace", None, guidance=guidance)
        text = first["text"]
        while first["next_offset"] is not None:
            first = read_document("workspace", first["sha256"], guidance=guidance, offset=first["next_offset"])
            text += first["text"]
        self.assertEqual(text, guidance["workspace"])

        schema = {"type": "object", "properties": {f"field_{number}": {"description": "x" * 200} for number in range(700)}}
        operations = [{"name": "large", "scope": "read", "input_schema": schema}]
        page = read_operation_contract("large", None, operations=operations)
        contract_text = page["text"]
        while page["next_offset"] is not None:
            page = read_operation_contract("large", page["sha256"], operations=operations, offset=page["next_offset"])
            contract_text += page["text"]
        self.assertEqual(json.loads(contract_text), operations[0])

        resource_context = {
            "capabilities": {
                "authorized_tools": [
                    {
                        "component_id": "large-resource",
                        "request_schema": {"properties": {f"field_{number}": {"description": "x" * 200} for number in range(700)}},
                    }
                ],
                "authorized_indexes": [],
            }
        }
        resource_id = read_resources(resource_context)["items"][0]["resource_id"]
        resource_page = read_resource_contract(resource_context, resource_id=resource_id, revision=None)
        resource_text = resource_page["text"]
        while resource_page["next_offset"] is not None:
            resource_page = read_resource_contract(
                resource_context, resource_id=resource_id, revision=resource_page["sha256"], offset=resource_page["next_offset"]
            )
            resource_text += resource_page["text"]
        self.assertEqual(json.loads(resource_text), resource_context["capabilities"]["authorized_tools"][0])

    def test_pages_round_trip_control_characters_and_unicode_within_json_budget(self) -> None:
        guidance = {"workspace": ("\x00\n\t" * 9000) + ("λ😀" * 4000)}
        page = read_document("workspace", None, guidance=guidance)
        text = ""
        while True:
            self.assertLessEqual(len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), MAX_MCP_RESPONSE_BYTES)
            text += page["text"]
            if page["next_offset"] is None:
                break
            page = read_document("workspace", page["sha256"], guidance=guidance, offset=page["next_offset"])
        self.assertEqual(text, guidance["workspace"])

    def test_pagers_reject_mid_codepoint_offsets(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            read_resource_description(
                {"capabilities": {"authorized_tools": [{"component_id": "tool", "description": "λ" * 5000}], "authorized_indexes": []}},
                resource_id="tool:" + content_hash("tool"),
                revision=None,
                offset=1,
            )
        self.assertEqual(raised.exception.status_code, 422)

    def test_resource_ids_are_type_namespaced_when_tool_and_index_names_collide(self) -> None:
        full = {
            "capabilities": {
                "authorized_tools": [{"component_id": "shared", "description": "tool"}],
                "authorized_indexes": [{"index_name": "shared", "description": "index"}],
            }
        }
        items = read_resources(full)["items"]
        self.assertEqual(len({item["resource_id"] for item in items}), 2)
        self.assertTrue(any(item["resource_id"].startswith("tool:") for item in items))
        self.assertTrue(any(item["resource_id"].startswith("index:") for item in items))

    def test_facts_page_is_pinned_to_static_facts_not_runtime_heartbeat(self) -> None:
        first_context = {
            "instruction_bundle": {
                "workspace": {"id": "w"},
                "user": {"id": "u"},
                "facts": {"key_files": ["a"]},
                "runtime": {"clock": 1},
                "diagnostics": {"heartbeat": 1},
            }
        }
        second_context = {
            "instruction_bundle": {
                "workspace": {"id": "w"},
                "user": {"id": "u"},
                "facts": {"key_files": ["a"]},
                "runtime": {"clock": 2},
                "diagnostics": {"heartbeat": 2},
            }
        }
        first = read_context_facts(first_context, None)
        second = read_context_facts(second_context, first["sha256"])
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["text"], second["text"])

    def test_resource_page_makes_progress_with_many_long_names(self) -> None:
        full = {
            "capabilities": {
                "authorized_tools": [{"component_id": "tool-" + str(number) + ("x" * 10000), "description": "d" * 10000} for number in range(10)],
                "authorized_indexes": [],
            }
        }
        page = read_resources(full, limit=100)
        self.assertGreater(len(page["items"]), 0)
        self.assertLessEqual(len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), MAX_MCP_RESPONSE_BYTES)


if __name__ == "__main__":
    unittest.main()
