"""Instruction skill delivery to request-scoped chat tools."""

import json
import unittest
from unittest import mock

from ragtime.rag.components import STREAM_DISPLAY_UNTRUNCATED_TOOL_NAMES, wrap_tools_with_content_protection
from ragtime.rag.prompts import build_tool_system_prompt
from ragtime.skills.chat_tool import build_read_skill_tool
from ragtime.skills.contract import render_skill_markdown
from ragtime.userspace.instruction_bundle import build_instruction_bundle
from tests.content_protection_support import use_disabled_content_protection
from tests.test_tool_skill_shared import make_noop_tool, make_rag_components, make_sql_bundle_tools


def instruction_skill(name, mode, content="Instructions"):
    return {"id": name, "name": name, "description": f"Description of {name}", "content": content, "mode": mode, "sort_order": 0}


def tool_config():
    return {
        "id": "tool-1",
        "name": "Demo SQL",
        "tool_type": "postgres",
        "description": "Reads demo SQL rows.",
        "skills": [instruction_skill("rules", "always"), instruction_skill("details", "on_demand")],
    }


class InstructionSkillPromptTests(unittest.TestCase):
    def test_prompt_and_legacy_output(self):
        config = tool_config()
        prompt = build_tool_system_prompt([config], instruction_skill_loader_available=True)
        self.assertIn("  - Skill `rules`: Description of rules (always applied; text below)", prompt)
        self.assertIn("  - Skill `details`: Description of details Load with `read_skill` (uri `skill://tools/tool-1/details/SKILL.md`).", prompt)
        self.assertIn("### Demo SQL · rules\n\nInstructions", prompt)
        prebuilt = build_tool_system_prompt([config])
        self.assertNotIn("`details`", prebuilt)
        self.assertNotIn("read_skill", prebuilt)
        legacy = {key: value for key, value in config.items() if key != "skills"}
        self.assertEqual(
            build_tool_system_prompt([legacy]),
            "\n\n## SYSTEM TOOLS\n\n- **Demo SQL** [PostgreSQL]: Reads demo SQL rows.\n\n"
            "Each tool connects to a different system. Read the description to choose the correct one.\n",
        )
        config["skills"][1]["description"] = "First line\n  second line"
        self.assertIn("  - Skill `details`: First line second line Load with", build_tool_system_prompt([config], instruction_skill_loader_available=True))
        config["skills"][1]["description"] = "x" * 301
        self.assertIn("x" * 300 + "…", build_tool_system_prompt([config], instruction_skill_loader_available=True))

    def test_bundle_without_skill_key_does_not_advertise_loader(self):
        bundle = build_instruction_bundle({"selected_tools": [{"component_id": "tool-1", "name": "Demo", "tool_type": "postgres"}]})
        self.assertNotIn("read_skill", str(bundle))


class InstructionSkillLoaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_loader_lookup_is_scoped(self):
        config = tool_config()
        loader = build_read_skill_tool([config])
        assert loader is not None
        self.assertIsNotNone(loader.coroutine)
        uri = "skill://tools/tool-1/details/SKILL.md"
        self.assertEqual(await loader.ainvoke({"uri": uri}), render_skill_markdown(config["skills"][1]))
        error = json.loads(await loader.ainvoke({"uri": "skill://tools/other/details/SKILL.md"}))
        self.assertIn(uri, error["valid_uris"])
        self.assertIn("error", error)
        self.assertIsNone(build_read_skill_tool([{**config, "skills": [config["skills"][0]]}]))


class InstructionSkillRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        use_disabled_content_protection(self)
        self.rag = make_rag_components(tool_skills_enabled=False)
        self.rag._tool_configs = [tool_config()]

    async def context(self, tools):
        with mock.patch.object(self.rag, "_create_ssh_transfer_tool", new=mock.AsyncMock(return_value=None)):
            return await self.rag._build_request_runtime_context(
                is_ui=False,
                executor=None,
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=False,
                runtime_tool_source_override=tools,
            )

    def prompt(self, tools, hidden=None):
        return self.rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=None,
            runtime_tools=tools,
            hidden_tool_config_ids=hidden,
        )

    async def test_binding_visibility_and_cache(self):
        context = await self.context(make_sql_bundle_tools())
        tools = context["runtime_tools"]
        self.assertIn("read_skill", [tool.name for tool in tools])
        self.assertIn("`details`", self.prompt(tools))
        self.assertNotIn("`details`", self.prompt(make_sql_bundle_tools()))
        self.assertNotIn("Instructions", self.prompt([], hidden=["tool-1"]))
        self.assertNotIn("Instructions", self.prompt(make_sql_bundle_tools(), hidden=["tool-1"]))

    async def test_collision_does_not_advertise_loader(self):
        collision = make_noop_tool("read_skill")
        with self.assertLogs("ragtime.rag.components", level="WARNING"):
            context = await self.context(make_sql_bundle_tools() + [collision])
        self.assertEqual(sum(tool.name == "read_skill" for tool in context["runtime_tools"]), 1)
        self.assertNotIn("`details`", self.prompt(context["runtime_tools"]))

    async def test_large_skill_survives_runtime_protection_wrapper(self):
        self.rag._app_settings["max_tool_output_chars"] = 5000
        self.rag._tool_configs[0]["skills"][1]["content"] = "x" * 7000
        context = await self.context(make_sql_bundle_tools())
        loader = next(tool for tool in context["runtime_tools"] if tool.name == "read_skill")
        wrapped = wrap_tools_with_content_protection([loader], self.rag._clone_structured_tool, tool_ids_by_name={})[0]
        output = await wrapped.ainvoke({"uri": "skill://tools/tool-1/details/SKILL.md"})
        self.assertIn("x" * 7000, output)
        self.assertIn("read_skill", STREAM_DISPLAY_UNTRUNCATED_TOOL_NAMES)
