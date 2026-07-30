import unittest

from ragtime.rag.prompts import (
    build_tool_system_prompt,
    build_userspace_diagnostics_turn_reminder_line,
    build_userspace_mode_prompt_addition,
)


class ToolSkillPromptTests(unittest.TestCase):
    def test_feature_disabled_preserves_legacy_prompt_text(self) -> None:
        configured_prompt = build_tool_system_prompt(
            [
                {
                    "id": "tool-1",
                    "name": "Demo SQL",
                    "tool_type": "postgres",
                    "description": "Reads demo rows.",
                }
            ],
            unavailable_tool_configs=[
                {
                    "id": "tool-2",
                    "name": "Offline SSH",
                    "tool_type": "ssh_shell",
                }
            ],
            tool_skill_mode="disabled",
            has_loadable_tool_skills=True,
        )

        self.assertEqual(
            configured_prompt,
            """

## SYSTEM TOOLS

- **Demo SQL** [PostgreSQL]: Reads demo rows.

Each tool connects to a different system. Read the description to choose the correct one.


## CONFIGURED BUT UNAVAILABLE IN THIS REQUEST

- **Offline SSH** [ssh_shell]
""",
        )

        no_selected_prompt = build_tool_system_prompt(
            [],
            no_tools_selected=True,
            tool_skill_mode="disabled",
            has_loadable_tool_skills=True,
        )
        self.assertEqual(
            no_selected_prompt,
            """

## SYSTEM TOOLS

No system tools are selected for this request. Do not claim live system access or attempt system-tool calls. Answer from indexed knowledge sources only (which may be code, docs, business records, or any other ingested content) unless tools are explicitly selected for this chat/workspace.
""",
        )

        no_configured_prompt = build_tool_system_prompt(
            [],
            tool_skill_mode="disabled",
            has_loadable_tool_skills=True,
        )
        self.assertEqual(
            no_configured_prompt,
            """

## SYSTEM TOOLS

No tools configured. Answer from indexed knowledge sources only (code, documentation, business records, or any other ingested content).
""",
        )

    def test_enabled_with_builtin_only_loadable_skills_shows_workflow_without_names(self) -> None:
        prompt = build_tool_system_prompt(
            [],
            tool_skill_mode="enabled",
            has_loadable_tool_skills=True,
        )

        self.assertIn("No optional system tools are currently loaded", prompt)
        self.assertIn("`search_tool_skills`", prompt)
        self.assertIn("`load_tool_skills`", prompt)
        self.assertIn("`unload_tool_skills`", prompt)
        self.assertNotIn("CONFIGURED BUT UNAVAILABLE", prompt)
        self.assertNotIn("Hidden SQL", prompt)

    def test_enabled_with_loaded_tools_and_hidden_loadable_skills_adds_guidance(self) -> None:
        prompt = build_tool_system_prompt(
            [
                {
                    "id": "tool-1",
                    "name": "Loaded SQL",
                    "tool_type": "postgres",
                    "description": "Loaded description.",
                }
            ],
            tool_skill_mode="enabled",
            has_loadable_tool_skills=True,
        )

        self.assertIn("Loaded SQL", prompt)
        self.assertIn("Loaded description.", prompt)
        self.assertIn("search_tool_skills", prompt)
        self.assertIn("load_tool_skills", prompt)
        self.assertIn("unload_tool_skills", prompt)

    def test_enabled_with_all_skills_loaded_omits_unneeded_workflow_guidance(self) -> None:
        prompt = build_tool_system_prompt(
            [
                {
                    "id": "tool-1",
                    "name": "Loaded SQL",
                    "tool_type": "postgres",
                    "description": "Loaded description.",
                }
            ],
            tool_skill_mode="enabled",
            has_loadable_tool_skills=False,
        )

        self.assertIn("Loaded SQL", prompt)
        self.assertNotIn("search_tool_skills", prompt)
        self.assertNotIn("load_tool_skills", prompt)
        self.assertNotIn("unload_tool_skills", prompt)
        self.assertNotIn("No optional system tools are currently loaded", prompt)

    def test_enabled_prompt_does_not_leak_hidden_tool_names_or_descriptions(self) -> None:
        prompt = build_tool_system_prompt(
            [
                {
                    "id": "tool-1",
                    "name": "Loaded SQL",
                    "tool_type": "postgres",
                    "description": "Loaded description.",
                }
            ],
            unavailable_tool_configs=[
                {
                    "id": "tool-2",
                    "name": "Selected But Broken",
                    "tool_type": "ssh_shell",
                }
            ],
            tool_skill_mode="enabled",
            has_loadable_tool_skills=True,
        )

        self.assertIn("Selected But Broken", prompt)
        self.assertNotIn("Hidden SQL", prompt)
        self.assertNotIn("Very secret description", prompt)

    def test_userspace_prompt_none_preserves_legacy_named_guidance(self) -> None:
        legacy_prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=True,
            workspace_continuity="### Workspace\n\n- Existing app.\n",
        )
        all_names_prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=True,
            workspace_continuity="### Workspace\n\n- Existing app.\n",
            available_tool_names={
                "discover_userspace_primitives",
                "run_terminal_command",
                "upsert_userspace_env_var",
            },
        )

        self.assertEqual(legacy_prompt, all_names_prompt)
        self.assertIn("discover_userspace_primitives", legacy_prompt)
        self.assertIn("run_terminal_command", legacy_prompt)
        self.assertIn("upsert_userspace_env_var", legacy_prompt)

    def test_userspace_prompt_empty_set_omits_optional_tool_names(self) -> None:
        prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=True,
            workspace_continuity="### Workspace\n\n- Existing app.\n",
            available_tool_names=set(),
        )

        self.assertNotIn("discover_userspace_primitives", prompt)
        self.assertNotIn("run_terminal_command", prompt)
        self.assertNotIn("upsert_userspace_env_var", prompt)
        self.assertIn("Preview apps may use same-origin `/__ragtime/*` primitives when useful", prompt)
        self.assertIn("Use terminal access for shell tasks such as installs, migrations, process checks", prompt)
        self.assertIn("If required keys are missing, create placeholder env vars", prompt)
        self.assertIn("### Workspace\n\n- Existing app.", prompt)
        self.assertIn("Persistent User Space dashboards must be live-wired", prompt)
        self.assertIn("Two-lane persistence contract", prompt)

    def test_userspace_prompt_selective_availability_names_only_visible_tools(self) -> None:
        prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=False,
            has_live_data_tools=False,
            workspace_continuity="### Workspace\n\n- Existing app.\n",
            available_tool_names={"run_terminal_command"},
        )

        self.assertIn("run_terminal_command", prompt)
        self.assertNotIn("discover_userspace_primitives", prompt)
        self.assertNotIn("upsert_userspace_env_var", prompt)
        self.assertIn("same-origin `/__ragtime/*` primitives when useful", prompt)
        self.assertIn("Workspace env vars are encrypted", prompt)
        self.assertNotIn("Persistent User Space dashboards must be live-wired", prompt)
        self.assertNotIn("Two-lane persistence contract", prompt)

    def test_userspace_diagnostics_none_and_all_names_match_legacy(self) -> None:
        diagnostics = [{"target_label": "preview", "count": 2, "max_ms": 1800, "avg_ms": 1200, "last_ms": 1500}]

        legacy_line = build_userspace_diagnostics_turn_reminder_line(diagnostics)
        all_names_line = build_userspace_diagnostics_turn_reminder_line(
            diagnostics,
            available_tool_names={"userspace_diagnostics"},
        )

        self.assertEqual(legacy_line, all_names_line)
        self.assertIn("userspace_diagnostics", legacy_line)

    def test_userspace_diagnostics_empty_set_suppresses_hidden_tool_name(self) -> None:
        line = build_userspace_diagnostics_turn_reminder_line(
            [{"target_label": "preview", "count": 2, "max_ms": 1800, "avg_ms": 1200, "last_ms": 1500}],
            available_tool_names=set(),
        )

        self.assertNotIn("userspace_diagnostics", line)
        self.assertIn("Use available diagnostics tooling for full execution times and errors.", line)
