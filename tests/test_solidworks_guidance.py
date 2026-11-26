"""The SolidWorks guide topics and their markdown must match the contract."""

import re
import unittest
from pathlib import Path

from ragtime.solidworks import guidance
from ragtime.solidworks.guidance import GUIDE_TOPICS, read_guide, read_topic_markdown

TOPIC_IDS = (
    "overview",
    "essentials",
    "find-model",
    "repair-assembly",
    "documents",
    "features-and-sketches",
    "assemblies",
    "paging",
    "pdm",
    "studying-models",
    "scripts",
    "previews",
)
GUIDANCE_DIR = Path(guidance.__file__).parent / "guidance"
MAX_LINES = 150
# The PDM topic carries the full operation table and error table; amendment 3 caps it at 170 lines.
# repair-assembly carries three complete C# scripts (with a using block, class header and try/finally each),
# the reference-plane helper and the verification prose.
MAX_LINES_BY_TOPIC = {"pdm": 170, "scripts": 170, "repair-assembly": 280}

# Every pdm.* operation of contract-pdm.md sections 2.1, 2.2 and 2.3.
PDM_OPERATIONS = (
    "pdm.search",
    "pdm.lookup",
    "pdm.status",
    "pdm.folder.list",
    "pdm.file.info",
    "pdm.variables.get",
    "pdm.file.get_latest",
    "pdm.folder.create",
    "pdm.file.add",
    "pdm.file.checkout",
    "pdm.file.undo_checkout",
    "pdm.file.checkin",
    "pdm.variables.set",
)

# New error codes of contract-pdm.md section 4, and the reasons they carry.
PDM_ERROR_CODES = (
    "pdm_not_linked",
    "pdm_unavailable",
    "pdm_folder_not_approved",
    "pdm_not_found",
    "pdm_conflict",
    "pdm_error",
)
PDM_REASONS = (
    "no_link",
    "linked_tool_missing",
    "linked_tool_disabled",
    "client_missing",
    "view_missing",
    "login_failed",
    "busy",
    "already_exists",
    "checked_out_by_other",
    "not_checked_out_here",
    "not_checked_out_by_tool",
    "never_checked_in",
    "local_file_exists",
)

# Skipped reasons of pdm.file.get_latest (amendment 3, section C).
PDM_SKIP_REASONS = (
    "checked_out_here",
    "local_untracked",
    "local_modified",
    "deadline",
    "not_in_vault",
)

# A pdm operation name as a token: not part of a longer identifier such as context.pdm.
PDM_OPERATION_TOKEN = re.compile(r"(?<![\w.])pdm\.[a-z_]+(?:\.[a-z_]+)*")

# Operations the studying routine reads, all from the existing guidance.
STUDY_OPERATIONS = (
    "model.summary",
    "configurations.list",
    "properties.get",
    "pdm.variables.get",
    "features.list",
    "feature.get",
    "sketch.get",
    "dimensions.list",
    "equations.list",
    "components.list",
    "mates.list",
    "document.close",
)

# Every operation name in contract section 5 (plus the Ragtime-handled ones).
CONTRACT_OPERATIONS = (
    "guide",
    "describe",
    "status",
    "documents.list",
    "document.open",
    "document.close",
    "model.summary",
    "configurations.list",
    "features.list",
    "feature.get",
    "sketch.get",
    "dimensions.list",
    "mates.list",
    "components.list",
    "properties.get",
    "mass_properties.get",
    "equations.list",
)

# Read-only operations that return a preview image or an exported file (previews topic).
PREVIEW_OPERATIONS = (
    "model.preview",
    "model.export",
)

# Error codes an agent can act on (overview table).
ACTIONABLE_ERROR_CODES = (
    "host_not_running",
    "host_updating",
    "host_stopping",
    "no_interactive_session",
    "version_not_installed",
    "version_mismatch",
    "solidworks_not_running",
    "instance_limit",
    "instance_timeout",
    "operation_timeout",
    "document_not_found",
    "document_not_open",
    "not_supported_for_document_type",
    "entity_not_found",
    "stale_snapshot",
    "result_too_large",
    "invalid_arguments",
    "scripts_not_allowed",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATION_CATALOG = REPO_ROOT / "dotnet/src/Ragtime.SolidWorks.Core/Operations/OperationCatalog.cs"
SCRIPT_CONTEXT = REPO_ROOT / "dotnet/src/Ragtime.SolidWorks.Scripting/ScriptContext.cs"
SERVICE_PY = REPO_ROOT / "ragtime/solidworks/service.py"
# Operations that Ragtime answers itself, plus document.copy (agent 0.6.0, not yet in the catalog).
RAGTIME_OPERATIONS = ("guide", "pdm.search", "pdm.lookup", "document.copy")
# First segments of operation names. A backticked dotted name starting with one of these must be an operation.
OPERATION_NAMESPACES = (
    "pdm",
    "document",
    "documents",
    "model",
    "feature",
    "features",
    "sketch",
    "dimensions",
    "mates",
    "components",
    "configurations",
    "properties",
    "mass_properties",
    "equations",
    "script",
    "host",
)
BACKTICKED_OPERATION = re.compile(r"`((?:" + "|".join(OPERATION_NAMESPACES) + r")\.[a-z_]+(?:\.[a-z_]+)*)`")
QUOTED_DOTTED_NAME = re.compile(r'"((?:[a-z_]+\.)+[a-z_]+)"')
# Built-in skill names that essentials.md must point to (spec section 5.1).
SKILL_NAMES = (
    "solidworks-find-model",
    "solidworks-repair-assembly",
    "solidworks-scripts",
    "solidworks-pdm",
)
# Mate error codes of swFeatureError_e that assemblies.md must name (values 38 to 48).
MATE_ERROR_NAMES = {
    38: "swFeatureErrorMateInvalidEdge",
    39: "swFeatureErrorMateInvalidFace",
    40: "swFeatureErrorMateFailedCreatingSurface",
    41: "swFeatureErrorMateInvalidEntity",
    42: "swFeatureErrorMateUnknownTangent",
    43: "swFeatureErrorMateDanglingGeometry",
    44: "swFeatureErrorMateEntityNotLinear",
    45: "swFeatureErrorMateEntityFailed",
    46: "swFeatureErrorMateOverdefined",
    47: "swFeatureErrorMateIlldefined",
    48: "swFeatureErrorMateBroken",
}
# Member names of ScriptContext that snippets may use (ScriptContext.cs declares each one).
SCRIPT_CONTEXT_MEMBERS = ("App", "Arguments", "WorkFolder", "Log", "NewWorkPath", "DefaultTemplate")
LIVE_CHECK_MARKER = "<!-- live-check: verified on SolidWorks 2024 SP5 -->"
# SolidWorks feature-walk calls return object; each call must be cast to Feature (CS0266 otherwise).
FEATURE_WALK_CALL = re.compile(r"\.(?:FirstFeature|GetFirstSubFeature|GetNextSubFeature|GetNextFeature)\(\)")
CAST_FEATURE_WALK_CALL = re.compile(r"\(Feature\)[\w.]+\.(?:FirstFeature|GetFirstSubFeature|GetNextSubFeature|GetNextFeature)\(\)")


def _topic_text(topic_id: str) -> str:
    return (GUIDANCE_DIR / f"{topic_id}.md").read_text(encoding="utf-8")


def _known_operations() -> set[str]:
    catalog = OPERATION_CATALOG.read_text(encoding="utf-8")
    return set(QUOTED_DOTTED_NAME.findall(catalog)) | set(RAGTIME_OPERATIONS)


def _csharp_snippets(topic_id: str) -> list[str]:
    return re.findall(r"```csharp\n(.*?)```", _topic_text(topic_id), flags=re.DOTALL)


class GuideTopicListTests(unittest.TestCase):
    def test_scripts_guidance_starts_with_csharp_limits_and_has_entry_point(self) -> None:
        text = _topic_text("scripts")
        self.assertLess(text.index('`$"…"'), text.index("public static class RagtimeScript"))
        self.assertIn("Run(ScriptContext ctx)", text)

    def test_topic_ids_are_in_contract_order(self) -> None:
        self.assertEqual(tuple(topic["id"] for topic in GUIDE_TOPICS), TOPIC_IDS)

    def test_every_topic_has_title_and_when_to_read(self) -> None:
        for topic in GUIDE_TOPICS:
            self.assertEqual(set(topic), {"id", "title", "when_to_read"})
            self.assertTrue(topic["title"].strip())
            self.assertTrue(topic["when_to_read"].strip())

    def test_read_guide_returns_topic_list_as_list(self) -> None:
        result = read_guide("paging")
        self.assertEqual([topic["id"] for topic in result["topics"]], list(TOPIC_IDS))
        self.assertIsInstance(result["topics"], list)


class ReadGuideTests(unittest.TestCase):
    def test_default_topic_is_overview(self) -> None:
        self.assertEqual(read_guide(None)["topic"], "overview")
        self.assertEqual(read_guide("")["topic"], "overview")
        self.assertEqual(read_guide("   ")["topic"], "overview")

    def test_topic_match_ignores_case_and_surrounding_whitespace(self) -> None:
        result = read_guide("  Features-And-Sketches \n")
        self.assertEqual(result["topic"], "features-and-sketches")
        self.assertEqual(result["markdown"], _topic_text("features-and-sketches"))

    def test_markdown_is_the_topic_file(self) -> None:
        for topic_id in TOPIC_IDS:
            with self.subTest(topic=topic_id):
                self.assertEqual(read_guide(topic_id)["markdown"], _topic_text(topic_id))

    def test_unknown_topic_raises_and_lists_valid_ids(self) -> None:
        with self.assertRaises(ValueError) as raised:
            read_guide("nonexistent")
        message = str(raised.exception)
        for topic_id in TOPIC_IDS:
            self.assertIn(topic_id, message)

    def test_path_like_topic_is_rejected(self) -> None:
        for candidate in ("../protocol", "..\\protocol", "guidance/overview", "overview.md"):
            with self.subTest(topic=candidate):
                with self.assertRaises(ValueError):
                    read_guide(candidate)


class TopicFileTests(unittest.TestCase):
    def test_every_topic_file_exists_is_non_empty_and_short(self) -> None:
        for topic_id in TOPIC_IDS:
            with self.subTest(topic=topic_id):
                path = GUIDANCE_DIR / f"{topic_id}.md"
                self.assertTrue(path.is_file())
                text = path.read_text(encoding="utf-8")
                self.assertTrue(text.strip())
                self.assertLessEqual(len(text.splitlines()), MAX_LINES_BY_TOPIC.get(topic_id, MAX_LINES))


class OverviewContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.overview = _topic_text("overview")

    def test_overview_states_required_behaviour(self) -> None:
        for phrase in ("read-only", "licence", "desktop session", "UNC", "metres"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.overview)

    def test_overview_explains_error_codes_it_is_expected_to_cover(self) -> None:
        for code in ACTIONABLE_ERROR_CODES:
            with self.subTest(code=code):
                self.assertIn(f"`{code}`", self.overview)

    def test_overview_names_the_envelope_fields(self) -> None:
        for field in ("`ok`", "`result`", "`error.code`", "`host`"):
            with self.subTest(field=field):
                self.assertIn(field, self.overview)

    def test_overview_defines_the_solidworks_agent(self) -> None:
        self.assertIn("The **SolidWorks agent** is the Ragtime program on the Windows machine", self.overview)


class ContractCoverageTests(unittest.TestCase):
    def test_every_contract_operation_is_documented_in_some_topic(self) -> None:
        all_text = "\n".join(_topic_text(topic_id) for topic_id in TOPIC_IDS)
        for operation in CONTRACT_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(f"`{operation}`", all_text)


class PdmTopicTests(unittest.TestCase):
    def test_permission_denied_is_not_retried(self):
        text = (GUIDANCE_DIR / "pdm.md").read_text(encoding="utf-8")
        self.assertIn("E_EDM_PERMISSION_DENIED", text)
        self.assertIn("PDM rights problem; report it and do not retry", text)

    def setUp(self) -> None:
        self.pdm = _topic_text("pdm")

    def test_pdm_topic_lists_every_pdm_operation(self) -> None:
        for operation in PDM_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(f"`{operation}`", self.pdm)

    def test_pdm_topic_documents_every_error_code_and_reason(self) -> None:
        for code in PDM_ERROR_CODES + PDM_REASONS:
            with self.subTest(code=code):
                self.assertIn(f"`{code}`", self.pdm)

    def test_pdm_topic_explains_unknown_operation_for_pdm(self) -> None:
        self.assertIn("`unknown_operation`", self.pdm)
        self.assertIn(
            "Open the SolidWorks agent panel in the tool's settings and choose Update agent to use PDM operations.",
            self.pdm,
        )

    def test_unknown_operation_row_points_to_the_agent_panel(self) -> None:
        row = next(line for line in self.pdm.splitlines() if line.startswith("| `unknown_operation` on `pdm.*`"))
        self.assertIn("SolidWorks agent panel", row)
        self.assertIn("`host_updating`", row)

    def test_pdm_topic_states_the_hard_limits(self) -> None:
        for phrase in (
            "delete",
            "rename",
            "move",
            "workflow state",
            "Parts before assemblies",
            "approved folders",
            "Ragtime adds a prefix",
            "Text read from the vault",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.pdm)

    def test_pdm_topic_states_claim_ascii_comment_and_shared_file_rules(self) -> None:
        for phrase in (
            "checked_out_by_tool",
            "leave it alone",
            "plain ASCII characters",
            "control characters",
            "one file",
            "`latest_version` 0",
            "not proof of who made the change",
            "`snapshot`",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.pdm)

    def test_pdm_topic_states_discovery_and_read_path_rules(self) -> None:
        for phrase in (
            "whenever the installed SolidWorks agent",
            "answers `pdm_not_linked` without a link",
            "appear only when a link resolves",
            "printable Unicode",
            "no wildcards",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.pdm)
        self.assertNotIn("Read operations accept any characters", self.pdm)

    def test_pdm_topic_documents_every_get_latest_skip_reason(self) -> None:
        for reason in PDM_SKIP_REASONS:
            with self.subTest(reason=reason):
                self.assertIn(f"`{reason}`", self.pdm)
        self.assertIn("never overwrites local work", self.pdm)
        self.assertIn("fails the whole call with `invalid_arguments`", self.pdm)
        self.assertNotIn("`link`", self.pdm)

    def test_not_in_vault_row_covers_virtual_and_embedded_components(self) -> None:
        row = next(line for line in self.pdm.splitlines() if line.startswith("| `not_in_vault`"))
        for phrase in ("virtual", "embedded"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, row)

    def test_pdm_topic_states_claim_expiry_and_source_rules(self) -> None:
        for phrase in (
            "24 hours",
            "must finish the check-in",
            "cannot be told apart from the tool's own check-out",
            "work_folder",
            "Save the file there first",
            "other programs running as the same Windows user",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.pdm)

    def test_pdm_topic_states_snapshot_lag_and_live_operations(self) -> None:
        self.assertIn("It can lag the vault", self.pdm)
        self.assertIn("asked from the PDM client", self.pdm)


class StudyingModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = _topic_text("studying-models")

    def test_routine_uses_the_pdm_search_and_get_latest_operations(self) -> None:
        for operation in ("pdm.search", "pdm.file.get_latest"):
            with self.subTest(operation=operation):
                self.assertIn(f"`{operation}`", self.study)

    def test_routine_reads_with_existing_operations(self) -> None:
        for operation in STUDY_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(f"`{operation}`", self.study)

    def test_routine_states_its_limits(self) -> None:
        for phrase in ("licence", "`.slddrw`", "does not hold feature data"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.study)


class PdmOperationNameTests(unittest.TestCase):
    def test_every_pdm_operation_token_in_guidance_is_a_contract_name(self) -> None:
        for topic_id in TOPIC_IDS:
            text = _topic_text(topic_id)
            for token in PDM_OPERATION_TOKEN.findall(text):
                with self.subTest(topic=topic_id, token=token):
                    self.assertIn(token, PDM_OPERATIONS)


class ScriptAwarenessTests(unittest.TestCase):
    def test_no_topic_denies_that_scripts_can_create_models(self) -> None:
        for topic_id in TOPIC_IDS:
            text = _topic_text(topic_id)
            for phrase in ("does not create models", "documents are never written"):
                with self.subTest(topic=topic_id, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_overview_points_to_script_run_and_scripts_enabled(self) -> None:
        overview = _topic_text("overview")
        for token in ("`script.run`", "`scripts_enabled`"):
            with self.subTest(token=token):
                self.assertIn(token, overview)


class SolidWorksAgentWordingTests(unittest.TestCase):
    """LLM-facing guidance says "SolidWorks agent", never a bare "agent" (copy.md 1.3)."""

    # Bold or quoted UI labels that keep the word "agent" verbatim (copy.md 1.3 item 2).
    ALLOWED_AGENT_PHRASES = ("SolidWorks agent", "Keep the agent up to date", "Update agent")

    def test_no_bare_agent_outside_the_allowed_phrases(self) -> None:
        for path in sorted(GUIDANCE_DIR.glob("*.md")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                remainder = line
                for phrase in self.ALLOWED_AGENT_PHRASES:
                    remainder = remainder.replace(phrase, "")
                with self.subTest(file=path.name, line=number):
                    self.assertIsNone(re.search(r"\bagent\b", remainder), line)


class OverviewPointerTests(unittest.TestCase):
    def test_overview_points_to_pdm_studying_models_and_previews(self) -> None:
        overview = _topic_text("overview")
        self.assertIn("`pdm`", overview)
        self.assertIn("`studying-models`", overview)
        self.assertIn("`previews`", overview)


class PreviewsTopicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previews = _topic_text("previews")

    def test_topic_names_both_operations(self) -> None:
        for operation in PREVIEW_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(f"`{operation}`", self.previews)

    def test_topic_lists_every_export_format_and_the_drawing_rule(self) -> None:
        for format_name in ("`step`", "`stl`", "`3mf`", "`glb`", "`native`"):
            with self.subTest(format=format_name):
                self.assertIn(format_name, self.previews)
        self.assertIn("Drawings support only `native` and `model.preview`", self.previews)
        self.assertIn("Assemblies support `step`, `3mf`, `glb` and `native`, not `stl`", self.previews)

    def test_topic_names_the_document_argument(self) -> None:
        self.assertIn("Arguments: `document`", self.previews)
        self.assertNotIn("`path`", self.previews)

    def test_topic_states_retention_size_and_sign_in(self) -> None:
        for phrase in ("7 days", "100 MB", "sign-in", "`page_url`"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.previews)

    def test_topic_states_open_document_rule_for_both_modes(self) -> None:
        self.assertIn("attach mode", self.previews)
        self.assertIn("`document.open`", self.previews)

    def test_topic_forbids_pasting_paths(self) -> None:
        self.assertIn("Do not paste file paths", self.previews)
        self.assertIn("Do not call `model.preview` or `model.export` in a loop", self.previews)


class GuideTopicRegistryTests(unittest.TestCase):
    def test_every_topic_in_guide_topics_has_a_file(self) -> None:
        for topic in GUIDE_TOPICS:
            with self.subTest(topic=topic["id"]):
                self.assertTrue((GUIDANCE_DIR / f"{topic['id']}.md").is_file())

    def test_new_topics_are_listed_and_readable(self) -> None:
        listed = {entry["id"] for entry in read_guide(None)["topics"]}
        for topic_id in ("essentials", "find-model", "repair-assembly"):
            with self.subTest(topic=topic_id):
                self.assertIn(topic_id, listed)
                result = read_guide(topic_id)
                self.assertEqual(result["topic"], topic_id)
                self.assertEqual(result["markdown"], _topic_text(topic_id))

    def test_read_topic_markdown_returns_the_topic_file(self) -> None:
        for topic_id in TOPIC_IDS:
            with self.subTest(topic=topic_id):
                self.assertEqual(read_topic_markdown(topic_id), _topic_text(topic_id))

    def test_read_topic_markdown_rejects_unknown_ids_and_lists_valid_ones(self) -> None:
        for candidate in ("nonexistent", "", "../protocol", "essentials.md"):
            with self.subTest(topic=candidate):
                with self.assertRaises(ValueError) as raised:
                    read_topic_markdown(candidate)
                self.assertIn("essentials", str(raised.exception))


class EssentialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("essentials")

    def test_essentials_fits_the_inline_limit(self) -> None:
        self.assertLessEqual(len(self.text), 1400)

    def test_essentials_points_to_each_task_skill(self) -> None:
        for skill in SKILL_NAMES:
            with self.subTest(skill=skill):
                self.assertIn(f"`{skill}`", self.text)

    def test_essentials_covers_refusals_units_and_describe(self) -> None:
        for phrase in ("`error.details.still_available`", "`error.details.admin_action`", "`describe`", "metres"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_essentials_says_to_call_status_first_and_what_a_missing_skill_means(self) -> None:
        self.assertIn("Call `status` first", self.text)
        self.assertIn("If a skill named here is not listed, that capability is off.", self.text)
        self.assertIn("`ragtime`", self.text)


class FindModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("find-model")

    def test_routine_names_the_identifier_forms_and_the_erp_limit(self) -> None:
        for phrase in ("SKU", "ERP", "`22-100-xxxxx`", "`CU-<catalog>-<yy><letter>-<rev>`", "XLSM"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_routine_uses_lookup_search_browsing_and_vault_views(self) -> None:
        for phrase in ("`pdm.lookup`", "`pdm.search`", "`pdm.folder.list`", "`pdm.status`", "`matched_on`"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_routine_stops_with_a_precise_ask_for_another_vault(self) -> None:
        self.assertIn("name that vault and stop", self.text)


class RepairAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("repair-assembly")
        self.snippets = _csharp_snippets("repair-assembly")

    def test_every_snippet_is_marked_as_verified_on_render_pc_sw_version(self) -> None:
        self.assertEqual(len(self.snippets), 3)
        marked = re.findall(re.escape(LIVE_CHECK_MARKER) + r"\n```csharp\n", self.text)
        self.assertEqual(len(marked), len(self.snippets))

    def test_every_feature_walk_call_is_cast_to_feature(self) -> None:
        walks = 0
        for snippet in self.snippets:
            with self.subTest(snippet=snippet[:60]):
                self.assertEqual(
                    len(CAST_FEATURE_WALK_CALL.findall(snippet)),
                    len(FEATURE_WALK_CALL.findall(snippet)),
                )
            walks += len(FEATURE_WALK_CALL.findall(snippet))
        self.assertGreater(walks, 0)

    def test_snippets_use_the_silent_read_write_open_and_the_mate_apis(self) -> None:
        joined = "\n".join(self.snippets)
        for token in (
            "OpenDoc6(",
            "swOpenDocOptions_e.swOpenDocOptions_Silent",
            "GetErrorCode2(out",
            '"MateGroup"',
            "EditSuppress2(",
            "SelectByID2(",
            "AddMate5(",
            "ForceRebuild3(",
            "Save3(",
        ):
            with self.subTest(token=token):
                self.assertIn(token, joined)

    def test_snippets_use_only_declared_script_context_members(self) -> None:
        declared = SCRIPT_CONTEXT.read_text(encoding="utf-8")
        for member in SCRIPT_CONTEXT_MEMBERS:
            with self.subTest(member=member):
                self.assertRegex(declared, rf"public [\w?<>\[\]]+ {member}\b")
        for snippet in self.snippets:
            used = set(re.findall(r"\bctx\.([A-Za-z]+)", snippet))
            self.assertLessEqual(used, set(SCRIPT_CONTEXT_MEMBERS))

    def test_every_snippet_works_only_on_a_copy_under_the_copies_folder(self) -> None:
        for snippet in self.snippets:
            with self.subTest(snippet=snippet[:60]):
                self.assertRegex(snippet, r"Path\.GetFullPath\(Path\.Combine\(ctx\.WorkFolder, \"copies\"\)\) \+ \"\\\\\"")
                self.assertIn("StartsWith(copies, StringComparison.OrdinalIgnoreCase)", snippet)
                self.assertIn("using System.IO;", snippet)

    def test_every_snippet_opens_activates_then_closes_in_finally(self) -> None:
        for snippet in self.snippets:
            with self.subTest(snippet=snippet[:60]):
                try_block = snippet.index("try\n")
                opened = snippet.index("OpenDoc6(")
                activated = snippet.index("ActivateDoc3(")
                self.assertGreater(opened, try_block)
                self.assertLess(opened, activated)
                self.assertGreater(snippet.index("finally\n"), activated)
                self.assertIn("ctx.App.CloseDoc(model.GetTitle())", snippet)

    def test_only_the_list_snippet_opens_read_only(self) -> None:
        read_only = [snippet for snippet in self.snippets if "swOpenDocOptions_ReadOnly" in snippet]
        self.assertEqual(len(read_only), 1)
        self.assertIn("GetErrorCode2(out", read_only[0])
        self.assertIn(
            "(int)swOpenDocOptions_e.swOpenDocOptions_Silent | (int)swOpenDocOptions_e.swOpenDocOptions_ReadOnly",
            read_only[0],
        )

    def test_recreate_snippet_finds_reference_planes_by_position_not_name(self) -> None:
        recreate = next(snippet for snippet in self.snippets if "AddMate5(" in snippet)
        self.assertIn('"RefPlane"', recreate)
        self.assertIn("FindPlane(assembly.GetComponentByName(", recreate)
        self.assertIn('int planeIndex = (int)ctx.Arguments["plane_index"];', recreate)
        self.assertNotIn('"FACE"', recreate)
        self.assertNotIn('"PLANE"', recreate)
        self.assertIn("firstPlane.Select2(false, 1)", recreate)
        self.assertIn("secondPlane.Select2(true, 1)", recreate)
        self.assertIn("Mate2 mate = (Mate2)assembly.AddMate5((int)swMateType_e.swMateCOINCIDENT", recreate)
        self.assertRegex(recreate, r"0, 0, 0, 1, 1, 0, 0, 0, false, false, 0, out mateError")
        self.assertIn("swAddMateError_NoError", recreate)

    def test_recreate_snippet_arguments_match_the_prose(self) -> None:
        recreate = next(snippet for snippet in self.snippets if "AddMate5(" in snippet)
        for key in ("replace", "first_component", "second_component", "plane_index"):
            with self.subTest(key=key):
                self.assertIn(f'ctx.Arguments["{key}"]', recreate)
                self.assertIn(f"`{key}`", self.text)

    def test_repair_prose_states_component_names_timing_and_no_face_names(self) -> None:
        self.assertIn("`components.list`", self.text)
        self.assertIn("16.5 s", self.text)
        self.assertIn("2 to 3 s", self.text)
        self.assertIn("1` (`swAddMateError_NoError`) is success", self.text)
        self.assertIn("does not do either", self.text)

    def test_routine_states_call_shape_timeout_and_no_vault_replacement(self) -> None:
        self.assertRegex(self.text, r"`timeout` 60 to 90")
        self.assertIn("`source`", self.text)
        self.assertIn("`purpose`", self.text)
        self.assertIn("No operation replaces an existing vault file with a copy", self.text)
        self.assertIn("`document.close`", self.text)

    def test_routine_delivers_the_copy_when_the_vault_refuses(self) -> None:
        self.assertIn("`model.export`", self.text)
        self.assertIn("native", self.text)
        self.assertIn("zip", self.text)
        self.assertIn("PDM vault: Allow changes", self.text)


class CSharpSnippetTests(unittest.TestCase):
    """Every fenced csharp block in every topic must be C# 5 (scripts.md, scripts compile as C# 5)."""

    # Modern-syntax markers that the script compiler rejects with a C# 5 hint.
    FORBIDDEN_CSHARP_5 = ('$"', "?.", "=>", "nameof", "out var", "{ get; } =")

    def test_every_csharp_block_avoids_modern_syntax(self) -> None:
        checked = 0
        for topic_id in TOPIC_IDS:
            for snippet in _csharp_snippets(topic_id):
                checked += 1
                for marker in self.FORBIDDEN_CSHARP_5:
                    with self.subTest(topic=topic_id, marker=marker):
                        self.assertNotIn(marker, snippet)
        self.assertGreaterEqual(checked, 4)

    def test_every_csharp_block_declares_the_script_entry_point(self) -> None:
        for topic_id in TOPIC_IDS:
            for snippet in _csharp_snippets(topic_id):
                with self.subTest(topic=topic_id):
                    self.assertIn("public static class RagtimeScript", snippet)
                    self.assertIn("public static object Run(ScriptContext ctx)", snippet)


class AssemblyFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("assemblies")

    def test_new_mate_and_component_fields_are_documented(self) -> None:
        for field in (
            "only_problems",
            "`error`",
            "`whats_wrong`",
            "`component_path`",
            "`resolved`",
            "`file_exists`",
            "`suppression`",
            "`missing_file`",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.text)

    def test_mate_error_table_lists_every_mate_code_with_its_name(self) -> None:
        for code, name in MATE_ERROR_NAMES.items():
            with self.subTest(code=code):
                self.assertRegex(self.text, rf"\| {code} \| `{name}` \|")

    def test_suppressed_components_are_not_assumed_deliberate(self) -> None:
        self.assertIn("another vault view", self.text)


class DocumentFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("documents")

    def test_open_warnings_summary_count_and_copy_are_documented(self) -> None:
        for phrase in ("`load_warnings`", "`missing_references`", "`missing_component_count`", "`document.copy`", "copies"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)


class PdmVaultOffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pdm = _topic_text("pdm")

    def test_the_copy_and_script_sentence_is_gone(self) -> None:
        self.assertNotIn("The user edits the model in SolidWorks", self.pdm)

    def test_existing_file_routine_has_no_checkout_edit_checkin_of_a_copy(self) -> None:
        section = self.pdm.split("### Existing file", 1)[1].split("## ", 1)[0]
        self.assertIn("No operation replaces an existing vault file with a copy", section)
        self.assertNotIn("Make the change", section)
        self.assertIn("`pdm.variables.set`", section)
        self.assertIn("`pdm.file.add`", section)

    def test_vault_off_section_points_to_the_copy_routine_and_the_setting(self) -> None:
        self.assertIn("## When vault changes are off", self.pdm)
        self.assertIn("`repair-assembly`", self.pdm)
        self.assertIn("PDM vault: Allow changes", self.pdm)
        self.assertIn("sandbox folder", self.pdm)


class ScriptsAndOverviewTests(unittest.TestCase):
    def test_scripts_allow_only_a_work_folder_copy_read_write(self) -> None:
        text = _topic_text("scripts")
        for phrase in ("work_folder", "read-write", "never save", "vault view"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_overview_explains_refusal_fields_and_points_to_skills(self) -> None:
        text = _topic_text("overview")
        for phrase in ("`error.details`", "`admin_action`", "`still_available`", "`solidworks-find-model`"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


class RefusalCodeTests(unittest.TestCase):
    def test_scripts_guidance_names_the_code_the_service_returns(self) -> None:
        service = SERVICE_PY.read_text(encoding="utf-8")
        scripts = _topic_text("scripts")
        self.assertIn('"scripts_not_allowed"', service)
        self.assertIn("`scripts_not_allowed`", scripts)
        self.assertNotIn("refused with `write_not_allowed`", scripts)

    def test_overview_error_table_has_a_scripts_not_allowed_row(self) -> None:
        rows = [line for line in _topic_text("overview").splitlines() if line.startswith("| `scripts_not_allowed` |")]
        self.assertEqual(len(rows), 1)
        self.assertIn("`error.details.admin_action`", rows[0])


class FindModelRoutineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _topic_text("find-model")

    def test_variant_fallback_and_paging_are_described(self) -> None:
        self.assertIn("`IDB0061-19` becomes `IDB0061`", self.text)
        self.assertIn("`CU-<base>`", self.text)
        self.assertIn("`offset`", self.text)
        self.assertIn("`next_offset`", self.text)

    def test_other_variant_is_a_candidate_needing_confirmation(self) -> None:
        self.assertIn("`CU-IDB0061-26A-A` for `IDB0061-19`", self.text)
        self.assertIn("candidate, not a confirmed match", self.text)
        self.assertIn("ask the user to confirm", self.text)

    def test_references_come_from_document_open_before_component_walks(self) -> None:
        self.assertLess(self.text.index("`load_warnings`"), self.text.index("`components.list`"))

    def test_vault_views_are_read_before_folder_browsing(self) -> None:
        self.assertLess(self.text.index("`pdm.status`"), self.text.index("`pdm.folder.list`"))


class OperationNameCatalogTests(unittest.TestCase):
    def test_every_backticked_operation_in_guidance_is_a_known_operation(self) -> None:
        known = _known_operations()
        self.assertIn("pdm.file.add", known)
        self.assertIn("mass_properties.get", known)
        for topic_id in TOPIC_IDS:
            for operation in BACKTICKED_OPERATION.findall(_topic_text(topic_id)):
                with self.subTest(topic=topic_id, operation=operation):
                    self.assertIn(operation, known)


if __name__ == "__main__":
    unittest.main()
