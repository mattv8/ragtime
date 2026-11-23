"""The SolidWorks guide topics and their markdown must match the contract."""

import re
import unittest
from pathlib import Path

from ragtime.solidworks import guidance
from ragtime.solidworks.guidance import GUIDE_TOPICS, read_guide

TOPIC_IDS = (
    "overview",
    "documents",
    "features-and-sketches",
    "assemblies",
    "paging",
    "pdm",
    "studying-models",
    "scripts",
)
GUIDANCE_DIR = Path(guidance.__file__).parent / "guidance"
MAX_LINES = 150
# The PDM topic carries the full operation table and error table; amendment 3 caps it at 170 lines.
MAX_LINES_BY_TOPIC = {"pdm": 170, "scripts": 170}

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

# Error codes an agent can act on (overview table).
ACTIONABLE_ERROR_CODES = (
    "host_not_running",
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
)


def _topic_text(topic_id: str) -> str:
    return (GUIDANCE_DIR / f"{topic_id}.md").read_text(encoding="utf-8")


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
        self.assertIn("Update the SolidWorks host to use PDM operations.", self.pdm)

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
            "whenever the installed host supports them",
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


class OverviewPointerTests(unittest.TestCase):
    def test_overview_points_to_pdm_and_studying_models(self) -> None:
        overview = _topic_text("overview")
        self.assertIn("`pdm`", overview)
        self.assertIn("`studying-models`", overview)


if __name__ == "__main__":
    unittest.main()
