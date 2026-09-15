import json
import unittest
from typing import Any, cast
from unittest import mock

from ragtime.indexer.models import SolidworksPdmConnectionConfig
from ragtime.indexer.pdm_service import PdmIndexerService, lookup_pdm_documents
from ragtime.indexer.pdm_source import (
    PdmConfigurationRef,
    PdmDocumentRecord,
    PdmRawValue,
    PdmVariableDef,
)


class FakeDb:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.query_calls: list[tuple[Any, ...]] = []
        self.execute_calls: list[tuple[Any, ...]] = []

    async def query_raw(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.query_calls.append((query, *params))
        return self.rows

    async def execute_raw(self, query: str, *params: Any) -> int:
        self.execute_calls.append((query, *params))
        return 1


class FakeJob:
    index_name = "pdm_test"
    total_chunks = 0
    processed_chunks = 0


class PdmIndexLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _config(self) -> SolidworksPdmConnectionConfig:
        return SolidworksPdmConnectionConfig(host="host", database="vault", user="user")

    def _record(self) -> PdmDocumentRecord:
        return PdmDocumentRecord(
            document_id=7597,
            filename="LAB0049.SLDPRT",
            latest_revision=8,
            folder_paths=["/vault/lights"],
            configurations=[PdmConfigurationRef(2, "@"), PdmConfigurationRef(2323, "LAB0049-15")],
            resolved_values={
                (2, 10): PdmRawValue(10, "Part Number", 2, 2, 8, "LAB0049", None, None, None),
                (2323, 11): PdmRawValue(11, "Description", 2323, 2, 5, "Single Pendant (BP)", None, None, None),
                (2323, 10): PdmRawValue(10, "Part Number", 2323, 2, 8, "LAB0049-15", None, None, None),
                (2323, 12): PdmRawValue(12, "Material", 2323, 2, 8, "", None, None, None),
            },
            bom_children=[],
            membership_fallback=False,
            has_beyond_latest_values=False,
        )

    async def test_lookup_escapes_patterns_clamps_and_binds_document_id(self) -> None:
        db = FakeDb()
        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=db):
            await lookup_pdm_documents("pdm_test", filename="a%_\\b", max_results=500)
            await lookup_pdm_documents("pdm_test", document_id=7597, max_results=0)
        self.assertIn("LIMIT 50", db.query_calls[0][0])
        self.assertEqual(db.query_calls[0][2], "%a\\%\\_\\\\b%")
        self.assertIn("LIMIT 1", db.query_calls[1][0])
        self.assertEqual(db.query_calls[1][2], 7597)

    async def test_lookup_requires_selector_without_database_call(self) -> None:
        with mock.patch("ragtime.indexer.pdm_service.get_db") as get_db:
            result = await lookup_pdm_documents("pdm_test")
        self.assertEqual(result, "Error: Provide document_id, filename, or part_number.")
        get_db.assert_not_called()

    async def test_state_mapping_lookup_filter_and_upsert(self) -> None:
        service = PdmIndexerService()
        record = self._record()
        config = self._config()
        defs = [
            PdmVariableDef(10, "part number", 0, False, False, False),
            PdmVariableDef(11, "DESCRIPTION", 0, False, False, False),
        ]
        roles = service._resolve_role_variables(defs, config)
        self.assertEqual(roles, {"part_number": 10, "description": 11})
        state = service._state_from_record(record, roles, {10: "Part Number", 11: "Description", 12: "Material"})
        self.assertIn("Part Number", state.document_values)
        self.assertEqual(state.configurations[0].name, "LAB0049-15")
        self.assertTrue(state.configurations[0].values["Material"].is_blank)

        db = FakeDb([{"state_json": state.model_dump_json(), "extracted_at": "2026-09-15"}])
        with mock.patch("ragtime.indexer.pdm_service.get_db", return_value=db):
            output = await lookup_pdm_documents("pdm_test", document_id=7597, configuration="lab0049-15")
            with mock.patch("ragtime.indexer.pdm_service.embed_documents_subbatched", return_value=[[0.1]]):
                await service._process_batch(cast(Any, FakeJob()), [record], object())
        self.assertIn("Configuration: LAB0049-15 (ID: 2323)", output)
        self.assertIn("Description: Single Pendant (BP) (v5)", output)
        state_call = next(call for call in db.execute_calls if "pdm_document_state" in call[0])
        self.assertIn("LAB0049-15", state_call[6])

    def test_resolve_role_variables_with_explicit_override(self) -> None:
        """Test that explicit part_number_variable override takes precedence over default name."""
        service = PdmIndexerService()
        defs = [
            PdmVariableDef(10, "Part Number", 0, False, False, False),
            PdmVariableDef(99, "PartNo", 0, False, False, False),
            PdmVariableDef(11, "Description", 0, False, False, False),
        ]

        # Config with explicit override: part_number_variable="PartNo"
        # Use a mock object that supports getattr with the override attribute
        config = mock.Mock(spec=SolidworksPdmConnectionConfig)
        config.variable_names = None
        config.part_number_variable = "PartNo"
        config.description_variable = None

        roles = service._resolve_role_variables(defs, config)
        # With override, PartNo (id=99) should win over Part Number (id=10)
        self.assertEqual(roles["part_number"], 99)
        self.assertEqual(roles["description"], 11)

    def test_resolve_role_variables_deterministic_with_duplicate_names(self) -> None:
        """Test that when multiple defs have same name, lowest variable_id is selected."""
        service = PdmIndexerService()
        # Create defs with unsorted IDs but same names
        defs = [
            PdmVariableDef(50, "Part Number", 0, False, False, False),
            PdmVariableDef(10, "Part Number", 0, False, False, False),  # Lower ID
            PdmVariableDef(11, "Description", 0, False, False, False),
        ]

        config = self._config()
        roles = service._resolve_role_variables(defs, config)
        # Should select the lowest ID (10) due to sorting
        self.assertEqual(roles["part_number"], 10)
        self.assertEqual(roles["description"], 11)


if __name__ == "__main__":
    unittest.main()
