import importlib
import random
import unittest
from datetime import datetime

from ragtime.indexer.pdm_source import build_pdm_extension_filter, resolve_latest_values


def _row(**overrides):
    row = {
        "ProjectID": 2,
        "ConfigurationID": 10,
        "VariableID": 20,
        "RevisionNo": 1,
        "ValueText": "value",
        "ValueInt": None,
        "ValueFloat": None,
        "ValueDate": None,
        "VariableName": "Description",
    }
    row.update(overrides)
    return row


class ResolveLatestValuesTests(unittest.TestCase):
    def test_is_independent_of_input_order(self):
        rows = [_row(RevisionNo=1, ValueText="old"), _row(RevisionNo=2, ValueText="new"), _row(ProjectID=7, RevisionNo=3, ValueText="other")]
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(resolve_latest_values(rows, 5, {2}), resolve_latest_values(shuffled, 5, {2}))

    def test_uses_latest_at_or_before_target_and_flags_later_values(self):
        resolved, beyond = resolve_latest_values(
            [_row(RevisionNo=4, ValueText="four"), _row(RevisionNo=5, ValueText="five"), _row(RevisionNo=13, ValueText="later")], 12, set()
        )
        self.assertEqual(resolved[(10, 20)].revision_no, 5)
        self.assertEqual(resolved[(10, 20)].value_text, "five")
        self.assertTrue(beyond)

    def test_blank_value_is_a_winning_clear(self):
        resolved, _ = resolve_latest_values([_row(RevisionNo=1, ValueText="old"), _row(RevisionNo=2, ValueText="")], 2, set())
        self.assertEqual(resolved[(10, 20)].value_text, "")

    def test_prefers_reserved_scope_then_lowest_scope(self):
        rows = [_row(ProjectID=7, ValueText="nonreserved"), _row(ProjectID=2, ValueText="reserved")]
        reserved, _ = resolve_latest_values(rows, 1, {2})
        lowest, _ = resolve_latest_values([_row(ProjectID=7), _row(ProjectID=9)], 1, set())
        self.assertEqual(reserved[(10, 20)].project_id, 2)
        self.assertEqual(lowest[(10, 20)].project_id, 7)

    def test_datetime_value_date_is_isoformatted(self):
        resolved, _ = resolve_latest_values([_row(ValueDate=datetime(2026, 9, 15, 10, 30, 0))], 1, set())
        self.assertEqual(resolved[(10, 20)].value_date, "2026-09-15T10:30:00")


class ExtensionFilterTests(unittest.TestCase):
    def test_rejects_sql_injection(self):
        with self.assertRaises(ValueError):
            build_pdm_extension_filter(["SLDPRT' OR 1=1 --"], "d.Filename")

    def test_builds_legacy_filter_shape(self):
        self.assertEqual(build_pdm_extension_filter(["SLDPRT", ".SLDASM"], "d.Filename"), "d.Filename LIKE '%SLDPRT' OR d.Filename LIKE '%.SLDASM'")
        self.assertEqual(build_pdm_extension_filter([], "Filename"), "1=1")
        self.assertEqual(build_pdm_extension_filter(None, "Filename"), "1=1")


class ImportTests(unittest.TestCase):
    def test_module_import_does_not_require_pymssql(self):
        module = importlib.import_module("ragtime.indexer.pdm_source")
        self.assertTrue(hasattr(module, "PdmSqlSource"))
