"""Pure contract tests for adaptive indexing resource settings."""

import unittest

from pydantic import ValidationError

from ragtime.indexer.models import AppSettings, UpdateSettingsRequest


class IndexingResourceSettingsTests(unittest.TestCase):
    def test_new_settings_default_to_adaptive_resource_limits(self) -> None:
        settings = AppSettings()

        self.assertEqual(settings.chunking_max_workers, 0)
        self.assertEqual(settings.chunking_max_batch_size, 0)
        self.assertEqual(settings.indexing_memory_budget_mb, 0)

    def test_resource_limits_accept_auto_and_documented_maxima(self) -> None:
        request = UpdateSettingsRequest(
            chunking_max_workers=16,
            chunking_max_batch_size=500,
            indexing_memory_budget_mb=1_048_576,
        )

        self.assertEqual(request.chunking_max_workers, 16)
        self.assertEqual(request.chunking_max_batch_size, 500)
        self.assertEqual(request.indexing_memory_budget_mb, 1_048_576)

    def test_memory_budget_rejects_values_between_auto_and_minimum(self) -> None:
        for value in (1, 255, 1_048_577):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    UpdateSettingsRequest(indexing_memory_budget_mb=value)

    def test_omitted_and_null_budget_remain_distinct_from_auto(self) -> None:
        self.assertNotIn("indexing_memory_budget_mb", UpdateSettingsRequest().model_dump(exclude_unset=True))
        self.assertEqual(
            UpdateSettingsRequest(indexing_memory_budget_mb=0).model_dump(exclude_unset=True)["indexing_memory_budget_mb"],
            0,
        )
        self.assertIsNone(UpdateSettingsRequest(indexing_memory_budget_mb=None).model_dump(exclude_unset=True)["indexing_memory_budget_mb"])
