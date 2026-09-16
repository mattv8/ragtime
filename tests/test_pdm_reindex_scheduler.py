import unittest
from datetime import datetime, timezone

from ragtime.core.scheduling import is_anchored_schedule_due


class PdmReindexSchedulerTests(unittest.TestCase):
    def test_anchored_schedule_is_not_due_before_next_slot_after_attempt(self):
        self.assertFalse(
            is_anchored_schedule_due(
                interval_seconds=3600,
                start_minute=60,
                timezone_name="UTC",
                last_run_at=datetime(2026, 1, 1, 0, 45, tzinfo=timezone.utc),
                now=datetime(2026, 1, 1, 0, 50, tzinfo=timezone.utc),
            )
        )
