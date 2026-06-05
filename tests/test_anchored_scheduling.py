import unittest
from datetime import datetime, timezone

from ragtime.core.scheduling import is_anchored_schedule_due, next_anchored_run_after


class AnchoredSchedulingTests(unittest.TestCase):
    def test_unanchored_schedule_returns_none(self) -> None:
        result = next_anchored_run_after(
            interval_seconds=3600,
            start_minute=None,
            timezone_name="UTC",
            after=datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc),
        )

        self.assertIsNone(result)

    def test_next_daily_run_uses_same_day_before_anchor(self) -> None:
        result = next_anchored_run_after(
            interval_seconds=24 * 3600,
            start_minute=9 * 60,
            timezone_name="UTC",
            after=datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result, datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc))

    def test_next_daily_run_uses_next_day_after_anchor(self) -> None:
        result = next_anchored_run_after(
            interval_seconds=24 * 3600,
            start_minute=9 * 60,
            timezone_name="UTC",
            after=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result, datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc))

    def test_subday_interval_walks_forward_from_anchor(self) -> None:
        result = next_anchored_run_after(
            interval_seconds=6 * 3600,
            start_minute=9 * 60,
            timezone_name="UTC",
            after=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result, datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc))

    def test_schedule_due_after_next_slot(self) -> None:
        result = is_anchored_schedule_due(
            interval_seconds=24 * 3600,
            start_minute=9 * 60,
            timezone_name="UTC",
            last_run_at=datetime(2026, 6, 1, 9, 5, tzinfo=timezone.utc),
            now=datetime(2026, 6, 2, 9, 1, tzinfo=timezone.utc),
        )

        self.assertTrue(result)

    def test_schedule_not_due_before_next_slot(self) -> None:
        result = is_anchored_schedule_due(
            interval_seconds=24 * 3600,
            start_minute=9 * 60,
            timezone_name="UTC",
            last_run_at=datetime(2026, 6, 1, 9, 5, tzinfo=timezone.utc),
            now=datetime(2026, 6, 2, 8, 59, tzinfo=timezone.utc),
        )

        self.assertFalse(result)

    def test_invalid_timezone_falls_back_to_legacy_schedule(self) -> None:
        result = is_anchored_schedule_due(
            interval_seconds=3600,
            start_minute=9 * 60,
            timezone_name="Not/A_Timezone",
            last_run_at=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
            now=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
