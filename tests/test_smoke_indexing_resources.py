import unittest

from tests.smoke_indexing_resources import parse_args, percentile95


class SmokeIndexingResourcesTests(unittest.TestCase):
    def test_parser_accepts_repeated_job_ids(self) -> None:
        args = parse_args(
            [
                "--base-url",
                "http://localhost:8000",
                "--job-id",
                "one",
                "--job-id",
                "two",
                "--deadline-seconds",
                "60",
                "--output",
                "/tmp/evidence.jsonl",
            ]
        )
        self.assertEqual(args.job_id, ["one", "two"])

    def test_percentile_uses_inclusive_p95(self) -> None:
        self.assertEqual(percentile95([0.1]), 0.1)
        self.assertAlmostEqual(percentile95(list(range(1, 101))) or 0, 95.05)
