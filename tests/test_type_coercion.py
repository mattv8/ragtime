import math
import unittest

from ragtime.core.type_coercion import coerce_nonnegative_int_metadata


class TypeCoercionTests(unittest.TestCase):
    def test_coerce_nonnegative_int_metadata_parses_numeric_metadata(self) -> None:
        self.assertEqual(coerce_nonnegative_int_metadata("12"), 12)
        self.assertEqual(coerce_nonnegative_int_metadata("12.9"), 12)
        self.assertEqual(coerce_nonnegative_int_metadata(3.8), 3)

    def test_coerce_nonnegative_int_metadata_clamps_negative_values(self) -> None:
        self.assertEqual(coerce_nonnegative_int_metadata(-5), 0)
        self.assertEqual(coerce_nonnegative_int_metadata("-5"), 0)

    def test_coerce_nonnegative_int_metadata_uses_nonnegative_default_for_bad_values(self) -> None:
        self.assertEqual(coerce_nonnegative_int_metadata("not a number", default=7), 7)
        self.assertEqual(coerce_nonnegative_int_metadata(math.inf, default=7), 7)
        self.assertEqual(coerce_nonnegative_int_metadata(None, default=-3), 0)
