import math
from collections.abc import Callable

import pytest

from ragtime.core.embedding_models import _coerce_int as coerce_embedding_int
from ragtime.core.model_limits import _coerce_int as coerce_limit_int
from ragtime.core.model_providers import coerce_positive_int
from ragtime.core.vision_models import _coerce_positive_int

Coercer = Callable[[object], int | None]
Outcome = tuple[str, type[object], object]


def _outcome(coercer: Coercer, value: object) -> Outcome:
    try:
        result = coercer(value)
    except Exception as error:
        return "raise", type(error), str(error)
    return "return", type(result), result


CASES: list[tuple[object, Outcome, Outcome]] = [
    (None, ("return", type(None), None), ("return", type(None), None)),
    (False, ("return", type(None), None), ("return", type(None), None)),
    (True, ("return", type(None), None), ("return", type(None), None)),
    (0, ("return", type(None), None), ("return", int, 0)),
    (-1, ("return", type(None), None), ("return", int, -1)),
    (1, ("return", int, 1), ("return", int, 1)),
    (1.9, ("return", int, 1), ("return", int, 1)),
    ("", ("return", type(None), None), ("return", type(None), None)),
    (" 1 ", ("return", int, 1), ("return", type(None), None)),
    ("1", ("return", int, 1), ("return", int, 1)),
    ("0", ("return", type(None), None), ("return", int, 0)),
    ("1.9", ("return", type(None), None), ("return", type(None), None)),
    ("1e3", ("return", type(None), None), ("return", type(None), None)),
    (
        "²",
        ("raise", ValueError, "invalid literal for int() with base 10: '²'"),
        ("raise", ValueError, "invalid literal for int() with base 10: '²'"),
    ),
    ("１２", ("return", int, 12), ("return", int, 12)),
    (
        math.nan,
        ("raise", ValueError, "cannot convert float NaN to integer"),
        ("raise", ValueError, "cannot convert float NaN to integer"),
    ),
    (
        math.inf,
        ("raise", OverflowError, "cannot convert float infinity to integer"),
        ("raise", OverflowError, "cannot convert float infinity to integer"),
    ),
    (
        -math.inf,
        ("raise", OverflowError, "cannot convert float infinity to integer"),
        ("raise", OverflowError, "cannot convert float infinity to integer"),
    ),
    (2**63, ("return", int, 2**63), ("return", int, 2**63)),
    ([], ("return", type(None), None), ("return", type(None), None)),
    ({}, ("return", type(None), None), ("return", type(None), None)),
]


@pytest.mark.parametrize(("value", "positive_expected", "integer_expected"), CASES)
def test_provider_metadata_coercers_preserve_characterized_contracts(
    value: object,
    positive_expected: Outcome,
    integer_expected: Outcome,
) -> None:
    for coercer in (coerce_positive_int, _coerce_positive_int):
        assert _outcome(coercer, value) == positive_expected
    for coercer in (coerce_limit_int, coerce_embedding_int):
        assert _outcome(coercer, value) == integer_expected
