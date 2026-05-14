from datetime import timezone

import pytest

runtime_utils = pytest.importorskip("runtime.core.utils")
get_positive_int_env = runtime_utils.get_positive_int_env
utc_now = runtime_utils.utc_now


def test_utc_now_returns_timezone_aware_utc_datetime() -> None:
    now = utc_now()

    assert now.tzinfo is timezone.utc
    assert now.utcoffset().total_seconds() == 0


def test_get_positive_int_env_uses_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNTIME_TEST_VALUE", raising=False)

    assert get_positive_int_env("RUNTIME_TEST_VALUE", 12) == 12


def test_get_positive_int_env_parses_positive_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_TEST_VALUE", " 42 ")

    assert get_positive_int_env("RUNTIME_TEST_VALUE", 12) == 42


@pytest.mark.parametrize("raw_value", ["0", "-1", "", "not-an-int"])
def test_get_positive_int_env_falls_back_for_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv("RUNTIME_TEST_VALUE", raw_value)

    assert get_positive_int_env("RUNTIME_TEST_VALUE", 12) == 12