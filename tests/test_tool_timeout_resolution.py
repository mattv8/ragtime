import pytest

from ragtime.core.tool_timeouts import resolve_effective_command_timeout, resolve_effective_tool_timeout
from ragtime.tools.influxdb import resolve_effective_timeout as resolve_influxdb_timeout
from ragtime.tools.mssql import resolve_effective_timeout as resolve_mssql_timeout
from ragtime.tools.mysql import resolve_effective_timeout as resolve_mysql_timeout

TOOL_TIMEOUT_RESOLVERS = [
    resolve_effective_tool_timeout,
    resolve_mysql_timeout,
    resolve_mssql_timeout,
    resolve_influxdb_timeout,
]


@pytest.mark.parametrize("resolve_timeout", TOOL_TIMEOUT_RESOLVERS, ids=lambda resolver: resolver.__module__)
@pytest.mark.parametrize(
    ("requested_timeout", "configured_timeout", "expected_timeout"),
    [
        (None, 60, 60),
        (0, 60, 0),
        (5, 60, 5),
        (120, 60, 60),
        (None, 0, 0),
        (0, 0, 0),
        (120, 0, 120),
    ],
)
def test_tool_timeout_resolution_matrices(resolve_timeout, requested_timeout, configured_timeout, expected_timeout) -> None:
    assert resolve_timeout(requested_timeout, configured_timeout) == expected_timeout


@pytest.mark.parametrize(
    ("requested_timeout", "configured_timeout", "expected_timeout"),
    [
        (None, 0, 0),
        (None, 60, 60),
        (0, 0, 0),
        (0, 60, 0),
        (120, 0, 120),
        (120, 60, 60),
    ],
)
def test_command_timeout_resolution_matrices(requested_timeout, configured_timeout, expected_timeout) -> None:
    assert resolve_effective_command_timeout(requested_timeout, configured_timeout) == expected_timeout
