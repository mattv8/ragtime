from ragtime.core.tool_timeouts import resolve_effective_command_timeout, resolve_effective_tool_timeout
from ragtime.tools.influxdb import resolve_effective_timeout as resolve_influxdb_timeout
from ragtime.tools.mssql import resolve_effective_timeout as resolve_mssql_timeout
from ragtime.tools.mysql import resolve_effective_timeout as resolve_mysql_timeout


def test_finite_tool_timeout_limit_is_not_bypassed_by_zero_or_null() -> None:
    resolvers = [
        resolve_effective_tool_timeout,
        resolve_mysql_timeout,
        resolve_mssql_timeout,
        resolve_influxdb_timeout,
    ]

    for resolve_timeout in resolvers:
        assert resolve_timeout(None, 60) == 60
        assert resolve_timeout(0, 60) == 0
        assert resolve_timeout(5, 60) == 5
        assert resolve_timeout(120, 60) == 60


def test_unlimited_tool_timeout_keeps_requested_timeout_semantics() -> None:
    resolvers = [
        resolve_effective_tool_timeout,
        resolve_mysql_timeout,
        resolve_mssql_timeout,
        resolve_influxdb_timeout,
    ]

    for resolve_timeout in resolvers:
        assert resolve_timeout(None, 0) == 0
        assert resolve_timeout(0, 0) == 0
        assert resolve_timeout(120, 0) == 120


def test_command_timeout_uses_configured_tool_max_when_omitted() -> None:
    assert resolve_effective_command_timeout(None, 0) == 0
    assert resolve_effective_command_timeout(None, 60) == 60


def test_command_timeout_preserves_explicit_unlimited_request() -> None:
    assert resolve_effective_command_timeout(0, 0) == 0
    assert resolve_effective_command_timeout(0, 60) == 0
    assert resolve_effective_command_timeout(120, 0) == 120
    assert resolve_effective_command_timeout(120, 60) == 60
