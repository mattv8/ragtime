"""Shared timeout resolution for runtime tools."""


def resolve_effective_tool_timeout(requested_timeout: int | None, timeout_max_seconds: int) -> int:
    """Resolve a requested timeout against a per-tool maximum.

    A max of 0 means the tool is configured as unlimited. For finite max values,
    omitted/null or explicit 0 requests use the configured maximum instead of
    bypassing it.
    """
    max_timeout = max(0, int(timeout_max_seconds or 0))

    if requested_timeout is None:
        return max_timeout

    requested = max(0, int(requested_timeout))

    if max_timeout == 0:
        return requested
    if requested == 0:
        return max_timeout
    return min(requested, max_timeout)
