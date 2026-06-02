"""Shared timeout resolution for runtime tools."""


def resolve_effective_tool_timeout(requested_timeout: int | None, timeout_max_seconds: int) -> int:
    """Resolve a requested timeout against a per-tool maximum.

    Rules:
    - max=0 (unlimited tool): pass the requested value through unchanged.
    - requested=None (omitted by model): return max (apply the configured cap).
    - requested=0 (explicit "no timeout"): return 0 — honours DB-query semantics
      where 0 means "run to proxy limit". Note: also bypasses SSH caps, which is
      acceptable because models use null rather than 0 to mean "use default".
    - otherwise: return min(requested, max).
    """
    max_timeout = max(0, int(timeout_max_seconds or 0))

    if requested_timeout is None:
        return max_timeout

    requested = max(0, int(requested_timeout))

    if max_timeout == 0:
        return requested

    if requested == 0:
        return 0

    return min(requested, max_timeout)


def resolve_effective_command_timeout(
    requested_timeout: int | None,
    timeout_max_seconds: int,
) -> int:
    """Resolve a shell-command timeout.

    Shell commands use the per-tool configured maximum as their default when the
    model omits a timeout. An explicit ``timeout=0`` means no timeout.
    """
    return resolve_effective_tool_timeout(requested_timeout, timeout_max_seconds)
