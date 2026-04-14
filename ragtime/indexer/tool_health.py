from typing import Any


def get_heartbeat_timeout_seconds(connection_config: dict[str, Any] | None) -> float:
    """Return the heartbeat timeout for a tool connection."""
    has_ssh_tunnel = bool((connection_config or {}).get("ssh_tunnel_enabled", False))
    return 15.0 if has_ssh_tunnel else 5.0
