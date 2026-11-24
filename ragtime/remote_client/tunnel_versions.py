"""Transient tunnel client version reporting."""

from ragtime.remote_client.relay import relay_manager
from ragtime.remote_client.tunnel_protocol import TUNNEL_CLIENT_VERSION


def tunnel_client_summary(connection_config: dict) -> dict | None:
    pairing_id = connection_config.get("pairing_id")
    if connection_config.get("transport") != "tunnel" or not isinstance(pairing_id, str) or not pairing_id:
        return None
    state = relay_manager.client_version(pairing_id)
    if state is None:
        return None
    connected, version = state
    return {
        "connected": connected,
        "version": version,
        "current_version": TUNNEL_CLIENT_VERSION,
        "outdated": version is None or version < TUNNEL_CLIENT_VERSION,
    }
