"""Computed runtime fields attached to tool configs in API responses."""

from ragtime.core.logging import get_logger
from ragtime.indexer.models import ToolConfig, ToolType

logger = get_logger(__name__)


def with_runtime_fields(config: ToolConfig) -> ToolConfig:
    """Return a copy of ``config`` carrying live host and tunnel state. Never raises."""
    return config.model_copy(
        update={
            "solidworks_host_update": _solidworks_host_update(config),
            "tunnel_client": _tunnel_client(config),
        }
    )


def _solidworks_host_update(config: ToolConfig) -> dict | None:
    if config.tool_type != ToolType.SOLIDWORKS or not config.id:
        return None
    try:
        from ragtime.solidworks.host_updater import host_updater  # inline-import: keep

        return host_updater.state_for(config.id)
    except Exception as exc:
        logger.warning(f"SolidWorks host update state unavailable for tool {config.id}: {type(exc).__name__}")
        return None


def _tunnel_client(config: ToolConfig) -> dict | None:
    connection_config = config.connection_config or {}
    if connection_config.get("transport") != "tunnel":
        return None
    try:
        from ragtime.remote_client.tunnel_versions import tunnel_client_summary  # inline-import: keep

        return tunnel_client_summary(connection_config)
    except Exception as exc:
        logger.warning(f"Tunnel client state unavailable for tool {config.id}: {type(exc).__name__}")
        return None
