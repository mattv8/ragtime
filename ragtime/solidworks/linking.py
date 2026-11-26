"""Visibility rules for linked SolidWorks and PDM tools."""

from collections.abc import Iterable


def hidden_pdm_config_ids(configs: Iterable[dict]) -> set[str]:
    configs = list(configs)
    pdm_ids = {str(config["id"]) for config in configs if config.get("tool_type") == "solidworks_pdm" and config.get("id")}
    linked_ids = {
        str(config["connection_config"]["pdm_tool_id"])
        for config in configs
        if config.get("tool_type") == "solidworks" and config.get("enabled", True) and (config.get("connection_config") or {}).get("pdm_tool_id")
    }
    return pdm_ids & linked_ids
