"""Shared SolidWorks tool metadata and LangChain factory."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.solidworks.service import solidworks_service

_DESCRIPTION_MAX_LENGTH = 1499
_TIMEOUT_MAX_SECONDS = 3600
_TIMEOUT_DESCRIPTION = "Timeout in seconds; omit to use the configured maximum. Values above the maximum are clamped to it."

SOLIDWORKS_TOOL_INPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "description": "Operation to execute; see the skills below and describe."},
        "arguments": {"type": "object", "description": "Arguments for the selected operation."},
        "timeout": {"type": "integer", "minimum": 1, "maximum": _TIMEOUT_MAX_SECONDS, "description": _TIMEOUT_DESCRIPTION},
    },
    "required": ["operation"],
}


def build_solidworks_tool_name(safe_name: str) -> str:
    return f"solidworks_{safe_name}"


def build_solidworks_tool_description(config: dict) -> str:
    prefix = (
        "Read SolidWorks model data (features, sketches, dimensions, mates, components, "
        f"properties, mass properties) from '{config.get('name', 'Unknown')}' as JSON."
    )
    connection = config.get("connection_config") or {}
    version = connection.get("solidworks_version")
    linked = bool(connection.get("pdm_tool_id"))
    folders = connection.get("pdm_approved_folders") or []
    vault_changes = linked and config.get("allow_write") and folders
    access = f"Vault changes: on in {len(folders)} approved folder(s)." if vault_changes else "Vault changes: off." if linked else "No PDM vault linked."
    pdm_description = (
        f" Linked to PDM vault '{config.get('pdm_link_name') or 'Unknown'}': find files with pdm.lookup and pdm.search, check state, get latest, check out and check in."
        if linked
        else ""
    )
    suffix = (
        f" Scripts: {'on' if connection.get('allow_scripts') else 'off'}. {access}{pdm_description}"
        f" Pinned to SolidWorks {version if version is not None else 'any installed version'}. "
        "Load the skills listed below before acting; call describe for argument schemas."
    )
    parts = []
    available = max(0, _DESCRIPTION_MAX_LENGTH - len(prefix) - len(suffix))
    for label, text in (("", config.get("description")), ("Vault index: ", (config.get("pdm_link") or {}).get("description"))):
        text = (text or "").strip().rstrip(".")
        if text and available > len(label) + 2:
            part = f" {label}{text[: available - len(label) - 2]}."
            parts.append(part)
            available -= len(part)
    return prefix + "".join(parts) + suffix


def create_solidworks_tool(config: dict, safe_name: str) -> StructuredTool:
    class SolidworksInput(BaseModel):
        operation: str = Field(description=SOLIDWORKS_TOOL_INPUT_SCHEMA["properties"]["operation"]["description"])
        arguments: dict | None = Field(default=None, description=SOLIDWORKS_TOOL_INPUT_SCHEMA["properties"]["arguments"]["description"])
        timeout: int | None = Field(
            default=None,
            ge=1,
            le=_TIMEOUT_MAX_SECONDS,
            description=SOLIDWORKS_TOOL_INPUT_SCHEMA["properties"]["timeout"]["description"],
        )

    async def execute_solidworks(operation: str, arguments: dict | None = None, timeout: int | None = None) -> str:
        return await solidworks_service.execute(config, operation, arguments, timeout)

    return StructuredTool.from_function(
        coroutine=execute_solidworks,
        name=build_solidworks_tool_name(safe_name),
        description=build_solidworks_tool_description(config),
        args_schema=SolidworksInput,
    )
