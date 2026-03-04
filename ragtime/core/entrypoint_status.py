"""Canonical entrypoint status model for workspace runtime-entrypoint.json.

This module is the **single source of truth** for framework metadata on the
ragtime-app side:

* ``FRAMEWORK_REQUIRED_PACKAGES`` — maps framework names to their package
  manifest and required dependency names.
* ``KNOWN_FRAMEWORKS`` — derived automatically from the package registry
  keys plus platform-level frameworks (``custom``, ``node``, ``static``)
  that don't have installable package requirements.

The runtime container (``runtime/shared.py``) keeps a **copy** of
``KNOWN_FRAMEWORKS`` because the two containers cannot cross-import.
When adding or removing a framework here, mirror ``KNOWN_FRAMEWORKS``
in ``runtime/shared.py``.

Keep changes to ``EntrypointStatus`` and ``parse_entrypoint_config``
mirrored in both files as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RUNTIME_ENTRYPOINT_CONFIG_PATH = ".ragtime/runtime-entrypoint.json"

EntrypointState = Literal["missing", "invalid", "valid"]

# ---------------------------------------------------------------------------
# Framework registry — canonical source for framework -> dependency mapping.
# Keys are framework names; values are (manifest_file, [required_packages]).
# Frameworks listed here are automatically included in KNOWN_FRAMEWORKS.
# ---------------------------------------------------------------------------
FRAMEWORK_REQUIRED_PACKAGES: dict[str, tuple[str, list[str]]] = {
    # Python frameworks -> requirements.txt
    "dash": ("requirements.txt", ["dash"]),
    "django": ("requirements.txt", ["django"]),
    "fastapi": ("requirements.txt", ["fastapi", "uvicorn"]),
    "flask": ("requirements.txt", ["flask"]),
    "gradio": ("requirements.txt", ["gradio"]),
    "streamlit": ("requirements.txt", ["streamlit"]),
    # Node frameworks -> package.json
    "express": ("package.json", ["express"]),
    "next": ("package.json", ["next"]),
    "nuxt": ("package.json", ["nuxt"]),
    "vite": ("package.json", ["vite"]),
}

# Platform-level framework names that don't require installable packages
# but are still valid values for the entrypoint "framework" field.
_EXTRA_KNOWN_FRAMEWORKS: frozenset[str] = frozenset({"custom", "node", "static"})

# All recognised framework names.  Derived from FRAMEWORK_REQUIRED_PACKAGES
# keys + platform-level extras.  Keep runtime/shared.py copy in sync.
KNOWN_FRAMEWORKS: frozenset[str] = (
    frozenset(FRAMEWORK_REQUIRED_PACKAGES) | _EXTRA_KNOWN_FRAMEWORKS
)


@dataclass(frozen=True, slots=True)
class EntrypointStatus:
    """Canonical parse result for ``.ragtime/runtime-entrypoint.json``.

    Attributes:
        state: ``missing`` (file absent), ``invalid`` (present but unusable),
            or ``valid`` (has a non-empty command).
        framework: Normalised framework string, or ``None`` when unknown.
        framework_known: Whether *framework* is in :data:`KNOWN_FRAMEWORKS`.
        command: The raw ``command`` field value, or empty string.
        cwd: The raw ``cwd`` field value, or ``"."``.
        error: Human-readable explanation when state is not ``valid``.
    """

    state: EntrypointState
    framework: str | None = None
    framework_known: bool = False
    command: str = ""
    cwd: str = "."
    error: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


def parse_entrypoint_config(workspace_files_path: Path) -> EntrypointStatus:
    """Parse the workspace runtime entrypoint and return a canonical status.

    Args:
        workspace_files_path: Host-side path to the workspace ``files/``
            directory (i.e. the directory that *contains* ``.ragtime/``).

    Returns:
        An :class:`EntrypointStatus` describing the current state.
    """
    config_path = workspace_files_path / RUNTIME_ENTRYPOINT_CONFIG_PATH
    if not config_path.exists() or not config_path.is_file():
        return EntrypointStatus(
            state="missing",
            error=(
                "No .ragtime/runtime-entrypoint.json found. "
                "Create one with a command, cwd, and framework to enable preview."
            ),
        )

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return EntrypointStatus(
            state="invalid",
            error=f"Failed to parse .ragtime/runtime-entrypoint.json: {exc}",
        )

    if not isinstance(raw, dict):
        return EntrypointStatus(
            state="invalid",
            error=".ragtime/runtime-entrypoint.json must be a JSON object.",
        )

    command = str(raw.get("command") or "").strip()
    cwd = str(raw.get("cwd") or ".").strip().replace("\\", "/")
    framework_raw = str(raw.get("framework") or "").strip().lower()
    framework = framework_raw or None
    framework_known = framework in KNOWN_FRAMEWORKS if framework else False

    parsed_raw = {"command": command, "cwd": cwd, "framework": framework_raw}

    if not command:
        return EntrypointStatus(
            state="invalid",
            framework=framework,
            framework_known=framework_known,
            command=command,
            cwd=cwd,
            error=(
                ".ragtime/runtime-entrypoint.json exists but has no command. "
                "Add a command field to define how the workspace starts."
            ),
            raw=parsed_raw,
        )

    return EntrypointStatus(
        state="valid",
        framework=framework,
        framework_known=framework_known,
        command=command,
        cwd=cwd,
        error=None,
        raw=parsed_raw,
    )
