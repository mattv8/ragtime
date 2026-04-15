"""Shared types and constants used across runtime manager, worker, and ragtime userspace.

The ``EntrypointStatus`` dataclass and ``parse_entrypoint_config`` function
are duplicated in ``ragtime/core/entrypoint_status.py`` for use by the
ragtime app container (which cannot import from the ``runtime`` package).
Keep the two copies in sync when modifying the entrypoint parsing contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .workspace_ops import normalize_runtime_file_path

RuntimeSessionState = Literal["starting", "running", "stopping", "stopped", "error"]

VALID_SESSION_STATES: set[str] = {"starting", "running", "stopping", "stopped", "error"}

RUNTIME_BOOTSTRAP_CONFIG_PATH = ".ragtime/runtime-bootstrap.json"
RUNTIME_BOOTSTRAP_STAMP_PATH = ".ragtime/.runtime-bootstrap.done"
RUNTIME_ENTRYPOINT_CONFIG_PATH = ".ragtime/runtime-entrypoint.json"
SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})

# ---------------------------------------------------------------------------
# Canonical entrypoint status – single source of truth for both prompt
# composition (ragtime app) and runtime launch (runtime worker).
# ---------------------------------------------------------------------------

EntrypointState = Literal["missing", "invalid", "valid"]

# Framework names recognised by the runtime auto-install and prompt nudge
# layers. Keep alphabetically sorted.
#
# This is a COPY of KNOWN_FRAMEWORKS from ragtime/core/entrypoint_status.py
# (the canonical source). The two containers cannot cross-import, so this
# must be kept manually in sync. The ragtime-side set is derived from
# FRAMEWORK_REQUIRED_PACKAGES keys + platform extras (custom, node, static).
KNOWN_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "custom",
        "dash",
        "django",
        "express",
        "fastapi",
        "flask",
        "gradio",
        "next",
        "node",
        "nuxt",
        "static",
        "streamlit",
        "vite",
    }
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


def has_cap_sys_admin() -> bool:
    """Return True when the current process has effective CAP_SYS_ADMIN.

    CAP_SYS_ADMIN is capability bit 21 in the ``CapEff`` bitmask from
    ``/proc/self/status``.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if not line.startswith("CapEff:"):
                continue
            cap_hex = line.split(":", 1)[1].strip()
            cap_bits = int(cap_hex, 16)
            return bool(cap_bits & (1 << 21))
    except Exception:
        return False
    return False


def normalize_file_path(
    file_path: str,
    *,
    check_reserved: bool = False,
    enforce_sqlite_managed: bool = False,
) -> str:
    """Normalize a workspace-relative file path and reject traversal attempts.

    Args:
        file_path: Raw file path string.
        check_reserved: When True, also reject paths under ``.ragtime/``.
        enforce_sqlite_managed: When True, require SQLite files to live under
            ``.ragtime/db/``.

    Returns:
        A clean, forward-slash-joined relative path.

    Raises:
        HTTPException: On invalid or traversal-attempting paths.
    """
    return normalize_runtime_file_path(
        file_path,
        check_reserved=check_reserved,
        enforce_sqlite_managed=enforce_sqlite_managed,
    )
