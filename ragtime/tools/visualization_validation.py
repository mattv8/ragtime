"""Validation feedback helpers for chat visualization tools."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


def _format_location(location: Any) -> str:
    if not isinstance(location, tuple) or not location:
        return "input"
    return ".".join(str(part) for part in location)


def format_visualization_validation_error(
    error: Exception,
    *,
    tool_name: str,
    expected_shape: str,
    guidance: list[str],
) -> str:
    """Return actionable tool-validation feedback for the model."""

    lines = [f"Tool input validation error for {tool_name}:"]
    if isinstance(error, ValidationError):
        details = error.errors()
        if details:
            for detail in details[:8]:
                location = _format_location(detail.get("loc"))
                message = str(detail.get("msg") or "Invalid value")
                error_type = str(detail.get("type") or "").strip()
                suffix = f" ({error_type})" if error_type else ""
                lines.append(f"- {location}: {message}{suffix}")
            if len(details) > 8:
                lines.append(f"- ... and {len(details) - 8} more validation errors")
        else:
            lines.append(f"- {error}")
    else:
        lines.append(f"- {error}")

    lines.append("")
    lines.append("Expected input shape:")
    lines.append(expected_shape)
    if guidance:
        lines.append("")
        lines.append("How to fix:")
        lines.extend(f"- {item}" for item in guidance)
    return "\n".join(lines)
