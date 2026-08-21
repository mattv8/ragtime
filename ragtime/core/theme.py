from __future__ import annotations


def canonicalize_theme_pack_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return "modern" if normalized == "vscode" else normalized
