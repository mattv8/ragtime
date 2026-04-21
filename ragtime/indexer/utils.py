"""Utility helpers for indexer components."""


import re
from typing import Optional

def safe_tool_name(
    raw_name: Optional[str],
    *,
    separator: str = "_",
    lowercase: bool = True,
) -> str:
    """Return a normalized name safe for identifiers and filenames."""
    if not separator:
        separator = "_"
    name = (raw_name or "").strip()
    normalized = re.sub(r"[^a-zA-Z0-9]+", separator, name).strip(separator)
    if lowercase:
        normalized = normalized.lower()
    return normalized
