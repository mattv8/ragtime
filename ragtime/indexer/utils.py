"""Utility helpers for indexer components."""


import re
from typing import Optional

def safe_tool_name(raw_name: Optional[str]) -> str:
    """Return a normalized tool name safe for index identifiers."""
    name = (raw_name or "").strip()
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
