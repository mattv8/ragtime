"""
Shared utilities for API and MCP accounting queries.

Provides WHERE-clause builders for constructing parameterized SQL queries
with optional filtering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def build_where_clause_for_since(
    since: Optional[datetime],
    column_name: str = "created_at",
) -> tuple[list[str], list[Any], int]:
    """
    Build a WHERE clause fragment for timestamp filtering.

    Args:
        since: Optional datetime for filtering (optional)
        column_name: Name of the timestamp column to filter on

    Returns:
        Tuple of (where_clauses, params, next_param_idx)
    """

    where_clauses: list[str] = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"{column_name} >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1

    return where_clauses, params, param_idx


def format_where_sql(where_clauses: list[str]) -> str:
    """
    Format WHERE clauses into SQL string.

    Args:
        where_clauses: List of WHERE clause fragments

    Returns:
        Formatted WHERE SQL string (empty if no clauses)
    """
    return f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
