"""
Centralized SQL utilities for database query tools.

Provides shared functionality for PostgreSQL, MSSQL, and future database tools.
Handles query validation, result limiting, and output formatting.
"""

from __future__ import annotations

import re
from typing import Any

from ragtime.core.logging import get_logger
from ragtime.core.security import sanitize_output
from ragtime.core.security import validate_sql_query as _validate_sql_query

logger = get_logger(__name__)


# =============================================================================
# DATABASE TYPE DEFINITIONS
# =============================================================================

# Supported database types for centralized handling
DB_TYPE_POSTGRES = "postgres"
DB_TYPE_MSSQL = "mssql"
DB_TYPE_MYSQL = "mysql"  # Reserved for future use


# =============================================================================
# LIMIT CLAUSE PATTERNS
# =============================================================================

# Database-specific LIMIT clause patterns
# PostgreSQL/MySQL: LIMIT n
# MSSQL: TOP n (after SELECT), or FETCH FIRST n ROWS ONLY (SQL:2008 standard)

LIMIT_PATTERNS: dict[str, dict[str, Any]] = {
    DB_TYPE_POSTGRES: {
        "detect": r"\bLIMIT\s+(\d+)",
        "replace": lambda n: f"LIMIT {n}",
    },
    DB_TYPE_MSSQL: {
        # MSSQL uses TOP n after SELECT, or OFFSET/FETCH
        "detect": r"\bTOP\s*\(?\s*(\d+)\s*\)?|\bFETCH\s+(FIRST|NEXT)\s+(\d+)\s+ROW",
        "replace": None,  # MSSQL TOP is position-dependent, handled separately
    },
    DB_TYPE_MYSQL: {
        "detect": r"\bLIMIT\s+(\d+)",
        "replace": lambda n: f"LIMIT {n}",
    },
}


def validate_sql_query(
    query: str,
    enable_write: bool = False,
    db_type: str = DB_TYPE_POSTGRES,
) -> tuple[bool, str]:
    """
    Validate SQL query for safety across database types.

    Wraps the core security validation and adds database-specific checks.

    Args:
        query: The SQL query to validate.
        enable_write: Whether write operations are allowed.
        db_type: Database type for dialect-specific validation.

    Returns:
        Tuple of (is_safe, reason_message).
    """
    query_upper = query.upper().strip()

    # For MSSQL, do our own validation instead of delegating to core
    # (core validation has PostgreSQL-specific LIMIT check)
    if db_type == DB_TYPE_MSSQL:
        # Check for SELECT/WITH only (unless write enabled)
        if not enable_write:
            if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
                return False, "Only SELECT queries are allowed"

            # Check for dangerous patterns (shared with PostgreSQL)
            dangerous_patterns = [
                r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)\b",
                r"\b(GRANT|REVOKE)\b",
                r";\s*--",  # Comment injection
                r";\s*(DROP|DELETE|UPDATE|INSERT)",  # Chained destructive commands
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, query_upper, re.IGNORECASE):
                    logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                    return False, "Query contains forbidden pattern"

        # MSSQL-specific dangerous patterns
        mssql_dangerous = [
            r"\bxp_cmdshell\b",
            r"\bsp_configure\b",
            r"\bOPENROWSET\b",
            r"\bOPENDATASOURCE\b",
            r"\bBULK\s+INSERT\b",
        ]
        for pattern in mssql_dangerous:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"MSSQL dangerous pattern detected: {pattern}")
                return False, "Query contains forbidden MSSQL pattern"

        # Check for result limiting (TOP or OFFSET/FETCH) - MSSQL style
        if query_upper.startswith("SELECT"):
            has_top = re.search(r"\bTOP\s*\(?\s*\d+\s*\)?", query_upper)
            has_fetch = re.search(r"\bFETCH\s+(FIRST|NEXT)\s+\d+\s+ROW", query_upper)
            if not has_top and not has_fetch:
                return (
                    False,
                    "SELECT queries must include TOP n or OFFSET/FETCH clause to limit results",
                )

        return True, "Query is safe"

    # For PostgreSQL and others, use core validation
    is_safe, reason = _validate_sql_query(query, enable_write)
    if not is_safe:
        return is_safe, reason

    return True, "Query is safe"


def extract_limit_value(query: str, db_type: str = DB_TYPE_POSTGRES) -> int | None:
    """
    Extract the limit value from a query.

    Args:
        query: SQL query to inspect.
        db_type: Database type for dialect-specific parsing.

    Returns:
        The limit value if found, None otherwise.
    """
    pattern_info = LIMIT_PATTERNS.get(db_type, LIMIT_PATTERNS[DB_TYPE_POSTGRES])
    detect_pattern = pattern_info["detect"]
    if not isinstance(detect_pattern, str):
        return None

    match = re.search(detect_pattern, query, re.IGNORECASE)

    if not match:
        return None

    # Handle MSSQL which has multiple capture groups
    if db_type == DB_TYPE_MSSQL:
        # Check for TOP first (group 1)
        if match.group(1):
            return int(match.group(1))
        # Check for FETCH (group 3)
        if match.lastindex is not None and match.lastindex >= 3 and match.group(3):
            return int(match.group(3))
        return None

    return int(match.group(1))


def enforce_max_results(
    query: str,
    max_results: int,
    db_type: str = DB_TYPE_POSTGRES,
) -> str:
    """
    Enforce maximum result limit on a query by reducing LIMIT/TOP if needed.

    Args:
        query: SQL query to modify.
        max_results: Maximum allowed results.
        db_type: Database type for dialect-specific handling.

    Returns:
        Modified query with enforced limit.
    """
    current_limit = extract_limit_value(query, db_type)

    if current_limit is None:
        logger.warning(f"No limit clause found in query for {db_type}")
        return query

    if current_limit <= max_results:
        return query

    logger.info(f"Reducing limit from {current_limit} to {max_results}")

    if db_type == DB_TYPE_MSSQL:
        # Replace TOP value
        query = re.sub(
            r"\bTOP\s*\(?\s*\d+\s*\)?",
            f"TOP ({max_results})",
            query,
            count=1,
            flags=re.IGNORECASE,
        )
        # Replace FETCH value if present
        query = re.sub(
            r"\bFETCH\s+(FIRST|NEXT)\s+\d+\s+ROW",
            lambda m: f"FETCH {m.group(1)} {max_results} ROW",
            query,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        # PostgreSQL/MySQL style
        query = re.sub(
            r"\bLIMIT\s+\d+",
            f"LIMIT {max_results}",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    return query


def format_query_result(
    rows: list[dict[str, Any]] | list[tuple[Any, ...]],
    columns: list[str] | None = None,
    max_output_length: int = 50000,
) -> str:
    """
    Format query results as a readable table string.

    Args:
        rows: Query result rows (list of dicts or tuples).
        columns: Column names (required if rows are tuples).
        max_output_length: Maximum output string length.

    Returns:
        Formatted table string.
    """
    if not rows:
        return "Query executed successfully (no results)"

    # Convert tuples to dicts if needed
    row_dicts: list[dict[str, Any]]
    if rows and isinstance(rows[0], tuple):
        if not columns:
            columns = [f"col{i}" for i in range(len(rows[0]))]
        row_dicts = [dict(zip(columns, row)) for row in rows]
    elif rows and isinstance(rows[0], dict):
        row_dicts = rows  # type: ignore[assignment]
        columns = list(rows[0].keys())  # type: ignore[union-attr]
    else:
        return "Query executed successfully (no results)"

    if not columns:
        return "Query executed successfully (no results)"

    # Build ASCII table
    lines = []

    # Calculate column widths
    widths = {col: len(str(col)) for col in columns}
    for row in row_dicts:
        for col in columns:
            val = str(row.get(col, ""))
            widths[col] = max(widths[col], min(len(val), 50))  # Cap at 50 chars

    # Header
    header = " | ".join(str(col).ljust(widths[col])[: widths[col]] for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    lines.append(header)
    lines.append(separator)

    # Rows
    for row in row_dicts:
        row_str = " | ".join(
            str(row.get(col, "")).ljust(widths[col])[: widths[col]] for col in columns
        )
        lines.append(row_str)

    result = "\n".join(lines)
    result += f"\n\n({len(row_dicts)} row{'s' if len(row_dicts) != 1 else ''})"

    return sanitize_output(result, max_output_length)


# =============================================================================
# ESCAPE HELPERS
# =============================================================================


def escape_shell_arg(value: str) -> str:
    """
    Escape a string for safe use in shell commands.

    Args:
        value: String to escape.

    Returns:
        Shell-escaped string.
    """
    return value.replace("'", "'\\''")
    return value.replace("'", "'\\''")
