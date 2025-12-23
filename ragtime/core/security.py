"""
Security utilities for SQL injection and command injection prevention.

Note: Write operations can be enabled via the Settings UI.
The enable_write_ops parameter should be passed from the database settings.
"""

import re
from typing import Tuple

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# SQL SECURITY PATTERNS
# =============================================================================

# Patterns that indicate potentially dangerous SQL
DANGEROUS_SQL_PATTERNS = [
    r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)\b",
    r"\b(GRANT|REVOKE)\b",
    r";\s*--",  # Comment injection
    r";\s*(DROP|DELETE|UPDATE|INSERT)",  # Chained destructive commands
    r"INTO\s+OUTFILE",
    r"LOAD_FILE",
    r"pg_read_file",
    r"pg_write_file",
    r"COPY\s+.*\s+TO",
    r"COPY\s+.*\s+FROM",
]

# Allowed read-only SQL keywords
SAFE_SQL_KEYWORDS = [
    "SELECT", "WITH", "FROM", "WHERE", "JOIN",
    "GROUP BY", "ORDER BY", "LIMIT", "HAVING",
    "UNION", "DISTINCT", "AS", "ON", "AND", "OR",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS",
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    "CASE", "WHEN", "THEN", "ELSE", "END",
    "COALESCE", "NULLIF", "CAST",
]

# =============================================================================
# ODOO SECURITY PATTERNS
# =============================================================================

# Patterns for safe Odoo ORM operations (read-only)
SAFE_ODOO_PATTERNS = [
    r"\.search\s*\(",
    r"\.browse\s*\(",
    r"\.read\s*\(",
    r"\.search_read\s*\(",
    r"\.search_count\s*\(",
    r"\.name_search\s*\(",
    r"\.fields_get\s*\(",
    r"\.mapped\s*\(",
    r"\.filtered\s*\(",
    r"\.sorted\s*\(",
    r"env\s*\[.*\]",
]

# Dangerous Odoo patterns (write operations)
DANGEROUS_ODOO_PATTERNS = [
    r"\.write\s*\(",
    r"\.create\s*\(",
    r"\.unlink\s*\(",
    r"\.copy\s*\(",
    r"\.sudo\s*\(\s*\)",
    r"os\.",
    r"subprocess\.",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\(",
    r"file\s*\(",
    r"compile\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"__builtins__",
    r"__class__",
    r"__mro__",
    r"__subclasses__",
]


def validate_sql_query(query: str, enable_write: bool = False) -> Tuple[bool, str]:
    """
    Validate SQL query for safety. Returns (is_safe, reason).
    Only allows read-only SELECT queries unless write ops are enabled.

    Args:
        query: The SQL query to validate.
        enable_write: Whether write operations are allowed.

    Returns:
        Tuple of (is_safe, reason_message).
    """
    query_upper = query.upper().strip()

    # If write operations are disabled, only allow SELECT/WITH
    if not enable_write:
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous patterns
        for pattern in DANGEROUS_SQL_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                return False, f"Query contains forbidden pattern"
    else:
        # Even with write enabled, block system-level operations
        system_patterns = [
            r"INTO\s+OUTFILE",
            r"LOAD_FILE",
            r"pg_read_file",
            r"pg_write_file",
            r"COPY\s+.*\s+TO",
            r"COPY\s+.*\s+FROM",
            r"\b(DROP|TRUNCATE|ALTER|CREATE)\b",
            r"\b(GRANT|REVOKE)\b",
        ]
        for pattern in system_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"System-level SQL pattern blocked: {pattern}")
                return False, "Query contains system-level operations that are not allowed"

    # Must have LIMIT clause for SELECT queries to prevent huge result sets
    if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
        return False, "SELECT queries must include a LIMIT clause"

    return True, "Query is safe"


def validate_odoo_code(code: str, enable_write_ops: bool = False) -> Tuple[bool, str]:
    """
    Validate Odoo shell code for safety. Returns (is_safe, reason).
    Only allows read-only ORM operations unless write ops are enabled.

    Args:
        code: The Python code to validate for Odoo shell execution.
        enable_write_ops: Whether write operations are allowed (from db settings).

    Returns:
        Tuple of (is_safe, reason_message).
    """
    # Check for dangerous patterns first
    for pattern in DANGEROUS_ODOO_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            if enable_write_ops:
                logger.warning(f"Write operation detected but allowed: {pattern}")
            else:
                logger.warning(f"Dangerous Odoo pattern detected: {pattern}")
                return False, f"Code contains forbidden pattern"

    # Must contain at least one safe pattern
    has_safe_pattern = any(
        re.search(pattern, code) for pattern in SAFE_ODOO_PATTERNS
    )

    if not has_safe_pattern:
        return False, "Code must contain valid ORM read operations"

    return True, "Code is safe"


def sanitize_output(output: str, max_length: int = 50000) -> str:
    """
    Sanitize and truncate output to prevent memory issues.

    Args:
        output: The output string to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized output string.
    """
    if len(output) > max_length:
        return output[:max_length] + f"\n\n... (truncated, {len(output) - max_length} chars omitted)"
    return output
