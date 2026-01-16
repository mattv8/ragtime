"""
Security utilities for SQL injection and command injection prevention.

Note: Write operations can be enabled via the Settings UI.
The enable_write_ops parameter should be passed from the database settings.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Tuple

from fastapi import Cookie, Depends, HTTPException, Request, status

from ragtime.core.auth import TokenData, validate_session
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

if TYPE_CHECKING:
    from prisma.models import User

logger = get_logger(__name__)


# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================


async def get_session_token(
    request: Request,
    session_cookie: Optional[str] = Cookie(None, alias="ragtime_session"),
) -> Optional[str]:
    """
    Extract session token from cookie or Authorization header.

    Prefers cookie (httpOnly), falls back to Bearer token header.
    """
    # Try cookie first
    if session_cookie:
        return session_cookie

    # Fall back to Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


async def get_current_user(
    token: Optional[str] = Depends(get_session_token),
) -> User:
    """
    Validate session and return current user.

    Raises 401 if not authenticated.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = await validate_session(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Fetch user from database
    db = await get_db()
    user = await db.user.find_unique(where={"id": token_data.user_id})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(get_session_token),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.

    Use for routes that work with or without auth.
    """
    if not token:
        return None

    token_data = await validate_session(token)
    if not token_data:
        return None

    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """
    Require admin role.

    Raises 403 if user is not an admin.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def get_token_data(
    token: Optional[str] = Depends(get_session_token),
) -> Optional[TokenData]:
    """Get token data without fetching user from database."""
    if not token:
        return None
    return await validate_session(token)


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
    "SELECT",
    "WITH",
    "FROM",
    "WHERE",
    "JOIN",
    "GROUP BY",
    "ORDER BY",
    "LIMIT",
    "HAVING",
    "UNION",
    "DISTINCT",
    "AS",
    "ON",
    "AND",
    "OR",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "CROSS",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "COALESCE",
    "NULLIF",
    "CAST",
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
    r"env\.registry",
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
    r"file\s*\(",
    r"compile\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"__builtins__",
    r"__class__",
    r"__mro__",
    r"__subclasses__",
]

# =============================================================================
# SSH COMMAND SECURITY PATTERNS
# =============================================================================

# Shell commands that modify system state
SSH_DANGEROUS_SHELL_PATTERNS = [
    # File system modifications
    r"\brm\s+(-[rf]+\s+)?/",  # rm with absolute paths
    r"\brm\s+-[rf]*\s+\*",  # rm with wildcards
    r"\brmdir\b",
    r"\bmkdir\b",
    r"\bmv\s+",
    r"\bcp\s+",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bln\s+",  # symlinks
    # Package management
    r"\bapt(-get)?\s+(install|remove|purge|autoremove)\b",
    r"\byum\s+(install|remove|erase)\b",
    r"\bdnf\s+(install|remove|erase)\b",
    r"\bzypper\s+(remove|rm)\b",
    r"\bpacman\s+-R\w*\b",
    r"\brpm\s+-e\b",
    r"\bpip\s+install\b",
    r"\bpip3\s+install\b",
    # Service management
    r"\bsystemctl\s+(start|stop|restart|enable|disable)\b",
    r"\bservice\s+\w+\s+(start|stop|restart)\b",
    # Process control
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    # User/group management
    r"\buseradd\b",
    r"\buserdel\b",
    r"\busermod\b",
    r"\bgroupadd\b",
    r"\bpasswd\b",
    # Dangerous redirections
    r">\s*/(?!dev/null\b)",  # Redirect to absolute path (allow /dev/null)
    r">>\s*/(?!dev/null\b)",  # Append to absolute path (allow /dev/null)
    r"\btee\s+/(?!dev/null\b)",  # tee to absolute path (allow /dev/null)
    # Dangerous system commands
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+[0-6]\b",
    # Filesystem / disk destructive operations
    r"\bdd\s+[^\n]*\bof=/dev/(sd[a-z]|hd[a-z]|nvme\d+n\d+|vd[a-z])\b",
    r"\bmkfs\.[a-z0-9]+\b",
    r"\bmkswap\b",
    r"\bwipefs\b",
    r"\bparted\b",
    r"\bfdisk\b",
    r"\bsfdisk\b",
    r"\bsgdisk\b",
    r"\blvremove\b",
    r"\bvgremove\b",
    r"\bpvremove\b",
    r"\bcryptsetup\s+luksFormat\b",
    r"\bmount\s+-o\s+remount,rw\b",
    r"\bfind\s+/\s+-delete\b",
    r"\bshred\b",
    r"\btruncate\b[^\n]*\s/dev/(sd[a-z]|hd[a-z]|nvme\d+n\d+|vd[a-z])\b",
    r"\bswapoff\b",
    # Destructive Docker commands
    r"\bdocker\s+(rm|rmi|stop|start|restart|kill|pause|unpause)\b",
    r"\bdocker\s+container\s+(rm|stop|kill|prune)\b",
    r"\bdocker\s+image\s+(rm|prune)\b",
    r"\bdocker\s+image\s+prune\b.*\b-a\b",
    r"\bdocker\s+volume\s+(rm|prune)\b",
    r"\bdocker\s+network\s+(rm|prune)\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+system\s+prune\b.*\b-a\b",
    r"\bdocker\s+builder\s+prune\b",
    r"\bdocker\s+swarm\s+leave\b.*\b--force\b",
    r"\bdocker\s+stack\s+rm\b",
    r"\bdocker-compose\s+(down|rm|stop|kill)\b",
    r"\bdocker\s+compose\s+(down|rm|stop|kill)\b",
    r"\bdocker\s+compose\s+down\b.*\b-v\b",
    r"\bdocker-compose\s+down\b.*\b-v\b",
    r"\bdocker\s+compose\s+down\b.*\b--volumes\b",
    r"\bdocker-compose\s+down\b.*\b--volumes\b",
    r"\bdocker\s+compose\s+down\b.*\b--rmi\s+(all|local)\b",
    r"\bdocker-compose\s+down\b.*\b--rmi\s+(all|local)\b",
    r"\bdocker\s+compose\s+rm\b.*\b-v\b",
    r"\bdocker-compose\s+rm\b.*\b-v\b",
    # Database CLI write detection (psql, mysql, etc.)
    r"\bpsql\b.*-c\s*['\"].*\b(UPDATE|INSERT|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    r"\bmysql\b.*-e\s*['\"].*\b(UPDATE|INSERT|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
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
    # Strip ALL SQL comments for validation to prevent false positives
    # and to catch patterns that might be obscured by comments
    query_stripped = query.strip()
    # Remove all block comments /* ... */
    query_stripped = re.sub(r"/\*.*?\*/", " ", query_stripped, flags=re.DOTALL)
    # Remove all single-line comments -- ...
    query_stripped = re.sub(r"--[^\n]*", " ", query_stripped)
    # Normalize whitespace
    query_stripped = " ".join(query_stripped.split())

    query_upper = query_stripped.upper()

    # If write operations are disabled, only allow SELECT/WITH
    if not enable_write:
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous patterns
        for pattern in DANGEROUS_SQL_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                return False, "Query contains forbidden pattern"
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
                return (
                    False,
                    "Query contains system-level operations that are not allowed",
                )

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
                return False, "Code contains forbidden pattern"

    # Allow read-only getattr over model fields (e.g., getattr(order, field_name))
    getattr_matches = re.findall(r"getattr\s*\(", code, re.IGNORECASE)
    if getattr_matches:
        safe_getattr = re.search(r"getattr\s*\(\s*\w+\s*,\s*field_name", code)
        if not safe_getattr:
            if enable_write_ops:
                logger.warning("getattr detected but allowed due to write flag")
            else:
                logger.warning("getattr detected and blocked (not whitelisted usage)")
                return False, "getattr usage is restricted to field_name inspection"

    # Allow open() only when clearly read-only; block write/append/update modes
    open_write_pattern = r"open\s*\([^)]*,\s*['\"][^'\"]*[wa\+][^'\"]*['\"]"
    if re.search(open_write_pattern, code, re.IGNORECASE):
        if enable_write_ops:
            logger.warning("open() with write/append mode detected but allowed")
        else:
            logger.warning("open() with write/append mode detected and blocked")
            return False, "File writes via open() are not allowed"

    # Must contain at least one safe pattern
    has_safe_pattern = any(re.search(pattern, code) for pattern in SAFE_ODOO_PATTERNS)

    if not has_safe_pattern:
        return False, "Code must contain valid ORM read operations"

    return True, "Code is safe"


def validate_ssh_command(command: str, allow_write: bool = False) -> Tuple[bool, str]:
    """
    Validate SSH shell command for safety. Returns (is_safe, reason).
    Blocks commands that modify system state unless write ops are enabled.

    Args:
        command: The shell command to validate.
        allow_write: Whether write operations are allowed.

    Returns:
        Tuple of (is_safe, reason_message).
    """
    if not command or not command.strip():
        return False, "Empty command"

    if allow_write:
        # When writes are allowed, only log but don't block
        logger.debug(f"SSH command allowed (write mode enabled): {command[:100]}...")
        return True, "Command allowed (write mode enabled)"

    # Check for embedded SQL write operations using existing patterns (case-insensitive)
    for pattern in DANGEROUS_SQL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            logger.warning(
                f"SSH command blocked - SQL write pattern detected: {pattern}"
            )
            return (
                False,
                "Command contains SQL write operation. Enable 'Allow Write Operations' to permit this.",
            )

    # Check for dangerous shell patterns (case-insensitive for commands)
    for pattern in SSH_DANGEROUS_SHELL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            logger.warning(
                f"SSH command blocked - dangerous pattern detected: {pattern}"
            )
            return (
                False,
                "Command contains write/modify operation. Enable 'Allow Write Operations' to permit this.",
            )

    return True, "Command is safe"


def sanitize_output(output: str, max_length: int = 50000) -> str:
    """
    Sanitize and truncate output to prevent memory and storage issues.

    Removes null bytes (\\u0000) which PostgreSQL cannot store in text columns,
    and truncates output to prevent memory issues.

    Args:
        output: The output string to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized output string.
    """
    # Remove null bytes - PostgreSQL text columns cannot store \u0000
    output = output.replace("\x00", "")

    if len(output) > max_length:
        return (
            output[:max_length]
            + f"\n\n... (truncated, {len(output) - max_length} chars omitted)"
        )
    return output
