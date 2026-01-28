"""
Security utilities for SQL injection and command injection prevention.

Note: Write operations can be enabled via the Settings UI.
The enable_write_ops parameter should be passed from the database settings.
"""

from __future__ import annotations

import posixpath
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
# SQL SECURITY PATTERNS (precompiled for performance)
# =============================================================================

# Patterns that indicate potentially dangerous SQL
_DANGEROUS_SQL_PATTERN_STRINGS = [
    r"\bDROP\s+(?:IF\s+EXISTS\s+)?(?:TABLE|DATABASE|SCHEMA|USER|ROLE|VIEW|INDEX|TRIGGER|FUNCTION|PROCEDURE|SEQUENCE|EXTENSION|DOMAIN|TYPE)\b",
    r"\bALTER\s+(?:IF\s+EXISTS\s+)?(?:TABLE|DATABASE|SCHEMA|USER|ROLE|VIEW|INDEX|TRIGGER|FUNCTION|PROCEDURE|SEQUENCE|EXTENSION|DOMAIN|TYPE)\b",
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?(?:TABLE|DATABASE|SCHEMA|USER|ROLE|VIEW|INDEX|TRIGGER|FUNCTION|PROCEDURE|SEQUENCE|UNIQUE\s+INDEX|EXTENSION|DOMAIN|TYPE)\b",
    r"\bTRUNCATE\s+(?:TABLE\s+)?",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+[^;&|\n]+\s+SET\b",
    r"\bDELETE\s+FROM\b",
    r"\bGRANT\s+.*\s+TO\b",
    r"\bREVOKE\s+.*\s+FROM\b",
    r";\s*--",  # Comment injection
    r"INTO\s+OUTFILE",
    r"LOAD_FILE",
    r"pg_read_file",
    r"pg_write_file",
    r"COPY\s+.*\s+TO",
    r"COPY\s+.*\s+FROM",
]
# Precompile with IGNORECASE since SQL is case-insensitive
DANGEROUS_SQL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _DANGEROUS_SQL_PATTERN_STRINGS
]

# SQL comment stripping patterns (precompiled)
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

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
# ODOO SECURITY PATTERNS (precompiled for performance)
# =============================================================================

# Patterns for safe Odoo ORM operations (read-only)
_SAFE_ODOO_PATTERN_STRINGS = [
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
    # Raw cursor operations (SQL validated separately)
    r"env\.cr\.execute\s*\(",
    r"env\.cr\.fetchall\s*\(",
    r"env\.cr\.fetchone\s*\(",
    r"env\.cr\.fetchmany\s*\(",
    r"env\.cr\.dictfetchall\s*\(",
    r"env\.cr\.dictfetchone\s*\(",
    r"env\.cr\.rowcount",
    r"env\.cr\.mogrify\s*\(",
]
SAFE_ODOO_PATTERNS = [re.compile(p) for p in _SAFE_ODOO_PATTERN_STRINGS]

# Dangerous Odoo patterns (write operations)
_DANGEROUS_ODOO_PATTERN_STRINGS = [
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
DANGEROUS_ODOO_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _DANGEROUS_ODOO_PATTERN_STRINGS
]

# Odoo-specific patterns (precompiled)
_ODOO_GETATTR_RE = re.compile(r"getattr\s*\(", re.IGNORECASE)
_ODOO_SAFE_GETATTR_RE = re.compile(r"getattr\s*\(\s*\w+\s*,\s*field_name")
_ODOO_OPEN_WRITE_RE = re.compile(
    r"open\s*\([^)]*,\s*['\"][^'\"]*[wa\+][^'\"]*['\"]", re.IGNORECASE
)
# Extract SQL from env.cr.execute() - matches triple-quoted and single-quoted strings
_ODOO_CR_EXECUTE_RE = re.compile(
    r"env\.cr\.execute\s*\(\s*"
    r"(?:"
    r'"""([^"]*(?:"(?!"")|""(?!"))*[^"]*)"""'
    r"|'''([^']*(?:'(?!'')|''(?!'))*[^']*)'''"
    r'|"([^"]*)"'
    r"|'([^']*)'"
    r")",
    re.DOTALL,
)

# =============================================================================
# SSH COMMAND SECURITY PATTERNS (precompiled for performance)
# =============================================================================

# Shell commands that modify system state
_SSH_DANGEROUS_SHELL_PATTERN_STRINGS = [
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
SSH_DANGEROUS_SHELL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _SSH_DANGEROUS_SHELL_PATTERN_STRINGS
]

# SSH directory constraint patterns (precompiled)
_SSH_DIR_TRAVERSAL_RE = re.compile(r"(?:^|[\s/])\.\.(?:/|[\s]|$)")
_SSH_HOME_EXPANSION_RE = re.compile(r"(?:^|[\s=:])~(?:\/|[\s]|$)")
_SSH_ENV_VAR_RE = re.compile(
    r"\$(?!\?|\$|\d)(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})"
)

# Write target extraction patterns - find paths that are destinations for writes
# These patterns extract the path argument from write commands
_SSH_WRITE_TARGET_PATTERNS = [
    # === Redirections ===
    # Redirect: > /path or >> /path
    re.compile(r">{1,2}\s*(/[^\s;|&<>]+)"),
    # tee: tee /path or tee -a /path
    re.compile(r"\btee\s+(?:-[a-z]\s+)*(/[^\s;|&<>]+)"),
    #
    # === File operations ===
    # ln: ln -s target /link (second path is write target)
    re.compile(r"\bln\s+(?:-[a-z]+\s+)*[^\s]+\s+(/[^\s;|&<>]+)"),
    # cp: cp [options] source(s) /dest - destination is the last absolute path
    re.compile(r"\bcp\s+(?:-[a-zA-Z]+\s+)*(?:[^\s]+\s+)+(/[^\s;|&<>]+)\s*(?:[;&|]|$)"),
    # mv: mv [options] source(s) /dest
    re.compile(r"\bmv\s+(?:-[a-zA-Z]+\s+)*(?:[^\s]+\s+)+(/[^\s;|&<>]+)\s*(?:[;&|]|$)"),
    # rm: rm [options] /path
    re.compile(r"\brm\s+(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"),
    # touch, mkdir, rmdir - path is direct argument
    re.compile(r"\btouch\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bmkdir\s+(?:-[a-z]+\s+)*(/[^\s;|&<>]+)"),
    re.compile(r"\brmdir\s+(?:-[a-z]+\s+)*(/[^\s;|&<>]+)"),
    # install: install [options] source /dest
    re.compile(r"\binstall\s+(?:-[a-zA-Z]+\s+)*(?:[^\s]+\s+)+(/[^\s;|&<>]+)"),
    # chmod/chown/chgrp: can modify file attributes
    re.compile(r"\bchmod\s+(?:-[a-zA-Z]+\s+)*[0-7]+\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bchmod\s+(?:-[a-zA-Z]+\s+)*[ugoa]*[+=-][rwxXst]+\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bchown\s+(?:-[a-zA-Z]+\s+)*[^\s]+\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bchgrp\s+(?:-[a-zA-Z]+\s+)*[^\s]+\s+(/[^\s;|&<>]+)"),
    #
    # === Sync/Transfer ===
    # rsync: rsync [options] source /dest (dangerous with --delete)
    re.compile(r"\brsync\s+[^\n]*\s(/[^\s;|&<>]+)\s*(?:[;&|]|$)"),
    # scp: scp [options] source /dest or scp source host:/dest
    re.compile(r"\bscp\s+(?:-[a-zA-Z]+\s+)*[^\s]+\s+(/[^\s;|&<>:]+)"),
    #
    # === Low-level disk/file operations ===
    # dd: of=/path - output file
    re.compile(r"\bdd\b[^\n]*\bof=(/[^\s;|&<>]+)"),
    # truncate: truncate [options] /path
    re.compile(r"\btruncate\s+(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"),
    # fallocate: fallocate [options] /path
    re.compile(r"\bfallocate\s+(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"),
    # shred: shred [options] /path
    re.compile(r"\bshred\s+(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"),
    #
    # === In-place editing ===
    # sed -i: sed -i[suffix] 'pattern' /path
    re.compile(r"\bsed\s+(?:[^\n]*\s)?-i[^\s]*\s+[^\n]*\s(/[^\s;|&<>]+)"),
    # perl -i: perl -i[suffix] -e 'code' /path or perl -pie 'code' /path
    re.compile(r"\bperl\s+(?:[^\n]*\s)?-[a-z]*i[^\s]*\s+[^\n]*\s(/[^\s;|&<>]+)"),
    # awk -i inplace (gawk): awk -i inplace 'code' /path
    re.compile(r"\b[gm]?awk\s+(?:[^\n]*\s)?-i\s+inplace\s+[^\n]*\s(/[^\s;|&<>]+)"),
    # ed: ed /path (line editor that modifies files)
    re.compile(r"\bed\s+(/[^\s;|&<>]+)"),
    # ex: ex /path (vi's line mode, can modify files)
    re.compile(r"\bex\s+(?:-[a-z]+\s+)*(/[^\s;|&<>]+)"),
    # patch: patch [options] /path < patchfile or patch -d /dir
    re.compile(r"\bpatch\s+(?:[^\n]*\s)?(?:-d\s+)?(/[^\s;|&<>]+)"),
    #
    # === Compression (in-place by default) ===
    # gzip/gunzip/bzip2/bunzip2/xz/unxz - modify files in place
    re.compile(
        r"\b(?:gzip|gunzip|bzip2|bunzip2|xz|unxz|lzma|unlzma)\s+"
        r"(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"
    ),
    # zip: zip archive.zip /path (adding to archive writes to archive location)
    re.compile(r"\bzip\s+(?:-[a-zA-Z]+\s+)*(/[^\s;|&<>]+)"),
    #
    # === Archive extraction ===
    # tar -x: tar -xf archive -C /dest or tar -xf archive (extracts to cwd or specified dir)
    re.compile(r"\btar\s+[^\n]*-C\s+(/[^\s;|&<>]+)"),
    re.compile(r"\btar\s+[^\n]*--directory[=\s]+(/[^\s;|&<>]+)"),
    # unzip: unzip archive -d /dest
    re.compile(r"\bunzip\s+[^\n]*-d\s+(/[^\s;|&<>]+)"),
    # 7z/7za: 7z x archive -o/dest
    re.compile(r"\b7z[a]?\s+[ex]\s+[^\n]*-o(/[^\s;|&<>]+)"),
    #
    # === Download ===
    # wget: wget -O /path or wget -P /dir
    re.compile(r"\bwget\s+[^\n]*-O\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bwget\s+[^\n]*-P\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bwget\s+[^\n]*--output-document[=\s]+(/[^\s;|&<>]+)"),
    re.compile(r"\bwget\s+[^\n]*--directory-prefix[=\s]+(/[^\s;|&<>]+)"),
    # curl: curl -o /path or curl -O (to cwd)
    re.compile(r"\bcurl\s+[^\n]*-o\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bcurl\s+[^\n]*--output[=\s]+(/[^\s;|&<>]+)"),
    #
    # === Text processing with output ===
    # sort -o: sort -o /output input
    re.compile(r"\bsort\s+[^\n]*-o\s+(/[^\s;|&<>]+)"),
    re.compile(r"\bsort\s+[^\n]*--output[=\s]+(/[^\s;|&<>]+)"),
    # split: split [options] input /prefix
    re.compile(r"\bsplit\s+(?:-[a-zA-Z]+\s+)*[^\s]+\s+(/[^\s;|&<>]+)"),
    # csplit: csplit [options] input /pattern/ -f /prefix
    re.compile(r"\bcsplit\s+[^\n]*-f\s+(/[^\s;|&<>]+)"),
    #
    # === Git operations (can write to worktree) ===
    # git clone: git clone url /dest
    re.compile(r"\bgit\s+clone\s+[^\n]*\s(/[^\s;|&<>]+)\s*(?:[;&|]|$)"),
    #
    # === Database CLI write with output files ===
    # psql with \o or COPY TO
    re.compile(r"\bpsql\s+[^\n]*-o\s+(/[^\s;|&<>]+)"),
    # mysql with output redirect is caught by > pattern
]
_SSH_PATH_EXTRACT_RE = re.compile(r"(?:^|[\s=:\"'])(/[a-zA-Z0-9_\-\./]+)")

# System-level SQL patterns for write mode (precompiled)
_SQL_SYSTEM_PATTERNS = [
    re.compile(r"INTO\s+OUTFILE", re.IGNORECASE),
    re.compile(r"LOAD_FILE", re.IGNORECASE),
    re.compile(r"pg_read_file", re.IGNORECASE),
    re.compile(r"pg_write_file", re.IGNORECASE),
    re.compile(r"COPY\s+.*\s+TO", re.IGNORECASE),
    re.compile(r"COPY\s+.*\s+FROM", re.IGNORECASE),
    re.compile(r"\b(DROP|TRUNCATE|ALTER|CREATE)\b", re.IGNORECASE),
    re.compile(r"\b(GRANT|REVOKE)\b", re.IGNORECASE),
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
    # Remove all block comments /* ... */ (using precompiled pattern)
    query_stripped = _SQL_BLOCK_COMMENT_RE.sub(" ", query_stripped)
    # Remove all single-line comments -- ... (using precompiled pattern)
    query_stripped = _SQL_LINE_COMMENT_RE.sub(" ", query_stripped)
    # Normalize whitespace
    query_stripped = " ".join(query_stripped.split())

    query_upper = query_stripped.upper()

    # If write operations are disabled, only allow SELECT/WITH
    if not enable_write:
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous patterns (using precompiled patterns)
        for pattern in DANGEROUS_SQL_PATTERNS:
            if pattern.search(query_upper):
                logger.warning(f"Dangerous SQL pattern detected: {pattern.pattern}")
                return False, "Query contains forbidden pattern"
    else:
        # Even with write enabled, block system-level operations (using precompiled)
        for pattern in _SQL_SYSTEM_PATTERNS:
            if pattern.search(query_upper):
                logger.warning(f"System-level SQL pattern blocked: {pattern.pattern}")
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
    # Check for dangerous patterns first (using precompiled patterns)
    for pattern in DANGEROUS_ODOO_PATTERNS:
        if pattern.search(code):
            if enable_write_ops:
                logger.warning(
                    f"Write operation detected but allowed: {pattern.pattern}"
                )
            else:
                logger.warning(f"Dangerous Odoo pattern detected: {pattern.pattern}")
                return False, "Code contains forbidden pattern"

    # Allow read-only getattr over model fields (e.g., getattr(order, field_name))
    if _ODOO_GETATTR_RE.search(code):
        if not _ODOO_SAFE_GETATTR_RE.search(code):
            if enable_write_ops:
                logger.warning("getattr detected but allowed due to write flag")
            else:
                logger.warning("getattr detected and blocked (not whitelisted usage)")
                return False, "getattr usage is restricted to field_name inspection"

    # Allow open() only when clearly read-only; block write/append/update modes
    if _ODOO_OPEN_WRITE_RE.search(code):
        if enable_write_ops:
            logger.warning("open() with write/append mode detected but allowed")
        else:
            logger.warning("open() with write/append mode detected and blocked")
            return False, "File writes via open() are not allowed"

    # Validate SQL inside env.cr.execute() calls
    for match in _ODOO_CR_EXECUTE_RE.finditer(code):
        # Extract SQL from whichever group matched (triple or single quoted)
        sql = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if sql:
            sql = sql.strip()
            is_safe, reason = validate_sql_query(sql, enable_write=enable_write_ops)
            if not is_safe:
                logger.warning(f"Unsafe SQL in env.cr.execute(): {reason}")
                return (
                    False,
                    f"SQL query in env.cr.execute() failed validation: {reason}",
                )

    # Must contain at least one safe pattern (using precompiled patterns)
    has_safe_pattern = any(pattern.search(code) for pattern in SAFE_ODOO_PATTERNS)

    if not has_safe_pattern:
        return False, "Code must contain valid ORM read operations"

    return True, "Code is safe"


def validate_ssh_command(
    command: str,
    allow_write: bool = False,
    allowed_directory: Optional[str] = None,
    expanded_command: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate SSH shell command for safety. Returns (is_safe, reason).
    Blocks commands that modify system state unless write ops are enabled.

    Args:
        command: The shell command to validate.
        allow_write: Whether write operations are allowed.
        allowed_directory: Optional directory path to constrain operations to.
        expanded_command: Optional pre-expanded command (env vars resolved via SSH).
            Required when command contains env vars and allowed_directory is set.
            Use expand_env_vars_via_ssh() from ragtime.core.ssh to obtain this.

    Returns:
        Tuple of (is_safe, reason_message).
    """
    if not command or not command.strip():
        return False, "Empty command"

    # Directory constraint check
    if allowed_directory:
        # 0. Validate allowed_directory is an absolute path (no ~ or relative paths)
        # This prevents misconfiguration where the constraint itself is ambiguous
        if not allowed_directory.startswith("/"):
            return (
                False,
                f"Configuration error: allowed_directory must be an absolute path, got '{allowed_directory}'",
            )
        if "~" in allowed_directory:
            return (
                False,
                "Configuration error: allowed_directory must not contain '~'. Use an absolute path.",
            )

        # 1. Block directory traversal via relative paths (using precompiled pattern)
        # We catch '..' only when it acts as a path component (bounded by / or whitespace)
        # This allows 'git log master..main' but blocks 'cd ..', 'cat ../file', '/foo/../bar'
        if _SSH_DIR_TRAVERSAL_RE.search(command):
            return (
                False,
                "Directory traversal (..) is not allowed when constrained to a directory.",
            )

        # 2. Block home directory expansion which might escape (using precompiled pattern)
        # Catch '~' when used as a path start (bounded by whitespace/=/: and followed by / or whitespace)
        # This allows 'HEAD~1' but blocks 'cd ~', 'cat ~/.ssh/id_rsa'
        if _SSH_HOME_EXPANSION_RE.search(command):
            return (
                False,
                "Home directory expansion (~) is not allowed when constrained to a directory.",
            )

        # 3. Check for unexpanded environment variables (using precompiled pattern)
        # If env vars are present and no expanded_command was provided, we can't validate paths
        # The caller should use expand_env_vars_via_ssh() first to get the expanded command
        has_env_vars = bool(_SSH_ENV_VAR_RE.search(command))
        if has_env_vars and expanded_command is None:
            return (
                False,
                "Command contains environment variables but no expanded form was provided. "
                "Use expand_env_vars_via_ssh() to pre-expand variables before validation.",
            )

        # Use expanded command for path validation if available
        command_for_path_check = expanded_command if expanded_command else command

        # 4. Check absolute paths (using precompiled pattern on expanded command if available)
        # Find paths starting with / (at start of string, or after space, =, :, ", ')
        # This covers: /path, arg=/path, "/path", etc.
        paths = _SSH_PATH_EXTRACT_RE.findall(command_for_path_check)

        # Base allowed path (normalized)
        base_allowed = posixpath.normpath(allowed_directory)

        # Safe system prefixes that are always allowed (executables, devices, temp)
        # This allows running tools (/usr/bin/git) and using safe resources (/dev/null)
        # Note: Write operations to these paths are still blocked by SSH_DANGEROUS_SHELL_PATTERNS
        SAFE_SYSTEM_PREFIXES = [
            # Device/kernel interfaces (read-safe, writes blocked by patterns)
            "/dev/",
            "/proc/",
            "/sys/",
            # Temp directories (commonly needed for intermediate files)
            "/tmp/",
            "/var/tmp/",
            # Executables (needed to run system tools)
            "/bin/",
            "/usr/bin/",
            "/usr/local/bin/",
            "/sbin/",
            "/usr/sbin/",
            # Libraries (needed by executables)
            "/lib/",
            "/lib64/",
            "/usr/lib/",
            "/usr/lib64/",
            "/usr/local/lib/",
            # Runtime/socket directories (for checking service status, etc.)
            "/run/",
            "/var/run/",
        ]

        for path in paths:
            # Normalize path to resolve .. inside strings
            # e.g. /app/../etc/passwd -> /etc/passwd
            normalized_path = posixpath.normpath(path)

            # Check for exact equality or prefix match with safe paths
            is_safe_system = False
            for safe_prefix in SAFE_SYSTEM_PREFIXES:
                # Check for exact match (e.g. /bin) or subpath (/bin/ls)
                # We strip trailing slash from prefix for exact match check
                clean_prefix = safe_prefix.rstrip("/")
                if normalized_path == clean_prefix or normalized_path.startswith(
                    safe_prefix
                ):
                    is_safe_system = True
                    break

            if is_safe_system:
                continue

            # If not a safe system path, it must be within allowed_directory
            if (
                not normalized_path.startswith(base_allowed + "/")
                and normalized_path != base_allowed
            ):
                return (
                    False,
                    f"Access forbidden: Path '{path}' is outside the allowed directory '{allowed_directory}'",
                )

        # 5. Check write targets specifically - writes must be within allowed_directory
        # Even when allow_write=True, we restrict writes to the allowed_directory
        # Safe system prefixes are for reading (executables, libs), not writing
        if allow_write:
            write_targets = []
            for pattern in _SSH_WRITE_TARGET_PATTERNS:
                write_targets.extend(pattern.findall(command_for_path_check))

            for target in write_targets:
                normalized_target = posixpath.normpath(target)

                # Only /dev/null is allowed as a write target outside the allowed_directory
                if normalized_target == "/dev/null":
                    continue

                # Write targets must be within allowed_directory
                if (
                    not normalized_target.startswith(base_allowed + "/")
                    and normalized_target != base_allowed
                ):
                    return (
                        False,
                        f"Write forbidden: Path '{target}' is outside the allowed directory '{allowed_directory}'",
                    )

    if allow_write:
        # When writes are allowed, only log but don't block
        logger.debug(f"SSH command allowed (write mode enabled): {command[:100]}...")
        return True, "Command allowed (write mode enabled)"

    # Check for embedded SQL write operations (using precompiled patterns)
    for pattern in DANGEROUS_SQL_PATTERNS:
        if pattern.search(command):
            logger.warning(
                f"SSH command blocked - SQL write pattern detected: {pattern.pattern}"
            )
            return (
                False,
                "Command contains SQL write operation. Enable 'Allow Write Operations' to permit this.",
            )

    # Check for dangerous shell patterns (using precompiled patterns)
    for pattern in SSH_DANGEROUS_SHELL_PATTERNS:
        if pattern.search(command):
            logger.warning(
                f"SSH command blocked - dangerous pattern detected: {pattern.pattern}"
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
