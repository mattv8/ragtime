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

            # Reject PostgreSQL-style LIMIT clause (not valid in MSSQL)
            if re.search(r"\bLIMIT\s+\d+", query_upper):
                return (
                    False,
                    "MSSQL does not support LIMIT clause. Use TOP n instead (e.g., SELECT TOP 10 * FROM table)",
                )

        return True, "Query is safe"

    # For MySQL/MariaDB, do validation similar to PostgreSQL but with MySQL-specific checks
    if db_type == DB_TYPE_MYSQL:
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

        # MySQL-specific dangerous patterns
        mysql_dangerous = [
            r"\bINTO\s+OUTFILE\b",  # Write files to server
            r"\bINTO\s+DUMPFILE\b",  # Binary dump to file
            r"\bLOAD_FILE\b",  # Read files from server
            r"\bLOAD\s+DATA\b",  # Load data from file
            r"\bSYSTEM\b",  # System commands (MariaDB)
            r"\bBENCHMARK\b",  # Timing attacks
            r"\bSLEEP\b",  # Timing attacks
        ]
        for pattern in mysql_dangerous:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"MySQL dangerous pattern detected: {pattern}")
                return False, "Query contains forbidden MySQL pattern"

        # Check for LIMIT clause (MySQL uses LIMIT like PostgreSQL)
        if query_upper.startswith("SELECT"):
            has_limit = re.search(r"\bLIMIT\s+\d+", query_upper)
            if not has_limit:
                return (
                    False,
                    "SELECT queries must include LIMIT clause to limit results",
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


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value to be JSON-compatible.

    Handles datetime, date, Decimal, bytes, and other non-JSON types.
    """
    from datetime import date, datetime
    from decimal import Decimal

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        # Convert to float for JSON, preserving reasonable precision
        return float(value)
    if isinstance(value, bytes):
        # Try to decode as UTF-8, otherwise return hex representation
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    # Fallback: convert to string
    return str(value)


def format_query_result(
    rows: list[dict[str, Any]] | list[tuple[Any, ...]],
    columns: list[str] | None = None,
    max_output_length: int = 50000,
) -> str:
    """
    Format query results as a readable table string with embedded metadata.

    The output includes a hidden JSON metadata block that UI clients can parse
    to render proper HTML tables. The ASCII table serves as fallback for
    non-UI clients (MCP, API).

    Format:
        <!--TABLEDATA:{"columns":[...],"rows":[...]}-->
        column_a | column_b
        ---------+---------
        value1   | value2

        (N rows)

    Args:
        rows: Query result rows (list of dicts or tuples).
        columns: Column names (required if rows are tuples).
        max_output_length: Maximum output string length.

    Returns:
        Formatted table string with embedded metadata.
    """
    import json

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

    # Build table metadata for UI clients
    # Serialize rows as list of lists (more compact than list of dicts)
    serialized_rows = [
        [_serialize_value(row.get(col)) for col in columns] for row in row_dicts
    ]
    table_metadata = {"columns": columns, "rows": serialized_rows}

    # Limit metadata size to avoid bloating output
    try:
        metadata_json = json.dumps(table_metadata, separators=(",", ":"))
        # If metadata is too large, skip it (ASCII table is still readable)
        if len(metadata_json) > 30000:
            metadata_line = ""
        else:
            metadata_line = f"<!--TABLEDATA:{metadata_json}-->\n"
    except (TypeError, ValueError):
        # JSON serialization failed, skip metadata
        metadata_line = ""

    # Build ASCII table (fallback display)
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

    ascii_table = "\n".join(lines)
    ascii_table += f"\n\n({len(row_dicts)} row{'s' if len(row_dicts) != 1 else ''})"

    result = metadata_line + ascii_table

    return sanitize_output(result, max_output_length)


def add_table_metadata_to_psql_output(psql_output: str) -> str:
    """
    Parse psql ASCII table output and add table metadata for UI rendering.

    psql output format:
        column_a | column_b | column_c
        ---------+----------+---------
        value1   | value2   | value3
        value4   | value5   | value6
        (N rows)

    This function extracts the column names and data, then prepends the
    <!--TABLEDATA:...}--> metadata for UI clients to render as HTML tables.

    TODO: MySQL/MariaDB Support
    --------------------------
    MySQL uses a different ASCII table format with box-drawing characters:

        +----------+----------+----------+
        | column_a | column_b | column_c |
        +----------+----------+----------+
        | value1   | value2   | value3   |
        | value4   | value5   | value6   |
        +----------+----------+----------+
        2 rows in set (0.00 sec)

    To add MySQL support:
    1. Detect MySQL format: first line starts with "+-" and ends with "-+"
    2. Header is on line 2 (between first two separator lines)
    3. Data rows are between lines 3 and last separator
    4. Use same position-based parsing (+ positions for column boundaries)
    5. Test with: docker exec -i mysql-container mysql -u user -ppass db -e "SELECT..."

    Test cases to verify:
    - Basic multi-column: SELECT id, name, value FROM test LIMIT 5;
    - Single column: SELECT name FROM test LIMIT 5;
    - NULL values: SELECT id, nullable_col FROM test WHERE nullable_col IS NULL;
    - Numeric types: SELECT int_col, decimal_col, float_col FROM test;
    - Text with pipe: SELECT description FROM test WHERE description LIKE '%|%';
    - JSON column: SELECT id, json_data FROM test; (MariaDB JSON or MySQL JSON)

    Args:
        psql_output: Raw output from psql command.

    Returns:
        Original output with table metadata prepended (if parseable).
    """
    import json

    if not psql_output or "Error" in psql_output[:50]:
        return psql_output

    lines = psql_output.strip().split("\n")
    if len(lines) < 3:  # Need at least header, separator, and one row
        return psql_output

    # Find the separator line (dashes with optional + for multi-column)
    # Single column: "-----" or " ----- "
    # Multi column: "---+---+---"
    separator_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match line that is only dashes and plus signs (with possible spaces)
        if stripped and re.match(r"^[-+]+$", stripped.replace(" ", "")):
            separator_idx = i
            break

    if separator_idx is None or separator_idx < 1:
        return psql_output

    # Get header and separator lines
    header_line = lines[separator_idx - 1]
    separator_line = lines[separator_idx]

    # Parse header using split on | (simple and reliable for headers)
    if "|" in header_line:
        columns = [col.strip() for col in header_line.split("|") if col.strip()]
    else:
        columns = [header_line.strip()]

    # Get column positions from SEPARATOR line for parsing DATA rows
    # Separator uses + at column boundaries: "---+---+---"
    # Data rows use | at same positions: " a | b | c "
    # This avoids issues with | appearing inside JSONB values
    if "+" in separator_line:
        # Multi-column: find positions of each +
        data_col_positions = [-1]  # Start before first char
        for i, char in enumerate(separator_line):
            if char == "+":
                data_col_positions.append(i)
        data_col_positions.append(len(separator_line))  # End after last char
    else:
        # Single column - entire line is one column
        data_col_positions = [-1, len(separator_line)]

    def parse_row_by_positions(line: str, positions: list[int]) -> list[str]:
        """Extract column values based on fixed positions from separator."""
        values = []
        for i in range(len(positions) - 1):
            start = positions[i] + 1  # Skip the delimiter
            end = positions[i + 1]
            if end > len(line):
                end = len(line)
            if start < len(line):
                val = line[start:end].strip()
                # Remove trailing | if present (from lines shorter than separator)
                if val.endswith("|"):
                    val = val[:-1].strip()
                values.append(val)
            else:
                values.append("")
        return values

    # Parse data rows (lines after separator, excluding footer)
    rows = []
    for line in lines[separator_idx + 1 :]:
        # Stop at row count line or empty line
        if line.strip().startswith("(") or not line.strip():
            break
        # Parse row values using fixed positions from separator
        values = parse_row_by_positions(line, data_col_positions)
        # Filter to match column count
        values = values[: len(columns)] if len(values) > len(columns) else values
        # Pad if needed
        while len(values) < len(columns):
            values.append("")
        # Convert numeric strings to numbers
        parsed_values: list[str | int | float | None] = []
        for val in values:
            if val == "" or val.lower() == "null":
                parsed_values.append(None)
            else:
                # Try to convert to int or float
                try:
                    if "." in val:
                        parsed_values.append(float(val))
                    else:
                        parsed_values.append(int(val))
                except ValueError:
                    parsed_values.append(val)
        rows.append(parsed_values)

    if not rows:
        return psql_output

    # Build metadata JSON
    try:
        table_metadata = {"columns": columns, "rows": rows}
        metadata_json = json.dumps(table_metadata, separators=(",", ":"))
        # Skip if metadata is too large
        if len(metadata_json) > 30000:
            return psql_output
        metadata_line = f"<!--TABLEDATA:{metadata_json}-->\n"
        return metadata_line + psql_output
    except (TypeError, ValueError):
        return psql_output


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


# =============================================================================
# MSSQL CONNECTION UTILITIES
# =============================================================================


class MssqlConnectionError(Exception):
    """Exception raised for MSSQL connection errors."""


class MssqlConnection:
    """
    Context manager for MSSQL database connections.

    Provides a clean interface for pymssql connections with proper
    error handling and resource cleanup.

    Example:
        async with mssql_connect(host, port, user, password, database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TOP 10 * FROM table")
            rows = cursor.fetchall()
    """

    def __init__(
        self,
        host: str,
        port: int | str,
        user: str,
        password: str,
        database: str,
        login_timeout: int = 10,
        timeout: int = 10,
        as_dict: bool = True,
    ):
        self.host = host
        self.port = str(port)
        self.user = user
        self.password = password
        self.database = database
        self.login_timeout = login_timeout
        self.timeout = timeout
        self.as_dict = as_dict
        self._conn: Any = None
        self._pymssql: Any = None

    def _import_pymssql(self) -> Any:
        """Import pymssql, raising a clear error if not available."""
        try:
            import pymssql  # type: ignore[import-untyped]

            return pymssql
        except ImportError as e:
            raise MssqlConnectionError(
                "pymssql package not installed. Install with: pip install pymssql"
            ) from e

    def connect(self) -> Any:
        """Establish connection synchronously (for use in thread pool)."""
        self._pymssql = self._import_pymssql()

        try:
            # pymssql is untyped - use getattr to avoid static type checker complaints
            connect_fn = getattr(self._pymssql, "connect")
            self._conn = connect_fn(
                server=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                login_timeout=self.login_timeout,
                timeout=self.timeout,
                as_dict=self.as_dict,
            )
            return self._conn
        except Exception as e:
            error_str = str(e)
            if "Login failed" in error_str:
                raise MssqlConnectionError(
                    "Login failed - check username and password"
                ) from e
            if "Cannot open database" in error_str:
                raise MssqlConnectionError(
                    f"Cannot open database '{self.database}' - check database name"
                ) from e
            raise MssqlConnectionError(f"Connection error: {error_str}") from e

    def close(self) -> None:
        """Close the connection if open."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> Any:
        return self.connect()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def mssql_connect(
    host: str,
    port: int | str,
    user: str,
    password: str,
    database: str,
    login_timeout: int = 10,
    timeout: int = 10,
    as_dict: bool = True,
) -> MssqlConnection:
    """
    Create an MSSQL connection context manager.

    Args:
        host: Database server hostname or IP.
        port: Database server port.
        user: Username for authentication.
        password: Password for authentication.
        database: Database name to connect to.
        login_timeout: Login timeout in seconds.
        timeout: Query timeout in seconds.
        as_dict: Whether to return rows as dictionaries.

    Returns:
        MssqlConnection context manager.

    Example:
        with mssql_connect(host, port, user, password, database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TOP 10 * FROM table")
            rows = cursor.fetchall()
    """
    return MssqlConnection(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        login_timeout=login_timeout,
        timeout=timeout,
        as_dict=as_dict,
    )


# =============================================================================
# MYSQL CONNECTION UTILITIES
# =============================================================================


class MysqlConnectionError(Exception):
    """Exception raised for MySQL connection errors."""


class MysqlConnection:
    """
    Context manager for MySQL/MariaDB database connections.

    Provides a clean interface for PyMySQL connections with proper
    error handling and resource cleanup.

    Example:
        with mysql_connect(host, port, user, password, database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table LIMIT 10")
            rows = cursor.fetchall()
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        connect_timeout: int = 10,
        read_timeout: int = 10,
        as_dict: bool = True,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.as_dict = as_dict
        self._conn: Any = None
        self._pymysql: Any = None

    def _import_pymysql(self) -> Any:
        """Import pymysql, raising a clear error if not available."""
        try:
            import pymysql  # type: ignore[import-untyped]

            return pymysql
        except ImportError as e:
            raise MysqlConnectionError(
                "pymysql package not installed. Install with: pip install pymysql"
            ) from e

    def connect(self) -> Any:
        """Establish connection synchronously (for use in thread pool)."""
        self._pymysql = self._import_pymysql()

        try:
            cursor_class = None
            if self.as_dict:
                cursors_module = getattr(self._pymysql, "cursors", None)
                if cursors_module:
                    cursor_class = getattr(cursors_module, "DictCursor", None)

            connect_fn = getattr(self._pymysql, "connect")
            kwargs: dict[str, Any] = {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "database": self.database,
                "connect_timeout": self.connect_timeout,
                "read_timeout": self.read_timeout,
                "write_timeout": self.read_timeout,
                "charset": "utf8mb4",
            }
            if cursor_class:
                kwargs["cursorclass"] = cursor_class

            self._conn = connect_fn(**kwargs)
            return self._conn
        except Exception as e:
            error_str = str(e)
            if "Access denied" in error_str:
                raise MysqlConnectionError(
                    "Access denied - check username and password"
                ) from e
            if "Unknown database" in error_str:
                raise MysqlConnectionError(
                    f"Unknown database '{self.database}' - check database name"
                ) from e
            if "Can't connect" in error_str:
                raise MysqlConnectionError(
                    f"Cannot connect to MySQL server at {self.host}:{self.port}"
                ) from e
            raise MysqlConnectionError(f"Connection error: {error_str}") from e

    def close(self) -> None:
        """Close the connection if open."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> Any:
        return self.connect()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def mysql_connect(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    connect_timeout: int = 10,
    read_timeout: int = 10,
    as_dict: bool = True,
) -> MysqlConnection:
    """
    Create a MySQL/MariaDB connection context manager.

    Args:
        host: Database server hostname or IP.
        port: Database server port (default 3306).
        user: Username for authentication.
        password: Password for authentication.
        database: Database name to connect to.
        connect_timeout: Connection timeout in seconds.
        read_timeout: Query read timeout in seconds.
        as_dict: Whether to return rows as dictionaries.

    Returns:
        MysqlConnection context manager.

    Example:
        with mysql_connect(host, port, user, password, database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table LIMIT 10")
            rows = cursor.fetchall()
    """
    return MysqlConnection(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        as_dict=as_dict,
    )
