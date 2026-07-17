from typing import Any


def sql_quote_literal(value: Any) -> str:
    """Quote a scalar value for trusted raw SQL construction."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"
