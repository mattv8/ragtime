"""Validation and deterministic assembly for typed OCR table recovery."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

OLLAMA_OCR_RECOVERY_REPEAT_PENALTY = 1.05

OCR_TABLE_RECOVERY_PROMPT = (
    "Extract only the tables visible in this image. Return each visible table once. "
    "Preserve the exact source grid and all visible cell text. Headers: one entry per physical column including the leftmost row-label column; combine merged group and subheader labels into each column header. "
    "Each row must have kind data or section. A data row has exactly as many cells as headers, including genuinely empty cells as empty strings. "
    "A section row is only a full-width source label spanning the whole table: use kind section and exactly one cell containing its exact text. "
    "Preserve the order of all section and data rows. Put all footnotes ONLY in footnotes, never also in rows. Include ALL visible rows and footnotes, but no prose outside the table. "
    "Never invent or repeat tables or rows. Stop at the visible table boundary."
)

OCR_TABLE_RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "caption": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string", "enum": ["data", "section"]},
                                "cells": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["kind", "cells"],
                        },
                    },
                    "footnotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["caption", "headers", "rows", "footnotes"],
            },
        }
    },
    "required": ["tables"],
}


class OllamaOcrIncompleteError(RuntimeError):
    """A sanitized incomplete Ollama OCR response signal."""

    def __init__(self, reason: str, eval_count: int | None = None, prompt_eval_count: int | None = None) -> None:
        self.reason = reason if reason in {"length", "incomplete", "error", "load", "malformed"} else "incomplete"
        self.eval_count = _valid_counter(eval_count)
        self.prompt_eval_count = _valid_counter(prompt_eval_count)
        super().__init__(str(self))

    def __str__(self) -> str:
        counters = []
        if self.eval_count is not None:
            counters.append(f"eval_count={self.eval_count}")
        if self.prompt_eval_count is not None:
            counters.append(f"prompt_eval_count={self.prompt_eval_count}")
        suffix = f" ({', '.join(counters)})" if counters else ""
        return f"Ollama OCR response was incomplete. reason={self.reason}{suffix}"


def _valid_counter(value: int | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def validate_ocr_recovery_response(ocr: dict[str, Any], *, include_classification: bool) -> None:
    """Validate a structured OCR response before starting table recovery work."""
    blocks = _validated_ocr_blocks(ocr, include_classification)
    try:
        _only_source_table(blocks)
    except ValueError:
        raise ValueError("Invalid OCR table recovery input.") from None


def recover_ocr_table_blocks(ocr: dict[str, Any], table_payload: dict[str, Any], *, include_classification: bool) -> dict[str, Any]:
    """Validate and replace one OCR table block with typed recovered table blocks."""
    source_blocks = _validated_ocr_blocks(ocr, include_classification)
    try:
        table = _validated_recovered_table(table_payload)
        table_index, source_table = _only_source_table(source_blocks)
        _validate_caption(table["caption"], source_blocks, table_index)
        recovered_rows = [row["cells"] for row in table["rows"] if row["kind"] == "data"]
        source_rows = _source_data_rows(
            source_table["content"], table["caption"], table["headers"], recovered_rows, table["footnotes"], _section_positions(table["rows"])
        )
        if _normalized_rows(source_rows) != _normalized_rows(recovered_rows):
            raise ValueError
        replacements = _render_table_blocks(table)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid recovered OCR table.") from None

    result = deepcopy(ocr)
    result_blocks = result["extracted_text"]
    result["extracted_text"] = [*result_blocks[:table_index], *replacements, *result_blocks[table_index + 1 :]]
    return result


_BLOCK_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",
    "Code",
    "Header",
    "Footer",
    "FigureCaption",
    "Formula",
    "Handwritten",
    "Annotation",
    "Uncategorized",
}
_CLASSIFICATION_STRING_FIELDS = ("image_type", "subject", "description", "setting")
_CLASSIFICATION_LIST_FIELDS = ("objects", "tags")
_CLASSIFICATION_FIELDS = {*_CLASSIFICATION_STRING_FIELDS, *_CLASSIFICATION_LIST_FIELDS}
_TABLE_FIELDS = {"caption", "headers", "rows", "footnotes"}
_ROW_FIELDS = {"kind", "cells"}
_WHITESPACE = re.compile(r"\s+")
_MARKDOWN_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_DIGIT = re.compile(r"\d")


def _validated_ocr_blocks(ocr: dict[str, Any], include_classification: bool) -> list[dict[str, str]]:
    if not isinstance(ocr, dict):
        raise ValueError("Invalid OCR table recovery input.")
    expected_fields = {"extracted_text", *_CLASSIFICATION_FIELDS} if include_classification else {"extracted_text"}
    if set(ocr) != expected_fields:
        raise ValueError("Invalid OCR table recovery input.")
    blocks = ocr.get("extracted_text")
    if not isinstance(blocks, list) or any(not _is_valid_block(block) for block in blocks):
        raise ValueError("Invalid OCR table recovery input.")
    if include_classification and not _has_valid_classification(ocr):
        raise ValueError("Invalid OCR table recovery input.")
    return blocks


def _is_valid_block(block: object) -> bool:
    return (
        isinstance(block, dict)
        and set(block) == {"type", "content"}
        and isinstance(block.get("type"), str)
        and block["type"] in _BLOCK_TYPES
        and isinstance(block.get("content"), str)
    )


def _has_valid_classification(ocr: dict[str, Any]) -> bool:
    return all(isinstance(ocr.get(field), str) for field in _CLASSIFICATION_STRING_FIELDS) and all(
        isinstance(ocr.get(field), list) and all(isinstance(value, str) for value in ocr[field]) for field in _CLASSIFICATION_LIST_FIELDS
    )


def _validated_recovered_table(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"tables"}:
        raise ValueError
    tables = payload.get("tables")
    if not isinstance(tables, list) or len(tables) != 1:
        raise ValueError
    table = tables[0]
    if not isinstance(table, dict) or set(table) != _TABLE_FIELDS:
        raise ValueError
    caption = table["caption"]
    headers = table["headers"]
    rows = table["rows"]
    footnotes = table["footnotes"]
    if not isinstance(caption, str) or not _is_string_list(headers) or len(headers) < 2 or not _is_string_list(footnotes):
        raise ValueError
    if len({_normalize_space(note) for note in footnotes}) != len(footnotes) or not isinstance(rows, list):
        raise ValueError
    validated_rows = [_validated_row(row, len(headers)) for row in rows]
    _validate_section_order(validated_rows)
    return {"caption": caption, "headers": headers, "rows": validated_rows, "footnotes": footnotes}


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validated_row(row: object, header_count: int) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != _ROW_FIELDS or row.get("kind") not in {"data", "section"} or not _is_string_list(row.get("cells")):
        raise ValueError
    cells = row["cells"]
    if row["kind"] == "data" and (len(cells) != header_count or not any(cell.strip() for cell in cells)):
        raise ValueError
    if row["kind"] == "section" and (len(cells) != 1 or not cells[0].strip()):
        raise ValueError
    return {"kind": row["kind"], "cells": cells}


def _validate_section_order(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row["kind"] == "section" and (index == len(rows) - 1 or rows[index + 1]["kind"] != "data"):
            raise ValueError


def _only_source_table(blocks: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    tables = [(index, block) for index, block in enumerate(blocks) if block["type"] == "Table"]
    if len(tables) != 1:
        raise ValueError
    return tables[0]


def _validate_caption(caption: str, blocks: list[dict[str, str]], table_index: int) -> None:
    first_line = caption.splitlines()[0] if caption else ""
    if not first_line.strip():
        raise ValueError
    normalized_caption = _normalize_space(first_line)
    matches = [
        index
        for index, block in enumerate(blocks)
        if index != table_index
        and block["type"] in {"Title", "FigureCaption"}
        and _caption_prefix_matches(_normalize_space(block["content"]), normalized_caption)
    ]
    if len(matches) != 1:
        raise ValueError
    caption_context = blocks[matches[0] : table_index]
    if any(not _caption_line_is_grounded(line, caption_context) for line in caption.splitlines()[1:] if line.strip()):
        raise ValueError


def _caption_prefix_matches(content: str, caption: str) -> bool:
    if not content.startswith(caption):
        return False
    return _has_identifier_boundary(content, 0, len(caption))


def _caption_line_is_grounded(line: str, context: list[dict[str, str]]) -> bool:
    normalized_line = _normalize_space(line)
    return any(_grounded_text_contains(_normalize_space(block["content"]), normalized_line) for block in context)


def _grounded_text_contains(content: str, value: str) -> bool:
    start = content.find(value)
    while start >= 0:
        end = start + len(value)
        if _has_identifier_boundary(content, start, end):
            return True
        start = content.find(value, start + 1)
    return False


def _has_identifier_boundary(content: str, start: int, end: int) -> bool:
    return _is_identifier_boundary(content, start, before=True) and _is_identifier_boundary(content, end, before=False)


def _is_identifier_boundary(content: str, index: int, *, before: bool) -> bool:
    adjacent_index = index - 1 if before else index
    if adjacent_index < 0 or adjacent_index >= len(content):
        return True
    adjacent = content[adjacent_index]
    if adjacent.isspace():
        return True
    if adjacent.isalnum():
        return False
    continuation_index = adjacent_index - 1 if before else adjacent_index + 1
    return continuation_index < 0 or continuation_index >= len(content) or not content[continuation_index].isalnum()


def _source_data_rows(
    content: str,
    caption: str,
    headers: list[str],
    recovered_rows: list[list[str]],
    footnotes: list[str],
    section_positions: list[tuple[str, int]],
) -> list[list[str]]:
    lines = content.splitlines()
    separator_index = next((index for index, line in enumerate(lines) if _is_separator(_parse_markdown_row(line))), None)
    if separator_index is None:
        raise ValueError
    header = _parse_markdown_row(lines[separator_index - 1]) if separator_index else None
    if header is None or len(header) < 2:
        raise ValueError
    _validate_preamble(lines[: separator_index - 1], caption)
    data_width = len(headers)
    header_parts = _validate_source_header(header, headers)
    rows: list[list[str]] = []
    seen_footnotes: list[str] = []
    seen_sections: list[tuple[str, int]] = []
    for line in lines[separator_index + 1 :]:
        row = _parse_markdown_row(line)
        if row is None:
            raise ValueError
        if len(row) == 1:
            _record_context_row(row[0], footnotes, section_positions, len(rows), seen_footnotes, seen_sections)
        elif len(row) == data_width and any(cell.strip() for cell in row):
            if len(rows) < len(recovered_rows) and _normalized_rows([row]) == _normalized_rows([recovered_rows[len(rows)]]):
                rows.append(row)
            elif not rows and _is_recovered_header_row(row, headers):
                additions = _full_header_parts(row, headers)
                if not _header_parts_can_extend(header_parts, additions, headers):
                    raise ValueError
                _append_header_parts(header_parts, additions)
            else:
                raise ValueError
        elif len(row) == data_width:
            continue
        else:
            raise ValueError
    if _normalized_rows(rows) != _normalized_rows(recovered_rows) or seen_footnotes != [_normalize_space(note) for note in footnotes]:
        raise ValueError
    if not _header_parts_match(header_parts, headers):
        raise ValueError
    source_labels = {label for label, _position in seen_sections}
    if seen_sections != [position for position in section_positions if position[0] in source_labels]:
        raise ValueError
    return rows


def _validate_preamble(lines: list[str], caption: str) -> None:
    caption_lines = {_normalize_space(line) for line in caption.splitlines() if line.strip()}
    if any(line.strip() and _normalize_space(line) not in caption_lines for line in lines):
        raise ValueError


def _is_recovered_header_row(row: list[str], headers: list[str]) -> bool:
    parts = _full_header_parts(row, headers)
    if parts is None or any(_DIGIT.search(cell) for cell in row if cell.strip()):
        return False
    is_full_header = all(_normalize_space(cell) == _normalize_space(header) for cell, header in zip(row, headers, strict=True))
    return not is_full_header


def _validate_source_header(source_header: list[str], headers: list[str]) -> list[list[str]]:
    if len(source_header) == len(headers):
        parts = _full_header_parts(source_header, headers)
        if parts is None:
            raise ValueError
        return parts
    if len(source_header) < len(headers):
        parts = _ragged_group_header_parts(source_header, headers)
        if parts is not None:
            return parts
    raise ValueError


def _full_header_parts(row: list[str], headers: list[str]) -> list[list[str]] | None:
    parts: list[list[str]] = []
    for index, (cell, header) in enumerate(zip(row, headers, strict=True)):
        if not cell.strip():
            parts.append([])
        elif index == 0 and _normalize_space(cell) != _header_parts(header)[0]:
            return None
        elif _normalize_space(cell) in _header_parts(header):
            parts.append([_normalize_space(cell)])
        else:
            return None
    return parts


def _ragged_group_header_parts(source_header: list[str], headers: list[str]) -> list[list[str]] | None:
    if source_header[0].strip() and _normalize_space(source_header[0]) != _normalize_space(headers[0]):
        return None
    group_runs = _header_group_runs(headers[1:])
    source_labels = [_normalize_space(cell) for cell in source_header[1:]]
    if not all(source_labels) or len(source_labels) != len(group_runs) or source_labels != [label for label, _width in group_runs]:
        return None
    parts: list[list[str]] = [[] for _header in headers]
    column = 1
    for label, width in group_runs:
        for index in range(column, column + width):
            parts[index].append(label)
        column += width
    return parts


def _header_group_runs(headers: list[str]) -> list[tuple[str, int]]:
    groups: list[tuple[str, int]] = []
    for header in headers:
        parts = _header_parts(header)
        if not parts:
            raise ValueError
        group = parts[0]
        if not groups or groups[-1][0] != group:
            groups.append((group, 1))
        else:
            groups[-1] = (group, groups[-1][1] + 1)
    return groups


def _append_header_parts(current: list[list[str]], additions: list[list[str]] | None) -> None:
    if additions is None:
        raise ValueError
    for existing, new_parts in zip(current, additions, strict=True):
        existing.extend(new_parts)


def _header_parts_can_extend(current: list[list[str]], additions: list[list[str]] | None, headers: list[str]) -> bool:
    if additions is None:
        return False
    for existing, new_parts, header in zip(current, additions, headers, strict=True):
        expected = _header_parts(header)
        if expected[len(existing) : len(existing) + len(new_parts)] != new_parts:
            return False
    return True


def _header_parts_match(source_parts: list[list[str]], headers: list[str]) -> bool:
    return source_parts == [_header_parts(header) for header in headers]


def _header_parts(header: str) -> list[str]:
    return [_normalize_space(part) for part in header.splitlines() if part.strip()]


def _section_positions(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    positions: list[tuple[str, int]] = []
    data_count = 0
    for row in rows:
        if row["kind"] == "section":
            positions.append((_normalize_space(row["cells"][0]), data_count))
        else:
            data_count += 1
    return positions


def _record_context_row(
    value: str,
    footnotes: list[str],
    section_positions: list[tuple[str, int]],
    data_position: int,
    seen_footnotes: list[str],
    seen_sections: list[tuple[str, int]],
) -> None:
    normalized_value = _normalize_space(value)
    if any(label == normalized_value for label, _position in section_positions):
        seen_sections.append((normalized_value, data_position))
        return
    if normalized_value in {_normalize_space(note) for note in footnotes}:
        seen_footnotes.append(normalized_value)
        return
    raise ValueError


def _parse_markdown_row(line: str) -> list[str] | None:
    if not line.lstrip().startswith("|") or not line.rstrip().endswith("|"):
        return None
    cells: list[str] = []
    cell: list[str] = []
    escaped = False
    for character in line.strip()[1:-1]:
        if escaped:
            cell.append(character if character in {"|", "\\"} else f"\\{character}")
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(character)
    if escaped:
        return None
    cells.append("".join(cell).strip())
    return cells


def _is_separator(row: list[str] | None) -> bool:
    if row is None:
        return False
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def _normalized_rows(rows: list[list[str]]) -> list[tuple[str, ...]]:
    return [tuple(_normalize_cell(cell) for cell in row) for row in rows]


def _normalize_space(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _normalize_cell(value: str) -> str:
    return _normalize_space(_MARKDOWN_BREAK.sub("\n", value))


def _render_table_blocks(table: dict[str, Any]) -> list[dict[str, str]]:
    groups: list[tuple[str | None, list[list[str]]]] = []
    section: str | None = None
    rows: list[list[str]] = []
    for row in table["rows"]:
        if row["kind"] == "section":
            if rows:
                groups.append((section, rows))
                rows = []
            section = row["cells"][0]
        else:
            rows.append(row["cells"])
    if rows:
        groups.append((section, rows))
    return [{"type": "Table", "content": _render_group(table, section, rows)} for section, rows in groups]


def _render_group(table: dict[str, Any], section: str | None, rows: list[list[str]]) -> str:
    context = [table["caption"]] if table["caption"] else []
    if section:
        context.append(section)
    context.extend(table["footnotes"])
    header = _markdown_row(table["headers"])
    separator = _markdown_row(["---"] * len(table["headers"]))
    return "\n".join([*context, header, separator, *(_markdown_row(row) for row in rows)])


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>") for cell in cells) + " |"
