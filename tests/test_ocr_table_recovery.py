"""Synthetic regression tests for typed OCR table recovery."""

from copy import deepcopy
from typing import Any

import pytest

from ragtime.core.ocr_table_recovery import (
    OCR_TABLE_RECOVERY_PROMPT,
    OCR_TABLE_RECOVERY_SCHEMA,
    OLLAMA_OCR_RECOVERY_REPEAT_PENALTY,
    OllamaOcrIncompleteError,
    recover_ocr_table_blocks,
    validate_ocr_recovery_response,
)
from ragtime.core.tokenization import count_tokens
from ragtime.indexer.chunking import chunk_semantic_segments, rechunk_oversized_content


def _ocr(
    table: str = "Caption\n| Clause | Value |\n| --- | --- |\n| 12.7.1.9 | 16.2 |\n| 12.7.1.9 | 16.3.1 |\n| prose | |\n| Synthetic note. |",
) -> dict[str, Any]:
    return {
        "extracted_text": [
            {"type": "Title", "content": "Caption — synthetic"},
            {"type": "Table", "content": table},
            {"type": "NarrativeText", "content": "Synthetic prose remains."},
        ],
        "image_type": "scan",
        "subject": "synthetic",
        "description": "synthetic description",
        "objects": ["table"],
        "setting": "test",
        "tags": ["synthetic"],
    }


def _payload(rows: list[dict[str, object]] | None = None) -> dict[str, Any]:
    return {
        "tables": [
            {
                "caption": "Caption",
                "headers": ["Clause", "Value"],
                "rows": rows
                or [
                    {"kind": "section", "cells": ["Normal operation"]},
                    {"kind": "data", "cells": ["12.7.1.9", "16.2"]},
                    {"kind": "data", "cells": ["12.7.1.9", "16.3.1"]},
                    {"kind": "data", "cells": ["prose", ""]},
                ],
                "footnotes": ["Synthetic note."],
            }
        ]
    }


def test_recovers_sections_blanks_digits_duplicates_and_preserves_input() -> None:
    ocr = _ocr()
    original = deepcopy(ocr)

    result = recover_ocr_table_blocks(ocr, _payload(), include_classification=True)

    assert result == {
        **original,
        "extracted_text": [
            original["extracted_text"][0],
            {
                "type": "Table",
                "content": "Caption\nNormal operation\nSynthetic note.\n| Clause | Value |\n| --- | --- |\n| 12.7.1.9 | 16.2 |\n| 12.7.1.9 | 16.3.1 |\n| prose |  |",
            },
            original["extracted_text"][2],
        ],
    }
    assert ocr == original
    assert result is not ocr
    result["extracted_text"][0]["content"] = "mutated"
    assert ocr == original


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tables": []},
        {"tables": [{"caption": "Caption"}]},
        {"tables": [{"caption": "Caption", "headers": ["A", "B"], "rows": [], "footnotes": [], "extra": "no"}]},
        {"tables": [{"caption": "Caption", "headers": ["A"], "rows": [], "footnotes": []}]},
    ],
)
def test_rejects_missing_or_extra_recovery_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), payload, include_classification=True)


@pytest.mark.parametrize(
    "rows",
    [
        [{"kind": "data", "cells": ["one"]}],
        [{"kind": "data", "cells": [None, "two"]}],
        [{"kind": "data", "cells": ["", ""]}],
        [{"kind": "unknown", "cells": ["one", "two"]}],
        [{"kind": "section", "cells": ["orphan"]}],
        [{"kind": "section", "cells": ["one"]}, {"kind": "section", "cells": ["two"]}, {"kind": "data", "cells": ["prose", ""]}],
    ],
)
def test_rejects_malformed_or_unsupported_rows(rows: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), _payload(rows), include_classification=True)


def test_rejects_duplicate_notes_caption_mismatch_and_ambiguous_caption() -> None:
    duplicate_notes = _payload()
    duplicate_notes["tables"][0]["footnotes"] = ["Note", "Note"]
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), duplicate_notes, include_classification=True)
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), {**_payload(), "tables": [{**_payload()["tables"][0], "caption": "Wrong"}]}, include_classification=True)
    ambiguous = _ocr()
    ambiguous["extracted_text"].insert(0, {"type": "Title", "content": "Caption duplicate"})
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(ambiguous, _payload(), include_classification=True)
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), {**_payload(), "tables": [{**_payload()["tables"][0], "caption": "  \n"}]}, include_classification=True)


@pytest.mark.parametrize(
    "rows",
    [
        [{"kind": "data", "cells": ["12.7.1.9", "16.3.1"]}, {"kind": "data", "cells": ["12.7.1.9", "16.2"]}, {"kind": "data", "cells": ["prose", ""]}],
        [{"kind": "data", "cells": ["12.7.1.9", "16.2"]}, {"kind": "data", "cells": ["prose", ""]}],
        [{"kind": "data", "cells": ["12.7.1.9", "16.2"]}, {"kind": "data", "cells": ["12.7.1.9", "changed"]}, {"kind": "data", "cells": ["prose", ""]}],
    ],
)
def test_rejects_shifted_missing_or_changed_source_cells(rows: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), _payload(rows), include_classification=True)


def test_accepts_legitimate_repeated_source_rows_and_escaped_pipes() -> None:
    table = "Caption\n| A | B |\n| --- | --- |\n| x\\|y | 1 |\n| x\\|y | 1 |"
    rows = [{"kind": "data", "cells": ["x|y", "1"]}, {"kind": "data", "cells": ["x|y", "1"]}]
    result = recover_ocr_table_blocks(
        _ocr(table), {"tables": [{"caption": "Caption", "headers": ["A", "B"], "rows": rows, "footnotes": []}]}, include_classification=True
    )
    assert result["extracted_text"][1]["content"].endswith("| x\\|y | 1 |\n| x\\|y | 1 |")


def test_rejects_multiple_source_or_recovered_tables_and_bad_classification() -> None:
    source = _ocr()
    source["extracted_text"].append({"type": "Table", "content": "| A | B |\n| --- | --- |\n| a | b |"})
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, _payload(), include_classification=True)
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), {"tables": [_payload()["tables"][0], _payload()["tables"][0]]}, include_classification=True)
    broken = _ocr()
    broken["tags"] = [1]
    with pytest.raises(ValueError, match="Invalid OCR table recovery input"):
        recover_ocr_table_blocks(broken, _payload(), include_classification=True)
    with pytest.raises(ValueError, match="Invalid OCR table recovery input"):
        recover_ocr_table_blocks({**_ocr(), "unexpected": "field"}, _payload(), include_classification=True)
    with pytest.raises(ValueError, match="Invalid OCR table recovery input"):
        recover_ocr_table_blocks({"extracted_text": _ocr()["extracted_text"], "unexpected": "field"}, _payload(), include_classification=False)


def test_public_contract_is_safe_and_strict() -> None:
    assert OLLAMA_OCR_RECOVERY_REPEAT_PENALTY == 1.05
    assert OCR_TABLE_RECOVERY_PROMPT.startswith("Extract only the tables visible")
    assert OCR_TABLE_RECOVERY_SCHEMA["additionalProperties"] is False
    error = OllamaOcrIncompleteError("provider said secret", eval_count=True, prompt_eval_count=4)
    assert error.reason == "incomplete"
    assert error.eval_count is None
    assert str(error) == "Ollama OCR response was incomplete. reason=incomplete (prompt_eval_count=4)"


def test_validates_http_boundary_source_contract_before_recovery() -> None:
    validate_ocr_recovery_response(_ocr(), include_classification=True)

    missing_classification = _ocr()
    missing_classification.pop("subject")
    malformed_block = _ocr()
    malformed_block["extracted_text"] = [{"type": "Unknown", "content": "synthetic"}]
    no_table = _ocr()
    no_table["extracted_text"] = [{"type": "Title", "content": "Caption"}]
    multiple_tables = _ocr()
    multiple_tables["extracted_text"].append({"type": "Table", "content": "| A | B |\n| --- | --- |\n| one | two |"})

    for invalid_source in (missing_classification, malformed_block, no_table, multiple_tables):
        with pytest.raises(ValueError, match="Invalid OCR table recovery input"):
            validate_ocr_recovery_response(invalid_source, include_classification=True)


def test_recovered_section_context_survives_semantic_and_embedding_rechunking() -> None:
    source = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n| C | 3 |\n| D | 4 |\n| Synthetic note. |")
    payload: dict[str, Any] = {
        "tables": [
            {
                "caption": "Caption",
                "headers": ["Clause", "Value"],
                "rows": [
                    {"kind": "section", "cells": ["First section"]},
                    {"kind": "data", "cells": ["A", "1"]},
                    {"kind": "data", "cells": ["B", "2"]},
                    {"kind": "section", "cells": ["Second section"]},
                    {"kind": "data", "cells": ["C", "3"]},
                    {"kind": "data", "cells": ["D", "4"]},
                ],
                "footnotes": ["Synthetic note."],
            }
        ]
    }
    recovered = recover_ocr_table_blocks(source, payload, include_classification=True)
    tables = [block["content"] for block in recovered["extracted_text"] if block["type"] == "Table"]
    header_with_row = "\n".join(tables[0].splitlines()[:6])
    budget = count_tokens(header_with_row)
    chunks = chunk_semantic_segments([("table", table) for table in tables], budget, 0, {"source": "synthetic.png"}, use_tokens=True)

    assert len(chunks) == 4
    assert all("Caption\n" in chunk.page_content and "Synthetic note." in chunk.page_content for chunk in chunks)
    assert any("First section" in chunk.page_content for chunk in chunks)
    assert any("Second section" in chunk.page_content for chunk in chunks)
    for table in tables:
        rechunked = rechunk_oversized_content(table, budget, metadata={"source": "synthetic.png", "segment_type": "table"})
        assert all("Caption\n" in chunk.page_content and "Synthetic note." in chunk.page_content for chunk in rechunked)
        with pytest.raises(ValueError, match="header is too large"):
            rechunk_oversized_content(table, 1, metadata={"source": "synthetic.png", "segment_type": "table"})


def test_recovers_ragged_group_headers_without_dropping_text_only_data_rows() -> None:
    source = _ocr(
        "Synthetic grid\n"
        "| | Group one | Group two | Group three |\n"
        "| --- | --- | --- | --- |\n"
        "| | A | B | C | D | E | F | G | H | I |\n"
        "| Type | | | | | | | | | |\n"
        "| Test label | 10.1 | 10.2 | 10.3 | 10.4 | 10.5 | 10.6 | 10.7 | 10.8 | 10.9 |\n"
        "| | | | | | | | | | |\n"
        "| | | | | | | | | | |\n"
        "| prose-only row | | | | | | | | | |\n"
        "| Test alpha | | 20.2 | | | | | | | |\n"
        "| Synthetic note. |"
    )
    source["extracted_text"][0]["content"] = "Synthetic grid — synthetic"
    rows = [
        {"kind": "section", "cells": ["Synthetic section"]},
        {"kind": "data", "cells": ["Test label", "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.8", "10.9"]},
        {"kind": "data", "cells": ["prose-only row", "", "", "", "", "", "", "", "", ""]},
        {"kind": "data", "cells": ["Test alpha", "", "20.2", "", "", "", "", "", "", ""]},
    ]
    payload: dict[str, Any] = {
        "tables": [
            {
                "caption": "Synthetic grid",
                "headers": [
                    "Type",
                    "Group one\nA",
                    "Group one\nB",
                    "Group one\nC",
                    "Group two\nD",
                    "Group two\nE",
                    "Group two\nF",
                    "Group three\nG",
                    "Group three\nH",
                    "Group three\nI",
                ],
                "rows": rows,
                "footnotes": ["Synthetic note."],
            }
        ]
    }

    recovered = recover_ocr_table_blocks(source, payload, include_classification=True)

    assert "| prose-only row |  |  |  |  |  |  |  |  |  |" in recovered["extracted_text"][1]["content"]
    missing_text_row = deepcopy(payload)
    missing_text_row["tables"][0]["rows"].pop(2)
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, missing_text_row, include_classification=True)


@pytest.mark.parametrize(
    "insertion",
    [
        "| Mandatory condition | retain this |  |  |  |  |  |  |  |  |\n",
        "| 9.9 | leading numeric row |\n",
    ],
)
def test_rejects_unknown_leading_source_rows_in_ragged_layout(insertion: str) -> None:
    source = _ocr(
        "Synthetic grid\n"
        "| | Group one | Group two | Group three |\n"
        "| --- | --- | --- | --- |\n"
        "| | A | B | C | D | E | F | G | H | I |\n"
        "| Type | | | | | | | | | |\n"
        f"{insertion}"
        "| Test label | 10.1 | 10.2 | 10.3 | 10.4 | 10.5 | 10.6 | 10.7 | 10.8 | 10.9 |\n"
        "| Synthetic note. |"
    )
    source["extracted_text"][0]["content"] = "Synthetic grid — synthetic"
    payload = {
        "tables": [
            {
                "caption": "Synthetic grid",
                "headers": ["Type", *[f"Group one\n{letter}" for letter in "ABCDEFGHI"]],
                "rows": [{"kind": "data", "cells": ["Test label", "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.8", "10.9"]}],
                "footnotes": ["Synthetic note."],
            }
        ]
    }

    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, payload, include_classification=True)


def test_rejects_unknown_preamble_before_source_table() -> None:
    source = _ocr(
        "Unexpected preamble\nCaption\n| Clause | Value |\n| --- | --- |\n| 12.7.1.9 | 16.2 |\n| 12.7.1.9 | 16.3.1 |\n| prose | |\n| Synthetic note. |"
    )

    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, _payload(), include_classification=True)


def test_rejects_swapped_normal_and_reassigned_ragged_headers() -> None:
    swapped = _payload()
    swapped["tables"][0]["headers"] = ["Value", "Clause"]
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), swapped, include_classification=True)

    source = _ocr("Synthetic grid\n| | North | South |\n| --- | --- |\n| | A | B | C | D |\n| Type | | | | |\n| Test | 1 | 2 | 3 | 4 |\n| Synthetic note. |")
    source["extracted_text"][0]["content"] = "Synthetic grid — synthetic"
    payload = {
        "tables": [
            {
                "caption": "Synthetic grid",
                "headers": ["Type", "South\nA", "South\nB", "North\nC", "North\nD"],
                "rows": [{"kind": "data", "cells": ["Test", "1", "2", "3", "4"]}],
                "footnotes": ["Synthetic note."],
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, payload, include_classification=True)


def test_preserves_header_looking_data_before_data_and_rejects_its_omission() -> None:
    source = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| Clause | Value |\n| actual | 1 |\n| Synthetic note. |")
    rows = [{"kind": "data", "cells": ["Clause", "Value"]}, {"kind": "data", "cells": ["actual", "1"]}]
    payload = {"tables": [{"caption": "Caption", "headers": ["Clause", "Value"], "rows": rows, "footnotes": ["Synthetic note."]}]}
    assert recover_ocr_table_blocks(source, payload, include_classification=True)
    payload["tables"][0]["rows"] = rows[1:]
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, payload, include_classification=True)


def test_rejects_caption_identifier_prefix_unverified_tail_and_moved_known_section() -> None:
    numeric_prefix = _ocr("Table 12\n| Clause | Value |\n| --- | --- |\n| A | 1 |")
    numeric_prefix["extracted_text"][0]["content"] = "Table 123 — synthetic"
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(
            numeric_prefix,
            {"tables": [{"caption": "Table 12", "headers": ["Clause", "Value"], "rows": [{"kind": "data", "cells": ["A", "1"]}], "footnotes": []}]},
            include_classification=True,
        )

    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(
            _ocr(),
            {
                "tables": [
                    {
                        "caption": "Caption\nInvented tail",
                        "headers": ["Clause", "Value"],
                        "rows": _payload()["tables"][0]["rows"],
                        "footnotes": ["Synthetic note."],
                    }
                ]
            },
            include_classification=True,
        )

    section_source = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| A | 1 |\n| Known section |\n| B | 2 |\n| Synthetic note. |")
    moved_section = {
        "tables": [
            {
                "caption": "Caption",
                "headers": ["Clause", "Value"],
                "rows": [{"kind": "section", "cells": ["Known section"]}, {"kind": "data", "cells": ["A", "1"]}, {"kind": "data", "cells": ["B", "2"]}],
                "footnotes": ["Synthetic note."],
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(section_source, moved_section, include_classification=True)


def test_rejects_invented_group_header_and_post_data_header_like_row() -> None:
    source = _ocr("Synthetic grid\n| | Invented | South |\n| --- | --- |\n| | A | B | C | D |\n| Type | | | | |\n| Test | 1 | 2 | 3 | 4 |\n| Synthetic note. |")
    source["extracted_text"][0]["content"] = "Synthetic grid — synthetic"
    payload = {
        "tables": [
            {
                "caption": "Synthetic grid",
                "headers": ["Type", "North\nA", "North\nB", "South\nC", "South\nD"],
                "rows": [{"kind": "data", "cells": ["Test", "1", "2", "3", "4"]}],
                "footnotes": ["Synthetic note."],
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, payload, include_classification=True)

    post_data_header = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| actual | 1 |\n| Clause | Value |\n| Synthetic note. |")
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(
            post_data_header,
            {
                "tables": [
                    {
                        "caption": "Caption",
                        "headers": ["Clause", "Value"],
                        "rows": [{"kind": "data", "cells": ["actual", "1"]}],
                        "footnotes": ["Synthetic note."],
                    }
                ]
            },
            include_classification=True,
        )


def test_normalizes_known_sections_notes_and_markdown_break_cells() -> None:
    source = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| A<br>B | 1 |\n| Known   section |\n| C | 2 |\n| Synthetic   note. |")
    rows = [
        {"kind": "data", "cells": ["A\nB", "1"]},
        {"kind": "section", "cells": ["Known section"]},
        {"kind": "data", "cells": ["C", "2"]},
    ]
    payload = {"tables": [{"caption": "Caption", "headers": ["Clause", "Value"], "rows": rows, "footnotes": ["Synthetic note."]}]}

    assert recover_ocr_table_blocks(source, payload, include_classification=True)


def test_rejects_ungrounded_header_parts_and_promoted_leading_data() -> None:
    source = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| A | 1 |")
    extra_normal_part = {
        "tables": [{"caption": "Caption", "headers": ["Clause", "Max\nValue"], "rows": [{"kind": "data", "cells": ["A", "1"]}], "footnotes": []}]
    }
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(source, extra_normal_part, include_classification=True)

    ragged_source = _ocr("Grid\n| | North | South |\n| --- | --- |\n| | A | B |\n| Type | | |\n| Test | 1 | 2 |")
    ragged_source["extracted_text"][0]["content"] = "Grid — synthetic"
    invented_subheader = {
        "tables": [
            {
                "caption": "Grid",
                "headers": ["Type", "North\nInvented\nA", "South\nB"],
                "rows": [{"kind": "data", "cells": ["Test", "1", "2"]}],
                "footnotes": [],
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(ragged_source, invented_subheader, include_classification=True)

    numeric_leading_data = _ocr("Caption\n| Type | V |\n| --- | --- |\n| | 16.2 |")
    numeric_header = {"tables": [{"caption": "Caption", "headers": ["Type", "V\n16.2"], "rows": [], "footnotes": []}]}
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(numeric_leading_data, numeric_header, include_classification=True)

    yes_leading_data = _ocr("Caption\n| Decision | Value |\n| --- | --- |\n| Yes | |")
    yes_header = {"tables": [{"caption": "Caption", "headers": ["Decision\nYes", "Value"], "rows": [], "footnotes": []}]}
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(yes_leading_data, yes_header, include_classification=True)


def test_rejects_dotted_identifier_prefixes_tails_and_normalized_duplicate_notes() -> None:
    dotted_prefix = _ocr("Table12\n| Clause | Value |\n| --- | --- |\n| A | 1 |")
    dotted_prefix["extracted_text"][0]["content"] = "Table12.7 — synthetic"
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(
            dotted_prefix,
            {"tables": [{"caption": "Table12", "headers": ["Clause", "Value"], "rows": [{"kind": "data", "cells": ["A", "1"]}], "footnotes": []}]},
            include_classification=True,
        )

    dotted_tail = _ocr("Caption\n| Clause | Value |\n| --- | --- |\n| A | 1 |")
    dotted_tail["extracted_text"].insert(1, {"type": "NarrativeText", "content": "SeeClause12.7.1.1"})
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(
            dotted_tail,
            {
                "tables": [
                    {"caption": "Caption\nSeeClause12.7", "headers": ["Clause", "Value"], "rows": [{"kind": "data", "cells": ["A", "1"]}], "footnotes": []}
                ]
            },
            include_classification=True,
        )

    duplicate_notes = _payload()
    duplicate_notes["tables"][0]["footnotes"] = ["Synthetic note.", "Synthetic  note."]
    with pytest.raises(ValueError, match="Invalid recovered OCR table"):
        recover_ocr_table_blocks(_ocr(), duplicate_notes, include_classification=True)
