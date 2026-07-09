"""Shared helpers for normalizing chart dataset field mappings.

Used by chat chart tooling and live visualization refresh to turn a flexible
result_mapping / visualization_mapping payload into a list of ``{field, label}``
dictionaries.
"""

from __future__ import annotations

from typing import Any


def normalize_dataset_mappings(mapping: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize a chart mapping payload into dataset field/label pairs.

    Supports ``datasets`` (list of mapping objects), ``dataset_fields`` /
    ``datasetFields`` (dict of label -> field or list of fields), and a
    root-level ``data_field`` / ``dataset`` / ``y_field`` / ``value_field``
    fallback used only when no explicit datasets or fields are provided.

    Returns a list of ``{"field": ..., "label": ...}`` dictionaries.
    """
    raw_datasets = mapping.get("datasets")
    normalized: list[dict[str, str]] = []
    root_field = str(
        mapping.get("data_field")
        or mapping.get("dataField")
        or mapping.get("dataset")
        or mapping.get("dataset_field")
        or mapping.get("datasetField")
        or mapping.get("y_field")
        or mapping.get("yField")
        or mapping.get("value_field")
        or mapping.get("valueField")
        or ""
    ).strip()

    root_mapping: dict[str, str] | None = None
    if root_field:
        root_mapping = {
            "field": root_field,
            "label": str(mapping.get("dataset_label") or mapping.get("label") or root_field),
        }

    if isinstance(raw_datasets, list):
        for entry in raw_datasets:
            if not isinstance(entry, dict):
                continue
            field = str(
                entry.get("data_field")
                or entry.get("dataField")
                or entry.get("dataset")
                or entry.get("dataset_field")
                or entry.get("datasetField")
                or entry.get("field")
                or entry.get("y_field")
                or entry.get("yField")
                or entry.get("value_field")
                or entry.get("valueField")
                or ""
            ).strip()
            if field:
                normalized.append({"field": field, "label": str(entry.get("label") or field)})

    raw_fields = mapping.get("dataset_fields") or mapping.get("datasetFields")
    if isinstance(raw_fields, dict):
        for label, field_value in raw_fields.items():
            field = str(field_value or "").strip()
            if field:
                normalized.append({"field": field, "label": str(label or field)})
    elif isinstance(raw_fields, list):
        for field_value in raw_fields:
            field = str(field_value or "").strip()
            if field:
                normalized.append({"field": field, "label": field})

    if not normalized and root_mapping:
        normalized.append(root_mapping)

    return normalized
