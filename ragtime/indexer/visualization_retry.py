"""Retry and repair helpers for chat visualization tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import TABLE_METADATA_END, TABLE_METADATA_START
from ragtime.indexer.models import (
    Conversation,
    RetryVisualizationRequest,
    RetryVisualizationResponse,
    ToolType,
)
from ragtime.indexer.repository import repository
from ragtime.rag import rag
from ragtime.tools.chart import create_chart
from ragtime.tools.datatable import create_datatable

logger = get_logger(__name__)

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "pie",
    "doughnut",
    "scatter",
    "radar",
    "polarArea",
}
SOURCE_RERUN_TOOL_TYPES = {
    ToolType.POSTGRES,
    ToolType.MYSQL,
    ToolType.MSSQL,
    ToolType.INFLUXDB,
}
MAX_CONTEXT_EVENTS = 12
MAX_CONTEXT_CHARS = 24000
MAX_OUTPUT_CONTEXT_CHARS = 8000
MAX_TABLE_ROWS = 5000
MAX_TABLE_COLUMNS = 100
MAX_CHART_POINTS = 1000
MAX_CHART_DATASETS = 8


@dataclass
class VisualizationRetryContext:
    conversation: Conversation
    user_id: str | None
    selected_tool_ids: set[str]


@dataclass
class _ValidatedPayload:
    source_data: dict[str, Any]
    strategy: str


def _truncate_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_table_metadata(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, str) or not output.startswith(TABLE_METADATA_START):
        return None
    line_end = output.find("\n")
    metadata_search_end = len(output) if line_end == -1 else line_end
    end = output.rfind(
        TABLE_METADATA_END,
        len(TABLE_METADATA_START),
        metadata_search_end,
    )
    if end == -1:
        return None
    try:
        parsed = json.loads(output[len(TABLE_METADATA_START) : end])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_visualization_payload(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    if parsed.get("__datatable__") is True:
        config = parsed.get("config")
        if isinstance(config, dict):
            columns = config.get("columns")
            rows = config.get("data")
            column_names: list[str] = []
            if isinstance(columns, list):
                for column in columns:
                    if isinstance(column, dict):
                        column_names.append(str(column.get("title") or column.get("data") or ""))
                    else:
                        column_names.append(str(column))
            return {"columns": column_names, "rows": rows, "title": parsed.get("title")}

    if parsed.get("__chart__") is True:
        config = parsed.get("config")
        if isinstance(config, dict):
            data = config.get("data")
            chart_type = config.get("type") or parsed.get("chart_type") or "bar"
            if isinstance(data, dict):
                return {
                    "labels": data.get("labels"),
                    "datasets": data.get("datasets"),
                    "chart_type": chart_type,
                    "title": parsed.get("title"),
                }
    return None


def _iter_candidate_sources(request: RetryVisualizationRequest) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(request.source_data, dict) and request.source_data:
        yield "provided_source_data", request.source_data
    if isinstance(request.failed_tool_input, dict) and request.failed_tool_input:
        yield "failed_tool_input", request.failed_tool_input

    failed_output_payload = _extract_table_metadata(request.failed_tool_output) or _extract_visualization_payload(
        request.failed_tool_output
    )
    if failed_output_payload:
        yield "failed_tool_output", failed_output_payload

    for index, event in enumerate(reversed(request.context_events[-MAX_CONTEXT_EVENTS:])):
        if not isinstance(event, dict):
            continue
        output = event.get("output")
        payload = _extract_table_metadata(output) or _extract_visualization_payload(output)
        if payload:
            yield f"context_event_{index}", payload
        event_input = event.get("input")
        if isinstance(event_input, dict) and event.get("tool") in {"create_datatable", "create_chart"}:
            yield f"context_event_input_{index}", event_input


def _coerce_row_list(row: Any, columns: list[str]) -> list[Any] | None:
    if isinstance(row, dict):
        return [row.get(column) for column in columns]
    if isinstance(row, (list, tuple)):
        values = list(row)
        if len(values) < len(columns):
            values = values + [None] * (len(columns) - len(values))
        return values[: len(columns)]
    return None


def _validate_datatable_source(source: dict[str, Any], strategy: str) -> _ValidatedPayload:
    raw_columns = source.get("columns")
    if raw_columns is None and isinstance(source.get("config"), dict):
        raw_columns = source["config"].get("columns")
    raw_rows = source.get("rows", source.get("data"))
    if raw_rows is None and isinstance(source.get("config"), dict):
        raw_rows = source["config"].get("data")

    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("No table columns found")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("No table rows found")

    columns: list[str] = []
    for column in raw_columns[:MAX_TABLE_COLUMNS]:
        if isinstance(column, dict):
            title = column.get("title") or column.get("name") or column.get("data")
            columns.append(str(title or "").strip())
        else:
            columns.append(str(column).strip())
    columns = [column or f"Column {idx + 1}" for idx, column in enumerate(columns)]

    rows: list[list[Any]] = []
    for raw_row in raw_rows[:MAX_TABLE_ROWS]:
        row = _coerce_row_list(raw_row, columns)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No valid table rows found")

    return _ValidatedPayload(
        source_data={"columns": columns, "rows": rows},
        strategy=strategy,
    )


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            number = float(cleaned)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def _validate_chart_source(source: dict[str, Any], strategy: str) -> _ValidatedPayload:
    chart_type = str(source.get("chart_type") or source.get("type") or "bar")
    if chart_type not in SUPPORTED_CHART_TYPES:
        chart_type = "bar"

    raw_labels = source.get("labels")
    raw_datasets = source.get("datasets")
    if raw_labels is None and isinstance(source.get("data"), dict):
        raw_labels = source["data"].get("labels")
        raw_datasets = raw_datasets or source["data"].get("datasets")
    if raw_datasets is None and "dataset" in source:
        raw_datasets = source.get("dataset")
    if raw_datasets is None and isinstance(source.get("data"), list):
        raw_datasets = [{"label": source.get("title") or "Series 1", "data": source.get("data")}]
    if isinstance(raw_datasets, dict):
        raw_datasets = [raw_datasets]

    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("No chart labels found")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("No chart datasets found")

    labels = [str(label) for label in raw_labels[:MAX_CHART_POINTS]]
    datasets: list[dict[str, Any]] = []
    for index, raw_dataset in enumerate(raw_datasets[:MAX_CHART_DATASETS]):
        if not isinstance(raw_dataset, dict):
            continue
        raw_data = raw_dataset.get("data")
        if not isinstance(raw_data, list):
            continue
        values: list[float | int] = []
        for value in raw_data[: len(labels)]:
            number = _coerce_number(value)
            if number is None:
                break
            values.append(number)
        if len(values) != len(labels):
            continue
        dataset = dict(raw_dataset)
        dataset["label"] = str(dataset.get("label") or f"Series {index + 1}")
        dataset["data"] = values
        datasets.append(dataset)
    if not datasets:
        raise ValueError("No valid numeric chart datasets found")

    return _ValidatedPayload(
        source_data={"labels": labels, "datasets": datasets, "chart_type": chart_type},
        strategy=strategy,
    )


def _table_to_chart_source(table_source: dict[str, Any], strategy: str) -> _ValidatedPayload:
    table_payload = _validate_datatable_source(table_source, strategy)
    columns = table_payload.source_data["columns"]
    rows = table_payload.source_data["rows"][:MAX_CHART_POINTS]
    label_index = 0
    numeric_columns: list[int] = []
    for column_index in range(len(columns)):
        numeric_values = [_coerce_number(row[column_index]) for row in rows]
        if numeric_values and all(value is not None for value in numeric_values):
            numeric_columns.append(column_index)
    if not numeric_columns:
        raise ValueError("No numeric column found for chart")
    if label_index in numeric_columns and len(columns) > 1:
        label_index = next((idx for idx in range(len(columns)) if idx not in numeric_columns), 0)
    datasets = []
    for column_index in numeric_columns[:MAX_CHART_DATASETS]:
        if column_index == label_index and len(numeric_columns) > 1:
            continue
        datasets.append(
            {
                "label": columns[column_index],
                "data": [_coerce_number(row[column_index]) for row in rows],
            }
        )
    if not datasets:
        raise ValueError("No numeric chart dataset found")
    return _ValidatedPayload(
        source_data={
            "labels": [str(row[label_index]) for row in rows],
            "datasets": datasets,
            "chart_type": "bar",
        },
        strategy=f"{strategy}_table_to_chart",
    )


def _validate_payload_for_tool(
    tool_type: str, source: dict[str, Any], strategy: str
) -> _ValidatedPayload:
    if tool_type == "datatable":
        return _validate_datatable_source(source, strategy)
    try:
        return _validate_chart_source(source, strategy)
    except ValueError:
        return _table_to_chart_source(source, strategy)


def _find_deterministic_payload(request: RetryVisualizationRequest) -> _ValidatedPayload | None:
    for strategy, source in _iter_candidate_sources(request):
        try:
            return _validate_payload_for_tool(request.tool_type, source, strategy)
        except ValueError:
            continue
    return None


async def _render_visualization(
    tool_type: str, title: str | None, validated: _ValidatedPayload
) -> str:
    if tool_type == "datatable":
        rows = validated.source_data["rows"]
        return await create_datatable(
            title=title or "Data",
            columns=validated.source_data["columns"],
            data=rows,
            description=f"Table with {len(rows)} rows",
        )

    labels = validated.source_data["labels"]
    return await create_chart(
        chart_type=validated.source_data.get("chart_type", "bar"),
        title=title or "Chart",
        labels=labels,
        datasets=validated.source_data["datasets"],
        description=f"Chart with {len(labels)} data points",
    )


def _candidate_source_events(request: RetryVisualizationRequest) -> Iterable[dict[str, Any]]:
    for event in reversed(request.context_events[-MAX_CONTEXT_EVENTS:]):
        if not isinstance(event, dict):
            continue
        connection = event.get("connection")
        if not isinstance(connection, dict):
            continue
        tool_config_id = str(connection.get("tool_config_id") or "").strip()
        event_input = event.get("input")
        if tool_config_id and isinstance(event_input, dict):
            yield event


async def _rerun_source_query(
    request: RetryVisualizationRequest,
    context: VisualizationRetryContext,
) -> tuple[str | None, dict[str, Any] | None]:
    if not request.allow_source_rerun:
        return None, None
    if not rag.is_ready:
        return None, None

    for event in _candidate_source_events(request):
        connection = event.get("connection") or {}
        tool_config_id = str(connection.get("tool_config_id") or "").strip()
        if tool_config_id not in context.selected_tool_ids:
            continue
        tool_config = await repository.get_tool_config(tool_config_id)
        if not tool_config or not tool_config.enabled:
            continue
        if tool_config.tool_type not in SOURCE_RERUN_TOOL_TYPES:
            continue
        try:
            tool = await rag.build_primary_runtime_tool_from_config(tool_config.model_dump())
            if tool is None:
                continue
            output = await tool.ainvoke(event.get("input") or {})
            output_text = str(output)
            payload = _extract_table_metadata(output_text) or _extract_visualization_payload(output_text)
            return output_text, payload
        except Exception:
            logger.exception("Visualization source rerun failed for tool %s", tool_config_id)
            continue
    return None, None


def _build_repair_prompt(
    request: RetryVisualizationRequest,
    rerun_output: str | None,
) -> tuple[str, str]:
    system_prompt = (
        "You repair failed UI visualization tool calls. Return one JSON object only. "
        "Do not invent data. Use only the supplied tool input, tool output, previous tool outputs, "
        "or source rerun output. If the data cannot be recovered, return {\"error\": \"...\"}."
    )
    if request.tool_type == "datatable":
        target_schema = (
            '{"source_data":{"columns":["Column"],"rows":[["value"]]},"title":"optional"}'
        )
    else:
        target_schema = (
            '{"source_data":{"chart_type":"bar","labels":["label"],'
            '"datasets":[{"label":"Series","data":[1]}]},"title":"optional"}'
        )
    context_events = []
    for event in request.context_events[-MAX_CONTEXT_EVENTS:]:
        if not isinstance(event, dict):
            continue
        context_events.append(
            {
                "tool": event.get("tool"),
                "input": event.get("input"),
                "output": _truncate_text(event.get("output") or "", MAX_OUTPUT_CONTEXT_CHARS),
                "connection": event.get("connection"),
            }
        )
    user_payload = {
        "target_tool_type": request.tool_type,
        "required_response_shape": target_schema,
        "title_hint": request.title,
        "failed_tool_input": request.failed_tool_input,
        "failed_tool_output": _truncate_text(request.failed_tool_output or "", MAX_OUTPUT_CONTEXT_CHARS),
        "provided_source_data": request.source_data,
        "source_rerun_output": _truncate_text(rerun_output or "", MAX_OUTPUT_CONTEXT_CHARS),
        "recent_tool_events": context_events,
    }
    return system_prompt, _truncate_text(user_payload, MAX_CONTEXT_CHARS)


async def _persist_repair_debug_record(
    context: VisualizationRetryContext,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    source_rerun_used: bool,
) -> None:
    if not settings.debug_mode:
        return
    try:
        await repository.create_provider_prompt_debug_record(
            conversation_id=context.conversation.id,
            chat_task_id=None,
            user_id=context.user_id,
            provider=provider or "unknown",
            model=model or context.conversation.model,
            mode="chat",
            request_kind="visualization_repair",
            rendered_system_prompt=system_prompt,
            rendered_user_input=user_prompt,
            rendered_provider_messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            rendered_chat_history=[],
            tool_scope_prompt="",
            prompt_additions="",
            turn_reminders="",
            debug_metadata={
                "tool_type": "visualization_repair",
                "conversation_model": context.conversation.model,
                "source_rerun_used": source_rerun_used,
                "response_preview": _truncate_text(response_text, 4000),
            },
        )
    except Exception:
        logger.exception("Failed to persist visualization repair debug record")


async def _repair_with_ai(
    request: RetryVisualizationRequest,
    context: VisualizationRetryContext,
    rerun_output: str | None,
) -> _ValidatedPayload | None:
    if not request.allow_ai_repair:
        return None
    if not rag.is_ready:
        raise RuntimeError("RAG service initializing, please retry")

    llm_resolution = await rag._get_request_scoped_llm(context.conversation.model)  # pyright: ignore[reportPrivateUsage]
    request_llm = getattr(llm_resolution, "llm", None)
    if request_llm is None:
        error_message = getattr(llm_resolution, "error_message", None)
        raise RuntimeError(error_message or "No LLM is available for visualization repair")

    system_prompt, user_prompt = _build_repair_prompt(request, rerun_output)
    response = await request_llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    response_text = _message_content_to_text(getattr(response, "content", response))
    await _persist_repair_debug_record(
        context,
        provider=getattr(llm_resolution, "provider", None) or "unknown",
        model=getattr(llm_resolution, "model", None) or context.conversation.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_text=response_text,
        source_rerun_used=bool(rerun_output),
    )
    parsed = _extract_json_object(response_text)
    if not parsed or parsed.get("error"):
        return None
    source_data = parsed.get("source_data")
    if not isinstance(source_data, dict):
        source_data = parsed
    try:
        return _validate_payload_for_tool(request.tool_type, source_data, "ai_repair")
    except ValueError as exc:
        logger.info("AI visualization repair returned invalid payload: %s", exc)
        return None


async def _persist_repaired_event(
    request: RetryVisualizationRequest,
    context: VisualizationRetryContext,
    new_output: str,
) -> None:
    """Write the repaired visualization output back to the conversation.

    Silently no-ops when the caller did not provide ``message_id`` and
    ``event_index``; logs a warning if the persistence call cannot locate
    the target event.
    """
    if not request.message_id and request.message_index is None:
        return
    if request.event_index is None:
        return
    expected_tool = (
        "create_chart" if request.tool_type == "chart" else "create_datatable"
    )
    try:
        ok = await repository.update_message_event_output(
            context.conversation.id,
            request.message_id,
            request.event_index,
            new_output,
            message_index=request.message_index,
            expected_tool=expected_tool,
        )
    except Exception:
        logger.exception(
            "Error persisting repaired visualization output for "
            "conversation=%s message_id=%s message_index=%s event_index=%s",
            context.conversation.id,
            request.message_id,
            request.message_index,
            request.event_index,
        )
        return
    if not ok:
        logger.warning(
            "Repaired visualization output not persisted: target event not "
            "found (conversation=%s message_id=%s message_index=%s "
            "event_index=%s tool=%s)",
            context.conversation.id,
            request.message_id,
            request.message_index,
            request.event_index,
            expected_tool,
        )


async def retry_visualization_with_repair(
    request: RetryVisualizationRequest,
    context: VisualizationRetryContext,
) -> RetryVisualizationResponse:
    deterministic = _find_deterministic_payload(request)
    if deterministic:
        output = await _render_visualization(request.tool_type, request.title, deterministic)
        await _persist_repaired_event(request, context, output)
        return RetryVisualizationResponse(
            success=True,
            output=output,
            repair_used=False,
            repair_strategy=deterministic.strategy,
            source_rerun_used=False,
        )

    rerun_output: str | None = None
    if request.allow_source_rerun:
        rerun_output, rerun_payload = await _rerun_source_query(request, context)
        if rerun_payload:
            try:
                validated = _validate_payload_for_tool(
                    request.tool_type, rerun_payload, "source_rerun"
                )
                output = await _render_visualization(request.tool_type, request.title, validated)
                await _persist_repaired_event(request, context, output)
                return RetryVisualizationResponse(
                    success=True,
                    output=output,
                    repair_used=False,
                    repair_strategy=validated.strategy,
                    source_rerun_used=True,
                )
            except ValueError:
                pass

    ai_payload = await _repair_with_ai(request, context, rerun_output)
    if ai_payload:
        output = await _render_visualization(request.tool_type, request.title, ai_payload)
        await _persist_repaired_event(request, context, output)
        return RetryVisualizationResponse(
            success=True,
            output=output,
            repair_used=True,
            repair_strategy=ai_payload.strategy,
            source_rerun_used=bool(rerun_output),
        )

    return RetryVisualizationResponse(
        success=False,
        error="Could not recover valid source data for this visualization.",
        repair_used=bool(request.allow_ai_repair),
        repair_strategy="failed",
        source_rerun_used=bool(rerun_output),
    )
