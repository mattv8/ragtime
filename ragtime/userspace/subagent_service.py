from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ragtime.core.event_bus import task_event_bus
from ragtime.core.logging import get_logger
from ragtime.indexer.models import ChatTaskStatus
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

MAX_SUBAGENT_FANOUT = 6
SUBAGENT_TOOL_NAME = "spawn_subagents"
SUBAGENT_HANDOFF_TOOL_NAME = "submit_subagent_handoff"
SUBAGENT_PRIVATE_PROMPT_CONTEXT_KEY = "subagent_private_prompt"
SUBAGENT_MAX_DEPTH = 1
SUBAGENT_REVIEW_ROLE = "review"

SUBAGENT_WRITE_TOOL_NAMES = {
    "upsert_userspace_file",
    "patch_userspace_file",
    "move_userspace_file",
    "delete_userspace_file",
    "create_userspace_snapshot",
    "run_terminal_command",
}


@dataclass
class SubAgentSpec:
    name: str
    role: str
    instructions: str
    file_scope: list[str] = field(default_factory=list)
    model: Optional[str] = None


class SubAgentService:
    class _SubagentTaskFailure(RuntimeError):
        def __init__(self, task_id: str) -> None:
            super().__init__(task_id)
            self.task_id = task_id

    def _normalize_scope_path(self, path: str) -> str:
        normalized = "/".join(part for part in str(path or "").strip().replace("\\", "/").split("/") if part and part != ".")
        if not normalized or normalized.startswith("../") or "/../" in f"/{normalized}/":
            raise ValueError(f"Invalid subagent file scope path: {path}")
        return normalized.rstrip("/")

    def _normalize_specs(self, raw_specs: list[dict[str, Any]]) -> list[SubAgentSpec]:
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("spawn_subagents requires between 1 and 6 subagent specs")
        if len(raw_specs) > MAX_SUBAGENT_FANOUT:
            raise ValueError("spawn_subagents supports at most 6 subagents per call")

        specs: list[SubAgentSpec] = []
        seen_scope_paths: dict[str, str] = {}
        for index, raw in enumerate(raw_specs, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"Subagent spec {index} must be an object")
            name = str(raw.get("name") or f"Subagent {index}").strip()[:80]
            role = str(raw.get("role") or "worker").strip().lower()[:40] or "worker"
            instructions = str(raw.get("instructions") or "").strip()
            if not instructions:
                raise ValueError(f"Subagent {name} must include instructions")

            scope_raw = raw.get("file_scope") or []
            if isinstance(scope_raw, str):
                scope_raw = [scope_raw]
            if not isinstance(scope_raw, list):
                raise ValueError(f"Subagent {name} file_scope must be a list of paths")
            file_scope = [self._normalize_scope_path(str(path)) for path in scope_raw if str(path or "").strip()]

            for scope_path in file_scope:
                for existing_path, existing_name in seen_scope_paths.items():
                    overlaps = scope_path == existing_path or scope_path.startswith(f"{existing_path}/") or existing_path.startswith(f"{scope_path}/")
                    if overlaps:
                        raise ValueError(f"Subagent file scopes overlap: {name} owns {scope_path}, {existing_name} owns {existing_path}")
                seen_scope_paths[scope_path] = name

            model = raw.get("model")
            specs.append(
                SubAgentSpec(
                    name=name,
                    role=role,
                    instructions=instructions,
                    file_scope=file_scope,
                    model=(str(model).strip() if model else None),
                )
            )
        return specs

    def _build_child_private_prompt(self, spec: SubAgentSpec) -> str:
        scope_text = ", ".join(spec.file_scope) if spec.file_scope else "read-only/no declared write scope"
        scope_rule = (
            "Modify only workspace files under the declared scope above."
            if spec.file_scope
            else "This is a read-only assignment: inspect, reason, and report findings without modifying files."
        )
        return (
            "\n\n### Subagent private execution context\n"
            f"- Role: {spec.role}.\n"
            f"- Declared file scope: {scope_text}.\n"
            f"- {scope_rule}\n"
            "- Stay focused on the parent-assigned task; do not perform unrelated cleanup or final parent synthesis.\n"
            "Before finishing, you MUST call the `submit_subagent_handoff` tool exactly once with a structured "
            "handoff. The parent agent only consumes the structured output from that tool; any freeform prose "
            "outside the tool call is ignored by the parent. "
            "Include a non-empty markdown `final_output`, plus structured fields when applicable: `summary`, "
            "`files_changed`, `files_reviewed`, `validation_performed`, and `remaining_risks`."
        )

    def _extract_subagent_handoff(self, task: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Extract the last submit_subagent_handoff tool call from task streaming state.

        Returns (handoff_dict, None) on success, or (None, error_message) when the
        handoff tool is missing, malformed, or has an empty final_output. Never falls
        back to raw response_content as a successful handoff.
        """
        streaming_state = getattr(task, "streaming_state", None) or {}
        tool_calls = getattr(streaming_state, "tool_calls", None) or []
        if not isinstance(tool_calls, list):
            return None, "Invalid streaming_state.tool_calls for subagent handoff extraction"

        handoff_call: Optional[dict[str, Any]] = None
        for call in tool_calls:
            if isinstance(call, dict) and call.get("tool") == SUBAGENT_HANDOFF_TOOL_NAME:
                handoff_call = call

        if handoff_call is None:
            return None, f"Subagent did not call the required `{SUBAGENT_HANDOFF_TOOL_NAME}` tool before finishing"

        input_data = handoff_call.get("input")
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except (json.JSONDecodeError, ValueError):
                return None, f"Subagent `{SUBAGENT_HANDOFF_TOOL_NAME}` tool input is not valid JSON"
        if not isinstance(input_data, dict):
            return None, f"Subagent `{SUBAGENT_HANDOFF_TOOL_NAME}` tool input is not an object"

        final_output = str(input_data.get("final_output") or "").strip()
        if not final_output:
            return None, f"Subagent `{SUBAGENT_HANDOFF_TOOL_NAME}` tool input missing non-empty `final_output`"

        def _normalize_str_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                if not value.strip():
                    return []
                return [value.strip()]
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item or "").strip()]
            return [str(value).strip()]

        handoff: dict[str, Any] = {
            "final_output": final_output,
            "summary": str(input_data.get("summary") or "").strip() or None,
            "files_changed": _normalize_str_list(input_data.get("files_changed")),
            "files_reviewed": _normalize_str_list(input_data.get("files_reviewed")),
            "validation_performed": _normalize_str_list(input_data.get("validation_performed")),
            "remaining_risks": _normalize_str_list(input_data.get("remaining_risks")),
        }
        return handoff, None

    async def spawn_subagents(
        self,
        *,
        parent_conversation_id: str,
        parent_task_id: Optional[str],
        workspace_id: str,
        user_id: str,
        parent_model: str,
        specs: list[dict[str, Any]],
        workspace_context: dict[str, Any],
        blocked_tool_names: set[str],
        disabled_builtin_tool_ids: Optional[set[str]] = None,
        current_time_context: Optional[dict[str, Any]] = None,
    ) -> str:
        normalized_specs = self._normalize_specs(specs)
        child_task_ids: list[str] = []

        from ragtime.indexer.background_tasks import background_task_service

        async def cancel_registered_children(*, exclude_task_ids: set[str] | None = None) -> None:
            excluded = exclude_task_ids or set()
            for task_id in dict.fromkeys(task_id for task_id in child_task_ids if task_id):
                if task_id in excluded:
                    continue
                background_task_service.cancel_task(task_id)
                task = await repository.get_chat_task(task_id)
                if task and task.status in {ChatTaskStatus.pending, ChatTaskStatus.running}:
                    await repository.cancel_chat_task(task_id)

        async def start_child(index: int, spec: SubAgentSpec) -> tuple[SubAgentSpec, str, str]:
            child = await repository.create_conversation(
                title=f"{spec.name} ({spec.role})",
                model=spec.model or parent_model,
                user_id=user_id,
                workspace_id=workspace_id,
                parent_conversation_id=parent_conversation_id,
                subagent_role=spec.role,
                subagent_index=index,
                subagents_enabled=False,
            )
            child_context = dict(workspace_context)
            child_context.update(
                {
                    "subagent_depth": SUBAGENT_MAX_DEPTH,
                    "subagent_parent_conversation_id": parent_conversation_id,
                    "subagent_file_scope": spec.file_scope,
                    "subagent_role": spec.role,
                }
            )
            child_blocked = set(blocked_tool_names)
            child_blocked.add(SUBAGENT_TOOL_NAME)
            if spec.role == SUBAGENT_REVIEW_ROLE or not spec.file_scope:
                child_blocked.update(SUBAGENT_WRITE_TOOL_NAMES)

            child_prompt = spec.instructions
            child_context[SUBAGENT_PRIVATE_PROMPT_CONTEXT_KEY] = self._build_child_private_prompt(spec)
            _updated_child, task, created = await repository.add_user_message_and_create_chat_task_if_idle(
                child.id,
                child_prompt,
            )
            if not task or not created:
                raise RuntimeError(f"Failed to create subagent task for {spec.name}")
            task_id = task.id
            child_task_ids.append(task_id)
            if parent_task_id:
                background_task_service.register_subagent_children(parent_task_id, [task_id])
            background_task_service.start_task(
                child.id,
                child_prompt,
                existing_task_id=task_id,
                blocked_tool_names=child_blocked,
                workspace_context=child_context,
                current_time_context=current_time_context,
                disabled_builtin_tool_ids=disabled_builtin_tool_ids,
            )
            await task_event_bus.publish(
                f"conversation:{parent_conversation_id}",
                {
                    "event": "subagent_spawned",
                    "parent_task_id": parent_task_id,
                    "conversation_id": child.id,
                    "task_id": task_id,
                    "name": spec.name,
                    "role": spec.role,
                    "index": index,
                },
            )
            return spec, child.id, task_id

        try:
            started = await asyncio.gather(*(start_child(index, spec) for index, spec in enumerate(normalized_specs, start=1)))

            async def wait_child(spec: SubAgentSpec, child_id: str, task_id: str) -> dict[str, Any]:
                try:
                    await background_task_service.await_task(task_id)
                except asyncio.CancelledError:
                    task = await repository.get_chat_task(task_id)
                    if not task or task.status != ChatTaskStatus.cancelled:
                        raise
                except Exception as exc:
                    raise self._SubagentTaskFailure(task_id) from exc
                task = await repository.get_chat_task(task_id)
                status = task.status.value if task else "unknown"
                handoff, handoff_error = self._extract_subagent_handoff(task)
                if handoff is None:
                    task_error = str(getattr(task, "error_message", "") or "").strip()
                    error_context = f" Error: {handoff_error}"
                    if task_error:
                        error_context += f" Child task error: {task_error}"
                    final_output = (
                        "## Subagent handoff missing or invalid\n\n"
                        f"The subagent completed without submitting a valid `{SUBAGENT_HANDOFF_TOOL_NAME}` tool call. "
                        f"{error_context}\n\n"
                        "Freeform child prose was not used as the parent handoff."
                    )
                    status = "failed"
                else:
                    final_output = handoff["final_output"]
                await task_event_bus.publish(
                    f"conversation:{parent_conversation_id}",
                    {
                        "event": "subagent_completed",
                        "parent_task_id": parent_task_id,
                        "conversation_id": child_id,
                        "task_id": task_id,
                        "name": spec.name,
                        "role": spec.role,
                        "status": status,
                    },
                )
                result: dict[str, Any] = {
                    "name": spec.name,
                    "role": spec.role,
                    "conversation_id": child_id,
                    "task_id": task_id,
                    "status": status,
                    "file_scope": spec.file_scope,
                    "final_output": final_output,
                }
                if handoff is not None:
                    result["handoff"] = handoff
                else:
                    result["handoff_error"] = handoff_error
                return result

            results = await asyncio.gather(*(wait_child(*item) for item in started))
            return json.dumps({"subagents": results}, indent=2)
        except self._SubagentTaskFailure as exc:
            await cancel_registered_children(exclude_task_ids={exc.task_id})
            logger.warning("Subagent child task failed for parent %s: %s", parent_conversation_id, exc.task_id)
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise exc
        except asyncio.CancelledError:
            await cancel_registered_children()
            raise
        except Exception as exc:
            await cancel_registered_children()
            logger.warning("Subagent spawn failed for parent %s: %s", parent_conversation_id, exc)
            raise


subagent_service = SubAgentService()
