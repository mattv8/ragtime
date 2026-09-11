"""Chat v1 fixture handler: selection, validation, export and historical clone."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from .common import FixtureError, database_cursor, json_parameter, read_json

CANDIDATE_LIMIT = 20
QUERY_LIMIT = CANDIDATE_LIMIT + 1
VALID_BRANCH_KINDS = {None, "edit", "delete", "replay"}
VALID_TOOL_OUTPUT_MODES = {"default", "show", "hide", "auto"}


class ChatSelectionError(FixtureError):
    pass


class ChatNotFoundError(ChatSelectionError):
    pass


class AmbiguousChatError(ChatSelectionError):
    def __init__(self, candidates: list[dict[str, Any]], has_more: bool = False):
        self.candidates = candidates[:CANDIDATE_LIMIT]
        self.has_more = has_more


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_CANDIDATE_SELECT = """
SELECT c.id, c.title, u.username, w.id, w.name FROM conversations c
LEFT JOIN users u ON u.id = c.user_id LEFT JOIN workspaces w ON w.id = c.workspace_id
"""


def _find_candidates(cursor, where: str, params: tuple[str, ...]) -> list[dict[str, Any]]:
    cursor.execute(f"{_CANDIDATE_SELECT} WHERE {where} ORDER BY c.updated_at DESC, c.id LIMIT {QUERY_LIMIT}", params)
    return [{"id": row[0], "title": row[1], "owner": row[2], "workspace_id": row[3], "workspace_name": row[4]} for row in cursor.fetchall()]


def _unique_or_ambiguous(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise AmbiguousChatError(candidates, len(candidates) > CANDIDATE_LIMIT)
    return None


def resolve_chat(cursor, selector: str) -> dict[str, Any]:
    """Resolve exact ID, exact title, then literal substring in that order."""
    if not selector or not selector.strip():
        raise ChatNotFoundError()
    selector = selector.strip()
    for where, params in (
        ("lower(c.id) = lower(%s)", (selector,)),
        ("lower(c.title) = lower(%s)", (selector,)),
        ("(c.id ILIKE %s ESCAPE '\\' OR c.title ILIKE %s ESCAPE '\\')", (f"%{escape_like(selector)}%",) * 2),
    ):
        selected = _unique_or_ambiguous(_find_candidates(cursor, where, params))
        if selected:
            return selected
    raise ChatNotFoundError()


def fixture_select() -> str:
    return """SELECT jsonb_build_object('version', 1, 'fixture_type', 'chat',
 'conversation', to_jsonb(c), 'branches', COALESCE((SELECT jsonb_agg(to_jsonb(b) ORDER BY b.created_at, b.id)
 FROM conversation_branches b WHERE b.conversation_id=c.id),'[]'::jsonb),
 'completed_tasks', COALESCE((SELECT jsonb_agg(jsonb_build_object('id',t.id,'user_message',t.user_message,
 'response_content',t.response_content,'streaming_state',t.streaming_state,'created_at',t.created_at,
 'started_at',t.started_at,'completed_at',t.completed_at,'last_update_at',t.last_update_at) ORDER BY t.created_at)
 FROM chat_tasks t WHERE t.conversation_id=c.id AND t.status='completed'),'[]'::jsonb)) FROM conversations c WHERE c.id=%s"""


def export_chat(selector: str, connect=None) -> dict[str, Any]:
    with database_cursor(read_only=True, connect=connect) as cursor:
        selected = resolve_chat(cursor, selector)
        cursor.execute(fixture_select(), (selected["id"],))
        row = cursor.fetchone()
        if not row or not row[0]:
            raise ChatNotFoundError()
        return row[0]


def _require(value: Any, description: str, expected: type | tuple[type, ...]) -> Any:
    if not isinstance(value, expected):
        raise FixtureError(f"Fixture {description} is invalid.")
    return value


def _timestamp(value: Any, description: str, *, fallback: str | None = None) -> str:
    if value is None and fallback is not None:
        return fallback
    if not isinstance(value, str):
        raise FixtureError(f"Fixture {description} is invalid.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FixtureError(f"Fixture {description} is invalid.") from error
    return value


def validate_document(document: object) -> dict[str, Any]:
    """Validate the entire untrusted v1 graph before any write is attempted."""
    doc = _require(document, "document", dict)
    if type(doc.get("version")) is not int or doc["version"] != 1 or doc.get("fixture_type", "chat") != "chat":
        raise FixtureError("Unsupported chat fixture.")
    conversation = _require(doc.get("conversation"), "conversation", dict)
    if not isinstance(conversation.get("id"), str) or not conversation["id"]:
        raise FixtureError("Fixture source conversation ID is invalid.")
    _require(conversation.get("title"), "conversation title", str)
    if conversation.get("model") is not None and not isinstance(conversation["model"], str):
        raise FixtureError("Fixture conversation model is invalid.")
    _require(conversation.get("messages"), "conversation messages", list)
    if not all(isinstance(message, dict) for message in conversation["messages"]):
        raise FixtureError("Fixture conversation messages are invalid.")
    tokens = conversation.get("total_tokens", 0)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise FixtureError("Fixture total tokens are invalid.")
    if conversation.get("tool_output_mode", "default") not in VALID_TOOL_OUTPUT_MODES:
        raise FixtureError("Fixture tool output mode is invalid.")
    disabled = conversation.get("disabled_builtin_tool_ids", [])
    if not isinstance(disabled, list) or not all(isinstance(item, str) for item in disabled):
        raise FixtureError("Fixture disabled builtins are invalid.")
    branches = _require(doc.get("branches"), "branches", list)
    ids: set[str] = set()
    parents: dict[str, str | None] = {}
    for branch in branches:
        _require(branch, "branch", dict)
        branch_id = branch.get("id")
        if not isinstance(branch_id, str) or not branch_id or branch_id in ids:
            raise FixtureError("Fixture branch IDs are invalid.")
        ids.add(branch_id)
        if "conversation_id" in branch and branch["conversation_id"] != conversation["id"]:
            raise FixtureError("Fixture branch conversation ID is invalid.")
        point = branch.get("branch_point_index")
        if not isinstance(point, int) or isinstance(point, bool) or point < 0 or branch.get("branch_kind") not in VALID_BRANCH_KINDS:
            raise FixtureError("Fixture branch metadata is invalid.")
        suffix = branch.get("preserved_messages")
        if not isinstance(suffix, list) or not all(isinstance(message, dict) for message in suffix):
            raise FixtureError("Fixture branch suffix is invalid.")
        base = branch.get("base_messages")
        if base is not None and (not isinstance(base, list) or len(base) != point or not all(isinstance(message, dict) for message in base)):
            raise FixtureError("Fixture branch base messages are invalid.")
        parent = branch.get("parent_branch_id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise FixtureError("Fixture branch parent is invalid.")
        parents[branch_id] = parent
        for field in ("created_at", "updated_at"):
            _timestamp(branch.get(field), f"branch {field}")
    for branch_id, parent in parents.items():
        if parent is not None and parent not in ids:
            raise FixtureError("Fixture branch parent is missing.")
    done: set[str] = set()
    for branch_id in parents:
        if branch_id in done:
            continue
        seen: set[str] = set()
        parent = branch_id
        while parent is not None:
            if parent in seen:
                raise FixtureError("Fixture branch graph contains a cycle.")
            if parent in done:
                break
            seen.add(parent)
            parent = parents[parent]
        done.update(seen)
    active = conversation.get("active_branch_id")
    if active is not None and (not isinstance(active, str) or active not in ids):
        raise FixtureError("Fixture active branch is missing.")
    tasks = _require(doc.get("completed_tasks", []), "completed tasks", list)
    task_ids: set[str] = set()
    for task in tasks:
        _require(task, "completed task", dict)
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids or not isinstance(task.get("user_message"), str):
            raise FixtureError("Fixture completed task is invalid.")
        task_ids.add(task_id)
        if task.get("status", "completed") != "completed":
            raise FixtureError("Fixture completed task status is invalid.")
        if "conversation_id" in task and task["conversation_id"] != conversation["id"]:
            raise FixtureError("Fixture completed task conversation ID is invalid.")
        if task.get("response_content") is not None and not isinstance(task.get("response_content"), str):
            raise FixtureError("Fixture completed task response is invalid.")
        if task.get("streaming_state") is not None and not isinstance(task.get("streaming_state"), dict):
            raise FixtureError("Fixture completed task streaming state is invalid.")
        created = _timestamp(task.get("created_at"), "completed task created_at")
        completed = _timestamp(task.get("completed_at"), "completed task completed_at", fallback=created)
        _timestamp(task.get("last_update_at"), "completed task last_update_at", fallback=completed)
        if task.get("started_at") is not None:
            _timestamp(task["started_at"], "completed task started_at")
    return doc


def import_chat(document: object, owner_username: str, title: str | None = None, connect=None) -> dict[str, Any]:
    """Clone a validated fixture in one transaction; no runnable state is copied."""
    doc = validate_document(document)
    if not isinstance(owner_username, str) or not owner_username:
        raise FixtureError("An existing owner username is required.")
    conversation = doc["conversation"]
    branches = doc["branches"]
    tasks = doc.get("completed_tasks", [])
    conversation_id = str(uuid.uuid4())
    branch_ids = {branch["id"]: str(uuid.uuid4()) for branch in branches}
    with database_cursor(read_only=False, connect=connect) as cursor:
        cursor.execute("SELECT id FROM users WHERE username = %s", (owner_username,))
        owner = cursor.fetchone()
        if not owner:
            raise FixtureError("Owner username does not exist.")
        owner_id = owner[0]
        target_title = title if title is not None else f"[Imported] {conversation['title']}"
        cursor.execute(
            """INSERT INTO conversations (id,title,model,messages,total_tokens,user_id,tool_output_mode,tool_selection_mode,
 disabled_builtin_tool_ids,loaded_tool_skill_ids,subagents_enabled,active_branch_id,workspace_id,parent_conversation_id,
 subagent_role,subagent_index,active_task_id,created_at,updated_at) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s::\"ConversationToolOutputMode\",
 'custom',%s::jsonb,'[]'::jsonb,false,%s,NULL,NULL,NULL,NULL,NULL,NOW(),NOW())""",
            (
                conversation_id,
                target_title,
                conversation.get("model") or "gpt-4-turbo",
                json_parameter(conversation["messages"]),
                conversation.get("total_tokens", 0),
                owner_id,
                conversation.get("tool_output_mode", "default"),
                json_parameter(conversation.get("disabled_builtin_tool_ids", [])),
                branch_ids.get(conversation.get("active_branch_id")),
            ),
        )
        for branch in branches:
            cursor.execute(
                """INSERT INTO conversation_branches (id,conversation_id,parent_branch_id,branch_point_index,branch_kind,
 preserved_messages,base_messages,associated_snapshot_id,created_by_user_id,created_at,updated_at)
 VALUES (%s,%s,%s,%s,%s::\"ConversationBranchKind\",%s::jsonb,%s::jsonb,NULL,%s,%s::timestamptz,%s::timestamptz)""",
                (
                    branch_ids[branch["id"]],
                    conversation_id,
                    branch_ids.get(branch.get("parent_branch_id")),
                    branch["branch_point_index"],
                    branch.get("branch_kind"),
                    json_parameter(branch["preserved_messages"]),
                    None if branch.get("base_messages") is None else json_parameter(branch["base_messages"]),
                    owner_id,
                    branch["created_at"],
                    branch["updated_at"],
                ),
            )
        for task in tasks:
            created = task["created_at"]
            completed = task.get("completed_at") or created
            updated = task.get("last_update_at") or completed
            cursor.execute(
                """INSERT INTO chat_tasks (id,conversation_id,status,user_message,response_content,streaming_state,
 created_at,started_at,completed_at,last_update_at) VALUES (%s,%s,'completed',%s,%s,%s::jsonb,%s::timestamptz,%s::timestamptz,%s::timestamptz,%s::timestamptz)""",
                (
                    str(uuid.uuid4()),
                    conversation_id,
                    task["user_message"],
                    task.get("response_content"),
                    None if task.get("streaming_state") is None else json_parameter(task["streaming_state"]),
                    created,
                    task.get("started_at"),
                    completed,
                    updated,
                ),
            )
    return {
        "fixture_type": "chat",
        "conversation_id": conversation_id,
        "title": target_title,
        "message_count": len(conversation["messages"]),
        "branch_count": len(branches),
        "completed_task_count": len(tasks),
    }


def _format_candidate(candidate: dict[str, Any]) -> str:
    details = [f"id={candidate['id']}", f"title={candidate['title']!r}"]
    if candidate["owner"]:
        details.append(f"owner={candidate['owner']}")
    if candidate["workspace_id"]:
        details.append(f"workspace={candidate['workspace_id']} ({candidate['workspace_name']!r})")
    return "  " + " ".join(details)


def run_export(args) -> int:
    if not args.selector or not args.selector.strip():
        raise ChatNotFoundError("No chat matches that selector.")
    try:
        document = export_chat(args.selector)
    except ChatNotFoundError:
        raise ChatNotFoundError("No chat matches that selector.") from None
    except AmbiguousChatError as error:
        print("Chat selector is ambiguous; refine it with an exact ID or title:", file=args.stderr)
        for candidate in error.candidates:
            print(_format_candidate(candidate), file=args.stderr)
        if error.has_more:
            print(f"  ... additional matches omitted (showing {CANDIDATE_LIMIT}).", file=args.stderr)
        return 1
    print(json.dumps(document, separators=(",", ":"), default=str), file=args.stdout)
    return 0


def run_import(args) -> int:
    document = read_json(args.file, args.stdin)
    result = import_chat(document, args.owner, args.title)
    print(json.dumps(result, separators=(",", ":")), file=args.stdout)
    return 0


def register_export(parser) -> None:
    parser.add_argument("selector", help="Exact ID, exact title, or literal ID/title substring")
    parser.set_defaults(run=run_export)


def register_import(parser) -> None:
    parser.add_argument("file", nargs="?", default="-", help="UTF-8 JSON file or - for stdin")
    parser.add_argument("--owner", required=True, help="Exact existing username")
    parser.add_argument("--title")
    parser.set_defaults(run=run_import)
