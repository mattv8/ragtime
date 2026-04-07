"""
RAG Components - FAISS Vector Store and LangChain Agent setup.
"""

import asyncio
import base64
import fnmatch
import hashlib
import io
import json
import math
import os
import posixpath
import re
import resource
import shlex
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional, Union, cast
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import StructuredTool, ToolException
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import (
    _construct_responses_api_payload, _get_last_messages)
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.core.copilot_auth import ensure_copilot_token_fresh
from ragtime.core.entrypoint_status import FRAMEWORK_REQUIRED_PACKAGES
from ragtime.core.file_constants import (USERSPACE_MODULE_SOURCE_EXTENSIONS,
                                         USERSPACE_STRICT_FRONTEND_EXTENSIONS,
                                         USERSPACE_THEME_AUDIT_EXTENSIONS,
                                         USERSPACE_TYPESCRIPT_EXTENSIONS)
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import (get_context_limit, get_output_limit,
                                       register_model_reasoning_capabilities,
                                       register_model_supported_endpoints,
                                       requires_responses_api,
                                       supports_reasoning,
                                       supports_responses_api,
                                       supports_thinking_budget)
from ragtime.core.ollama import (DEFAULT_WARMUP_TIMEOUT_SECONDS, KEEP_ALIVE,
                                 NUM_GPU, get_model_context_length,
                                 get_model_details, has_capability,
                                 warmup_embedding_model, warmup_model)
from ragtime.core.security import (_SSH_ENV_VAR_RE, sanitize_output,
                                   validate_odoo_code, validate_sql_query,
                                   validate_ssh_command)
from ragtime.core.sql_utils import add_table_metadata_to_psql_output
from ragtime.core.ssh import (SSHConfig, SSHTunnel, build_ssh_tunnel_config,
                              execute_ssh_command, expand_env_vars_via_ssh,
                              ssh_tunnel_config_from_dict)
from ragtime.core.tokenization import truncate_to_token_budget
from ragtime.indexer.pdm_service import pdm_indexer, search_pdm_index
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import schema_indexer, search_schema_index
from ragtime.indexer.vector_backends import FaissBackend, get_faiss_backend
from ragtime.rag.prompts import (
    BASE_CHAT_SYSTEM_PROMPT, BASE_USERSPACE_SYSTEM_PROMPT,
    SQLITE_INCLUDE_MODE_HINT, TOOL_OUTPUT_VISIBILITY_PROMPT,
    TOOL_USAGE_REMINDER, UI_VISUALIZATION_CHAT_PROMPT,
    UI_VISUALIZATION_COMMON_PROMPT, UI_VISUALIZATION_USERSPACE_PROMPT,
    build_index_system_prompt, build_tool_system_prompt,
    build_userspace_entrypoint_nudge, build_userspace_mode_prompt_addition,
    build_userspace_mounts_prompt_fragment,
    build_userspace_object_storage_prompt_fragment,
    build_userspace_turn_reminder, build_userspace_turn_reminder_with_env_vars,
    build_workspace_continuity_context)
from ragtime.tools import get_all_tools, get_enabled_tools
from ragtime.tools.chart import (CHAT_CHART_DESCRIPTION_SUFFIX,
                                 USERSPACE_CHART_DESCRIPTION_SUFFIX,
                                 create_chart_tool)
from ragtime.tools.datatable import (CHAT_DATATABLE_DESCRIPTION_SUFFIX,
                                     USERSPACE_DATATABLE_DESCRIPTION_SUFFIX,
                                     create_datatable_tool)
from ragtime.tools.filesystem_indexer import search_filesystem_index
from ragtime.tools.git_history import (_is_shallow_repository,
                                       create_aggregate_git_history_tool,
                                       create_per_index_git_history_tool)
from ragtime.tools.influxdb import create_influxdb_tool
from ragtime.tools.mssql import create_mssql_tool
from ragtime.tools.mysql import create_mysql_tool
from ragtime.tools.odoo_shell import filter_odoo_output
from ragtime.userspace.models import (ArtifactType,
                                      UpsertWorkspaceEnvVarRequest,
                                      UpsertWorkspaceFileRequest,
                                      UserSpaceLiveDataCheck,
                                      UserSpaceLiveDataConnection)
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

# Maximum timeout for any tool execution (5 minutes)
# AI can request up to this limit; configured per-tool timeout is the default
MAX_TOOL_TIMEOUT_SECONDS = 300

# Shared FAISS/MMR search tuning constants.
SEARCH_RESULTS_K_MAX = 50
SEARCH_MAX_CHARS_PER_RESULT = 10_000
MMR_FETCH_K_MULTIPLIER = 4
MMR_MIN_FETCH_K = 20

# Maximum idle time (seconds) to wait between streamed chunks from the LLM.
# httpx interprets a flat float as per-operation timeout (connect, read, write, pool),
# NOT as a total request duration.  During streaming, "read=300" means we timeout only
# if 300 s pass with zero data — safe for long reasoning/thinking phases.
LLM_REQUEST_TIMEOUT_SECONDS: float = 300

# Shared image payload caps for all vision-related message content.
# Keep a single source of truth to avoid drift between userspace and chat paths.
IMAGE_PAYLOAD_LIMITS = {
    "max_width": 1024,
    "max_height": 1024,
    "max_pixels": 786_432,
    "max_bytes": 350_000,
}

# Keep at most this many concurrent image downsampling jobs.
IMAGE_DOWNSAMPLE_MAX_CONCURRENCY = 2

INTERNAL_AGENT_CONTINUATION_PROMPT = (
    "Continue the task from the gathered context above. If more verification or "
    "changes are needed, keep working and call additional tools when necessary. "
    "Once you have enough information, send the next user-facing assistant "
    "reply. Do not stop without a user-visible answer."
)
INTERNAL_AGENT_FINAL_SYNTHESIS_PROMPT = (
    "Write the next assistant reply for the user using the verified results "
    "already gathered above. Do not mention internal continuation, hidden "
    "prompts, tool availability, or that tools are unavailable. Answer the "
    "user directly. If the task is incomplete, state the verified status and "
    "the most concrete next step instead of describing internal limitations."
)
MAX_INTERNAL_AGENT_CONTINUATIONS = 2

_USERSPACE_LIVE_DATA_BINDING_VALIDATOR_JS_PATH = Path(__file__).with_name(
    "userspace_live_data_binding_validator.js"
)
_USERSPACE_TYPESCRIPT_VALIDATOR_JS_PATH = Path(__file__).with_name(
    "userspace_typescript_validator.js"
)

USERSPACE_RECENT_FAILURE_LIMIT = 6
USERSPACE_RECENT_FAILURE_PROMPT_LIMIT = 3
USERSPACE_TOOL_REPEAT_THRESHOLD = 2
USERSPACE_TOOL_ROUND_SUMMARY_LIMIT = 10


def escape_prompt_template_braces(text: str) -> str:
    """Escape single braces for LangChain prompt templates while preserving doubles."""
    escaped_open = re.sub(r"(?<!\{)\{(?!\{)", "{{", text)
    return re.sub(r"(?<!\})\}(?!\})", "}}", escaped_open)


def resolve_effective_timeout(requested_timeout: int, timeout_max_seconds: int) -> int:
    """Resolve runtime timeout using per-tool max (0 = unlimited)."""
    requested = max(0, int(requested_timeout))
    max_timeout = max(0, int(timeout_max_seconds))

    if max_timeout == 0:
        return requested
    return min(requested, max_timeout)


def clamp_search_parameters(k: int, max_chars_per_result: int) -> tuple[int, int]:
    """Clamp search parameters to safe limits."""
    clamped_k = max(1, min(SEARCH_RESULTS_K_MAX, int(k)))
    clamped_chars = max(0, min(SEARCH_MAX_CHARS_PER_RESULT, int(max_chars_per_result)))
    return clamped_k, clamped_chars


# =============================================================================
# TOKEN OPTIMIZATION UTILITIES
# =============================================================================


def truncate_tool_output(output: str, max_chars: int) -> str:
    """
    Truncate tool output to a maximum character limit.

    Preserves the beginning and end of the output for context, adding a
    truncation notice in the middle. This helps maintain useful information
    while reducing token consumption.

    Args:
        output: The tool output string to truncate.
        max_chars: Maximum allowed characters (0 = no limit).

    Returns:
        Truncated output string, or original if under limit.
    """
    if max_chars <= 0 or len(output) <= max_chars:
        return output

    # Keep 60% from start, 30% from end (leaving 10% for truncation notice)
    head_chars = int(max_chars * 0.6)
    tail_chars = int(max_chars * 0.3)
    omitted = len(output) - head_chars - tail_chars

    return (
        output[:head_chars]
        + f"\n\n... [{omitted:,} characters omitted] ...\n\n"
        + output[-tail_chars:]
    )


def wrap_tool_with_truncation(tool: StructuredTool, max_chars: int) -> StructuredTool:
    """
    Wrap a LangChain tool to truncate its output before returning to the agent.

    This reduces token consumption in the agent's scratchpad by limiting
    how much of each tool's output is retained for context.

    Args:
        tool: The LangChain StructuredTool to wrap.
        max_chars: Maximum characters for tool output (0 = no limit).

    Returns:
        A new StructuredTool with truncated output behavior.
    """
    if max_chars <= 0:
        return tool

    original_func = tool.func
    original_coroutine = tool.coroutine

    def truncating_func(*args, **kwargs):
        result = original_func(*args, **kwargs)
        if isinstance(result, str):
            return truncate_tool_output(result, max_chars)
        return result

    async def truncating_coroutine(*args, **kwargs):
        result = await original_coroutine(*args, **kwargs)
        if isinstance(result, str):
            return truncate_tool_output(result, max_chars)
        return result

    tool_kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "func": truncating_func if original_func else None,
        "coroutine": truncating_coroutine if original_coroutine else None,
        "args_schema": tool.args_schema,
        "return_direct": tool.return_direct,
        "handle_tool_error": tool.handle_tool_error,
    }
    if hasattr(tool, "handle_validation_error"):
        tool_kwargs["handle_validation_error"] = getattr(
            tool,
            "handle_validation_error",
            False,
        )

    # Some langchain_core versions may not expose handle_validation_error.
    try:
        return StructuredTool(**tool_kwargs)
    except TypeError:
        tool_kwargs.pop("handle_validation_error", None)
        return StructuredTool(**tool_kwargs)


def compress_intermediate_step(
    action: Any, observation: str, max_summary_chars: int = 200
) -> str:
    """
    Compress a single intermediate step into a brief summary.

    Used for the rolling window to summarize older tool calls while
    keeping recent ones in full detail.

    Args:
        action: The agent action (tool call).
        observation: The tool output.
        max_summary_chars: Maximum characters for the observation summary.

    Returns:
        A compressed summary of the step.
    """
    tool_name = getattr(action, "tool", "unknown")
    tool_input = getattr(action, "tool_input", {})

    # Summarize input
    if isinstance(tool_input, dict):
        # Extract key fields for common tool types
        input_summary_parts = []
        for key in ["query", "code", "command", "prompt", "sql"]:
            if key in tool_input:
                val = str(tool_input[key])[:100]
                input_summary_parts.append(f"{key}={val}...")
                break
        if not input_summary_parts:
            input_summary = str(tool_input)[:100]
        else:
            input_summary = ", ".join(input_summary_parts)
    else:
        input_summary = str(tool_input)[:100]

    # Summarize output
    obs_str = str(observation)
    if len(obs_str) > max_summary_chars:
        obs_summary = obs_str[:max_summary_chars] + "..."
    else:
        obs_summary = obs_str

    return (
        f'<tool_use name="{tool_name}">{input_summary}</tool_use>'
        f"<tool_result>{obs_summary}</tool_result>"
    )


def format_scratchpad_with_window(
    intermediate_steps: list,
    window_size: int,
    format_func: Any,
) -> list:
    """
    Format intermediate steps with a rolling window compression.

    Keeps the last `window_size` steps in full detail, compresses older
    steps into brief summaries to reduce token consumption.

    Args:
        intermediate_steps: List of (action, observation) tuples.
        window_size: Number of recent steps to keep in full detail (0 = keep all).
        format_func: Original formatting function for full-detail steps.

    Returns:
        List of messages representing the formatted scratchpad.
    """
    if window_size <= 0 or len(intermediate_steps) <= window_size:
        # No compression needed
        return format_func(intermediate_steps)

    # Split into old (to compress) and recent (keep full)
    old_steps = intermediate_steps[:-window_size]
    recent_steps = intermediate_steps[-window_size:]

    messages = []

    # Add compressed summary of older steps
    if old_steps:
        summaries = []
        for action, observation in old_steps:
            summaries.append(compress_intermediate_step(action, observation))
        summary_text = (
            f"[Prior tool calls ({len(old_steps)} steps, summarized for brevity)]\n"
            + "\n".join(summaries)
        )
        # Add as a system-like message in the scratchpad
        messages.append(AIMessage(content=summary_text))

    # Add recent steps in full detail
    recent_formatted = format_func(recent_steps)
    messages.extend(recent_formatted)

    return messages


def get_process_memory_bytes() -> int:
    """Get current process RSS memory in bytes."""
    try:
        # Try reading from /proc for more accurate Linux stats
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, IndexError):
        pass

    # Fallback to resource module (less accurate on some systems)
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if os.uname().sysname == "Darwin":
            return rusage.ru_maxrss
        return rusage.ru_maxrss * 1024
    except Exception:
        return 0


# Source file extensions eligible for hardcoded data pattern scanning.
_HARDCODED_DATA_SOURCE_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".html",
)


def _resolve_entrypoint_source_file(runtime_config: dict[str, Any]) -> str | None:
    """Best-effort extraction of the main source file from a runtime-entrypoint config.

    Parses the ``command`` field to identify the primary application source file
    for common framework patterns.  Returns ``None`` when the command cannot be
    parsed or no recognisable source file is found.
    """
    if not runtime_config:
        return None
    command = str(runtime_config.get("command") or "").strip()
    if not command:
        return None

    # Strip leading env-var assignments (e.g. FLASK_APP=app.py ...)
    env_source_file: str | None = None
    tokens: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    clean_tokens: list[str] = []
    for token in tokens:
        if "=" in token and not token.startswith("-"):
            # Check for FLASK_APP=app.py pattern
            key, _, value = token.partition("=")
            if key.upper() == "FLASK_APP" and value:
                env_source_file = value
            continue
        clean_tokens.append(token)

    if not clean_tokens:
        return env_source_file

    # Skip the interpreter (python3, node, npx, etc.)
    binary = clean_tokens[0].rsplit("/", 1)[-1]  # basename
    rest = clean_tokens[1:]

    if binary in ("python3", "python"):
        if rest and rest[0] == "-m":
            # python3 -m flask run ... -> use FLASK_APP env or None
            # python3 -m uvicorn main:app ... -> main.py
            if len(rest) >= 2:
                module = rest[1]
                if module == "uvicorn" and len(rest) >= 3:
                    # uvicorn main:app -> main.py
                    app_ref = rest[2].split(":")[0]  # "main:app" -> "main"
                    if app_ref and not app_ref.startswith("-"):
                        return f"{app_ref}.py"
                elif module == "flask":
                    return env_source_file  # rely on FLASK_APP
                elif module == "gunicorn" and len(rest) >= 3:
                    app_ref = rest[2].split(":")[0]
                    if app_ref and not app_ref.startswith("-"):
                        return f"{app_ref}.py"
            return env_source_file
        else:
            # python3 app.py ... or python3 manage.py runserver ...
            for arg in rest:
                if arg.endswith(".py") and not arg.startswith("-"):
                    return arg
        return env_source_file

    if binary in ("node", "nodejs"):
        # node server.js ...
        for arg in rest:
            if not arg.startswith("-") and "." in arg:
                return arg
        return None

    if binary in ("gunicorn", "uvicorn"):
        # gunicorn app:application --bind ... -> app.py
        # uvicorn main:app --host ... -> main.py
        for arg in rest:
            if not arg.startswith("-") and ":" in arg:
                app_ref = arg.split(":")[0]
                if app_ref:
                    return f"{app_ref}.py"
        return env_source_file

    if binary == "npx":
        # npx esbuild dashboard/main.ts --bundle ...
        if rest and rest[0] == "esbuild" and len(rest) >= 2:
            candidate = rest[1]
            if not candidate.startswith("-") and "." in candidate:
                return candidate
        return None

    # Fallback: look for a recognisable source file in the remaining tokens
    for arg in rest:
        if arg.startswith("-"):
            continue
        lower = arg.lower()
        if any(lower.endswith(ext) for ext in _HARDCODED_DATA_SOURCE_EXTENSIONS):
            return arg

    return env_source_file


_HEX_COLOR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})(?![A-Za-z0-9_])"
)
_IMPORT_SPECIFIER_PATTERN = re.compile(
    r"^\s*import(?:\s+type)?(?:[\s\w{},*]+from\s*)?[\"']([^\"']+)[\"']",
    re.MULTILINE,
)

# Patterns that indicate hardcoded mock/sample data in module source.
# Each tuple: (compiled regex, human-readable description)
_HARDCODED_DATA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # getMockData / mockData / sampleData / dummyData function/variable names
    (re.compile(r"\b(?:get)?[Mm]ock[Dd]ata\b"), "mock data function/variable"),
    (re.compile(r"\b(?:get)?[Ss]ample[Dd]ata\b"), "sample data function/variable"),
    (re.compile(r"\b(?:get)?[Dd]ummy[Dd]ata\b"), "dummy data function/variable"),
    (re.compile(r"\b(?:get)?[Ff]ake[Dd]ata\b"), "fake data function/variable"),
    (re.compile(r"\b(?:get)?[Ss]tatic[Dd]ata\b"), "static data function/variable"),
    (re.compile(r"\bplaceholder[Dd]ata\b"), "placeholder data variable"),
    # Large inline array literals with 5+ object elements (e.g., [{...}, {...}, ...])
    (
        re.compile(
            r"=\s*\[\s*(?:\{[^}]{8,}\}\s*,\s*){4,}",
            re.DOTALL,
        ),
        "large inline data array literal (5+ objects)",
    ),
]


def find_hard_coded_hex_colors(content: str) -> list[str]:
    """Return unique hard-coded hex color literals found in content."""
    if not content:
        return []
    seen: set[str] = set()
    results: list[str] = []
    for match in _HEX_COLOR_PATTERN.finditer(content):
        literal = match.group(0)
        normalized = literal.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(literal)
    return results


def find_hardcoded_data_patterns(content: str) -> list[str]:
    """Detect hardcoded mock/sample data patterns in module source code.

    Returns a list of human-readable descriptions of detected patterns.
    """
    if not content:
        return []
    findings: list[str] = []
    for pattern, description in _HARDCODED_DATA_PATTERNS:
        if pattern.search(content):
            findings.append(description)
    return findings


def is_userspace_module_source_path(path: str) -> bool:
    """Return True when the path is a userspace module source file."""
    return (path or "").lower().endswith(USERSPACE_MODULE_SOURCE_EXTENSIONS)


def is_userspace_typescript_path(path: str) -> bool:
    """Return True when the path is a TypeScript userspace file."""
    return (path or "").lower().endswith(USERSPACE_TYPESCRIPT_EXTENSIONS)


def is_userspace_theme_audit_path(path: str) -> bool:
    """Return True when the file should be audited for hardcoded theme colors."""
    return (path or "").lower().endswith(USERSPACE_THEME_AUDIT_EXTENSIONS)


def is_userspace_strict_frontend_path(path: str) -> bool:
    """Return True when strict runtime probe should apply to this frontend file."""
    return (path or "").lower().endswith(USERSPACE_STRICT_FRONTEND_EXTENSIONS)


def validate_userspace_runtime_contract(content: str, file_path: str) -> list[str]:
    """Validate constraints required by the User Space runtime/tooling path."""
    del file_path
    violations: list[str] = []

    # User Space module tooling expects workspace-local import paths.
    imports = _IMPORT_SPECIFIER_PATTERN.findall(content or "")
    unsupported_imports = [
        spec for spec in imports if not spec.startswith(("./", "../", "/"))
    ]
    if unsupported_imports:
        sample = ", ".join(sorted(set(unsupported_imports))[:8])
        violations.append(
            "Unsupported bare imports for userspace module tooling: "
            f"{sample}. Use local workspace modules only."
        )

    # Avoid regex-only JSX heuristics here: template literals can contain
    # HTML-like text in valid .ts code and should not be flagged as JSX.
    # Syntax-level JSX issues are reported by the TypeScript validator.

    return violations


def normalize_runtime_console_errors(console_errors: Any) -> list[str]:
    """Normalize browser console/page errors returned by runtime probes."""
    if not isinstance(console_errors, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in console_errors:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def is_serious_runtime_console_error(message: str) -> bool:
    """Return True for console errors that indicate runtime execution failure."""
    lowered = (message or "").strip().lower()
    if not lowered:
        return False

    serious_markers = (
        "typeerror",
        "referenceerror",
        "syntaxerror",
        "uncaught",
        "unhandled",
        "is not a function",
        "cannot read properties",
        "failed to resolve module specifier",
        "does not provide an export named",
        "cannot use import statement outside a module",
    )
    return any(marker in lowered for marker in serious_markers)


def explain_runtime_console_error(message: str) -> str:
    """Provide a targeted remediation hint for common runtime probe failures."""
    lowered = (message or "").strip().lower()
    if not lowered:
        return ""

    if "getcontext" in lowered and "is not a function" in lowered:
        return (
            "This usually means the DOM lookup returned a non-canvas element. "
            "Create a real <canvas> node and call getContext() on that canvas, "
            "not on the container div."
        )
    if "cannot read properties" in lowered and "null" in lowered:
        return (
            "This usually means a selector lookup returned null. Verify the "
            "target element exists before using it and keep IDs/selectors in sync."
        )
    if "failed to resolve module specifier" in lowered:
        return (
            "Use workspace-local import paths only and ensure the served bundle "
            "uses browser-resolvable module specifiers."
        )
    if "does not provide an export named" in lowered:
        return (
            "The imported module/export shape does not match the code. Verify the "
            "export name and the bundle format served by the runtime."
        )
    return ""


async def validate_live_data_binding(
    content: str,
    file_path: str,
    declared_component_ids: set[str] | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Validate dashboard module content for live-data execute() binding via TypeScript AST.

    Uses the TypeScript compiler to walk the AST and verify that the module
    source structurally contains ``context.components[componentId].execute()``
    call patterns.  This is a deterministic, non-regex check that cannot be
    satisfied by fabricating metadata alone.

    Returns a dict with:
      - has_execute_calls: bool
      - found_component_ids: list[str]
      - has_local_imports: bool
      - has_context_components_access: bool
      - missing_component_ids: list[str]  (declared IDs not found in code)
    """
    _default_fail = {
        "ok": False,
        "validator_available": False,
        "has_execute_calls": False,
        "found_component_ids": [],
        "has_local_imports": False,
        "has_context_components_access": False,
        "missing_component_ids": [],
    }

    if not _USERSPACE_LIVE_DATA_BINDING_VALIDATOR_JS_PATH.exists():
        return {
            **_default_fail,
            "message": "Userspace validator script not found",
        }

    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            str(_USERSPACE_LIVE_DATA_BINDING_VALIDATOR_JS_PATH),
            file_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            **_default_fail,
            "message": "Node runtime unavailable for AST validation",
        }

    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(content.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return {**_default_fail, "message": "AST validation timed out"}

    if process.returncode != 0:
        return {**_default_fail, "message": "AST validation process failed"}

    raw = stdout.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {**_default_fail, "message": "AST validation returned invalid output"}

    if not isinstance(parsed, dict):
        return {
            **_default_fail,
            "message": "AST validation returned unexpected payload",
        }

    # Cross-check declared component IDs against IDs found via AST.
    if declared_component_ids and parsed.get("ok"):
        found_ids = set(parsed.get("found_component_ids", []))
        parsed["missing_component_ids"] = sorted(declared_component_ids - found_ids)
    else:
        parsed.setdefault("missing_component_ids", [])

    return parsed


async def validate_userspace_typescript_content(
    content: str,
    file_path: str,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Validate TypeScript content using frontend's TypeScript compiler diagnostics."""
    contract_violations = validate_userspace_runtime_contract(content, file_path)

    if not _USERSPACE_TYPESCRIPT_VALIDATOR_JS_PATH.exists():
        return {
            "ok": False,
            "validator_available": False,
            "message": "Userspace TypeScript validator script not found",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            str(_USERSPACE_TYPESCRIPT_VALIDATOR_JS_PATH),
            file_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "validator_available": False,
            "message": "Node runtime not available for TypeScript validation",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(content.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return {
            "ok": False,
            "validator_available": False,
            "message": "TypeScript validation timed out",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    if process.returncode != 0:
        return {
            "ok": False,
            "validator_available": False,
            "message": (
                stderr.decode("utf-8", errors="replace")
                or "TypeScript validation failed"
            ).strip(),
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    raw = stdout.decode("utf-8", errors="replace").strip()
    if not raw:
        return {
            "ok": False,
            "validator_available": False,
            "message": "TypeScript validator returned no output",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "validator_available": False,
            "message": "TypeScript validator returned invalid JSON",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "validator_available": False,
            "message": "TypeScript validator returned unexpected payload",
            "errors": contract_violations,
            "contract_errors": contract_violations,
            "contract_error_count": len(contract_violations),
        }

    existing_errors = parsed.get("errors")
    merged_errors = list(existing_errors) if isinstance(existing_errors, list) else []
    for violation in contract_violations:
        if violation not in merged_errors:
            merged_errors.append(violation)

    parsed["errors"] = merged_errors
    parsed["contract_errors"] = contract_violations
    parsed["contract_error_count"] = len(contract_violations)
    parsed["error_count"] = len(merged_errors)
    if contract_violations:
        parsed["ok"] = False

    return parsed


class _CopilotChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant with Copilot-specific enhancements.

    1. **reasoning_text preservation** — Copies the Copilot-specific
       ``reasoning_text`` streaming delta field into ``additional_kwargs``
       so the thinking pipeline can surface it (chat completions path only).

    2. **Automatic Responses API fallback** — If the first request fails with
       ``unsupported_api_for_model`` (model only supports ``/responses``), the
       instance flips ``use_responses_api=True`` and retries transparently.
       The result is cached in ``model_limits`` so future LLM builds pick the
       right endpoint immediately.

     3. **Reasoning parameter downgrade** — If a model rejects strict reasoning
         options (for example ``summary`` or high effort levels), retry once with
         safer reasoning settings instead of failing the whole request.
    """

    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ):
        result = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if result is None or result.message is None:
            return result

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning_text = delta.get("reasoning_text")
            if reasoning_text:
                result.message.additional_kwargs["reasoning_text"] = reasoning_text

        return result

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text:
                        text_parts.append(str(text))
                else:
                    text_parts.append(str(block))
            return "\n".join(part for part in text_parts if part)
        if content is None:
            return ""
        return str(content)

    def _get_request_payload(
        self,
        input_,
        *,
        stop=None,
        **kwargs,
    ) -> dict:
        """Build Responses payloads with system prompts lifted into instructions.

        Copilot/OpenAI reasoning models can suppress reasoning summaries when the
        request includes a system message alongside tool schemas. Translating
        leading system messages into the Responses API ``instructions`` field
        preserves equivalent guidance while restoring reasoning summary events.
        """
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}
        if not self._use_responses_api(payload):
            chat_payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            return chat_payload

        leading_system_messages: list[SystemMessage] = []
        remaining_messages = list(messages)
        while remaining_messages and isinstance(remaining_messages[0], SystemMessage):
            leading_system_messages.append(remaining_messages.pop(0))

        if leading_system_messages:
            instructions_text = "\n\n".join(
                text
                for text in (
                    self._message_content_to_text(message.content)
                    for message in leading_system_messages
                )
                if text
            )
            if instructions_text:
                existing_instructions = payload.get("instructions")
                if existing_instructions:
                    instructions_text = (
                        f"{existing_instructions}\n\n{instructions_text}"
                    )
                payload["instructions"] = instructions_text

        if self.use_previous_response_id:
            last_messages, previous_response_id = _get_last_messages(remaining_messages)
            payload_to_use = (
                last_messages if previous_response_id else remaining_messages
            )
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id
            return _construct_responses_api_payload(payload_to_use, payload)

        return _construct_responses_api_payload(remaining_messages, payload)

    @staticmethod
    def _get_error_body(exc: Exception) -> dict[str, Any] | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            return body

        response = getattr(exc, "response", None)
        if response is not None:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return None

        return None

    @staticmethod
    def _get_error_code(exc: Exception) -> str:
        code = getattr(exc, "code", None)
        if code:
            return str(code).lower()

        body = _CopilotChatOpenAI._get_error_body(exc)
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            return str(error_obj.get("code", "") or "").lower()

        return ""

    @staticmethod
    def _is_unsupported_api_error(exc: Exception) -> bool:
        """Return True if *exc* is an OpenAI ``unsupported_api_for_model`` error."""
        code = _CopilotChatOpenAI._get_error_code(exc)
        if code == "unsupported_api_for_model":
            return True

        body = _CopilotChatOpenAI._get_error_body(exc)
        if isinstance(body, dict):
            error_obj = body.get("error", {})

            # Prefer structured endpoint capability hints when providers include
            # them in error payloads.
            supported_endpoints = error_obj.get("supported_endpoints")
            if isinstance(supported_endpoints, list):
                endpoints = {str(item).lower() for item in supported_endpoints}
                if "/responses" in endpoints and "/chat/completions" not in endpoints:
                    return True

            details = error_obj.get("details")
            if isinstance(details, dict):
                detail_endpoints = details.get("supported_endpoints")
                if isinstance(detail_endpoints, list):
                    endpoints = {str(item).lower() for item in detail_endpoints}
                    if (
                        "/responses" in endpoints
                        and "/chat/completions" not in endpoints
                    ):
                        return True

        return False

    @staticmethod
    def _looks_like_summary_error(exc: Exception) -> bool:
        code = _CopilotChatOpenAI._get_error_code(exc)
        if code in {"unsupported_reasoning_summary", "invalid_reasoning_summary"}:
            return True

        body = _CopilotChatOpenAI._get_error_body(exc)
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            param = str(error_obj.get("param", "") or "").lower()
            if param == "reasoning.summary":
                return True
        return False

    @staticmethod
    def _looks_like_effort_error(exc: Exception) -> bool:
        code = _CopilotChatOpenAI._get_error_code(exc)
        if code in {"invalid_reasoning_effort", "unsupported_reasoning_effort"}:
            return True

        body = _CopilotChatOpenAI._get_error_body(exc)
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            param = str(error_obj.get("param", "") or "").lower()
            if param == "reasoning.effort":
                return True
        return False

    def _request_uses_reasoning_controls(self) -> bool:
        """Return True when this request shape includes reasoning-specific params.

        Restrict speculative ``invalid_request_body`` -> Responses probing to
        requests that actually exercise reasoning controls. This avoids routing
        unrelated request-shape errors for models like Claude Haiku 4.5 onto the
        Responses API and incorrectly caching that endpoint choice.
        """
        reasoning = getattr(self, "reasoning", None)
        if isinstance(reasoning, dict) and reasoning:
            return True

        direct_effort = getattr(self, "reasoning_effort", None)
        if direct_effort:
            return True

        default_params = getattr(self, "_default_params", None)
        if isinstance(default_params, dict):
            if default_params.get("reasoning"):
                return True
            if default_params.get("reasoning_effort"):
                return True

        return False

    def _should_probe_responses_fallback(self, exc: Exception) -> bool:
        """Retry on /responses when chat+tools+reasoning fails structurally."""
        if self.use_responses_api:
            return False

        if not self._request_uses_reasoning_controls():
            return False

        body = self._get_error_body(exc)
        code = self._get_error_code(exc)
        if code == "invalid_request_body":
            return True

        if isinstance(body, dict):
            error_obj = body.get("error", {})
            return (
                str(error_obj.get("code", "") or "").lower() == "invalid_request_body"
            )

        return False

    def _switch_to_responses_api(self, *, cache_result: bool = True) -> None:
        """Flip this instance to use the Responses API.

        When the switch is based on a speculative ``invalid_request_body`` probe,
        do not update the shared capability cache. Only authoritative endpoint
        signals should be persisted across future requests.
        """
        self.use_responses_api = True
        self.output_version = "responses/v1"
        if cache_result:
            register_model_supported_endpoints(self.model_name, ["/responses"])
        logger.info(
            "Model %s does not support /chat/completions — "
            "switched to Responses API for this and future requests",
            self.model_name,
        )

    def _downgrade_reasoning_parameters(self, exc: Exception) -> bool:
        """Relax reasoning options for provider/model compatibility.

        Returns True when parameters were changed and a retry should be
        attempted.
        """
        reasoning = getattr(self, "reasoning", None)
        if not isinstance(reasoning, dict):
            return False
        reasoning_payload: dict[str, Any] = reasoning

        changed = False
        if self._looks_like_summary_error(exc) and "summary" in reasoning_payload:
            reasoning_payload.pop("summary", None)
            changed = True
            logger.info(
                "Model %s rejected reasoning summary; retrying without summary",
                self.model_name,
            )

        if self._looks_like_effort_error(exc):
            effort = str(reasoning_payload.get("effort", "")).lower()
            if effort in {"high", "xhigh"}:
                reasoning_payload["effort"] = "medium"
                changed = True
                logger.info(
                    "Model %s rejected reasoning effort=%s; retrying with effort=medium",
                    self.model_name,
                    effort,
                )

        return changed

    async def _retry_after_reasoning_downgrade(
        self,
        call,
        *args,
        **kwargs,
    ):
        try:
            return await call(*args, **kwargs)
        except Exception as exc:
            if self.use_responses_api and self._downgrade_reasoning_parameters(exc):
                return await call(*args, **kwargs)
            raise

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return await self._retry_after_reasoning_downgrade(
                super()._agenerate,
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except Exception as exc:
            unsupported_api = self._is_unsupported_api_error(exc)
            probe_responses = self._should_probe_responses_fallback(exc)
            if not self.use_responses_api and (unsupported_api or probe_responses):
                self._switch_to_responses_api(cache_result=unsupported_api)
                return await self._retry_after_reasoning_downgrade(
                    super()._agenerate,
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            raise

    async def _astream(self, *args, **kwargs):
        try:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk
        except Exception as exc:
            if self.use_responses_api and self._downgrade_reasoning_parameters(exc):
                async for chunk in super()._astream(*args, **kwargs):
                    yield chunk
                return

            unsupported_api = self._is_unsupported_api_error(exc)
            probe_responses = self._should_probe_responses_fallback(exc)
            if not self.use_responses_api and (unsupported_api or probe_responses):
                self._switch_to_responses_api(cache_result=unsupported_api)
                try:
                    async for chunk in super()._astream(*args, **kwargs):
                        yield chunk
                except Exception as retry_exc:
                    if self.use_responses_api and self._downgrade_reasoning_parameters(
                        retry_exc
                    ):
                        async for chunk in super()._astream(*args, **kwargs):
                            yield chunk
                    else:
                        raise
            else:
                raise


class RAGComponents:
    """Container for RAG components initialized at startup.

    Supports progressive initialization with background FAISS loading:
    - core_ready: LLM and settings loaded, can serve non-indexed queries
    - indexes_ready: All FAISS indexes loaded, full knowledge search available
    - is_ready: Alias for core_ready (API can serve requests)

    UI vs API/MCP:
    - agent_executor: Standard agent for API/MCP requests
    - agent_executor_ui: Agent with UI-only tools (charts) for chat UI requests
    """

    def __init__(self):
        self.retrievers: dict[str, Any] = {}
        self.faiss_dbs: dict[str, Any] = (
            {}
        )  # Raw FAISS vectorstores for dynamic k searches
        self.agent_executor: Optional[AgentExecutor] = None
        self.agent_executor_ui: Optional[AgentExecutor] = (
            None  # UI-only agent with chart tool
        )
        self.llm: Optional[Any] = None  # ChatOpenAI, ChatAnthropic, or ChatOllama
        self._core_ready: bool = False  # LLM/settings ready
        self._indexes_ready: bool = False  # All FAISS indexes loaded
        self._indexes_loading: bool = False  # Background loading in progress
        self._indexes_total: int = 0  # Total indexes to load
        self._indexes_loaded: int = 0  # Indexes loaded so far
        self._app_settings: Optional[dict] = None
        self._tool_configs: Optional[List[dict]] = None
        self._index_metadata: Optional[List[dict]] = None
        self._system_prompt: str = BASE_CHAT_SYSTEM_PROMPT
        self._system_prompt_ui: str = BASE_CHAT_SYSTEM_PROMPT  # Includes UI additions
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._init_in_progress: bool = False
        self._embedding_model: Optional[Any] = None  # Cached for background loading
        # Detailed index loading tracking
        self._index_details: dict[str, dict] = (
            {}
        )  # name -> {status, size_mb, chunk_count, load_time, error}
        self._loading_index: Optional[str] = None  # Currently loading index name
        self._faiss_loading_task: Optional[asyncio.Task] = None  # prevent GC
        self._ollama_warmup_task: Optional[asyncio.Task] = None  # prevent GC
        # Token optimization settings
        self._scratchpad_window_size: int = 6  # Default, updated from settings
        # Request-scoped prompt fragments cache
        self._request_prompt_cache: dict[tuple[Any, ...], str] = {}
        # Track the Copilot HMAC token baked into the cached LLM so we can
        # detect when it changes (expiry refresh or re-authorization) and
        # rebuild the LLM transparently.
        self._copilot_llm_token: Optional[str] = None
        self._image_downsample_semaphore = asyncio.Semaphore(
            IMAGE_DOWNSAMPLE_MAX_CONCURRENCY
        )
        self._userspace_recent_failures: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def _stable_tool_signature(
        tool_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str:
        payload = {
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _parse_json_object(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw.startswith("{"):
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def _classify_userspace_failure(cls, *messages: Any) -> str:
        combined = " ".join(str(message or "") for message in messages).lower()
        if not combined:
            return "validation_failed"
        if "repeated identical tool call" in combined or "strategy failure" in combined:
            return "repeated_tool_call"
        if (
            "no server-verified execution proof" in combined
            or "execution proof" in combined
        ):
            return "execution_proof_missing"
        if (
            "no runnable web entrypoint found" in combined
            or "missing runtime entrypoint" in combined
        ):
            return "entrypoint_missing"
        if ".ragtime/runtime-entrypoint.json" in combined and (
            "invalid" in combined
            or "parsing failed" in combined
            or "invalid json" in combined
        ):
            return "entrypoint_invalid"
        if "entry-point wiring required" in combined:
            return "entrypoint_wiring_required"
        if "live_data_connections" in combined:
            return "live_connections_missing"
        if "live_data_checks" in combined:
            return "live_checks_missing"
        if "context.components" in combined or "live data binding" in combined:
            return "live_binding_missing"
        if "hardcoded data" in combined or "mock data" in combined:
            return "hardcoded_data_detected"
        if (
            "unable to resolve local import" in combined
            or "unsupported bare imports" in combined
        ):
            return "import_contract_error"
        if "typescript" in combined or "contract violation" in combined:
            return "typescript_error"
        if "blank page" in combined or "blank screen" in combined:
            return "runtime_blank_preview"
        if "browser console" in combined or "javascript exception" in combined:
            return "runtime_console_error"
        if "directory listing" in combined:
            return "runtime_entrypoint_misconfigured"
        if "devserver is not running" in combined:
            return "runtime_unavailable"
        if (
            "invalid live_data_connections component_id" in combined
            or "invalid live_data_checks component_id" in combined
        ):
            return "invalid_component_id"
        if "invalid file path" in combined or "does not exist" in combined:
            return "path_invalid"
        return "validation_failed"

    @staticmethod
    def _next_best_tool_for_failure(
        failure_class: str, fallback_tool: str = "patch_userspace_file"
    ) -> str:
        if failure_class in {"repeated_tool_call", "path_invalid"}:
            return "read_userspace_file"
        if failure_class in {
            "live_connections_missing",
            "live_checks_missing",
            "live_binding_missing",
            "execution_proof_missing",
        }:
            return "assay_userspace_code"
        if failure_class in {
            "runtime_blank_preview",
            "runtime_console_error",
            "runtime_unavailable",
        }:
            return "run_terminal_command"
        if failure_class in {
            "entrypoint_missing",
            "entrypoint_invalid",
            "entrypoint_wiring_required",
        }:
            return "upsert_userspace_file"
        if failure_class in {
            "typescript_error",
            "import_contract_error",
            "hardcoded_data_detected",
        }:
            return fallback_tool
        return fallback_tool

    def _record_userspace_failure(
        self,
        workspace_id: str,
        *,
        failure_class: str,
        summary: str,
        tool_name: str | None = None,
        resolved: bool = False,
    ) -> None:
        if not workspace_id or not summary.strip():
            return

        entries = self._userspace_recent_failures.setdefault(workspace_id, [])
        normalized_summary = summary.strip()
        for entry in reversed(entries):
            if (
                entry.get("failure_class") == failure_class
                and entry.get("summary") == normalized_summary
                and entry.get("tool_name") == tool_name
            ):
                entry["resolved"] = resolved
                entry["updated_at"] = time.time()
                return

        entries.append(
            {
                "failure_class": failure_class,
                "summary": normalized_summary,
                "tool_name": tool_name,
                "resolved": resolved,
                "updated_at": time.time(),
            }
        )
        if len(entries) > USERSPACE_RECENT_FAILURE_LIMIT:
            del entries[:-USERSPACE_RECENT_FAILURE_LIMIT]

    def _mark_userspace_failures_resolved(
        self,
        workspace_id: str,
        resolution_summary: str,
    ) -> None:
        entries = self._userspace_recent_failures.get(workspace_id)
        if not entries:
            return
        now = time.time()
        for entry in entries:
            if not entry.get("resolved"):
                entry["resolved"] = True
                entry["resolution_summary"] = resolution_summary
                entry["updated_at"] = now

    def _get_userspace_recent_failure_summaries(self, workspace_id: str) -> list[str]:
        entries = self._userspace_recent_failures.get(workspace_id, [])
        summaries: list[str] = []
        for entry in sorted(
            entries,
            key=lambda item: float(item.get("updated_at", 0)),
            reverse=True,
        )[:USERSPACE_RECENT_FAILURE_PROMPT_LIMIT]:
            failure_class = str(entry.get("failure_class", "validation_failed"))
            summary = str(entry.get("summary", "")).strip()
            if not summary:
                continue
            status = "resolved" if entry.get("resolved") else "unresolved"
            summaries.append(f"{failure_class}: {summary} [{status}]")
        return summaries

    @classmethod
    def _group_userspace_validation_diagnostics(
        cls,
        *,
        errors: list[str],
        contract_errors: list[str],
        runtime_errors: list[str],
        runtime_warnings: list[str],
    ) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "entrypoint": [],
            "live_data": [],
            "imports": [],
            "typescript": [],
            "runtime": [],
            "policy": [],
            "other": [],
        }

        def add(group: str, message: str) -> None:
            text = str(message or "").strip()
            if text and text not in groups[group]:
                groups[group].append(text)

        for message in errors:
            failure_class = cls._classify_userspace_failure(message)
            if failure_class.startswith("entrypoint"):
                add("entrypoint", message)
            elif failure_class in {
                "live_connections_missing",
                "live_checks_missing",
                "live_binding_missing",
                "execution_proof_missing",
                "invalid_component_id",
            }:
                add("live_data", message)
            elif failure_class == "import_contract_error":
                add("imports", message)
            elif failure_class in {
                "runtime_blank_preview",
                "runtime_console_error",
                "runtime_unavailable",
                "runtime_entrypoint_misconfigured",
            }:
                add("runtime", message)
            elif failure_class == "hardcoded_data_detected":
                add("policy", message)
            elif failure_class == "typescript_error":
                add("typescript", message)
            else:
                add("other", message)

        for message in contract_errors:
            add(
                (
                    "live_data"
                    if "live_data" in message.lower()
                    or "context.components" in message.lower()
                    else "typescript"
                ),
                message,
            )
        for message in runtime_errors:
            add("runtime", message)
        for message in runtime_warnings:
            add("runtime", message)

        return {key: value for key, value in groups.items() if value}

    def _wrap_runtime_tools_with_request_state(
        self,
        tools: list[Any],
        *,
        mode: str,
        workspace_id: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        request_state: dict[str, Any] = {
            "tool_calls": cast(list[dict[str, Any]], []),
            "signature_counts": cast(dict[str, int], {}),
            "blocked_repeat_calls": 0,
            "max_iterations_reached": False,
            "internal_continue_attempts": 0,
            "internal_continue_stop_reason": "",
            "tool_free_synthesis_used": False,
        }
        if mode != "userspace" or not tools:
            return tools, request_state

        repeat_threshold = USERSPACE_TOOL_REPEAT_THRESHOLD
        wrapped_tools: list[Any] = []

        for tool in tools:
            tool_name = getattr(tool, "name", "")
            original_coroutine = getattr(tool, "coroutine", None)
            if not tool_name or original_coroutine is None:
                wrapped_tools.append(tool)
                continue

            async def guarded_coroutine(
                *args: Any,
                _tool_name: str = tool_name,
                _orig=original_coroutine,
                **kwargs: Any,
            ) -> Any:
                signature = self._stable_tool_signature(_tool_name, args, kwargs)
                signature_counts = cast(
                    dict[str, int], request_state["signature_counts"]
                )
                call_count = int(signature_counts.get(signature, 0)) + 1
                signature_counts[signature] = call_count

                if call_count > repeat_threshold:
                    failure_class = "repeated_tool_call"
                    payload = {
                        "tool": _tool_name,
                        "status": "rejected_not_persisted",
                        "rejected": True,
                        "persisted": False,
                        "retryable": False,
                        "failure_class": failure_class,
                        "message": (
                            f"Repeated identical tool call blocked for {_tool_name}. "
                            "Read the latest tool output and change strategy before retrying."
                        ),
                        "action_required": (
                            "Use the latest diagnostics or file contents to choose a different next step before calling this tool again."
                        ),
                        "next_best_tool": self._next_best_tool_for_failure(
                            failure_class,
                            fallback_tool="assay_userspace_code",
                        ),
                        "diagnostics": {
                            "tool": _tool_name,
                            "repeat_count": call_count,
                            "signature": signature,
                        },
                    }
                    request_state["blocked_repeat_calls"] += 1
                    request_state["tool_calls"].append(
                        {
                            "tool": _tool_name,
                            "signature": signature,
                            "repeat_count": call_count,
                            "status": payload["status"],
                            "failure_class": failure_class,
                            "blocked": True,
                        }
                    )
                    if workspace_id:
                        self._record_userspace_failure(
                            workspace_id,
                            failure_class=failure_class,
                            summary=f"Repeated {_tool_name} without new inputs.",
                            tool_name=_tool_name,
                        )
                    return json.dumps(payload, indent=2)

                result = await _orig(*args, **kwargs)
                parsed = self._parse_json_object(result)
                status = "completed"
                failure_class = None
                if parsed:
                    status = str(parsed.get("status") or status)
                    failure_class = (
                        str(parsed.get("failure_class") or "").strip() or None
                    )
                    if not failure_class and (
                        parsed.get("rejected")
                        or parsed.get("persisted_with_violations")
                    ):
                        failure_class = self._classify_userspace_failure(
                            parsed.get("message"),
                            parsed.get("error"),
                            parsed.get("contract_violations"),
                        )

                request_state["tool_calls"].append(
                    {
                        "tool": _tool_name,
                        "signature": signature,
                        "repeat_count": call_count,
                        "status": status,
                        "failure_class": failure_class,
                        "blocked": False,
                    }
                )

                if (
                    workspace_id
                    and failure_class
                    and parsed
                    and (
                        parsed.get("rejected")
                        or parsed.get("persisted_with_violations")
                    )
                ):
                    summary = str(
                        parsed.get("message") or parsed.get("error") or failure_class
                    )
                    self._record_userspace_failure(
                        workspace_id,
                        failure_class=failure_class,
                        summary=summary,
                        tool_name=_tool_name,
                    )

                return result

            wrapped_tools.append(
                self._clone_structured_tool(
                    tool,
                    coroutine=guarded_coroutine,
                    func=getattr(tool, "func", None),
                )
            )

        return wrapped_tools, request_state

    def _build_request_debug_metadata(
        self,
        *,
        mode: str,
        request_tool_state: dict[str, Any] | None,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "mode": mode,
        }
        state = request_tool_state or {}
        if state:
            tool_calls = list(state.get("tool_calls") or [])
            metadata["tool_rounds"] = tool_calls[-USERSPACE_TOOL_ROUND_SUMMARY_LIMIT:]
            metadata["blocked_repeat_calls"] = int(state.get("blocked_repeat_calls", 0))
            metadata["max_iterations_reached"] = bool(
                state.get("max_iterations_reached", False)
            )
            metadata["internal_continue_attempts"] = int(
                state.get("internal_continue_attempts", 0)
            )
            metadata["internal_continue_stop_reason"] = str(
                state.get("internal_continue_stop_reason", "")
            )
            metadata["tool_free_synthesis_used"] = bool(
                state.get("tool_free_synthesis_used", False)
            )

        if mode == "userspace":
            metadata["userspace_agent"] = {
                "tool_repeat_threshold": USERSPACE_TOOL_REPEAT_THRESHOLD,
                "recent_failures": (
                    self._get_userspace_recent_failure_summaries(workspace_id or "")
                    if workspace_id
                    else []
                ),
            }
        return metadata

    def _get_image_payload_limits(self) -> dict[str, int]:
        """Resolve image payload limits from settings with sane bounds."""
        app_settings = self._app_settings or {}

        def _get_int(
            settings_key: str, limit_key: str, minimum: int, maximum: int
        ) -> int:
            raw = app_settings.get(settings_key, IMAGE_PAYLOAD_LIMITS[limit_key])
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = IMAGE_PAYLOAD_LIMITS[limit_key]
            return min(maximum, max(minimum, value))

        return {
            "max_width": _get_int("image_payload_max_width", "max_width", 320, 4096),
            "max_height": _get_int("image_payload_max_height", "max_height", 240, 4096),
            "max_pixels": _get_int(
                "image_payload_max_pixels", "max_pixels", 76_800, 8_000_000
            ),
            "max_bytes": _get_int(
                "image_payload_max_bytes", "max_bytes", 50_000, 5_000_000
            ),
        }

    def _schedule_ollama_warmup(self) -> None:
        """Start best-effort Ollama warmup without blocking API startup."""
        if self._ollama_warmup_task and not self._ollama_warmup_task.done():
            return
        self._ollama_warmup_task = asyncio.create_task(self._warmup_ollama_models())

    async def _warmup_ollama_models(self) -> None:
        """Warm LLM/embedding Ollama models with bounded timeout and no raises."""
        if not self._app_settings:
            return

        llm_provider = self._app_settings.get("llm_provider", "openai").lower()
        embedding_provider = self._app_settings.get(
            "embedding_provider", "ollama"
        ).lower()

        tasks: list[Any] = []

        if llm_provider == "ollama":
            llm_model = self._app_settings.get("llm_model", "qwen3.5:latest")
            llm_base_url = self._app_settings.get(
                "llm_ollama_base_url",
                self._app_settings.get("ollama_base_url", "http://localhost:11434"),
            )
            tasks.append(
                warmup_model(
                    llm_model,
                    llm_base_url,
                    timeout_seconds=DEFAULT_WARMUP_TIMEOUT_SECONDS,
                )
            )

        if embedding_provider == "ollama":
            embedding_model = self._app_settings.get(
                "embedding_model", "nomic-embed-text"
            )
            embedding_base_url = self._app_settings.get(
                "ollama_base_url", "http://localhost:11434"
            )
            tasks.append(
                warmup_embedding_model(
                    embedding_model,
                    embedding_base_url,
                    timeout_seconds=DEFAULT_WARMUP_TIMEOUT_SECONDS,
                )
            )

        if not tasks:
            return

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            # Defensive safety: warmup is an optimization only.
            logger.debug(f"Ollama warmup background task failed: {e}")

    @property
    def is_ready(self) -> bool:
        """API can serve requests when core is ready."""
        return self._core_ready

    @is_ready.setter
    def is_ready(self, value: bool):
        """Setter for backwards compatibility."""
        self._core_ready = value

    @property
    def indexes_ready(self) -> bool:
        """All FAISS indexes are loaded."""
        return self._indexes_ready

    @property
    def loading_status(self) -> dict:
        """Get current loading status for health endpoint."""
        return {
            "core_ready": self._core_ready,
            "indexes_ready": self._indexes_ready,
            "indexes_loading": self._indexes_loading,
            "indexes_total": self._indexes_total,
            "indexes_loaded": self._indexes_loaded,
            "retrievers_available": list(self.retrievers.keys()),
            "index_details": list(self._index_details.values()),
            "loading_index": self._loading_index,
            "sequential_loading": (
                self._app_settings.get("sequential_index_loading", False)
                if self._app_settings
                else False
            ),
        }

    def unload_index(self, name: str) -> bool:
        """Remove an index from memory.

        Args:
            name: Index name to unload

        Returns:
            True if the index was unloaded, False if not found
        """
        unloaded = False
        if name in self.retrievers:
            del self.retrievers[name]
            logger.info(f"Unloaded index '{name}' from retrievers")
            unloaded = True

        if name in self.faiss_dbs:
            del self.faiss_dbs[name]
            logger.debug(f"Removed index '{name}' from faiss_dbs")

        if name in self._index_details:
            del self._index_details[name]
            logger.debug(f"Removed index '{name}' from index_details tracking")

        return unloaded

    def rename_index(self, old_name: str, new_name: str) -> bool:
        """Rename an index in memory (reuse loaded FAISS data).

        Args:
            old_name: Current index name
            new_name: New index name

        Returns:
            True if the index was renamed, False if not found
        """
        if old_name not in self.retrievers:
            return False

        # Move retriever to new key
        self.retrievers[new_name] = self.retrievers.pop(old_name)
        logger.info(f"Renamed index '{old_name}' to '{new_name}' in retrievers")

        # Also move faiss_dbs
        if old_name in self.faiss_dbs:
            self.faiss_dbs[new_name] = self.faiss_dbs.pop(old_name)
            logger.debug(f"Renamed index in faiss_dbs")

        # Also move index_details tracking
        if old_name in self._index_details:
            self._index_details[new_name] = self._index_details.pop(old_name)
            self._index_details[new_name]["name"] = new_name
            logger.debug(f"Renamed index in index_details tracking")

        return True

    async def rebuild_agent(self) -> None:
        """Rebuild the agent with current tools and retrievers.

        Use this instead of full initialize() when only tool/index changes
        need to be reflected without reloading all indexes from disk.
        """
        await self._create_agent()

    async def initialize(self):
        """Initialize all RAG components.

        Uses a lock to prevent concurrent initializations - if another init
        is in progress, this call will wait for it to complete rather than
        starting a duplicate initialization.
        """
        # Fast path: if init is already in progress, just wait for it
        if self._init_in_progress:
            logger.debug("RAG initialization already in progress, waiting...")
            async with self._init_lock:
                # Lock acquired means the other init finished
                logger.debug("RAG initialization completed by another caller")
                return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._init_in_progress:
                logger.debug("RAG initialization completed by another caller")
                return

            self._init_in_progress = True
            try:
                await self._do_initialize()
            finally:
                self._init_in_progress = False

    async def _do_initialize(self):
        """Perform the actual RAG initialization.

        Uses a two-phase approach:
        1. Core init (blocking): Load settings, LLM, tools - marks core_ready
        2. Index loading (background): Load FAISS indexes in parallel - marks indexes_ready

        This allows the API to serve requests immediately while indexes load.
        """
        start_time = time.time()
        logger.info("Initializing RAG components (core)...")

        # Load settings from database
        self._app_settings = await get_app_settings()
        self._tool_configs = await get_tool_configs()
        self._index_metadata = await self._load_index_metadata()
        self._request_prompt_cache.clear()

        # Build system prompts with tool and index descriptions
        tool_prompt_section = build_tool_system_prompt(self._tool_configs)
        index_prompt_section = build_index_system_prompt(self._index_metadata or [])

        # Base system prompt (for API/MCP)
        self._system_prompt = (
            BASE_CHAT_SYSTEM_PROMPT + index_prompt_section + tool_prompt_section
        )

        # UI system prompt (includes common visualization instructions)
        self._system_prompt_ui = (
            BASE_CHAT_SYSTEM_PROMPT
            + index_prompt_section
            + tool_prompt_section
            + UI_VISUALIZATION_COMMON_PROMPT
        )

        # Add visibility prompt when mode is 'auto' (AI decides)
        if self._app_settings.get("tool_output_mode", "default") == "auto":
            self._system_prompt_ui += TOOL_OUTPUT_VISIBILITY_PROMPT

        # Initialize LLM based on provider from database settings
        await self._init_llm()

        # Load embedding model (needed for FAISS loading)
        self._embedding_model = await self._get_embedding_model()

        # Create the agent with tools (without FAISS retrievers for now)
        # This allows non-indexed queries to work immediately
        await self._create_agent()

        # Mark core as ready - API can now serve requests
        self._core_ready = True
        core_time = time.time() - start_time
        logger.info(
            f"RAG core initialized in {core_time:.1f}s - API ready (indexes loading in background)"
        )

        # Warmup is a best-effort optimization and should never block startup.
        self._schedule_ollama_warmup()

        # Start background FAISS loading — hold strong reference to prevent GC
        self._faiss_loading_task = asyncio.create_task(
            self._load_faiss_indexes_background()
        )

    async def _load_faiss_indexes_background(self):
        """Load FAISS indexes in background.

        This runs after core init completes, loading indexes without blocking
        the API. Supports both parallel loading (faster, higher peak RAM) and
        sequential loading (slower, lower peak RAM - loads smallest first).
        """
        start_time = time.time()

        if not self._embedding_model:
            logger.warning("No embedding model available for FAISS loading")
            self._indexes_ready = True
            return

        try:
            self._indexes_loading = True
            sequential = self._app_settings.get("sequential_index_loading", False)
            if sequential:
                await self._load_faiss_indexes_sequential(self._embedding_model)
            else:
                await self._load_faiss_indexes_parallel(self._embedding_model)
        except Exception as e:
            logger.error(f"Error in background FAISS loading: {e}")
        finally:
            self._indexes_loading = False
            self._indexes_ready = True

            # Rebuild agent with updated retrievers
            try:
                await self._create_agent()
            except Exception as e:
                logger.error(f"Failed to rebuild agent after index loading: {e}")

            elapsed = time.time() - start_time
            logger.info(
                f"FAISS indexes loaded in background ({elapsed:.1f}s): "
                f"{len(self.retrievers)} document index(es) ready"
            )

            # Also load any FAISS-based filesystem indexes
            await self._load_filesystem_faiss_indexes()

    async def _load_filesystem_faiss_indexes(self):
        """Load FAISS-based filesystem indexes at startup.

        Filesystem indexes can use either pgvector or FAISS. For FAISS-based
        ones, we need to load them into memory like document indexes.

        Note: Since filesystem and document indexes share the same /data/ folder,
        we skip any indexes that exist in index_metadata (those are document indexes).
        """
        if not self._embedding_model:
            return

        try:
            faiss_backend = get_faiss_backend()
            all_disk_indexes = FaissBackend.list_disk_indexes()

            if not all_disk_indexes:
                logger.debug("No FAISS indexes found on disk")
                return

            # Filter out document indexes (those in index_metadata)
            document_index_names = set()
            if self._index_metadata:
                document_index_names = {
                    idx.get("name") for idx in self._index_metadata if idx.get("name")
                }

            # Filesystem indexes are those not in document index metadata
            disk_indexes = [
                name for name in all_disk_indexes if name not in document_index_names
            ]

            if not disk_indexes:
                logger.debug("No FAISS filesystem indexes to load")
                return

            logger.info(f"Loading {len(disk_indexes)} FAISS filesystem index(es)...")
            loaded = 0
            for index_name in disk_indexes:
                try:
                    start_time = time.time()
                    success = await faiss_backend.load_index(
                        index_name, self._embedding_model
                    )
                    if success:
                        loaded += 1
                        load_time = time.time() - start_time
                        # Get stats for the loaded index
                        stats = await faiss_backend.get_index_stats(index_name)
                        # Track in index_details (same namespace as document indexes)
                        self._index_details[index_name] = {
                            "name": index_name,
                            "display_name": index_name,
                            "status": "loaded",
                            "type": "filesystem_faiss",
                            "chunk_count": stats.get("embedding_count", 0),
                            "size_mb": stats.get("size_mb"),
                            "load_time_seconds": load_time,
                            "error": None,
                        }
                except Exception as e:
                    logger.warning(
                        f"Failed to load FAISS filesystem index {index_name}: {e}"
                    )
                    # Track failed load
                    self._index_details[index_name] = {
                        "name": index_name,
                        "display_name": index_name,
                        "status": "error",
                        "type": "filesystem_faiss",
                        "chunk_count": None,
                        "size_mb": None,
                        "load_time_seconds": None,
                        "error": str(e),
                    }

            if loaded > 0:
                logger.info(f"Loaded {loaded} FAISS filesystem index(es)")

        except Exception as e:
            logger.error(f"Error loading FAISS filesystem indexes: {e}")

    async def _init_llm(self):
        """Initialize LLM based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("llm_provider", "openai").lower()
        model = self._app_settings.get("llm_model", "gpt-4-turbo")
        max_tokens = await self._resolve_llm_max_tokens(provider, model)
        self.llm = await self._build_llm(provider, model, max_tokens)

        if self.llm is None:
            logger.warning(
                "No usable LLM configured - chat features will be disabled until provider credentials and model settings are valid"
            )
            return

        if provider == "ollama":
            logger.info(f"Using Ollama LLM: {model}")
        elif provider == "anthropic":
            logger.info(f"Using Anthropic LLM: {model}")
        elif provider == "github_models":
            logger.info(f"Using GitHub Models LLM: {model}")
        elif provider == "github_copilot":
            logger.info(f"Using GitHub Copilot LLM: {model}")
        elif hasattr(self.llm, "model_name"):
            logger.info(f"Using OpenAI LLM: {self.llm.model_name}")
        else:
            logger.info(f"Using OpenAI LLM: {model}")

    async def _resolve_llm_max_tokens(self, provider: str, model: str) -> int:
        """Resolve max tokens for an LLM request using configured limits."""
        assert self._app_settings is not None
        max_tokens = self._app_settings.get("llm_max_tokens", 4096)

        if max_tokens < 100000:
            return max_tokens

        if provider == "ollama":
            base_url = self._app_settings.get(
                "llm_ollama_base_url",
                self._app_settings.get("ollama_base_url", "http://localhost:11434"),
            )
            detected_limit = await get_model_context_length(model, base_url)
            if detected_limit:
                logger.info(
                    f"Using detected context limits for Ollama model {model}: {detected_limit}"
                )
                return detected_limit

            logger.warning(
                f"Could not detect limits for Ollama model {model}, using default 4096"
            )
            return 4096

        detected_limit = await get_output_limit(model)
        if detected_limit:
            logger.info(f"Using detected output limit for {model}: {detected_limit}")
            return detected_limit

        logger.warning(f"Could not detect output limit for {model}, using default 4096")
        return 4096

    async def _build_llm(
        self,
        provider: str,
        model: str,
        max_tokens: int,
    ) -> Optional[Any]:
        """Build an LLM client for a provider/model pair, or return None when unavailable."""
        assert self._app_settings is not None
        provider_normalized = provider.lower().strip()

        async def _hydrate_openai_compatible_capabilities(
            *,
            metadata_urls: list[str],
            headers: dict[str, str],
            requested_model: str,
        ) -> None:
            """Populate endpoint/reasoning capability caches from provider APIs."""

            def _extract_rows(payload: Any) -> list[dict[str, Any]]:
                if isinstance(payload, dict):
                    data = payload.get("data", payload)
                    if isinstance(data, list):
                        return [row for row in data if isinstance(row, dict)]
                    return []
                if isinstance(payload, list):
                    return [row for row in payload if isinstance(row, dict)]
                return []

            def _normalized_model_id(raw_model_id: str) -> str:
                value = str(raw_model_id or "").strip()
                value = re.sub(r"/+", "/", value).lstrip("/")
                if "/" in value:
                    _, _, short_id = value.partition("/")
                    return short_id or value
                return value

            requested_variants = {
                _normalized_model_id(requested_model).lower(),
                str(requested_model or "").strip().lower(),
            }

            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    for url in metadata_urls:
                        try:
                            response = await client.get(url, headers=headers)
                            if response.status_code == 404:
                                continue
                            response.raise_for_status()
                            rows = _extract_rows(response.json())
                        except Exception:
                            continue

                        for row in rows:
                            row_model_id = str(row.get("id", "") or "").strip()
                            if not row_model_id:
                                continue

                            normalized_row_id = _normalized_model_id(row_model_id)
                            row_variants = {
                                row_model_id.lower(),
                                normalized_row_id.lower(),
                            }
                            if requested_variants.isdisjoint(row_variants):
                                continue

                            supported_endpoints = row.get("supportedEndpoints")
                            if isinstance(supported_endpoints, list):
                                register_model_supported_endpoints(
                                    normalized_row_id, supported_endpoints
                                )

                            capabilities = row.get("capabilities")
                            if isinstance(capabilities, dict):
                                supports_obj = capabilities.get("supports")
                                supports_flags: list[str] = []
                                if isinstance(supports_obj, list):
                                    supports_flags = [
                                        str(flag).lower() for flag in supports_obj
                                    ]
                                elif isinstance(supports_obj, dict):
                                    supports_flags = [
                                        str(flag).lower()
                                        for flag, enabled in supports_obj.items()
                                        if enabled
                                    ]

                                if supports_flags:
                                    register_model_reasoning_capabilities(
                                        normalized_row_id,
                                        reasoning_supported=(
                                            "reasoning_effort" in supports_flags
                                            or "reasoning" in supports_flags
                                        ),
                                        thinking_budget_supported=(
                                            "thinking_budget" in supports_flags
                                            or "max_thinking_budget" in supports_flags
                                        ),
                                    )

                            return
            except Exception:
                # Capability hydration is best-effort and must not block request setup.
                return

        if provider_normalized in {"github_copilot", "github_models"}:
            model = model.lstrip("/")

        if provider_normalized == "ollama":
            try:
                base_url = self._app_settings.get(
                    "llm_ollama_base_url",
                    self._app_settings.get("ollama_base_url", "http://localhost:11434"),
                )
                # Detect thinking capability so we can surface reasoning
                # tokens to the UI instead of silently dropping them.
                reasoning = None
                try:
                    details = await get_model_details(model, base_url)
                    if has_capability(details, "thinking"):
                        reasoning = True
                        logger.info(
                            f"Ollama model '{model}' supports thinking; enabling reasoning mode"
                        )
                except Exception:
                    pass  # Non-critical; default (None) lets model decide
                return ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=0,
                    num_predict=max_tokens,
                    num_gpu=NUM_GPU,
                    keep_alive=KEEP_ALIVE,
                    reasoning=reasoning,
                )
            except ImportError:
                logger.warning("langchain-ollama not installed")
                return None

        if provider_normalized == "anthropic":
            api_key = self._app_settings.get("anthropic_api_key", "")
            if not api_key:
                logger.warning("Anthropic selected but no API key configured")
                return None
            try:
                return ChatAnthropic(
                    model=model,
                    temperature=0,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    default_request_timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                )
            except ImportError:
                logger.warning("langchain-anthropic not installed")
                return None

        if provider_normalized == "github_copilot":
            # GitHub Copilot uses OAuth flow. Proactively refresh the
            # short-lived HMAC token if near expiry.
            token = await ensure_copilot_token_fresh()
            if not token:
                logger.warning(
                    "GitHub Copilot selected but OAuth token is missing/expired. "
                    "Reconnect GitHub Copilot in Settings."
                )
                return None

            # Track the token baked into this LLM instance so
            # _ensure_copilot_llm_fresh can detect when it changes.
            self._copilot_llm_token = token

            base_url = self._app_settings.get(
                "github_copilot_base_url", "https://api.githubcopilot.com"
            )
            base_url = str(base_url or "https://api.githubcopilot.com").rstrip("/")

            copilot_headers = {
                "Openai-Intent": "conversation-panel",
                "Copilot-Integration-Id": "vscode-chat",
                "Editor-Version": "vscode/1.99.0",
                "Editor-Plugin-Version": "copilot-chat/0.26.3",
                "User-Agent": "GitHubCopilotChat/0.26.3",
                "Accept": "application/json",
            }

            copilot_metadata_headers = {
                **copilot_headers,
                "Authorization": f"Bearer {token}",
            }

            await _hydrate_openai_compatible_capabilities(
                metadata_urls=[f"{base_url}/models", f"{base_url}/v1/models"],
                headers=copilot_metadata_headers,
                requested_model=model,
            )

            # Build kwargs for _CopilotChatOpenAI — it handles both
            # /chat/completions and /responses via use_responses_api,
            # with automatic runtime fallback if the endpoint is wrong.
            copilot_kwargs: dict[str, Any] = {
                "model": model,
                "temperature": 0,
                "streaming": True,
                "api_key": token,
                "base_url": base_url,
                "max_tokens": max_tokens,
                "request_timeout": LLM_REQUEST_TIMEOUT_SECONDS,
                "default_headers": copilot_headers,
            }

            model_supports_reasoning = await supports_reasoning(model)
            model_supports_responses = await supports_responses_api(model)
            model_requires_responses = await requires_responses_api(model)

            # Prefer Responses API for reasoning-capable models when supported.
            # This improves reasoning quality and unlocks summary deltas.
            use_responses_api = model_requires_responses or (
                model_supports_reasoning and model_supports_responses
            )

            if use_responses_api:
                copilot_kwargs["use_responses_api"] = True
                # LangChain's legacy Responses v0 adapter can drop GPT reasoning
                # summary deltas once tools are bound. responses/v1 preserves
                # reasoning items in streamed content blocks.
                copilot_kwargs["output_version"] = "responses/v1"
                if model_supports_reasoning:
                    copilot_kwargs["reasoning"] = {
                        "effort": "high",
                        "summary": "auto",
                    }
                if model_requires_responses:
                    logger.info("Model %s requires Responses API (pre-detected)", model)
                else:
                    logger.info(
                        "Model %s supports Responses API; preferring it for reasoning",
                        model,
                    )
            else:
                if model_supports_reasoning:
                    copilot_kwargs["reasoning_effort"] = "high"
                if await supports_thinking_budget(model):
                    copilot_kwargs["extra_body"] = {"thinking_budget": 16384}

            return _CopilotChatOpenAI(
                **copilot_kwargs,
            )

        if provider_normalized == "github_models":
            token = self._app_settings.get("github_models_api_token", "")
            if not token:
                logger.warning("GitHub Models selected but no PAT token configured")
                return None

            await _hydrate_openai_compatible_capabilities(
                metadata_urls=["https://models.github.ai/catalog/models"],
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "ragtime",
                },
                requested_model=model,
            )

            github_models_kwargs: dict[str, Any] = {
                "model": model,
                "temperature": 0,
                "streaming": True,
                "api_key": token,
                "base_url": "https://models.github.ai/inference",
                "max_tokens": max_tokens,
                "request_timeout": LLM_REQUEST_TIMEOUT_SECONDS,
                "default_headers": {
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "ragtime",
                },
            }

            if await supports_reasoning(model):
                github_models_kwargs["reasoning_effort"] = "high"

            # GitHub Models exposes OpenAI-compatible chat completions at /inference.
            return _CopilotChatOpenAI(
                **github_models_kwargs,
            )

        api_key = self._app_settings.get("openai_api_key", "")
        if not api_key:
            logger.warning("OpenAI selected but no API key configured")
            return None

        await _hydrate_openai_compatible_capabilities(
            metadata_urls=[
                "https://api.openai.com/v1/models",
                f"https://api.openai.com/v1/models/{quote(model, safe=':/_-')}",
            ],
            headers={"Authorization": f"Bearer {api_key}"},
            requested_model=model,
        )

        openai_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "streaming": True,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "request_timeout": LLM_REQUEST_TIMEOUT_SECONDS,
        }

        if await supports_reasoning(model):
            openai_kwargs["reasoning_effort"] = "high"

        return _CopilotChatOpenAI(
            **openai_kwargs,
        )

    async def _get_embedding_model(self):
        """Get embedding model based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("embedding_provider", "ollama").lower()
        model = self._app_settings.get("embedding_model", "nomic-embed-text")

        if provider == "ollama":
            base_url = self._app_settings.get(
                "ollama_base_url", "http://localhost:11434"
            )
            logger.info(f"Using Ollama embeddings: {model} at {base_url}")
            return OllamaEmbeddings(
                model=model, base_url=base_url, num_gpu=NUM_GPU, keep_alive=KEEP_ALIVE
            )
        elif provider == "openai":
            api_key = self._app_settings.get("openai_api_key", "")
            if not api_key:
                logger.warning("OpenAI embeddings selected but no API key configured")
            logger.info(f"Using OpenAI embeddings: {model}")
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _create_retriever_from_faiss(self, db: FAISS, index_name: str) -> Any:
        """Create a retriever from a FAISS vectorstore with appropriate settings.

        Supports MMR (Max Marginal Relevance) for result diversification when enabled.
        MMR reduces near-duplicate results by balancing relevance with diversity.

        Args:
            db: FAISS vectorstore instance
            index_name: Name of the index (for logging)

        Returns:
            A retriever configured with current settings
        """
        search_k = self._app_settings.get("search_results_k", 5)
        use_mmr = self._app_settings.get("search_use_mmr", True)
        mmr_lambda = self._app_settings.get("search_mmr_lambda", 0.5)

        if use_mmr:
            # MMR retriever: fetch_k gets more candidates, then MMR selects k diverse ones
            # fetch_k should be larger than k to give MMR choices to diversify from
            fetch_k = max(
                search_k * MMR_FETCH_K_MULTIPLIER,
                MMR_MIN_FETCH_K,
            )
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_k,
                    "fetch_k": fetch_k,
                    "lambda_mult": mmr_lambda,  # 0=max diversity, 1=max relevance
                },
            )
            logger.debug(
                f"Created MMR retriever for {index_name} "
                f"(k={search_k}, fetch_k={fetch_k}, lambda={mmr_lambda})"
            )
        else:
            # Standard similarity retriever
            retriever = db.as_retriever(search_kwargs={"k": search_k})
            logger.debug(
                f"Created similarity retriever for {index_name} (k={search_k})"
            )

        return retriever

    @staticmethod
    def _add_timeout_field_to_schema(
        schema_class: type[BaseModel],
        default_timeout: int,
        timeout_max_seconds: int,
        timeout_label: str,
    ) -> type[BaseModel]:
        """Inject a standard timeout field into dynamic tool schemas."""
        timeout_field: Any = Field(
            default=default_timeout,
            ge=0,
            le=86400,
            description=(
                f"{timeout_label} timeout in seconds (default: {default_timeout}, "
                f"max: {'unlimited' if timeout_max_seconds == 0 else timeout_max_seconds}). "
                "Use 0 for no timeout."
            ),
        )
        schema_class.model_fields["timeout"] = timeout_field
        schema_class.model_rebuild()
        return schema_class

    def _search_faiss_databases(
        self,
        *,
        query: str,
        dbs_to_search: dict[str, Any],
        k: int,
        max_chars_per_result: int,
        use_mmr: bool,
        mmr_lambda: float,
        include_index_name: bool,
        include_index_name_in_errors: bool = True,
        ollama_error_message: str,
    ) -> tuple[list[str], list[str]]:
        """Search one or more FAISS indexes with shared formatting/error handling."""
        results: list[str] = []
        errors: list[str] = []
        k, max_chars_per_result = clamp_search_parameters(k, max_chars_per_result)

        for name, db in dbs_to_search.items():
            try:
                logger.debug(
                    f"Searching index '{name}' with query: {query[:50]}..., k={k}"
                )
                if use_mmr:
                    fetch_k = max(k * MMR_FETCH_K_MULTIPLIER, MMR_MIN_FETCH_K)
                    docs = db.max_marginal_relevance_search(
                        query,
                        k=k,
                        fetch_k=fetch_k,
                        lambda_mult=mmr_lambda,
                    )
                else:
                    docs = db.similarity_search(query, k=k)

                logger.debug(f"Index '{name}' returned {len(docs)} documents")
                for doc in docs:
                    source = doc.metadata.get("source", "unknown")
                    content = doc.page_content
                    if max_chars_per_result > 0 and len(content) > max_chars_per_result:
                        content = content[:max_chars_per_result] + "... (truncated)"
                    if include_index_name:
                        results.append(f"[{name}] {source}:\n{content}")
                    else:
                        results.append(f"{source}:\n{content}")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Error searching {name}: {e}", exc_info=True)
                if (
                    "ollama" in error_msg.lower()
                    or "failed to connect" in error_msg.lower()
                ):
                    if include_index_name_in_errors:
                        errors.append(f"[{name}] {ollama_error_message}")
                    else:
                        errors.append(ollama_error_message)
                else:
                    if include_index_name_in_errors:
                        errors.append(f"[{name}] Search error: {error_msg}")
                    else:
                        errors.append(f"Search error: {error_msg}")

        return results, errors

    async def _load_faiss_indexes(self, embedding_model):
        """Load FAISS indexes from database metadata (sequential, for backwards compat).

        Uses the index_metadata table to discover available indexes and loads
        only those that are enabled. The path is read directly from the database
        metadata, which was saved by the indexer service.

        For parallel loading, use _load_faiss_indexes_parallel instead.
        """
        assert self._app_settings is not None  # Set by initialize()
        # Try to load from database metadata (preferred)
        if self._index_metadata:
            enabled_indexes = [
                idx
                for idx in self._index_metadata
                if idx.get("enabled", True) and idx.get("chunk_count", 0) > 0
            ]

            if enabled_indexes:
                for idx in enabled_indexes:
                    index_name = idx.get("name")
                    if not index_name:
                        continue

                    # Use the path stored in the database by the indexer
                    index_path_str = idx.get("path")
                    if not index_path_str:
                        logger.warning(
                            f"Index {index_name} has no path in metadata, skipping"
                        )
                        continue

                    index_path = Path(index_path_str)
                    if index_path.exists():
                        try:
                            db = FAISS.load_local(
                                str(index_path),
                                embedding_model,
                                allow_dangerous_deserialization=True,
                            )
                            # Create retriever with MMR support if enabled
                            self.retrievers[index_name] = (
                                self._create_retriever_from_faiss(db, index_name)
                            )
                            self.faiss_dbs[index_name] = (
                                db  # Store for dynamic k searches
                            )
                            logger.info(
                                f"Loaded FAISS index: {index_name} from {index_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load FAISS index {index_name}: {e}"
                            )
                    else:
                        logger.warning(f"FAISS index path not found: {index_path}")

                if self.retrievers:
                    logger.info(
                        f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
                    )
                else:
                    logger.info("No enabled FAISS indexes found in database metadata")
            else:
                logger.info("No indexes found in database metadata")
        else:
            logger.info("No index metadata available (database not initialized)")

    async def _load_faiss_indexes_parallel(self, embedding_model):
        """Load FAISS indexes in parallel using asyncio.to_thread.

        This offloads the blocking FAISS.load_local calls to a thread pool,
        allowing multiple indexes to load concurrently and not blocking the
        event loop. Tracks memory usage for each index.
        """
        if not self._index_metadata:
            logger.info("No index metadata available for parallel loading")
            return

        enabled_indexes = [
            idx
            for idx in self._index_metadata
            if idx.get("enabled", True) and idx.get("chunk_count", 0) > 0
        ]

        if not enabled_indexes:
            logger.info("No enabled indexes to load")
            return

        self._indexes_total = len(enabled_indexes)
        self._indexes_loaded = 0

        # Get current embedding dimension by probing the embedding model
        # This is more reliable than tracked app_settings when provider changes
        current_embedding_dim = None
        try:
            test_embedding = await asyncio.to_thread(
                embedding_model.embed_query, "test"
            )
            current_embedding_dim = len(test_embedding)
            logger.info(
                f"Detected embedding dimension: {current_embedding_dim} "
                f"(will check indexes for mismatch)"
            )
        except Exception as e:
            # Fall back to tracked dimension if probe fails
            current_embedding_dim = self._app_settings.get("embedding_dimension")
            logger.warning(
                f"Could not probe embedding dimension: {e}. "
                f"Using tracked dimension: {current_embedding_dim}"
            )

        # Initialize index details for all indexes
        for idx in enabled_indexes:
            index_name = idx.get("name")
            if index_name:
                self._index_details[index_name] = {
                    "name": index_name,
                    "status": "pending",
                    "type": "document",  # Distinguish from filesystem_faiss
                    "size_mb": (
                        idx.get("size_bytes", 0) / (1024 * 1024)
                        if idx.get("size_bytes")
                        else None
                    ),
                    "chunk_count": idx.get("chunk_count"),
                    "load_time_seconds": None,
                    "error": None,
                }

        async def load_single_index(
            idx: dict,
        ) -> tuple[str, Any, dict] | None:
            """Load a single FAISS index in a thread and measure memory."""
            index_name = idx.get("name")
            if not index_name:
                return None

            # Mark as loading
            if index_name in self._index_details:
                self._index_details[index_name]["status"] = "loading"

            index_path_str = idx.get("path")
            if not index_path_str:
                logger.warning(f"Index {index_name} has no path in metadata, skipping")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = "No path in metadata"
                return None

            index_path = Path(index_path_str)
            if not index_path.exists():
                logger.warning(f"FAISS index path not found: {index_path}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name][
                        "error"
                    ] = f"Index files missing: {index_path} - re-index required"
                return None

            try:
                start = time.time()
                mem_before = get_process_memory_bytes()

                # Offload blocking I/O to thread pool
                db = await asyncio.to_thread(
                    FAISS.load_local,
                    str(index_path),
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )

                elapsed = time.time() - start
                mem_after = get_process_memory_bytes()

                # Get embedding dimension from the loaded index
                embedding_dim = db.index.d if hasattr(db, "index") else None

                # Check for dimension mismatch before creating retriever
                if (
                    current_embedding_dim
                    and embedding_dim
                    and embedding_dim != current_embedding_dim
                ):
                    mismatch_msg = (
                        f"Embedding dimension mismatch: index has {embedding_dim} dims, "
                        f"but current model produces {current_embedding_dim} dims. "
                        f"Re-index required."
                    )
                    logger.warning(f"Index {index_name}: {mismatch_msg}")
                    if index_name in self._index_details:
                        self._index_details[index_name]["status"] = "error"
                        self._index_details[index_name]["error"] = mismatch_msg
                        self._index_details[index_name][
                            "embedding_dimension"
                        ] = embedding_dim
                    # Return None to skip adding this retriever
                    return None

                # Memory used by this index (approximate - may include GC overhead)
                steady_mem = max(0, mem_after - mem_before)

                # Create retriever with MMR support if enabledfrom ragtime.core.ssh import
                retriever = self._create_retriever_from_faiss(db, index_name)
                logger.info(
                    f"Loaded FAISS index: {index_name} from {index_path} "
                    f"({elapsed:.1f}s, ~{steady_mem / 1024**3:.2f}GB)"
                )

                # Update index detail
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "loaded"
                    self._index_details[index_name]["load_time_seconds"] = elapsed

                memory_stats = {
                    "embedding_dimension": embedding_dim,
                    "steady_memory_bytes": steady_mem,
                    "load_time_seconds": elapsed,
                }

                return (index_name, db, retriever, memory_stats)
            except Exception as e:
                logger.warning(f"Failed to load FAISS index {index_name}: {e}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = str(e)
                return None

        # Load all indexes in parallel
        logger.info(f"Loading {len(enabled_indexes)} FAISS index(es) in parallel...")
        results = await asyncio.gather(
            *[load_single_index(idx) for idx in enabled_indexes],
            return_exceptions=True,
        )

        # Process results and update memory stats in database
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Index loading exception: {result}")
            elif result is not None:
                index_name, db, retriever, memory_stats = result
                self.retrievers[index_name] = retriever
                self.faiss_dbs[index_name] = db  # Store for dynamic k searches
                self._indexes_loaded += 1

                # Update memory stats in database (best effort)
                try:
                    await repository.update_index_memory_stats(index_name, memory_stats)
                except Exception as e:
                    logger.debug(f"Failed to update memory stats for {index_name}: {e}")

        if self.retrievers:
            logger.info(
                f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
            )

    async def _load_faiss_indexes_sequential(self, embedding_model):
        """Load FAISS indexes sequentially, smallest first.

        This reduces peak memory usage compared to parallel loading by loading
        one index at a time. Indexes are sorted by size (smallest first) so
        the system becomes partially functional faster.
        """

        if not self._index_metadata:
            logger.info("No index metadata available for sequential loading")
            return

        enabled_indexes = [
            idx
            for idx in self._index_metadata
            if idx.get("enabled", True) and idx.get("chunk_count", 0) > 0
        ]

        if not enabled_indexes:
            logger.info("No enabled indexes to load")
            return

        # Sort by size_bytes (smallest first) for faster initial availability
        enabled_indexes.sort(key=lambda x: x.get("size_bytes", 0))

        self._indexes_total = len(enabled_indexes)
        self._indexes_loaded = 0
        search_k = self._app_settings.get("search_results_k", 5)

        # Get current embedding dimension by probing the embedding model
        # This is more reliable than tracked app_settings when provider changes
        current_embedding_dim = None
        try:
            test_embedding = await asyncio.to_thread(
                embedding_model.embed_query, "test"
            )
            current_embedding_dim = len(test_embedding)
            logger.info(
                f"Detected embedding dimension: {current_embedding_dim} "
                f"(will check indexes for mismatch)"
            )
        except Exception as e:
            # Fall back to tracked dimension if probe fails
            current_embedding_dim = self._app_settings.get("embedding_dimension")
            logger.warning(
                f"Could not probe embedding dimension: {e}. "
                f"Using tracked dimension: {current_embedding_dim}"
            )

        # Initialize index details for all indexes
        for idx in enabled_indexes:
            index_name = idx.get("name")
            if index_name:
                self._index_details[index_name] = {
                    "name": index_name,
                    "status": "pending",
                    "type": "document",  # Distinguish from filesystem_faiss
                    "size_mb": (
                        idx.get("size_bytes", 0) / (1024 * 1024)
                        if idx.get("size_bytes")
                        else None
                    ),
                    "chunk_count": idx.get("chunk_count"),
                    "load_time_seconds": None,
                    "error": None,
                }

        logger.info(
            f"Loading {len(enabled_indexes)} FAISS index(es) sequentially "
            "(smallest first)..."
        )

        for idx in enabled_indexes:
            index_name = idx.get("name")
            if not index_name:
                continue

            # Mark as currently loading
            self._loading_index = index_name
            if index_name in self._index_details:
                self._index_details[index_name]["status"] = "loading"

            index_path_str = idx.get("path")
            if not index_path_str:
                logger.warning(f"Index {index_name} has no path in metadata, skipping")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = "No path in metadata"
                continue

            index_path = Path(index_path_str)
            if not index_path.exists():
                logger.warning(f"FAISS index path not found: {index_path}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name][
                        "error"
                    ] = f"Index files missing: {index_path} - re-index required"
                continue

            try:
                start = time.time()
                mem_before = get_process_memory_bytes()

                # Track peak memory during loading
                peak_mem = mem_before

                def load_and_track_peak():
                    nonlocal peak_mem
                    db = FAISS.load_local(
                        str(index_path),
                        embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    # Check memory after loading (this is approximate)
                    current_mem = get_process_memory_bytes()
                    peak_mem = max(peak_mem, current_mem)
                    return db

                db = await asyncio.to_thread(load_and_track_peak)

                elapsed = time.time() - start
                mem_after = get_process_memory_bytes()

                # Get embedding dimension from the loaded index
                embedding_dim = db.index.d if hasattr(db, "index") else None

                # Check for dimension mismatch before creating retriever
                if (
                    current_embedding_dim
                    and embedding_dim
                    and embedding_dim != current_embedding_dim
                ):
                    mismatch_msg = (
                        f"Embedding dimension mismatch: index has {embedding_dim} dims, "
                        f"but current model produces {current_embedding_dim} dims. "
                        f"Re-index required."
                    )
                    logger.warning(f"Index {index_name}: {mismatch_msg}")
                    if index_name in self._index_details:
                        self._index_details[index_name]["status"] = "error"
                        self._index_details[index_name]["error"] = mismatch_msg
                        self._index_details[index_name][
                            "embedding_dimension"
                        ] = embedding_dim
                    continue  # Skip this index

                # Calculate memory stats
                steady_mem = max(0, mem_after - mem_before)
                observed_peak = max(0, peak_mem - mem_before)

                # Create retriever with MMR support if enabled
                retriever = self._create_retriever_from_faiss(db, index_name)
                self.retrievers[index_name] = retriever
                self.faiss_dbs[index_name] = db  # Store for dynamic k searches
                self._indexes_loaded += 1

                # Update index detail
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "loaded"
                    self._index_details[index_name]["load_time_seconds"] = elapsed

                logger.info(
                    f"Loaded FAISS index: {index_name} from {index_path} "
                    f"(k={search_k}, {elapsed:.1f}s, ~{steady_mem / 1024**3:.2f}GB)"
                )

                # Update memory stats in database
                try:
                    await repository.update_index_memory_stats(
                        index_name,
                        {
                            "embedding_dimension": embedding_dim,
                            "steady_memory_bytes": steady_mem,
                            "peak_memory_bytes": observed_peak,
                            "load_time_seconds": elapsed,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to update memory stats for {index_name}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load FAISS index {index_name}: {e}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = str(e)

        # Clear loading index indicator
        self._loading_index = None

        if self.retrievers:
            logger.info(
                f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
            )

    async def _load_index_metadata(self) -> list[dict]:
        """Load index metadata from database for system prompt."""
        try:
            metadata_list = await repository.list_index_metadata()
            return [
                {
                    "name": m.name,
                    "path": m.path,
                    "description": getattr(m, "description", ""),
                    "enabled": m.enabled,
                    "search_weight": getattr(m, "searchWeight", 1.0),
                    "document_count": m.documentCount,
                    "chunk_count": m.chunkCount,
                    "source_type": m.sourceType,
                    "size_bytes": m.sizeBytes,
                    "embedding_dimension": getattr(m, "embeddingDimension", None),
                    "steady_memory_bytes": getattr(m, "steadyMemoryBytes", None),
                    "peak_memory_bytes": getattr(m, "peakMemoryBytes", None),
                }
                for m in metadata_list
            ]
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return []

    async def _create_agent(self):
        """Create the LangChain agents with tools.

        Creates two agents:
        - agent_executor: Standard agent for API/MCP requests
        - agent_executor_ui: Agent with UI-only tools (charts) for chat UI requests
        """
        assert self._app_settings is not None  # Set by initialize()
        self._request_prompt_cache.clear()
        # Skip agent creation if LLM is not configured
        if self.llm is None:
            logger.warning("Skipping agent creation - no LLM configured")
            self.agent_executor = None
            self.agent_executor_ui = None
            return

        tools = []

        # Add knowledge search tool(s) if we have FAISS retrievers
        if self.retrievers:
            aggregate_search = self._app_settings.get("aggregate_search", True)
            if aggregate_search:
                # Single aggregated search_knowledge tool
                tools.append(self._create_knowledge_search_tool())
                logger.info(
                    f"Added search_knowledge tool for {len(self.retrievers)} index(es)"
                )
            else:
                # Separate search_<index_name> tools for each index
                knowledge_tools = self._create_per_index_search_tools()
                tools.extend(knowledge_tools)
                logger.info(f"Added {len(knowledge_tools)} per-index search tools")

            # Add file access tools (read_file_from_index, list_files_in_index)
            file_access_tools = self._create_file_access_tools()
            if file_access_tools:
                tools.extend(file_access_tools)
                logger.info(f"Added {len(file_access_tools)} file access tools")

        # Add git history search tool(s) if we have git repos
        git_history_tools = await self._create_git_history_tools()
        if git_history_tools:
            tools.extend(git_history_tools)

        # Get tools from the new ToolConfig system
        if self._tool_configs:
            config_tools = await self._build_tools_from_configs(
                skip_knowledge_tool=True
            )
            tools.extend(config_tools)
            logger.info(f"Built {len(config_tools)} tools from configurations")
        else:
            # Fallback to legacy enabled_tools system
            app_settings = await get_app_settings()
            enabled_list = app_settings["enabled_tools"]
            if enabled_list:
                legacy_tools = get_enabled_tools(enabled_list)
                tools.extend(legacy_tools)
                logger.info(f"Using legacy tool configuration: {enabled_list}")

        if not tools:
            available = list(get_all_tools().keys())
            logger.warning(
                f"No tools configured. Available tool types: {available}. "
                f"Configure via Tools tab at /indexes/ui?view=tools"
            )

        # Respect admin-configured iteration limit; fall back to 15 if invalid
        try:
            max_iterations = int(self._app_settings.get("max_iterations", 15))
            if max_iterations < 1:
                max_iterations = 1
        except (TypeError, ValueError):
            max_iterations = 15

        # Get token optimization settings
        max_tool_output_chars = int(
            self._app_settings.get("max_tool_output_chars", 5000)
        )
        scratchpad_window_size = int(
            self._app_settings.get("scratchpad_window_size", 6)
        )

        # Wrap all tools with output truncation (if enabled)
        # This reduces token consumption in the agent's scratchpad
        if max_tool_output_chars > 0:
            tools = [wrap_tool_with_truncation(t, max_tool_output_chars) for t in tools]
            logger.info(
                f"Wrapped {len(tools)} tools with output truncation "
                f"(max {max_tool_output_chars:,} chars)"
            )

        # Store window size for scratchpad compression
        self._scratchpad_window_size = scratchpad_window_size

        # Create windowed message formatter for scratchpad compression
        # This reduces token consumption by compressing older tool results
        def create_windowed_formatter(window_size: int):
            """
            Create a message formatter that compresses old tool results.

            Anthropic requires strict tool_use/tool_result pairing, so we can't
            drop old steps entirely. Instead, we keep all pairs but aggressively
            compress the content of old tool_result messages.
            """

            def windowed_format(intermediate_steps):
                grouped_steps = self._group_intermediate_steps_by_message(
                    intermediate_steps
                )
                num_steps = len(intermediate_steps)
                num_groups = len(grouped_steps)

                if window_size <= 0 or num_groups <= window_size:
                    result = self._format_intermediate_steps_for_agent(
                        intermediate_steps
                    )
                    if num_steps > 0:
                        total_chars = sum(len(str(m.content)) for m in result)
                        logger.debug(
                            f"Scratchpad: {num_steps} steps, {len(result)} messages, "
                            f"{total_chars:,} chars (no compression needed)"
                        )
                    return result

                full_chars = sum(
                    len(str(m.content))
                    for m in self._format_intermediate_steps_for_agent(
                        intermediate_steps
                    )
                )

                old_groups = grouped_steps[:-window_size]
                recent_groups = grouped_steps[-window_size:]
                old_steps = [step for group in old_groups for step in group]
                recent_steps = [step for group in recent_groups for step in group]

                compressed_msgs: list[BaseMessage] = []
                for msg in format_to_tool_messages(old_steps):
                    if isinstance(msg, ToolMessage):
                        content = str(msg.content)
                        if len(content) > 200:
                            content = content[:150] + "... [truncated]"
                        compressed_msgs.append(
                            ToolMessage(
                                content=content,
                                tool_call_id=msg.tool_call_id,
                                additional_kwargs=(
                                    dict(msg.additional_kwargs)
                                    if getattr(msg, "additional_kwargs", None)
                                    else None
                                ),
                            )
                        )
                    else:
                        compressed_msgs.append(msg)

                all_msgs = compressed_msgs + self._format_intermediate_steps_for_agent(
                    recent_steps
                )

                compressed_chars = sum(len(str(m.content)) for m in all_msgs)
                reduction = (
                    100 * (1 - compressed_chars / full_chars) if full_chars > 0 else 0
                )

                if old_steps:
                    logger.info(
                        f"Scratchpad compression: {num_steps} steps | "
                        f"{len(old_steps)} old results compressed | "
                        f"{full_chars:,} -> {compressed_chars:,} chars ({reduction:.1f}% reduction)"
                    )

                return all_msgs

            return windowed_format

        # Create the message formatter (or None for default behavior)
        message_formatter = (
            create_windowed_formatter(scratchpad_window_size)
            if scratchpad_window_size > 0
            else self._format_intermediate_steps_for_agent
        )

        if scratchpad_window_size > 0:
            logger.info(
                f"Using windowed scratchpad formatter (window={scratchpad_window_size})"
            )

        # Create standard agent (for API/MCP)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", escape_prompt_template_braces(self._system_prompt)),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                MessagesPlaceholder(variable_name="user_input"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if tools:
            # Use thinking-aware agent for Ollama models with reasoning,
            # standard create_tool_calling_agent for everything else.
            agent = self._create_thinking_aware_agent(
                self.llm,
                tools,
                prompt,
                message_formatter=message_formatter,
            )
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=False,
            )
        else:
            self.agent_executor = None

        # Create UI agent (with visualization tools and UI prompt)
        # Note: create_chart_tool and create_datatable_tool are NOT wrapped with
        # truncation because their JSON output must be complete for rendering
        ui_tools = tools + [create_chart_tool, create_datatable_tool]

        prompt_ui = ChatPromptTemplate.from_messages(
            [
                ("system", escape_prompt_template_braces(self._system_prompt_ui)),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                MessagesPlaceholder(variable_name="user_input"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if ui_tools:
            # Pass same message_formatter for consistent behavior
            agent_ui = self._create_thinking_aware_agent(
                self.llm,
                ui_tools,
                prompt_ui,
                message_formatter=message_formatter,
            )
            self.agent_executor_ui = AgentExecutor(
                agent=agent_ui,
                tools=ui_tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=False,
            )
            logger.info("Created UI agent with chart and datatable tools")
        else:
            self.agent_executor_ui = None

    async def _build_tools_from_configs(
        self, skip_knowledge_tool: bool = False
    ) -> List[Any]:
        """
        Build LangChain tools from ToolConfig entries.

        Creates dynamic tool wrappers for each configured tool instance.
        Tool configs are built in parallel to avoid sequential delays.

        Args:
            skip_knowledge_tool: If True, don't add the search_knowledge tool
                (used when caller has already added it)
        """
        tools = []

        # Add knowledge search tool if we have FAISS retrievers (unless skipped)
        if self.retrievers and not skip_knowledge_tool:
            tools.append(self._create_knowledge_search_tool())
            # Also add file access tools
            tools.extend(self._create_file_access_tools())

        sql_tool_types = {"postgres", "mssql", "mysql"}
        tool_builders = {
            "postgres": self._create_postgres_tool,
            "mssql": self._create_mssql_tool,
            "mysql": self._create_mysql_tool,
            "influxdb": self._create_influxdb_tool,
            "odoo_shell": self._create_odoo_tool,
            "ssh_shell": self._create_ssh_tool,
            "filesystem_indexer": self._create_filesystem_tool,
            "solidworks_pdm": self._create_pdm_search_tool,
        }

        async def _build_single_tool(config: dict) -> List[Any]:
            """Build tool(s) from a single config entry with timeout."""
            tool_type = config.get("tool_type")
            raw_name = (config.get("name", "") or "").strip()
            tool_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
            tool_id = config.get("id") or ""
            result_tools = []

            try:
                builder = tool_builders.get(str(tool_type))
                if not builder:
                    logger.warning(f"Unknown tool type: {tool_type}")
                    return []

                tool = await builder(config, tool_name, tool_id)

                if str(tool_type) in sql_tool_types:
                    schema_tool = await self._create_schema_search_tool(
                        config,
                        tool_name,
                        tool_id,
                    )
                    if schema_tool:
                        result_tools.append(schema_tool)

                if tool:
                    result_tools.insert(0, tool)
                return result_tools

            except Exception as e:
                logger.warning(f"Failed to build {tool_type} tool '{raw_name}': {e}")
                return []

        # Build all tool configs in parallel
        if self._tool_configs:
            build_tasks = [_build_single_tool(config) for config in self._tool_configs]
            results = await asyncio.gather(*build_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                config = self._tool_configs[i]
                if isinstance(result, BaseException):
                    logger.warning(
                        f"Tool '{config.get('name', '?')}' ({config.get('tool_type')}) "
                        f"build failed: {result}"
                    )
                elif isinstance(result, list) and result:
                    tools.extend(result)

        return tools

    async def _create_schema_search_tool(
        self, config: dict, tool_name: str, tool_id: str
    ):
        """Create a schema search tool for SQL database tools.

        This tool allows the agent to search the indexed database schema
        to find relevant table/column information before writing queries.
        """
        conn_config = config.get("connection_config", {})

        # Check if schema indexing is enabled
        schema_index_enabled = conn_config.get("schema_index_enabled", False)
        logger.debug(
            f"Schema tool check for {tool_name}: enabled={schema_index_enabled}"
        )
        if not schema_index_enabled:
            return None

        # Check if there are any schema embeddings for this tool
        embedding_count = await schema_indexer.get_embedding_count(tool_id, tool_name)
        if embedding_count == 0:
            logger.debug(
                f"Schema indexing enabled but no embeddings found for {tool_name}"
            )
            # Still create the tool - it will just return "no results" until indexed
            # This is better than no tool at all

        # Use tool_name (safe name) for index lookup - matches how trigger_index creates it
        index_name = f"schema_{tool_name}"
        description = config.get("description", "")

        class SchemaSearchInput(BaseModel):
            query: str = Field(
                description="Search query to find relevant tables, columns, or relationships in the database schema"
            )

        async def search_schema(query: str) -> str:
            """Search the database schema for relevant table information."""
            logger.debug(f"[{tool_name}] Schema search: {query[:100]}")

            result = await search_schema_index(
                query=query,
                index_name=index_name,
                max_results=5,
            )
            return result

        tool_description = (
            f"Search the schema of the {config.get('name', 'database')} database "
            f"to find table names, column definitions, relationships, and indexes. "
            f"Use this BEFORE writing SQL queries when you need to understand the database structure."
        )
        if description:
            tool_description += f" Database contains: {description}"

        schema_tool = StructuredTool.from_function(
            coroutine=search_schema,
            name=f"search_{tool_name}_schema",
            description=tool_description,
            args_schema=SchemaSearchInput,
        )
        logger.info(f"Created schema search tool: search_{tool_name}_schema")
        return schema_tool

    def _create_knowledge_search_tool(self):
        """Create a tool for on-demand FAISS knowledge search.

        This allows the agent to search the indexed documentation at any point
        during its reasoning, not just at the beginning of the query.

        The agent can control:
        - k: Number of results to retrieve (default from settings, max 50)
        - max_chars_per_result: How much content to show per result (default 500, 0=unlimited)
        """
        # Build index_name description with available indexes
        index_names = list(self.retrievers.keys())
        index_name_desc = (
            "Optional: specific index to search (leave empty to search all indexes)"
        )
        if index_names:
            index_name_desc += f". Available indexes: {', '.join(index_names)}"

        # Get default k from settings
        default_k = (
            self._app_settings.get("search_results_k", 5) if self._app_settings else 5
        )
        use_mmr = (
            self._app_settings.get("search_use_mmr", True)
            if self._app_settings
            else True
        )
        mmr_lambda = (
            self._app_settings.get("search_mmr_lambda", 0.5)
            if self._app_settings
            else 0.5
        )

        class KnowledgeSearchInput(BaseModel):
            query: str = Field(
                description="Search query to find relevant documentation, code, or technical information"
            )
            index_name: str = Field(
                default="",
                description=index_name_desc,
            )
            k: int = Field(
                default=default_k,
                ge=1,
                le=50,
                description=f"Number of results to retrieve (default: {default_k}). Increase for broader searches, decrease for focused results.",
            )
            max_chars_per_result: int = Field(
                default=500,
                ge=0,
                le=10000,
                description="Maximum characters per result (default: 500). Use 0 for full content when you need complete code/file content. Increase when results are truncated.",
            )

        def search_knowledge(
            query: str,
            index_name: str = "",
            k: int = default_k,
            max_chars_per_result: int = 500,
        ) -> str:
            """Search indexed documentation for relevant information."""
            k, max_chars_per_result = clamp_search_parameters(k, max_chars_per_result)

            # Log the search attempt for debugging
            logger.debug(
                f"search_knowledge called with query='{query[:50]}...', index_name='{index_name}', k={k}, max_chars={max_chars_per_result}"
            )
            logger.debug(f"Available FAISS dbs: {list(self.faiss_dbs.keys())}")

            # Determine which indexes to search
            if index_name and index_name in self.faiss_dbs:
                dbs_to_search = {index_name: self.faiss_dbs[index_name]}
            else:
                dbs_to_search = self.faiss_dbs

            if not dbs_to_search:
                # Check if indexes are still loading
                if self._indexes_loading:
                    logger.info("Knowledge search called while indexes still loading")
                    return (
                        "Knowledge indexes are currently loading in the background. "
                        f"Progress: {self._indexes_loaded}/{self._indexes_total} loaded. "
                        "Please try again in a moment, or use other available tools."
                    )
                logger.warning("No FAISS dbs available for search_knowledge")
                return "No knowledge indexes are currently loaded. Please index some documents first."

            results, errors = self._search_faiss_databases(
                query=query,
                dbs_to_search=dbs_to_search,
                k=k,
                max_chars_per_result=max_chars_per_result,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                include_index_name=True,
                ollama_error_message=(
                    "Embedding service unavailable - Cannot connect to Ollama. "
                    "Check that Ollama is running and the URL in Settings is accessible from the server "
                    "(use 'host.docker.internal' instead of 'localhost' when running in Docker)."
                ),
            )

            if results:
                logger.debug(f"search_knowledge found {len(results)} results")
                return (
                    f"Found {len(results)} relevant documents:\n\n"
                    + "\n\n---\n\n".join(results)
                )

            # Return errors if we had any, otherwise generic no results message
            if errors:
                logger.warning(f"search_knowledge failed with errors: {errors}")
                return "Search failed:\n" + "\n".join(errors)

            logger.debug("search_knowledge found no results")
            return "No relevant documentation found for this query."

        # Build description with available indexes
        index_names = list(self.retrievers.keys())
        description = (
            "Search the indexed documentation and codebase for relevant information. "
            "Use this to find code examples, schema definitions, configuration details, or technical documentation. "
            f"Available indexes: {', '.join(index_names)}. "
            "The query should describe what you're looking for. "
            "Use 'k' to control number of results (increase for broader searches). "
            "Use 'max_chars_per_result' to control content length (use 0 for full content when results are truncated)."
        )

        async def _search_knowledge_async(
            query: str,
            index_name: str = "",
            k: int = default_k,
            max_chars_per_result: int = 500,
        ) -> str:
            return await asyncio.to_thread(
                search_knowledge, query, index_name, k, max_chars_per_result
            )

        return StructuredTool.from_function(
            func=search_knowledge,
            coroutine=_search_knowledge_async,
            name="search_knowledge",
            description=description,
            args_schema=KnowledgeSearchInput,
        )

    def _create_file_access_tools(self) -> list:
        """Create tools for direct file access from indexed repositories.

        Provides two tools:
        - read_file_from_index: Read all chunks of a specific file by path
        - list_files_in_index: List all indexed files in a repository

        These complement the search tools by allowing direct file access
        when the AI knows which file it needs.
        """
        index_names = list(self.retrievers.keys())
        if not index_names:
            return []

        tools = []

        # Tool 1: Read file by path
        class ReadFileInput(BaseModel):
            file_path: str = Field(
                description="Relative path to the file (e.g., 'src/utils/helper.py' or 'ragtime/rag/components.py')"
            )
            index_name: str = Field(
                default="",
                description=f"Index to read from. Available: {', '.join(index_names)}. Leave empty to search all.",
            )

        def read_file_from_index(file_path: str, index_name: str = "") -> str:
            """Read all chunks of a file from an indexed repository by its path."""
            results = []
            errors = []
            target_indexes = [index_name] if index_name else index_names

            for idx_name in target_indexes:
                faiss_db = self.faiss_dbs.get(idx_name)

                if not faiss_db:
                    continue

                try:
                    # Get all documents from the FAISS index
                    docstore = faiss_db.docstore
                    index_to_docstore_id = faiss_db.index_to_docstore_id

                    # Find all chunks matching the file path
                    matching_chunks = []
                    for idx, doc_id in index_to_docstore_id.items():
                        doc = docstore.search(doc_id)
                        if doc and hasattr(doc, "metadata"):
                            source = doc.metadata.get("source", "")
                            # Match exact path or path ending
                            if source == file_path or source.endswith(f"/{file_path}"):
                                chunk_index = doc.metadata.get("chunk_index", idx)
                                matching_chunks.append((chunk_index, doc.page_content))

                    if matching_chunks:
                        # Sort by chunk_index
                        matching_chunks.sort(key=lambda x: x[0] if x[0] != -1 else -999)

                        # Build result
                        result_parts = [
                            f"[{idx_name}] {file_path} ({len(matching_chunks)} chunks):"
                        ]
                        for chunk_idx, content in matching_chunks:
                            if chunk_idx == -1:
                                result_parts.append(
                                    f"\n--- File Summary ---\n{content}"
                                )
                            else:
                                result_parts.append(
                                    f"\n--- Chunk {chunk_idx} ---\n{content}"
                                )

                        results.append("\n".join(result_parts))

                except Exception as e:
                    logger.warning(f"Error reading file from {idx_name}: {e}")
                    errors.append(f"[{idx_name}] Error: {str(e)}")

            if results:
                return "\n\n".join(results)

            # File not found - suggest similar files
            if errors:
                return f"File '{file_path}' not found.\nErrors: " + "; ".join(errors)
            return f"File '{file_path}' not found in indexed repositories. Use list_files_in_index to see available files, or search_knowledge to find relevant files."

        async def _read_file_from_index_async(
            file_path: str, index_name: str = ""
        ) -> str:
            return await asyncio.to_thread(read_file_from_index, file_path, index_name)

        read_file_tool = StructuredTool.from_function(
            func=read_file_from_index,
            coroutine=_read_file_from_index_async,
            name="read_file_from_index",
            description=(
                "Read the complete content of a specific file from an indexed repository by its path. "
                "Use this when you know the exact file path and need to see all of its content. "
                "Returns all chunks of the file in order. "
                f"Available indexes: {', '.join(index_names)}."
            ),
            args_schema=ReadFileInput,
        )
        tools.append(read_file_tool)

        # Tool 2: List files in index
        class ListFilesInput(BaseModel):
            index_name: str = Field(
                default="",
                description=f"Index to list files from. Available: {', '.join(index_names)}. Leave empty to list all.",
            )
            pattern: str = Field(
                default="",
                description="Optional filter pattern (e.g., '*.py', 'src/', 'components'). Matches against file path.",
            )
            limit: int = Field(
                default=50,
                ge=1,
                le=500,
                description="Maximum number of files to return (default: 50)",
            )

        def list_files_in_index(
            index_name: str = "", pattern: str = "", limit: int = 50
        ) -> str:
            """List all indexed files in a repository."""
            results = []
            target_indexes = [index_name] if index_name else index_names

            for idx_name in target_indexes:
                faiss_db = self.faiss_dbs.get(idx_name)
                if not faiss_db:
                    continue

                try:
                    docstore = faiss_db.docstore
                    index_to_docstore_id = faiss_db.index_to_docstore_id

                    # Collect unique file paths
                    file_paths: set[str] = set()
                    for doc_id in index_to_docstore_id.values():
                        doc = docstore.search(doc_id)
                        if doc and hasattr(doc, "metadata"):
                            source = doc.metadata.get("source", "")
                            if source and not source.startswith("git:"):
                                file_paths.add(source)

                    # Apply pattern filter
                    if pattern:
                        pattern_lower = pattern.lower()
                        filtered = []
                        for fp in file_paths:
                            fp_lower = fp.lower()
                            # Support glob patterns and simple substring match
                            if "*" in pattern or "?" in pattern:
                                if fnmatch.fnmatch(fp_lower, pattern_lower):
                                    filtered.append(fp)
                            elif pattern_lower in fp_lower:
                                filtered.append(fp)
                        file_paths = set(filtered)

                    # Sort and limit
                    sorted_files = sorted(file_paths)[:limit]

                    if sorted_files:
                        results.append(
                            f"[{idx_name}] {len(sorted_files)} files"
                            + (
                                f" (of {len(file_paths)} matching)"
                                if len(file_paths) > limit
                                else ""
                            )
                            + ":\n"
                            + "\n".join(f"  {fp}" for fp in sorted_files)
                        )
                    else:
                        results.append(
                            f"[{idx_name}] No files found matching '{pattern}'"
                            if pattern
                            else f"[{idx_name}] No files indexed"
                        )

                except Exception as e:
                    logger.warning(f"Error listing files from {idx_name}: {e}")
                    results.append(f"[{idx_name}] Error: {str(e)}")

            return "\n\n".join(results) if results else "No indexes available."

        async def _list_files_in_index_async(
            index_name: str = "", pattern: str = "", limit: int = 50
        ) -> str:
            return await asyncio.to_thread(
                list_files_in_index, index_name, pattern, limit
            )

        list_files_tool = StructuredTool.from_function(
            func=list_files_in_index,
            coroutine=_list_files_in_index_async,
            name="list_files_in_index",
            description=(
                "List all files in an indexed repository. Use this to discover what files are available, "
                "or to find files matching a pattern (e.g., '*.py', 'components/', 'test'). "
                f"Available indexes: {', '.join(index_names)}."
            ),
            args_schema=ListFilesInput,
        )
        tools.append(list_files_tool)

        return tools

    async def _create_git_history_tools(self) -> List[Any]:
        """Create git history search tool(s) for git-based indexes.

        When aggregate_search is enabled: creates a single search_git_history tool
        When aggregate_search is disabled: creates search_git_history_<name> per index
        """
        tools: List[Any] = []
        index_base = Path(settings.index_data_path)

        # Find all git repos and match them with index metadata
        # Only include repos where git_history_depth != 1 (shallow clone has no history)
        git_repos: List[tuple[str, Path, str]] = []  # (name, path, description)

        def _discover_git_repos() -> List[tuple[str, Path]]:
            """Discover git repo directories on disk (blocking I/O)."""
            repos: List[tuple[str, Path]] = []
            if index_base.exists():
                for index_dir in index_base.iterdir():
                    if not index_dir.is_dir():
                        continue
                    git_repo = index_dir / ".git_repo"
                    if git_repo.exists() and (git_repo / ".git").exists():
                        repos.append((index_dir.name, git_repo))
            return repos

        disk_repos = await asyncio.to_thread(_discover_git_repos)

        for repo_name, git_repo in disk_repos:
            # Get metadata including config_snapshot to check git_history_depth
            description = ""
            git_history_depth = 0  # Default to full history
            for idx in self._index_metadata or []:
                if idx.get("name") == repo_name:
                    description = idx.get("description", "")
                    # Get git_history_depth from config_snapshot
                    config = idx.get("config_snapshot") or {}
                    git_history_depth = config.get("git_history_depth", 0)
                    break

            # Only expose git history tool if depth != 1
            # depth=0 means full history, depth>1 means we have commits to search
            # depth=1 is a shallow clone with only the latest commit (not useful)
            if git_history_depth == 1:
                logger.debug(
                    f"Skipping git history tool for {repo_name}: "
                    "shallow clone (depth=1 in config)"
                )
                continue

            # Also check actual repo state - it may have minimal history
            # even if config doesn't reflect this (e.g., cloned externally)
            # Note: _is_shallow_repository checks commit count, not just
            # whether it's technically shallow - depth > 1 is still useful
            if await _is_shallow_repository(git_repo):
                logger.debug(
                    f"Skipping git history tool for {repo_name}: "
                    "minimal commit history (1-2 commits)"
                )
                continue

            git_repos.append((repo_name, git_repo, description))

        if not git_repos:
            return []

        aggregate_search = (self._app_settings or {}).get("aggregate_search", True)

        if aggregate_search:
            # Single tool for all git repos - include available repos in description
            repo_names = [name for name, _, _ in git_repos]
            tools.append(create_aggregate_git_history_tool(repo_names))
            logger.info(
                f"Added search_git_history tool for {len(git_repos)} repo(s): {repo_names}"
            )
        else:
            # Separate tool per repo
            for name, repo_path, description in git_repos:
                tool = create_per_index_git_history_tool(name, repo_path, description)
                tools.append(tool)
            logger.info(f"Added {len(tools)} per-index git history search tools")

        return tools

    def _create_per_index_search_tools(self) -> List[Any]:
        """Create separate search tools for each index.

        When aggregate_search is disabled, this creates search_<index_name>
        tools that give the AI granular control over which index to search.

        Supports dynamic k and max_chars_per_result parameters like the aggregate search.
        """
        tools = []

        # Get settings
        default_k = (
            self._app_settings.get("search_results_k", 5) if self._app_settings else 5
        )
        use_mmr = (
            self._app_settings.get("search_use_mmr", True)
            if self._app_settings
            else True
        )
        mmr_lambda = (
            self._app_settings.get("search_mmr_lambda", 0.5)
            if self._app_settings
            else 0.5
        )

        # Get index metadata for descriptions and weights
        index_weights = {}
        index_descriptions = {}
        for idx in self._index_metadata or []:
            if idx.get("enabled", True):
                name = idx.get("name", "")
                index_weights[name] = idx.get("search_weight", 1.0)
                index_descriptions[name] = idx.get("description", "")

        for index_name, db in self.faiss_dbs.items():
            # Create a closure to capture the current index_name and db
            def make_search_func(
                idx_name: str,
                idx_db,
                use_mmr_: bool,
                mmr_lambda_: float,
                default_k_: int,
            ):
                def search_index(
                    query: str,
                    k: int = default_k_,
                    max_chars_per_result: int = 500,
                ) -> str:
                    """Search this specific index for relevant information."""
                    k, max_chars_per_result = clamp_search_parameters(
                        k,
                        max_chars_per_result,
                    )

                    logger.debug(
                        f"search_{idx_name} called with query='{query[:50]}...', k={k}, max_chars={max_chars_per_result}"
                    )

                    results, errors = self._search_faiss_databases(
                        query=query,
                        dbs_to_search={idx_name: idx_db},
                        k=k,
                        max_chars_per_result=max_chars_per_result,
                        use_mmr=use_mmr_,
                        mmr_lambda=mmr_lambda_,
                        include_index_name=False,
                        include_index_name_in_errors=False,
                        ollama_error_message=(
                            "Embedding service unavailable - Cannot connect to Ollama. "
                            "Check that Ollama is running and accessible."
                        ),
                    )

                    if errors:
                        return errors[0]

                    if results:
                        return (
                            f"Found {len(results)} relevant documents:\n\n"
                            + "\n\n---\n\n".join(results)
                        )

                    return (
                        f"No relevant documents found in {idx_name} for query: {query}"
                    )

                return search_index

            # Create input schema for this tool with k and max_chars_per_result
            class IndexSearchInput(BaseModel):
                query: str = Field(
                    description="Search query to find relevant documentation or code"
                )
                k: int = Field(
                    default=default_k,
                    ge=1,
                    le=50,
                    description=f"Number of results (default: {default_k}). Increase for broader searches.",
                )
                max_chars_per_result: int = Field(
                    default=500,
                    ge=0,
                    le=10000,
                    description="Max chars per result (default: 500). Use 0 for full content.",
                )

            # Build description including the index description and weight hint
            weight = index_weights.get(index_name, 1.0)
            idx_desc = index_descriptions.get(index_name, "")

            tool_description = (
                f"Search the '{index_name}' index for relevant information. "
                "Use 'k' for result count, 'max_chars_per_result' for content length (0=full)."
            )
            if idx_desc:
                tool_description += f" {idx_desc}"
            if weight != 1.0:
                if weight > 1.0:
                    tool_description += f" [Priority: High (weight={weight})]"
                elif weight > 0:
                    tool_description += f" [Priority: Low (weight={weight})]"

            # Sanitize index name for tool name (replace invalid chars)
            safe_name = index_name.replace("-", "_").replace(" ", "_").lower()
            tool_name = f"search_{safe_name}"

            _sync_func = make_search_func(
                index_name, db, use_mmr, mmr_lambda, default_k
            )

            async def _async_search_index(
                query: str,
                k: int = default_k,
                max_chars_per_result: int = 500,
                _fn=_sync_func,
            ) -> str:
                return await asyncio.to_thread(_fn, query, k, max_chars_per_result)

            tool = StructuredTool.from_function(
                func=_sync_func,
                coroutine=_async_search_index,
                name=tool_name,
                description=tool_description,
                args_schema=IndexSearchInput,
            )
            tools.append(tool)

        return tools

    async def _create_postgres_tool(
        self,
        config: dict,
        tool_name: str,
        _tool_id: str,
        include_metadata: bool = True,
    ):
        """Create a PostgreSQL query tool from config."""
        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 5432)

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)

        # Create input schema with captured timeout value
        _default_timeout = timeout  # Capture for closure

        class PostgresInput(BaseModel):
            query: str = Field(
                default="",
                description="SQL query to execute. Must include LIMIT clause.",
            )
            reason: str = Field(
                default="", description="Brief description of what this query retrieves"
            )

        self._add_timeout_field_to_schema(
            PostgresInput,
            default_timeout=_default_timeout,
            timeout_max_seconds=timeout_max_seconds,
            timeout_label="Query",
        )

        async def execute_query(
            query: str = "", reason: str = "", timeout: int = _default_timeout, **_: Any
        ) -> str:
            """Execute PostgreSQL query using this tool's configuration."""
            # Validate required fields
            if not query or not query.strip():
                return "Error: 'query' parameter is required. Provide a SQL query to execute."
            if not reason:
                reason = "SQL query"

            logger.info(f"[{tool_name}] Query: {reason}")

            effective_timeout = resolve_effective_timeout(timeout, timeout_max_seconds)
            db_connect_timeout = effective_timeout if effective_timeout > 0 else 30

            # Validate query
            is_safe, validation_reason = validate_sql_query(
                query, enable_write=allow_write
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            # Build command
            user = conn_config.get("user", "")
            password = conn_config.get("password", "")
            database = conn_config.get("database", "")
            container = conn_config.get("container", "")

            # SSH tunnel mode uses psycopg2
            if ssh_tunnel_config:

                def run_tunnel_query() -> str:
                    try:
                        import psycopg2  # type: ignore[import-untyped]
                        import psycopg2.extras  # type: ignore[import-untyped]
                    except ImportError:
                        return "Error: psycopg2 package not installed. Install with: pip install psycopg2-binary"

                    tunnel: SSHTunnel | None = None
                    conn = None
                    try:
                        if ssh_tunnel_config is None:
                            return "Error: SSH tunnel configuration is missing"
                        tunnel_cfg = ssh_tunnel_config_from_dict(
                            ssh_tunnel_config, default_remote_port=5432
                        )
                        if not tunnel_cfg:
                            return "Error: Invalid SSH tunnel configuration"

                        tunnel = SSHTunnel(tunnel_cfg)
                        local_port = tunnel.start()

                        conn = psycopg2.connect(
                            host="127.0.0.1",
                            port=local_port,
                            user=user,
                            password=password,
                            dbname=database,
                            connect_timeout=db_connect_timeout,
                        )
                        cursor = conn.cursor(
                            cursor_factory=psycopg2.extras.RealDictCursor
                        )
                        cursor.execute(query)

                        if cursor.description:
                            rows = cursor.fetchall()
                            if not rows and not include_metadata:
                                return "Query executed successfully (no results)"

                            # Format as psql-like output
                            columns = [col.name for col in cursor.description]
                            lines = []
                            lines.append(" | ".join(columns))
                            lines.append("-+-".join(["-" * len(c) for c in columns]))
                            for row in rows:
                                lines.append(
                                    " | ".join(str(row.get(c, "")) for c in columns)
                                )
                            lines.append(
                                f"({len(rows)} row{'s' if len(rows) != 1 else ''})"
                            )
                            output = "\n".join(lines)
                            output = add_table_metadata_to_psql_output(
                                output, include_metadata=include_metadata
                            )
                            return sanitize_output(output)
                        else:
                            return "Query executed successfully (no results)"

                    except Exception as e:
                        return f"Error: {str(e)}"
                    finally:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
                        if tunnel:
                            try:
                                tunnel.stop()
                            except Exception:
                                pass

                try:
                    if effective_timeout > 0:
                        return await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, run_tunnel_query
                            ),
                            timeout=effective_timeout + 5,
                        )
                    return await asyncio.get_event_loop().run_in_executor(
                        None, run_tunnel_query
                    )
                except asyncio.TimeoutError:
                    return f"Error: Query timed out after {effective_timeout}s"

            escaped_query = query.replace("'", "'\\''")

            if host:
                cmd = [
                    "psql",
                    "-h",
                    host,
                    "-p",
                    str(port),
                    "-U",
                    user,
                    "-d",
                    database,
                    "-c",
                    query,
                ]
                env = {"PGPASSWORD": password}
            elif container:
                cmd = [
                    "docker",
                    "exec",
                    "-i",
                    container,
                    "bash",
                    "-c",
                    f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \'{escaped_query}\'',
                ]
                env = None
            else:
                return "Error: No connection configured"

            try:
                if effective_timeout > 0:
                    process = await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env,
                        ),
                        timeout=effective_timeout,
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    return f"Error: {stderr.decode('utf-8', errors='replace').strip()}"

                output = stdout.decode("utf-8", errors="replace").strip()
                if not output:
                    return "Query executed successfully (no results)"

                if not include_metadata and "(0 rows)" in output[-20:]:
                    return "Query executed successfully (no results)"

                # Add table metadata for UI rendering BEFORE sanitizing
                # so metadata is extracted from complete data
                output = add_table_metadata_to_psql_output(
                    output, include_metadata=include_metadata
                )

                # Now sanitize the combined output
                output = sanitize_output(output)
                return output

            except asyncio.TimeoutError:
                return f"Error: Query timed out after {effective_timeout}s"
            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = (
            f"Query the {config.get('name', 'PostgreSQL')} database using SQL."
        )
        if description:
            tool_description += f" This database contains: {description}"
        tool_description += " Include LIMIT clause to restrict results. SELECT queries only unless writes are enabled."

        return StructuredTool.from_function(
            coroutine=execute_query,
            name=f"query_{tool_name}",
            description=tool_description,
            args_schema=PostgresInput,
        )

    async def _create_mssql_tool(
        self,
        config: dict,
        tool_name: str,
        _tool_id: str,
        include_metadata: bool = True,
    ):
        """Create an MSSQL/SQL Server query tool from config."""

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        max_results = config.get("max_results", 100)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 1433)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
        if not ssh_tunnel_config and not host:
            logger.error(f"MSSQL tool {tool_name} missing host configuration")
            return None

        return create_mssql_tool(
            name=config.get("name", tool_name),
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=timeout,
            timeout_max_seconds=timeout_max_seconds,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
        )

    async def _create_mysql_tool(
        self,
        config: dict,
        tool_name: str,
        _tool_id: str,
        include_metadata: bool = True,
    ):
        """Create a MySQL/MariaDB query tool from config."""

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        max_results = config.get("max_results", 100)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 3306)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
        if not ssh_tunnel_config and not host:
            logger.error(f"MySQL tool {tool_name} missing host configuration")
            return None

        return create_mysql_tool(
            name=config.get("name", tool_name),
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=timeout,
            timeout_max_seconds=timeout_max_seconds,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
        )

    async def _create_influxdb_tool(
        self,
        config: dict,
        tool_name: str,
        _tool_id: str,
        include_metadata: bool = True,
    ):
        """Create an InfluxDB 2.x (Flux) query tool from config."""

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        max_results = config.get("max_results", 100)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = int(conn_config.get("port", 8086) or 8086)
        use_https = bool(conn_config.get("use_https", False))
        token = conn_config.get("token", "")
        org = conn_config.get("org", "")

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)

        if not host or not token or not org:
            logger.error(f"InfluxDB tool {tool_name} missing required configuration")
            return None

        return create_influxdb_tool(
            name=config.get("name", tool_name),
            host=host,
            port=port,
            use_https=use_https,
            token=token,
            org=org,
            timeout=timeout,
            timeout_max_seconds=timeout_max_seconds,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
        )

    async def _create_odoo_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an Odoo shell tool from config (Docker or SSH mode)."""
        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 60)  # Odoo shell needs more time to initialize
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")
        mode = conn_config.get("mode", "docker")  # docker or ssh

        # Capture timeout for closure
        _default_timeout = timeout

        class OdooInput(BaseModel):
            code: str = Field(
                default="",
                description="Python code to execute in Odoo shell using ORM methods",
            )
            reason: str = Field(
                default="", description="Brief description of what this code does"
            )

        self._add_timeout_field_to_schema(
            OdooInput,
            default_timeout=_default_timeout,
            timeout_max_seconds=timeout_max_seconds,
            timeout_label="Execution",
        )

        def _build_docker_command(
            container: str, database: str, config_path: str
        ) -> list:
            """Build Docker exec command for Odoo shell."""
            cmd = [
                "docker",
                "exec",
                "-i",
                container,
                "odoo",
                "shell",
                "--no-http",
                "-d",
                database,
            ]
            if config_path:
                cmd.extend(["-c", config_path])
            return cmd

        async def execute_odoo(
            code: str = "", reason: str = "", timeout: int = _default_timeout, **_: Any
        ) -> str:
            """Execute Odoo shell command using this tool's configuration."""
            # Validate required fields
            if not code or not code.strip():
                return "Error: 'code' parameter is required. Provide Python code to execute in the Odoo shell."
            if not reason:
                reason = "Odoo query"

            logger.info(f"[{tool_name}] Odoo ({mode}): {reason}")

            effective_timeout = resolve_effective_timeout(timeout, timeout_max_seconds)

            # Validate code
            is_safe, validation_reason = validate_odoo_code(
                code, enable_write_ops=allow_write
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            database = conn_config.get("database", "odoo")
            config_path = conn_config.get("config_path", "")

            # Wrap user code with env setup and error handling
            wrapped_code = f"""
env = self.env
try:
{chr(10).join("    " + line for line in code.strip().split(chr(10)))}
except Exception as e:
    print(f"ODOO_ERROR: {{type(e).__name__}}: {{e}}")
"""
            # Add exit command
            full_input = wrapped_code + "\nexit()\n"

            async def _run_with_cmd(cmd: list) -> str:
                """Execute command and return filtered output."""
                try:
                    if effective_timeout > 0:
                        process = await asyncio.wait_for(
                            asyncio.create_subprocess_exec(
                                *cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                            ),
                            timeout=effective_timeout,
                        )
                    else:
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,  # Merge stderr into stdout
                        )
                    stdout, _ = await process.communicate(input=full_input.encode())
                    output = stdout.decode("utf-8", errors="replace")

                    result = filter_odoo_output(output)
                    return (
                        sanitize_output(result)
                        if result
                        else "Query executed successfully (no output)"
                    )

                except asyncio.TimeoutError:
                    return f"Error: Query timed out after {effective_timeout}s"
                except FileNotFoundError:
                    cmd_name = "SSH" if mode == "ssh" else "Docker"
                    return f"Error: {cmd_name} command not found"
                except Exception as e:
                    logger.exception(f"Odoo shell error: {e}")
                    return f"Error: {str(e)}"

            # Build command based on mode
            if mode == "ssh":
                ssh_host = conn_config.get("ssh_host", "")
                if not ssh_host:
                    return "Error: No SSH host configured"

                # Use Paramiko for SSH connection
                ssh_config = SSHConfig(
                    host=ssh_host,
                    port=conn_config.get("ssh_port", 22),
                    user=conn_config.get("ssh_user", ""),
                    password=conn_config.get("ssh_password"),
                    key_path=conn_config.get("ssh_key_path"),
                    key_content=conn_config.get("ssh_key_content"),
                    key_passphrase=conn_config.get("ssh_key_passphrase"),
                    timeout=effective_timeout if effective_timeout > 0 else 3600,
                )

                # Build remote Odoo shell command
                odoo_bin_path = conn_config.get("odoo_bin_path", "odoo-bin")
                odoo_config_path = conn_config.get("config_path", "")
                working_directory = conn_config.get("working_directory", "")
                run_as_user = conn_config.get("run_as_user", "")

                odoo_cmd = f"{odoo_bin_path} shell --no-http -d {database}"
                if odoo_config_path:
                    odoo_cmd = f"{odoo_cmd} -c {odoo_config_path}"
                if run_as_user:
                    odoo_cmd = f"sudo -u {run_as_user} {odoo_cmd}"
                if working_directory:
                    odoo_cmd = f"cd {working_directory} && {odoo_cmd}"

                # Use heredoc to pass code to shell
                remote_command = f"{odoo_cmd} <<'ODOO_EOF'\n{full_input}ODOO_EOF"

                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: execute_ssh_command(ssh_config, remote_command)
                    )

                    if not result.success and "ODOO_ERROR" not in result.output:
                        return f"Error (exit {result.exit_code}): {result.stderr or result.stdout}"

                    # For SSH, filter with ssh_mode=True to strip STDERR section
                    filtered = filter_odoo_output(result.output, ssh_mode=True)
                    return (
                        sanitize_output(filtered)
                        if filtered
                        else "Query executed successfully (no output)"
                    )

                except Exception as e:
                    logger.exception(f"Odoo SSH error: {e}")
                    return f"Error: {str(e)}"

            else:  # docker mode
                container = conn_config.get("container", "")
                if not container:
                    return "Error: No container configured"
                cmd = _build_docker_command(container, database, config_path)
                return await _run_with_cmd(cmd)

        mode_label = "SSH" if mode == "ssh" else "Docker"
        tool_description = f"Query {config.get('name', 'Odoo')} ERP using Python ORM code ({mode_label} connection)."
        if description:
            tool_description += f" This system contains: {description}"
        tool_description += (
            " Use env['model'].search_read(domain, fields, limit=N) for data retrieval."
        )

        return StructuredTool.from_function(
            coroutine=execute_odoo,
            name=f"odoo_{tool_name}",
            description=tool_description,
            args_schema=OdooInput,
        )

    async def _create_ssh_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an SSH shell tool from config."""
        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        timeout_max_seconds = int(
            config.get("timeout_max_seconds", MAX_TOOL_TIMEOUT_SECONDS) or 0
        )
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")
        working_directory = conn_config.get("working_directory", "")

        # Capture timeout for closure
        _default_timeout = timeout

        class SSHInput(BaseModel):
            command: str = Field(
                default="", description="Shell command to execute on the remote server"
            )
            reason: str = Field(
                default="", description="Brief description of what this command does"
            )

        self._add_timeout_field_to_schema(
            SSHInput,
            default_timeout=_default_timeout,
            timeout_max_seconds=timeout_max_seconds,
            timeout_label="Command",
        )

        async def execute_ssh(
            command: str = "",
            reason: str = "",
            timeout: int = _default_timeout,
            **_: Any,
        ) -> str:
            """Execute SSH command using this tool's configuration."""
            # Validate required fields
            if not command or not command.strip():
                return "Error: 'command' parameter is required. Provide a shell command to execute."
            if not reason:
                reason = "SSH command"

            host = conn_config.get("host", "")
            port = conn_config.get("port", 22)
            user = conn_config.get("user", "")
            key_path = conn_config.get("key_path")
            key_content = conn_config.get("key_content")
            key_passphrase = conn_config.get("key_passphrase")
            password = conn_config.get("password")
            command_prefix = conn_config.get("command_prefix", "")

            if not host or not user:
                return "Error: Host and user are required"

            effective_timeout = resolve_effective_timeout(timeout, timeout_max_seconds)

            # Build SSH config for potential env var expansion
            ssh_config = SSHConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                key_path=key_path,
                key_content=key_content,
                key_passphrase=key_passphrase,
                timeout=effective_timeout if effective_timeout > 0 else 3600,
            )

            # If working_directory is set, expand env vars for path validation
            expanded_command = None
            if working_directory:
                # Check if command contains env vars that need expansion (using precompiled pattern)
                if _SSH_ENV_VAR_RE.search(command):
                    # Expand env vars on the remote host
                    loop = asyncio.get_event_loop()
                    expanded_command, expand_error = await loop.run_in_executor(
                        None, lambda: expand_env_vars_via_ssh(ssh_config, command)
                    )
                    if expand_error:
                        return f"Error: {expand_error}"

            # Validate command for dangerous patterns
            is_safe, validation_reason = validate_ssh_command(
                command,
                allow_write=allow_write,
                allowed_directory=working_directory or None,
                expanded_command=expanded_command,
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            logger.info(f"[{tool_name}] SSH: {reason}")

            full_command = f"{command_prefix}{command}"

            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: execute_ssh_command(ssh_config, full_command)
                )

                if not result.success:
                    return f"Error (exit {result.exit_code}): {result.stderr or result.stdout}"

                output = result.output.strip()
                return (
                    sanitize_output(output)
                    if output
                    else "Command executed successfully (no output)"
                )

            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = (
            f"Execute shell commands on {config.get('name', 'remote server')} via SSH."
        )
        if description:
            tool_description += f" This server provides access to: {description}"
        if not allow_write:
            tool_description += " Read-only mode: write operations are blocked."

        return StructuredTool.from_function(
            coroutine=execute_ssh,
            name=f"ssh_{tool_name}",
            description=tool_description,
            args_schema=SSHInput,
        )

    async def _create_filesystem_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a filesystem search tool from config."""
        conn_config = config.get("connection_config", {})
        description = config.get("description", "")
        index_name = conn_config.get("index_name", "")

        # Store tool_id for potential future use in logging/tracking
        _tool_id = tool_id  # noqa: F841

        class FilesystemSearchInput(BaseModel):
            query: str = Field(
                description="Natural language search query to find relevant documents/files"
            )
            max_results: int = Field(
                default=10,
                ge=1,
                le=50,
                description="Maximum number of results to return (1-50, default 10)",
            )
            max_chars_per_result: int = Field(
                default=500,
                ge=0,
                le=10000,
                description="Maximum characters per result (default: 500). Use 0 for full content when you need complete file content. Increase when results are truncated.",
            )

        async def search_filesystem(
            query: str,
            max_results: int = 10,
            max_chars_per_result: int = 500,
            **_: Any,
        ) -> str:
            """Search the filesystem index."""
            logger.info(f"[{tool_name}] Filesystem search: {query[:100]}...")
            return await search_filesystem_index(
                query=query,
                index_name=index_name,
                max_results=max_results,
                max_chars_per_result=max_chars_per_result,
            )

        tool_description = (
            f"Search indexed documents from {config.get('name', 'filesystem')}."
        )
        if description:
            tool_description += f" {description}"
        if index_name:
            tool_description += f" (Index: {index_name})"

        return StructuredTool.from_function(
            coroutine=search_filesystem,
            name=f"search_{tool_name}",
            description=tool_description,
            args_schema=FilesystemSearchInput,
        )

    async def _create_pdm_search_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a SolidWorks PDM search tool from config."""
        description = config.get("description", "")

        # Use tool_name for index lookup - matches how trigger_index creates it
        index_name = f"pdm_{tool_name}"

        # Store tool_id for potential future use in logging/tracking
        _tool_id = tool_id  # noqa: F841

        # Check if there are any PDM embeddings for this tool
        embedding_count = await pdm_indexer.get_embedding_count(tool_id, tool_name)
        if embedding_count == 0:
            logger.debug(f"PDM tool configured but no embeddings found for {tool_name}")
            # Still create the tool - it will just return "no results" until indexed

        class PdmSearchInput(BaseModel):
            query: str = Field(
                description=(
                    "Natural language search query to find PDM documents. "
                    "Search for parts, assemblies, drawings by part number, "
                    "material, description, author, folder, or BOM relationships."
                )
            )
            document_type: Optional[str] = Field(
                default=None,
                description=(
                    "Optional filter by document type: SLDPRT (parts), "
                    "SLDASM (assemblies), SLDDRW (drawings), or None for all"
                ),
            )

        async def search_pdm(
            query: str, document_type: Optional[str] = None, **_: Any
        ) -> str:
            """Search the PDM index."""
            logger.info(f"[{tool_name}] PDM search: {query[:100]}...")
            return await search_pdm_index(
                query=query,
                index_name=index_name,
                document_type=document_type,
                max_results=10,
            )

        tool_description = (
            f"Search SolidWorks PDM metadata from {config.get('name', 'PDM vault')}. "
            f"Find parts, assemblies, and drawings by part number, material, "
            f"description, author, folder path, or BOM relationships."
        )
        if description:
            tool_description += f" Database contains: {description}"

        logger.info(f"Created PDM search tool: search_{tool_name}")
        return StructuredTool.from_function(
            coroutine=search_pdm,
            name=f"search_{tool_name}",
            description=tool_description,
            args_schema=PdmSearchInput,
        )

    def get_context_from_retrievers(
        self, query: str, max_docs: int = 5
    ) -> tuple[str, list[dict]]:
        """
        Retrieve relevant context from all FAISS indexes.

        Applies token budget from settings to prevent context overflow.
        When token budget is exceeded, earlier retrieved chunks take priority.

        Args:
            query: The search query.
            max_docs: Maximum documents per index.

        Returns:
            Tuple of (combined context string, list of source metadata).
        """
        all_docs = []
        sources = []
        for name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query)
                for doc in docs[:max_docs]:
                    source = doc.metadata.get("source", "unknown")
                    all_docs.append(f"[{name}:{source}]\n{doc.page_content}")
                    sources.append(
                        {
                            "index": name,
                            "source": source,
                            "preview": (
                                doc.page_content[:200] + "..."
                                if len(doc.page_content) > 200
                                else doc.page_content
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error retrieving from {name}: {e}")

        if not all_docs:
            return "", sources

        # Apply token budget if configured (0 = unlimited)
        token_budget = (
            self._app_settings.get("context_token_budget", 4000)
            if self._app_settings
            else 4000
        )

        if token_budget > 0:
            context, actual_tokens = truncate_to_token_budget(
                all_docs, max_tokens=token_budget
            )
            if actual_tokens >= token_budget:
                logger.debug(
                    f"Context truncated to {actual_tokens} tokens (budget: {token_budget})"
                )
        else:
            context = "\n\n---\n\n".join(all_docs)

        return context, sources

    def _prepend_reminder_to_content(self, content: Any, mode: str = "chat") -> Any:
        """Prepend tool usage reminder to content (string or multimodal list).

        For string content: prepends the reminder text.
        For multimodal content: prepends a text block with the reminder.

        Args:
            content: String or list of content parts (LangChain format)

        Returns:
            Content with reminder prepended
        """
        reminder_text = self._build_turn_reminder_text(mode)

        if isinstance(content, str):
            return f"{reminder_text}{content}"

        if isinstance(content, list):
            # Prepend reminder as first text block
            reminder_block = {"type": "text", "text": reminder_text}
            return [reminder_block] + content

        # Fallback
        return f"{reminder_text}{content}"

    @staticmethod
    def _build_turn_reminder_text(
        mode: str,
        *,
        include_sqlite_persistence: bool = False,
        userspace_env_var_turn_hint: str = "",
    ) -> str:
        """Build per-turn reminder text prepended to user input."""
        reminder_text = TOOL_USAGE_REMINDER
        if mode == "userspace":
            if userspace_env_var_turn_hint:
                reminder_text += build_userspace_turn_reminder_with_env_vars(
                    include_sqlite_persistence=include_sqlite_persistence,
                    env_var_reminder_line=userspace_env_var_turn_hint,
                )
            else:
                reminder_text += build_userspace_turn_reminder(
                    include_sqlite_persistence=include_sqlite_persistence,
                )
        return reminder_text

    @staticmethod
    def _build_userspace_env_var_prompt_fragment(env_vars: list[Any]) -> str:
        """Render a safe env-var inventory section (keys + set/missing, never values)."""
        if not env_vars:
            return (
                "\n### Configured workspace environment variables\n\n"
                "- None configured yet. If code needs secrets, create placeholder keys first and ask the user to fill values in the Environment Variables modal.\n"
            )

        max_items = 40
        lines: list[str] = []
        for item in env_vars[:max_items]:
            key = str(getattr(item, "key", "") or "").strip()
            if not key:
                continue
            has_value = bool(getattr(item, "has_value", False))
            status = "set" if has_value else "placeholder (value missing)"
            lines.append(f"- `{key}`: {status}")

        omitted = max(0, len(env_vars) - max_items)
        if omitted > 0:
            lines.append(f"- ...and {omitted} more")

        body = "\n".join(lines) if lines else "- None configured yet."
        return "\n### Configured workspace environment variables\n\n" + body + "\n"

    @staticmethod
    def _build_userspace_env_var_turn_hint(env_vars: list[Any]) -> str:
        """Build a short env-var line for the per-turn checklist."""
        if not env_vars:
            return (
                "- No workspace env vars are configured. If the implementation needs secrets, "
                "create placeholder keys and instruct the user to fill values in Environment Variables.\n"
            )

        max_items = 10
        parts: list[str] = []
        for item in env_vars[:max_items]:
            key = str(getattr(item, "key", "") or "").strip()
            if not key:
                continue
            has_value = bool(getattr(item, "has_value", False))
            parts.append(f"{key}({'set' if has_value else 'missing'})")

        if not parts:
            return ""

        suffix = (
            "" if len(env_vars) <= max_items else f", +{len(env_vars) - max_items} more"
        )
        return "- Workspace env vars (keys only): " + ", ".join(parts) + suffix + ".\n"

    @staticmethod
    def _build_userspace_mount_prompt_fragment(
        mountable_sources: list[Any],
        mounts: list[Any],
    ) -> str:
        """Render workspace mount context with workspace-root-relative paths."""
        mount_capability_enabled = bool(mountable_sources)
        if not mount_capability_enabled and not mounts:
            return ""

        mount_items: list[dict[str, str]] = []
        for mount in mounts:
            target_path = str(getattr(mount, "target_path", "") or "").strip() or "/"
            source_name = str(getattr(mount, "source_name", "") or "").strip()
            tool_name = str(getattr(mount, "tool_name", "") or "").strip()
            display_name = source_name or tool_name or "Unknown tool"
            source_path = str(getattr(mount, "source_path", "") or "").strip() or "."
            sync_status = str(getattr(mount, "sync_status", "") or "pending")
            normalized_target = (
                target_path if target_path.startswith("/") else f"/{target_path}"
            )
            workspace_relative = posixpath.relpath(normalized_target, "/workspace")
            enabled = bool(getattr(mount, "enabled", True))
            mount_items.append(
                {
                    "workspace_relative_path": workspace_relative,
                    "target_path": normalized_target,
                    "tool_name": display_name,
                    "source_name": source_name,
                    "source_path": source_path,
                    "sync_status": sync_status,
                    "description": str(getattr(mount, "description", "") or "").strip(),
                    "enabled": "true" if enabled else "false",
                }
            )

        return build_userspace_mounts_prompt_fragment(
            mounts_enabled=mount_capability_enabled,
            mounts=mount_items,
        )

    @staticmethod
    def _build_userspace_object_storage_prompt_fragment(
        object_storage_config: Any | None,
    ) -> str:
        """Render workspace object-storage availability for userspace mode."""

        if object_storage_config is None:
            return ""

        buckets = getattr(object_storage_config, "buckets", None) or []
        bucket_items: list[dict[str, str]] = []
        for bucket in buckets:
            bucket_name = str(getattr(bucket, "name", "") or "").strip()
            if not bucket_name:
                continue
            public_prefix = (
                str(getattr(bucket, "public_prefix", "") or "public").strip()
                or "public"
            )
            private_prefix = (
                str(getattr(bucket, "private_prefix", "") or "private").strip()
                or "private"
            )
            bucket_items.append(
                {
                    "name": bucket_name,
                    "public_root": f"/{bucket_name}/{public_prefix}",
                    "private_root": f"/{bucket_name}/{private_prefix}",
                    "description": str(
                        getattr(bucket, "description", "") or ""
                    ).strip(),
                    "is_default": (
                        "true"
                        if bool(getattr(bucket, "is_default", False))
                        else "false"
                    ),
                }
            )

        return build_userspace_object_storage_prompt_fragment(
            object_storage_enabled=True,
            buckets=bucket_items,
        )

    @staticmethod
    def _serialize_prompt_content(content: Any) -> Any:
        """Serialize LangChain content to JSON-safe values for prompt debug storage."""
        if content is None:
            return ""
        if isinstance(content, (str, int, float, bool)):
            return content
        if isinstance(content, list):
            serialized: list[Any] = []
            for item in content:
                if isinstance(item, dict):
                    serialized.append(item)
                elif hasattr(item, "model_dump"):
                    serialized.append(item.model_dump(mode="python"))
                else:
                    serialized.append(str(item))
            return serialized
        if isinstance(content, dict):
            return content
        return str(content)

    @classmethod
    def _serialize_base_message(cls, message: BaseMessage) -> dict[str, Any]:
        """Serialize BaseMessage for persisted prompt-debug payloads."""
        role = "assistant"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"

        return {
            "role": role,
            "type": message.__class__.__name__,
            "content": cls._serialize_prompt_content(getattr(message, "content", "")),
        }

    @classmethod
    def _has_non_empty_serialized_content(cls, content: Any) -> bool:
        """Return True when serialized content has meaningful data for debug display."""
        if content is None:
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            if not content:
                return False
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        return True
                    if item.get("type") != "text":
                        return True
                elif isinstance(item, str):
                    if item.strip():
                        return True
                elif item is not None:
                    return True
            return False
        if isinstance(content, dict):
            return bool(content)
        return bool(str(content).strip())

    @classmethod
    def _filter_debug_messages_with_content(
        cls, messages: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Drop debug messages with empty serialized content before persistence."""
        filtered: List[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if cls._has_non_empty_serialized_content(content):
                filtered.append(message)
        return filtered

    async def _persist_provider_prompt_debug_record(
        self,
        *,
        conversation_id: Optional[str],
        user_id: Optional[str],
        chat_task_id: Optional[str],
        provider: str,
        model: str,
        mode: str,
        request_kind: str,
        system_prompt: str,
        rendered_user_input: Any,
        chat_history: List[BaseMessage],
        provider_messages: List[dict[str, Any]],
        tool_scope_prompt: str,
        prompt_additions: str,
        turn_reminders: str,
        debug_metadata: dict[str, Any] | None = None,
        message_index: Optional[int] = None,
    ) -> None:
        """Persist a provider input debug row when DEBUG_MODE is enabled."""
        if not settings.debug_mode or not conversation_id:
            return

        rendered_user_input_serialized = self._serialize_prompt_content(
            rendered_user_input
        )
        if isinstance(rendered_user_input_serialized, str):
            rendered_user_input_text = rendered_user_input_serialized
        else:
            rendered_user_input_text = json.dumps(
                rendered_user_input_serialized,
                ensure_ascii=True,
                default=str,
            )

        try:
            serialized_chat_history = [
                self._serialize_base_message(message) for message in chat_history
            ]
            await repository.create_provider_prompt_debug_record(
                conversation_id=conversation_id,
                chat_task_id=chat_task_id,
                user_id=user_id,
                provider=provider,
                model=model,
                mode=mode,
                request_kind=request_kind,
                rendered_system_prompt=system_prompt,
                rendered_user_input=rendered_user_input_text,
                rendered_provider_messages=self._filter_debug_messages_with_content(
                    provider_messages
                ),
                rendered_chat_history=self._filter_debug_messages_with_content(
                    serialized_chat_history
                ),
                tool_scope_prompt=tool_scope_prompt,
                prompt_additions=prompt_additions,
                turn_reminders=turn_reminders,
                debug_metadata=debug_metadata,
                message_index=message_index,
            )
        except Exception:
            logger.exception("Failed to persist provider prompt debug record")

    def _convert_message_to_langchain(self, message: Any) -> Any:
        """
        Convert a Message object (from schemas.py) to LangChain HumanMessage format.

        Handles both string content and multimodal content (text + images).
        LangChain expects multimodal content in this format:
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]

        Args:
            message: Either a string or Message object with content attribute

        Returns:
            str or list suitable for LangChain HumanMessage content
        """
        # If already a string, return as-is
        if isinstance(message, str):
            return message

        # Get content from Message object
        content = getattr(message, "content", message)

        # If content is a string, return it
        if isinstance(content, str):
            return content

        # If content is a list (multimodal), convert to LangChain format
        if isinstance(content, list):
            langchain_content = []
            for item in content:
                if isinstance(item, dict):
                    # Already in dict format
                    if item.get("type") == "text":
                        langchain_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
                    elif item.get("type") == "image_url":
                        langchain_content.append(self._normalize_image_part(item))
                elif hasattr(item, "type"):
                    # Pydantic model
                    if item.type == "text":
                        langchain_content.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        image_part = {
                            "type": "image_url",
                            "image_url": {"url": item.image_url.url},
                        }
                        langchain_content.append(self._normalize_image_part(image_part))
            return langchain_content if langchain_content else ""

        # Fallback: convert to string
        return str(content)

    async def _convert_message_to_langchain_async(self, message: Any) -> Any:
        """Async message conversion with non-blocking image downsampling."""
        if isinstance(message, str):
            return message

        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            langchain_content: list[Any] = []
            image_tasks: list[tuple[int, asyncio.Task[dict[str, Any]]]] = []

            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        langchain_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
                    elif item.get("type") == "image_url":
                        index = len(langchain_content)
                        langchain_content.append(None)
                        image_tasks.append(
                            (
                                index,
                                asyncio.create_task(
                                    self._normalize_image_part_async(item)
                                ),
                            )
                        )
                elif hasattr(item, "type"):
                    if item.type == "text":
                        langchain_content.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        image_part = {
                            "type": "image_url",
                            "image_url": {"url": item.image_url.url},
                        }
                        index = len(langchain_content)
                        langchain_content.append(None)
                        image_tasks.append(
                            (
                                index,
                                asyncio.create_task(
                                    self._normalize_image_part_async(image_part)
                                ),
                            )
                        )

            if image_tasks:
                for index, task in image_tasks:
                    langchain_content[index] = await task
                langchain_content = [
                    item for item in langchain_content if item is not None
                ]

            return langchain_content if langchain_content else ""

        return str(content)

    def _strip_images_from_content(self, content: Any) -> Any:
        """
        Strip image_url parts from multimodal content.

        Used to prevent rate limit issues when using tool-calling agents,
        since each agent iteration resends the full input including images.
        Images are replaced with [image attached] placeholder text.

        Args:
            content: String or list of content parts

        Returns:
            Content with images replaced by placeholder text
        """
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return content

        stripped = []
        image_count = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_count += 1
                stripped.append({"type": "text", "text": "[image attached]"})
            else:
                stripped.append(part)

        if image_count > 0:
            logger.info(
                f"Stripped {image_count} image(s) from input to reduce token usage "
                "(tool-calling agents resend input on each iteration)"
            )

        return stripped if stripped else content

    def _fetch_http_image_as_data_url(self, url: str) -> str:
        """Fetch an HTTP/HTTPS image URL and return it as a base64 data URL.

        Clients like OpenWebUI may send image URLs pointing to their internal
        file storage instead of inline base64 data URLs.  LLM providers cannot
        reach those addresses, so we fetch the image server-side and convert it
        to a data URL that can be processed by the normal downsampling pipeline.
        """
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = (
                    resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                )
                if not content_type.startswith("image/"):
                    logger.debug(
                        "HTTP URL %s returned non-image content-type: %s",
                        url[:120],
                        content_type,
                    )
                    return url
                encoded = base64.b64encode(resp.content).decode("ascii")
                data_url = f"data:{content_type};base64,{encoded}"
                logger.info(
                    "Fetched HTTP image URL (%d bytes) and converted to data URL",
                    len(resp.content),
                )
                return data_url
        except Exception as exc:
            logger.warning("Failed to fetch HTTP image URL %s: %s", url[:120], exc)
            return url

    async def _fetch_http_image_as_data_url_async(self, url: str) -> str:
        """Async version of HTTP image URL fetching."""
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = (
                    resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                )
                if not content_type.startswith("image/"):
                    logger.debug(
                        "HTTP URL %s returned non-image content-type: %s",
                        url[:120],
                        content_type,
                    )
                    return url
                encoded = base64.b64encode(resp.content).decode("ascii")
                data_url = f"data:{content_type};base64,{encoded}"
                logger.info(
                    "Fetched HTTP image URL (%d bytes) and converted to data URL",
                    len(resp.content),
                )
                return data_url
        except Exception as exc:
            logger.warning("Failed to fetch HTTP image URL %s: %s", url[:120], exc)
            return url

    def _downsample_image_data_url(self, url: str) -> str:
        """Downsample and compress base64 image data URLs to control prompt payload size."""
        if not isinstance(url, str):
            return url

        # Fetch HTTP/HTTPS image URLs and convert to data URLs first
        if url.lower().startswith(("http://", "https://")):
            url = self._fetch_http_image_as_data_url(url)

        header, sep, payload = url.partition(",")
        if not sep or not header.lower().startswith("data:image/"):
            return url
        if ";base64" not in header.lower() or not payload:
            return url

        try:
            raw_bytes = base64.b64decode(payload, validate=False)
            if not raw_bytes:
                return url

            with Image.open(io.BytesIO(raw_bytes)) as original:
                image = ImageOps.exif_transpose(original)
                width, height = image.size
                limits = self._get_image_payload_limits()

                if width <= 0 or height <= 0:
                    return url

                scale = min(
                    limits["max_width"] / width,
                    limits["max_height"] / height,
                    math.sqrt(limits["max_pixels"] / (width * height)),
                    1.0,
                )

                if scale < 1.0:
                    resized_width = max(1, int(width * scale))
                    resized_height = max(1, int(height * scale))
                    image = image.resize(
                        (resized_width, resized_height),
                        Image.Resampling.LANCZOS,
                    )

                if image.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    alpha = image.split()[-1]
                    background.paste(image, mask=alpha)
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                best_bytes = b""
                for quality in (80, 70, 60, 50):
                    buffer = io.BytesIO()
                    image.save(
                        buffer,
                        format="JPEG",
                        quality=quality,
                        optimize=True,
                        progressive=True,
                    )
                    candidate = buffer.getvalue()
                    if not candidate:
                        continue
                    best_bytes = candidate
                    if len(candidate) <= limits["max_bytes"]:
                        break

                if not best_bytes:
                    return url

                encoded = base64.b64encode(best_bytes).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"

        except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
            logger.debug(f"Skipping image downsample for invalid data URL: {exc}")
            return url
        except Exception:
            logger.exception("Unexpected error downsampling image attachment")
            return url

    async def _downsample_image_data_url_async(self, url: str) -> str:
        """Downsample data URL off the event loop, bounded by semaphore."""
        if not isinstance(url, str):
            return url

        # Fetch HTTP/HTTPS image URLs and convert to data URLs first
        if url.lower().startswith(("http://", "https://")):
            url = await self._fetch_http_image_as_data_url_async(url)

        if not url.lower().startswith("data:image/"):
            return url

        async with self._image_downsample_semaphore:
            return await asyncio.to_thread(self._downsample_image_data_url, url)

    def _normalize_image_part(self, part: dict[str, Any]) -> dict[str, Any]:
        """Normalize image_url content part and downsample inline data URLs."""
        normalized = dict(part)
        image_url = normalized.get("image_url")

        if isinstance(image_url, dict):
            image_url_dict = dict(image_url)
            current_url = image_url_dict.get("url")
            if isinstance(current_url, str):
                image_url_dict["url"] = self._downsample_image_data_url(current_url)
            normalized["image_url"] = image_url_dict
        elif isinstance(image_url, str):
            normalized["image_url"] = {
                "url": self._downsample_image_data_url(image_url)
            }

        return normalized

    async def _normalize_image_part_async(self, part: dict[str, Any]) -> dict[str, Any]:
        """Async image part normalization/downsampling that avoids loop blocking."""
        normalized = dict(part)
        image_url = normalized.get("image_url")

        if isinstance(image_url, dict):
            image_url_dict = dict(image_url)
            current_url = image_url_dict.get("url")
            if isinstance(current_url, str):
                image_url_dict["url"] = await self._downsample_image_data_url_async(
                    current_url
                )
            normalized["image_url"] = image_url_dict
        elif isinstance(image_url, str):
            normalized["image_url"] = {
                "url": await self._downsample_image_data_url_async(image_url)
            }

        return normalized

    def _has_image_content(self, content: Any) -> bool:
        """Return True when content contains one or more image_url parts."""
        if not isinstance(content, list):
            return False

        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
            if hasattr(part, "type") and getattr(part, "type", None) == "image_url":
                return True

        return False

    def _build_local_image_data_url(self, file_path: str) -> str | None:
        """Load a local image file and convert it to a downsampled data URL."""
        raw_path = (file_path or "").strip()
        if not raw_path:
            return None

        try:
            resolved_path = Path(raw_path).resolve()
        except Exception:
            logger.debug(
                "Ignoring unreadable image path for tool context: %s", raw_path
            )
            return None

        try:
            data_root = Path(settings.index_data_path).resolve()
            resolved_path.relative_to(data_root)
        except Exception:
            logger.debug(
                "Ignoring out-of-bounds image path for tool context: %s",
                resolved_path,
            )
            return None

        if not resolved_path.exists() or not resolved_path.is_file():
            return None

        suffix = resolved_path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix)
        if not content_type:
            return None

        try:
            encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
        except Exception:
            logger.exception(
                "Failed reading local image for tool context: %s",
                resolved_path,
            )
            return None

        data_url = f"data:{content_type};base64,{encoded}"
        return self._downsample_image_data_url(data_url)

    def _build_screenshot_reference_content(
        self, observation: Any
    ) -> list[dict[str, Any]] | None:
        """Attach the latest userspace screenshot as multimodal tool context."""
        raw = observation if isinstance(observation, str) else str(observation or "")
        if not raw:
            return None

        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        image_data_url = self._build_local_image_data_url(
            str(payload.get("screenshot_path") or "")
        )
        if not image_data_url:
            return None

        preview_path = str(payload.get("preview_path") or "").strip() or "/"
        return [
            {
                "type": "text",
                "text": (
                    "Reference image from the latest capture_userspace_screenshot "
                    f"tool result for path '{preview_path}'. Use this image when "
                    "describing what was actually captured."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
        ]

    def _format_intermediate_steps_for_agent(
        self,
        intermediate_steps: list[Any],
        *,
        include_latest_screenshot_image: bool = True,
    ) -> list[BaseMessage]:
        """Format agent scratchpad steps and preserve the latest screenshot visually."""
        if not intermediate_steps:
            return []

        latest_screenshot_tool_call_id: str | None = None
        latest_screenshot_observation: Any = None
        if include_latest_screenshot_image:
            for step in intermediate_steps:
                action = step[0] if isinstance(step, tuple) and step else None
                if getattr(action, "tool", "") == "capture_userspace_screenshot":
                    latest_screenshot_tool_call_id = (
                        str(getattr(action, "tool_call_id", "") or "") or None
                    )
                    latest_screenshot_observation = (
                        step[1] if isinstance(step, tuple) and len(step) > 1 else None
                    )

        formatted_messages = list(format_to_tool_messages(intermediate_steps))
        reference_content = self._build_screenshot_reference_content(
            latest_screenshot_observation
        )
        if latest_screenshot_tool_call_id and reference_content is not None:
            augmented_messages: list[BaseMessage] = []
            for msg in formatted_messages:
                if (
                    isinstance(msg, ToolMessage)
                    and msg.tool_call_id == latest_screenshot_tool_call_id
                ):
                    tool_text = str(msg.content or "").strip()
                    content: list[dict[str, Any]] = []
                    if tool_text:
                        content.append({"type": "text", "text": tool_text})
                    content.extend(reference_content)
                    augmented_messages.append(
                        ToolMessage(
                            content=content,
                            tool_call_id=msg.tool_call_id,
                            additional_kwargs=(
                                dict(msg.additional_kwargs)
                                if getattr(msg, "additional_kwargs", None)
                                else None
                            ),
                        )
                    )
                else:
                    augmented_messages.append(msg)
            formatted_messages = augmented_messages

        return formatted_messages

    def _group_intermediate_steps_by_message(
        self,
        intermediate_steps: list[Any],
    ) -> list[list[Any]]:
        """Group tool steps that came from the same assistant tool-call message.

        Anthropic requires that all tool results for a single assistant message
        appear immediately after that message. Grouping prevents scratchpad
        compression from splitting a multi-tool assistant turn across the
        old/recent boundary.
        """
        grouped_steps: list[list[Any]] = []
        last_signature: str | None = None

        for step in intermediate_steps:
            action = step[0] if isinstance(step, tuple) and step else None
            message_log = getattr(action, "message_log", None) or []
            signature_parts: list[str] = []
            for message in message_log:
                tool_calls = getattr(message, "tool_calls", None)
                signature_parts.append(
                    json.dumps(
                        {
                            "type": type(message).__name__,
                            "content": getattr(message, "content", None),
                            "tool_calls": tool_calls,
                        },
                        default=str,
                        sort_keys=True,
                    )
                )
            signature = "||".join(signature_parts)

            if grouped_steps and signature == last_signature:
                grouped_steps[-1].append(step)
            else:
                grouped_steps.append([step])
                last_signature = signature

        return grouped_steps

    @staticmethod
    def _truncate_prompt_preview(text: str, max_chars: int = 320) -> str:
        """Return a compact single-line preview for internal prompt context."""
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(0, max_chars - 16)] + "... (truncated)"

    @classmethod
    def _summarize_tool_output_for_synthesis(
        cls,
        output: Any,
        *,
        max_chars: int = 800,
    ) -> str:
        """Convert raw tool output into synthesis-friendly plain text."""
        text = sanitize_output(str(output or "")).strip()
        if not text:
            return "(no output)"

        parsed = cls._parse_json_object(text)
        if isinstance(parsed, dict):
            parts: list[str] = []
            status = str(parsed.get("status") or "").strip()
            if status:
                parts.append(f"status={status}")

            message = str(parsed.get("message") or parsed.get("error") or "").strip()
            if message:
                parts.append(message)

            diagnostics = parsed.get("diagnostics")
            if isinstance(diagnostics, dict) and diagnostics:
                diagnostic_keys = (
                    "path",
                    "error_count",
                    "contract_error_count",
                    "total_files",
                    "inspected_file_count",
                    "has_dashboard_entry",
                    "repeat_count",
                    "signature",
                )
                diagnostic_parts = [
                    f"{key}={diagnostics[key]}"
                    for key in diagnostic_keys
                    if key in diagnostics
                ]
                if diagnostic_parts:
                    parts.append("diagnostics: " + ", ".join(diagnostic_parts))

            next_best_tool = str(parsed.get("next_best_tool") or "").strip()
            if next_best_tool:
                parts.append(f"next={next_best_tool}")

            summary = " | ".join(part for part in parts if part)
            if summary:
                return cls._truncate_prompt_preview(summary, max_chars)

        return cls._truncate_prompt_preview(text, max_chars)

    @classmethod
    def _build_internal_synthesis_tool_context(
        cls,
        *,
        intermediate_steps: list[Any],
        replay_messages: list[BaseMessage],
        max_items: int = 8,
    ) -> SystemMessage | None:
        """Build a plain-language context summary for tool-free final synthesis."""
        lines: list[str] = []

        if intermediate_steps:
            for step in intermediate_steps[-max_items:]:
                action = step[0] if isinstance(step, tuple) and step else None
                observation = step[1] if isinstance(step, tuple) and len(step) > 1 else ""
                tool_name = str(getattr(action, "tool", "unknown") or "unknown")
                tool_input = getattr(action, "tool_input", {})
                args_preview = cls._truncate_prompt_preview(
                    json.dumps(tool_input, ensure_ascii=True, default=str)
                    if isinstance(tool_input, dict)
                    else str(tool_input),
                    220,
                )
                output_summary = cls._summarize_tool_output_for_synthesis(observation)
                lines.append(
                    f"- {tool_name} args={args_preview}; result={output_summary}"
                )
        elif replay_messages:
            index = 0
            while index < len(replay_messages):
                message = replay_messages[index]
                if not isinstance(message, AIMessage):
                    index += 1
                    continue

                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    index += 1
                    continue

                tool_call = tool_calls[0]
                tool_name = str(tool_call.get("name") or "unknown")
                tool_args = tool_call.get("args") or {}
                args_preview = cls._truncate_prompt_preview(
                    json.dumps(tool_args, ensure_ascii=True, default=str)
                    if isinstance(tool_args, dict)
                    else str(tool_args),
                    220,
                )
                tool_output = ""
                if index + 1 < len(replay_messages) and isinstance(
                    replay_messages[index + 1], ToolMessage
                ):
                    tool_output = str(replay_messages[index + 1].content or "")
                    index += 1
                output_summary = cls._summarize_tool_output_for_synthesis(tool_output)
                lines.append(
                    f"- {tool_name} args={args_preview}; result={output_summary}"
                )
                if len(lines) >= max_items:
                    break
                index += 1

        if not lines:
            return None

        return SystemMessage(
            content=(
                "Verified tool activity summary for final answer generation. "
                "Use this as factual context, but do not quote raw tool protocol, "
                "function-call metadata, or JSON wrappers unless the user explicitly "
                "asked for them.\n\n"
                + "\n".join(lines)
            )
        )

    def _extract_text_from_message(self, message: Any) -> str:
        """
        Extract text content from a message (string or multimodal).

        Args:
            message: Either a string or Message object

        Returns:
            Extracted text content
        """
        if isinstance(message, str):
            return message

        # Use get_text_content if available
        if hasattr(message, "get_text_content"):
            return message.get_text_content()

        # Get content attribute
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content

        # Extract from multimodal list
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif hasattr(item, "type") and item.type == "text":
                    text_parts.append(item.text)
            return " ".join(text_parts)

        return str(content)

    @staticmethod
    def _extract_text_from_stream_content(content: Any) -> str:
        """Extract plain text from streaming content payloads."""
        if not content:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(str(block.get("text", "")))
                else:
                    text_parts.append(str(block))
            return "".join(text_parts)

        return str(content)

    @staticmethod
    def _extract_text_from_responses_output_items(output_items: Any) -> str:
        """Extract assistant-visible text from Responses API output items."""
        if not isinstance(output_items, list):
            return ""

        text_parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).lower()

            if item_type == "message":
                content_blocks = item.get("content")
                if not isinstance(content_blocks, list):
                    continue

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue

                    block_type = str(block.get("type", "")).lower()
                    if block_type in {"output_text", "text"}:
                        text = block.get("text")
                        if text:
                            text_parts.append(str(text))
                    elif block_type in {"output_refusal", "refusal"}:
                        refusal = block.get("refusal") or block.get("text")
                        if refusal:
                            text_parts.append(str(refusal))
                continue

            if item_type in {"output_text", "text"}:
                text = item.get("text")
                if text:
                    text_parts.append(str(text))
            elif item_type in {"output_refusal", "refusal"}:
                refusal = item.get("refusal") or item.get("text")
                if refusal:
                    text_parts.append(str(refusal))

        return "".join(text_parts)

    @classmethod
    def _extract_reasoning_text_from_content_list(cls, content: Any) -> str:
        """Extract reasoning text from content blocks across provider formats."""
        if not isinstance(content, list):
            return ""

        reasoning_parts: list[str] = []
        reasoning_block_types = {
            "thinking",
            "reasoning",
            "reasoning_content",
            "reasoning_summary",
            "reasoning_text",
            "reasoning_summary_text",
            "redacted_thinking",
        }
        reasoning_delta_suffixes = (
            "reasoning.delta",
            "thinking.delta",
            "reasoning_text.delta",
            "reasoning_summary_text.delta",
        )
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "")).lower()
            is_reasoning_block_type = block_type in reasoning_block_types or any(
                block_type.endswith(suffix) for suffix in reasoning_delta_suffixes
            )
            if (
                not is_reasoning_block_type
                and block.get("thought") is not True
                and not any(
                    key in block
                    for key in (
                        "reasoning_text",
                        "reasoning_summary_text",
                        "cot_summary",
                    )
                )
            ):
                continue

            text = cls._extract_reasoning_text_from_payload(
                block.get("cot_summary")
                or block.get("text")
                or block.get("thinking")
                or block.get("reasoning")
                or block.get("reasoning_text")
                or block.get("reasoning_summary_text")
                or block.get("delta")
            )
            if text:
                reasoning_parts.append(text)

            summary = block.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    summary_text = cls._extract_reasoning_text_from_payload(part)
                    if summary_text:
                        reasoning_parts.append(summary_text)
            elif isinstance(summary, str) and summary:
                reasoning_parts.append(summary)

        return "".join(reasoning_parts)

    @classmethod
    def _extract_reasoning_text_from_payload(cls, payload: Any) -> str:
        """Extract reasoning text from provider-specific structured payloads."""
        if not payload:
            return ""

        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            return cls._extract_reasoning_text_from_content_list(payload)

        if not isinstance(payload, dict):
            return str(payload)

        reasoning_parts: list[str] = []
        seen_parts: set[str] = set()

        def append_reasoning_text(value: Any) -> None:
            if not value:
                return

            if isinstance(value, dict):
                text = cls._extract_reasoning_text_from_payload(value)
            elif isinstance(value, list):
                text = cls._extract_reasoning_text_from_content_list(value)
            else:
                text = str(value)

            if text and text not in seen_parts:
                seen_parts.add(text)
                reasoning_parts.append(text)

        if payload.get("thought") is True:
            append_reasoning_text(payload.get("text"))

        for key in (
            "cot_summary",
            "text",
            "delta",
            "summary_text",
            "thinking",
            "reasoning",
            "reasoning_text",
            "reasoning_content",
            "reasoning_summary",
            "reasoning_summary_text",
        ):
            append_reasoning_text(payload.get(key))

        summary = payload.get("summary")
        if isinstance(summary, list):
            for part in summary:
                append_reasoning_text(part)
        else:
            append_reasoning_text(summary)

        content = payload.get("content")
        append_reasoning_text(content)

        return "".join(reasoning_parts)

    @classmethod
    def _extract_reasoning_from_stream_chunk(cls, chunk: Any) -> str:
        """Extract reasoning/thinking text from a streamed model chunk.

        Ollama thinking models stream tokens in `message.thinking` at the API
        level. Depending on provider adapters, this may appear as
        `additional_kwargs.thinking` or `additional_kwargs.reasoning_content`.
        """
        if not chunk:
            return ""

        additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
        if isinstance(additional_kwargs, dict):
            reasoning_text = cls._extract_reasoning_text_from_payload(
                additional_kwargs.get("reasoning")
                or additional_kwargs.get("reasoning_content")
                or additional_kwargs.get("reasoning_text")
                or additional_kwargs.get("reasoning_summary")
                or additional_kwargs.get("reasoning_summary_text")
                or additional_kwargs.get("thinking")
            )
            if reasoning_text:
                return reasoning_text

        content = getattr(chunk, "content", None)
        reasoning_from_content = (
            RAGComponents._extract_reasoning_text_from_content_list(content)
        )
        if reasoning_from_content:
            return reasoning_from_content

        return ""

    @classmethod
    def _extract_text_from_chat_model_output(cls, output: Any) -> str:
        """Extract text from LangChain chat model end payloads."""
        if output is None:
            return ""

        if hasattr(output, "content"):
            return cls._extract_text_from_stream_content(output.content)

        if isinstance(output, dict):
            # Responses API often places final assistant text under output[] items.
            output_items = output.get("output")
            text_from_output_items = cls._extract_text_from_responses_output_items(
                output_items
            )
            if text_from_output_items:
                return text_from_output_items

            content = output.get("content")
            if content is not None:
                return cls._extract_text_from_stream_content(content)

            generations = output.get("generations")
            if isinstance(generations, list) and generations:
                first_generation = generations[0]
                if isinstance(first_generation, list) and first_generation:
                    first_generation = first_generation[0]

                if isinstance(first_generation, dict):
                    message = first_generation.get("message")
                    if isinstance(message, dict):
                        return cls._extract_text_from_stream_content(
                            message.get("content")
                        )
                    if message is not None and hasattr(message, "content"):
                        return cls._extract_text_from_stream_content(message.content)

                    text = first_generation.get("text")
                    if text is not None:
                        return cls._extract_text_from_stream_content(text)

                if hasattr(first_generation, "message"):
                    message = first_generation.message
                    if hasattr(message, "content"):
                        return cls._extract_text_from_stream_content(message.content)

        if hasattr(output, "generations"):
            generations = output.generations
            if isinstance(generations, list) and generations:
                first_generation = generations[0]
                if isinstance(first_generation, list) and first_generation:
                    first_generation = first_generation[0]
                if hasattr(first_generation, "message"):
                    message = first_generation.message
                    if hasattr(message, "content"):
                        return cls._extract_text_from_stream_content(message.content)

        return ""

    @classmethod
    def _extract_reasoning_from_chat_model_output(cls, output: Any) -> str:
        """Extract final reasoning/thinking text from chat model end payloads."""
        if output is None:
            return ""

        if hasattr(output, "additional_kwargs"):
            additional_kwargs = getattr(output, "additional_kwargs", {}) or {}
            if isinstance(additional_kwargs, dict):
                reasoning_text = cls._extract_reasoning_text_from_payload(
                    additional_kwargs.get("reasoning")
                    or additional_kwargs.get("reasoning_content")
                    or additional_kwargs.get("reasoning_text")
                    or additional_kwargs.get("reasoning_summary")
                    or additional_kwargs.get("reasoning_summary_text")
                    or additional_kwargs.get("thinking")
                )
                if reasoning_text:
                    return reasoning_text

        if isinstance(output, dict):
            output_items = output.get("output")
            reasoning_from_items = cls._extract_reasoning_text_from_content_list(
                output_items
            )
            if reasoning_from_items:
                return reasoning_from_items

            message = output.get("message")
            if isinstance(message, dict):
                message_reasoning = cls._extract_reasoning_text_from_content_list(
                    message.get("content")
                )
                if message_reasoning:
                    return message_reasoning
                thinking = message.get("thinking")
                if thinking:
                    return str(thinking)

            generations = output.get("generations")
            if isinstance(generations, list) and generations:
                first_generation = generations[0]
                if isinstance(first_generation, list) and first_generation:
                    first_generation = first_generation[0]

                if isinstance(first_generation, dict):
                    message = first_generation.get("message")
                    if isinstance(message, dict):
                        message_reasoning = (
                            cls._extract_reasoning_text_from_content_list(
                                message.get("content")
                            )
                        )
                        if message_reasoning:
                            return message_reasoning
                        thinking = message.get("thinking")
                        if thinking:
                            return str(thinking)
                    elif message is not None and hasattr(message, "additional_kwargs"):
                        message_kwargs = getattr(message, "additional_kwargs", {}) or {}
                        if isinstance(message_kwargs, dict):
                            reasoning_text = cls._extract_reasoning_text_from_payload(
                                message_kwargs.get("reasoning")
                                or message_kwargs.get("reasoning_content")
                                or message_kwargs.get("reasoning_text")
                                or message_kwargs.get("reasoning_summary")
                                or message_kwargs.get("reasoning_summary_text")
                                or message_kwargs.get("thinking")
                            )
                            if reasoning_text:
                                return reasoning_text

                    if isinstance(message, dict):
                        message_reasoning = (
                            cls._extract_reasoning_text_from_content_list(
                                message.get("content")
                            )
                        )
                        if message_reasoning:
                            return message_reasoning

                if hasattr(first_generation, "message"):
                    message = first_generation.message
                    if hasattr(message, "content"):
                        message_reasoning = (
                            cls._extract_reasoning_text_from_content_list(
                                message.content
                            )
                        )
                        if message_reasoning:
                            return message_reasoning
                    if hasattr(message, "additional_kwargs"):
                        message_kwargs = getattr(message, "additional_kwargs", {}) or {}
                        if isinstance(message_kwargs, dict):
                            reasoning_text = cls._extract_reasoning_text_from_payload(
                                message_kwargs.get("reasoning")
                                or message_kwargs.get("reasoning_content")
                                or message_kwargs.get("reasoning_text")
                                or message_kwargs.get("reasoning_summary")
                                or message_kwargs.get("reasoning_summary_text")
                                or message_kwargs.get("thinking")
                            )
                            if reasoning_text:
                                return reasoning_text

        return ""

    @staticmethod
    def _compute_missing_suffix(emitted: str, final_text: str) -> str:
        """Return text present in final_text but not yet emitted."""
        if not final_text:
            return ""

        if not emitted:
            return final_text

        if final_text.startswith(emitted):
            return final_text[len(emitted) :]

        max_overlap = min(len(emitted), len(final_text))
        for overlap_size in range(max_overlap, 0, -1):
            if emitted.endswith(final_text[:overlap_size]):
                return final_text[overlap_size:]

        return ""

    def _derive_config_tool_names(self, config: dict) -> set[str]:
        """Derive runtime tool names that are generated for a ToolConfig entry."""
        tool_type = config.get("tool_type")
        raw_name = (config.get("name", "") or "").strip()
        tool_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
        if not tool_name:
            return set()

        names: set[str] = set()
        if tool_type in {"postgres", "mssql", "mysql"}:
            names.add(f"query_{tool_name}")
            names.add(f"search_{tool_name}_schema")
        elif tool_type == "influxdb":
            names.add(f"query_{tool_name}")
        elif tool_type == "odoo_shell":
            names.add(f"odoo_{tool_name}")
        elif tool_type == "ssh_shell":
            names.add(f"ssh_{tool_name}")
        elif tool_type in {"filesystem_indexer", "solidworks_pdm"}:
            names.add(f"search_{tool_name}")
        return names

    def _get_tool_connection_metadata(
        self, runtime_tool_name: str
    ) -> dict[str, str] | None:
        """Resolve tool connection metadata for a runtime tool name."""
        if not self._tool_configs:
            return None

        for config in self._tool_configs:
            tool_names = self._derive_config_tool_names(config)
            if runtime_tool_name in tool_names:
                tool_id = (config.get("id") or "").strip()
                tool_name = (config.get("name") or "").strip()
                tool_type = (config.get("tool_type") or "").strip()
                if not tool_id:
                    return None
                timeout = config.get("timeout") or 30
                timeout_max = config.get("timeout_max_seconds") or 300
                return {
                    "tool_config_id": tool_id,
                    "tool_config_name": tool_name,
                    "tool_type": tool_type,
                    "timeout": str(timeout),
                    "timeout_max_seconds": str(timeout_max),
                }
        return None

    def _build_request_tool_scope_prompt(
        self,
        tools: list[Any],
        mode: str = "chat",
    ) -> str:
        """Build a request-scoped prompt section listing active tool connections.

        Only includes tools that have connection metadata (i.e. tools backed by
        a ToolConfig entry).  Userspace file tools and other built-in tools are
        already described in the system prompt and do not need to be repeated.
        """
        if not tools:
            return ""

        lines: list[str] = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")
            if not tool_name:
                continue

            connection_meta = self._get_tool_connection_metadata(tool_name)
            if not connection_meta:
                # Skip tools without connection metadata — they are already
                # described in the system prompt and repeating them wastes tokens.
                continue

            timeout_str = connection_meta.get("timeout", "30")
            timeout_max_str = connection_meta.get("timeout_max_seconds", "300")
            line = (
                f"- `{tool_name}` -> {connection_meta.get('tool_config_name') or tool_name} "
                f"(id={connection_meta.get('tool_config_id')}, type={connection_meta.get('tool_type')}, "
                f"timeout={timeout_str}s, max_timeout={timeout_max_str}s)"
            )
            lines.append(line)

        if not lines:
            return ""

        prompt = (
            "\n\n## ACTIVE TOOL CONNECTIONS FOR THIS REQUEST\n\n"
            + "Use only these active tool connections in this request context.\n"
        )
        if mode == "userspace":
            prompt += (
                "When creating reusable dashboards/charts/tables, preserve the tool name and input as the stable data connection reference.\n"
                "Dashboard component execution via `context.components[id].execute()` shares the tool's configured timeout. "
                "If a complex query risks exceeding the timeout, optimize the query or advise the user to ask an admin to increase the tool timeout in Settings > Tools.\n"
            )

        return prompt + "\n" + "\n".join(lines)

    @staticmethod
    def _extract_query_text_from_tool_call(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str | None:
        if args:
            first = args[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, BaseModel):
                payload = first.model_dump(mode="python")
                if isinstance(payload, dict):
                    for key in ("query", "sql", "request", "command"):
                        value = payload.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
            if isinstance(first, dict):
                for key in ("query", "sql", "request", "command"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        for key in ("query", "sql", "request", "command"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_row_count_from_tool_output(output: Any) -> int:
        output_str = output if isinstance(output, str) else str(output)
        if not output_str or output_str.startswith("Error:"):
            return 0

        match = re.search(r"\((\d+)\s+rows?\)\s*$", output_str)
        if match:
            return int(match.group(1))

        if "no results" in output_str.lower():
            return 0

        return 0

    def _wrap_userspace_runtime_tools_for_execution_proofs(
        self,
        runtime_tools: list[Any],
        workspace_id: str,
        allowed_tool_config_ids: list[str],
    ) -> list[Any]:
        if not runtime_tools or not workspace_id or not allowed_tool_config_ids:
            return runtime_tools

        allowed_ids = set(allowed_tool_config_ids)
        wrapped_tools: list[Any] = []
        for tool in runtime_tools:
            tool_name = getattr(tool, "name", "")
            if not tool_name.startswith("query_"):
                wrapped_tools.append(tool)
                continue

            connection_meta = self._get_tool_connection_metadata(tool_name)
            component_id = (
                (connection_meta or {}).get("tool_config_id", "").strip()
                if connection_meta
                else ""
            )
            if not component_id or component_id not in allowed_ids:
                wrapped_tools.append(tool)
                continue

            original_coroutine = getattr(tool, "coroutine", None)
            original_func = getattr(tool, "func", None)

            if original_coroutine is not None:

                async def proofing_coroutine(
                    *args: Any,
                    _orig=original_coroutine,
                    _component_id=component_id,
                    **kwargs: Any,
                ) -> Any:
                    result = await _orig(*args, **kwargs)
                    query_text = self._extract_query_text_from_tool_call(
                        args,
                        kwargs,
                    )
                    if query_text and not str(result).startswith("Error:"):
                        row_count = self._extract_row_count_from_tool_output(result)
                        userspace_service.record_execution_proof(
                            workspace_id,
                            _component_id,
                            row_count,
                            query_text,
                        )
                    return result

                wrapped_tools.append(
                    self._clone_structured_tool(
                        tool,
                        coroutine=proofing_coroutine,
                        func=original_func,
                    )
                )
                continue

            wrapped_tools.append(tool)

        return wrapped_tools

    def _apply_mode_specific_tool_description_overrides(
        self,
        tools: list[Any],
        mode: str,
    ) -> list[Any]:
        """Return tools with mode-specific descriptions for this request."""
        if not tools:
            return tools

        live_wiring_suffix = (
            "\n\nUser Space mode override:\n"
            "- Query/search tools should provide source data and request payloads for live-wired dashboard components.\n"
            "- Do not assume static snapshots are acceptable persistence for dashboard artifacts.\n"
            "- Use the exact successful query payload as the baseline connection request for live wiring.\n"
            "- Chat query tools may enforce LIMIT for safe exploration, but persisted live_data_connections request payloads do not require LIMIT unless intentionally desired."
        )

        chat_query_suffix = (
            "\n\nChat mode override:\n"
            "- Use this tool for current-response analysis and include explicit payload values per call."
        )

        overridden_tools: list[Any] = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")
            description = getattr(tool, "description", None)
            if not tool_name or not isinstance(description, str):
                overridden_tools.append(tool)
                continue

            description_suffix = ""
            if tool_name == "create_chart":
                if mode == "userspace":
                    description_suffix = USERSPACE_CHART_DESCRIPTION_SUFFIX
                elif mode == "chat":
                    description_suffix = CHAT_CHART_DESCRIPTION_SUFFIX
            elif tool_name == "create_datatable":
                if mode == "userspace":
                    description_suffix = USERSPACE_DATATABLE_DESCRIPTION_SUFFIX
                elif mode == "chat":
                    description_suffix = CHAT_DATATABLE_DESCRIPTION_SUFFIX
            elif mode == "userspace" and tool_name.startswith(("query_", "search_")):
                description_suffix = live_wiring_suffix
            elif mode == "chat" and tool_name.startswith(("query_", "search_")):
                description_suffix = chat_query_suffix

            if not description_suffix:
                overridden_tools.append(tool)
                continue

            overridden_tools.append(
                self._clone_structured_tool(
                    tool,
                    description=(
                        description.rstrip() + "\n" + description_suffix.strip()
                    ),
                )
            )

        return overridden_tools

    async def _create_userspace_file_tools(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[StructuredTool]:
        """Create request-scoped User Space file tools for agentic artifact editing."""

        def _create_userspace_tool(
            *,
            coroutine: Any,
            name: str,
            description: str,
            args_schema: Any,
            handle_tool_error: bool = True,
        ) -> StructuredTool:
            kwargs: dict[str, Any] = {
                "coroutine": coroutine,
                "name": name,
                "description": description,
                "args_schema": args_schema,
                "handle_tool_error": handle_tool_error,
            }

            # Keep invalid tool inputs as tool-level failures (instead of
            # chat-level exceptions) where supported by langchain_core.
            try:
                return StructuredTool.from_function(
                    **kwargs,
                    handle_validation_error=True,
                )
            except TypeError:
                return StructuredTool.from_function(**kwargs)

        class ListUserSpaceFilesInput(BaseModel):
            reason: str = Field(
                default="",
                description="Brief description of why files are being listed",
            )

        class ListUserSpaceEnvVarsInput(BaseModel):
            reason: str = Field(
                default="",
                description="Brief description of why env vars are being listed",
            )

        class UpsertUserSpaceEnvVarInput(BaseModel):
            key: str = Field(
                description=(
                    "Environment variable key (for example: OPENAI_API_KEY). "
                    "Must match [A-Za-z_][A-Za-z0-9_]*."
                ),
            )
            value: str | None = Field(
                default=None,
                description=(
                    "Optional secret value. Omit to create/update placeholder keys "
                    "that users can fill in the Environment Variables modal."
                ),
            )
            description: str | None = Field(
                default=None,
                description="Optional human-readable description shown in the UI.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why this env var is needed",
            )

        class AssayUserSpaceCodeInput(BaseModel):
            max_files: int = Field(
                default=12,
                ge=1,
                le=50,
                description=(
                    "Maximum number of workspace files to inspect in the assay pass. "
                    "Prefer dashboard entry + recently updated files."
                ),
            )
            max_chars_per_file: int = Field(
                default=1600,
                ge=200,
                le=12000,
                description=(
                    "Maximum number of UTF-8 characters to include per inspected file preview."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why the code assay is needed",
            )

        class ReadUserSpaceFileInput(BaseModel):
            path: str = Field(
                default="dashboard/main.ts",
                description=(
                    "Relative path from the workspace files root to read. "
                    "Default entry file: dashboard/main.ts"
                ),
            )
            start_line: Optional[int] = Field(
                default=None,
                description="Optional starting line number (1-indexed, inclusive) to read a partial file. Useful to focus on specific sections.",
            )
            end_line: Optional[int] = Field(
                default=None,
                description="Optional ending line number (1-indexed, inclusive) to read a partial file.",
            )
            search_query: Optional[str] = Field(
                default=None,
                description="Optional exact string or keyword to locate relevant context lines. If provided, returns matching lines and their surrounding context. 'start_line'/'end_line' take precedence if both are strictly specified, otherwise they can filter the search range.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why this file is being read",
            )

        class DeleteUserSpaceFileInput(BaseModel):
            path: str = Field(
                description="Relative path from the workspace files root to delete.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why this file is being deleted",
            )

        class MoveUserSpaceFileInput(BaseModel):
            old_path: str = Field(
                description="Existing relative path from the workspace files root.",
            )
            new_path: str = Field(
                description="Destination relative path from the workspace files root.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why this file is being moved",
            )

        class UpsertUserSpaceFileInput(BaseModel):
            class LiveDataConnectionInput(BaseModel):
                component_kind: str = Field(
                    default="tool_config",
                    description="Connection kind. Must be tool_config for User Space persistence.",
                )
                component_id: str = Field(
                    description="Admin-configured tool config ID selected for this workspace.",
                )
                request: dict[str, Any] | str = Field(
                    description=(
                        "Query/command payload used to fetch or refresh live data. "
                        "For persisted live data wiring, LIMIT is optional and should only be used when intentional."
                    ),
                )
                component_name: str | None = Field(
                    default=None,
                    description="Optional friendly connection name.",
                )
                component_type: str | None = Field(
                    default=None,
                    description="Optional tool type (postgres, mssql, mysql, odoo_shell, ssh_shell, etc).",
                )
                refresh_interval_seconds: int | None = Field(
                    default=None,
                    ge=1,
                    description="Optional refresh interval in seconds.",
                )

            class LiveDataCheckInput(BaseModel):
                component_id: str | None = Field(
                    default=None,
                    description=(
                        "Tool config ID verified during live data check. "
                        "If omitted, upsert may infer it when exactly one "
                        "live_data_connections entry is provided."
                    ),
                )
                connection_check_passed: bool = Field(
                    description="True only when a live connection test succeeded.",
                )
                transformation_check_passed: bool = Field(
                    description="True only when transformation/shape checks succeeded.",
                )
                input_row_count: int | None = Field(
                    default=None,
                    ge=0,
                    description="Optional input row count seen during validation.",
                )
                output_row_count: int | None = Field(
                    default=None,
                    ge=0,
                    description="Optional output row count after transformation.",
                )
                note: str | None = Field(
                    default=None,
                    description="Optional validation note.",
                )

            path: str = Field(
                default="",
                description=(
                    "Relative path from the workspace files root to create/update "
                    "(for example: dashboard/main.ts)."
                ),
            )
            content: str = Field(description="Full UTF-8 file content to write")

            @field_validator("content", mode="before")
            @classmethod
            def _coerce_content_to_str(cls, v: Any) -> str:
                if isinstance(v, str):
                    return v
                # LLM sometimes passes a dict/list instead of a string
                return json.dumps(v, indent=2)

            @field_validator("live_data_connections", "live_data_checks", mode="before")
            @classmethod
            def _coerce_json_str_to_list(cls, v: Any) -> Any:
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, list):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                return v

            @field_validator("artifact_type", mode="before")
            @classmethod
            def _normalize_artifact_type(cls, value: Any) -> Any:
                if value is None:
                    return None

                normalized = str(value).strip().lower()
                if not normalized:
                    return None

                if normalized == "module_ts":
                    return "module_ts"

                # Treat unknown values as unspecified to avoid bubbling
                # pydantic literal validation errors into chat-level failures.
                return None

            artifact_type: ArtifactType | None = Field(
                default="module_ts",
                description=(
                    "Artifact type for preview/rendering. Use module_ts for interactive reports."
                ),
            )
            live_data_requested: bool = Field(
                default=False,
                description=(
                    "Automatically inferred to true when workspace has selected tools "
                    "and writes target dashboard modules. Explicit true is also accepted. "
                    "When effective, live_data_connections, live_data_checks, and "
                    "context.components[componentId].execute() calls are all required."
                ),
            )
            live_data_connections: list[LiveDataConnectionInput] | None = Field(
                default=None,
                description=(
                    "Required when workspace has selected tools and write targets a dashboard module. "
                    "Each connection must reference a workspace-selected tool via component_id. "
                    "The module source must structurally call context.components[component_id].execute() "
                    "for each declared connection (AST-verified)."
                ),
            )
            live_data_checks: list[LiveDataCheckInput] | None = Field(
                default=None,
                description=(
                    "Required when live_data_connections are provided. Each check must report "
                    "connection_check_passed=true and transformation_check_passed=true for a "
                    "component_id that matches a declared live_data_connections entry."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why this file is being updated",
            )

        class PatchUserSpaceFileInput(BaseModel):
            class ReplacementInput(BaseModel):
                old_text: str = Field(
                    description=(
                        "Exact text to replace. Must match existing file content."
                    ),
                )
                new_text: str = Field(
                    description="Replacement text for this occurrence.",
                )
                max_replacements: int = Field(
                    default=1,
                    ge=1,
                    le=100,
                    description=(
                        "Maximum replacements for this old_text (default 1 for surgical edits)."
                    ),
                )
                required: bool = Field(
                    default=True,
                    description=("When true, fail if old_text is not found."),
                )

            path: str = Field(
                default="dashboard/main.ts",
                description="Relative path from workspace root to patch.",
            )
            replacements: list[Any] = Field(
                default_factory=list,
                max_length=50,
                description=(
                    "Ordered replacement operations to apply. Prefer a JSON array of objects; "
                    "quoted JSON strings are tolerated and parsed best-effort."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why this patch is being applied",
            )

            @field_validator("replacements", mode="before")
            @classmethod
            def _coerce_replacements(cls, value: Any) -> Any:
                if isinstance(value, str):
                    raw = value.strip()
                    if not raw:
                        return []
                    parsed: Any | None = None
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = None

                    if isinstance(parsed, str):
                        nested_raw = parsed.strip()
                        if nested_raw:
                            try:
                                parsed = json.loads(nested_raw)
                            except Exception:
                                parsed = None

                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                    return []
                if isinstance(value, tuple):
                    return list(value)
                if isinstance(value, dict):
                    return [value]
                return value

        class CreateUserSpaceSnapshotInput(BaseModel):
            message: str = Field(
                default="AI checkpoint",
                description=(
                    "Short summary for this snapshot checkpoint. "
                    "Use milestone-oriented messages (e.g., 'wired sales chart + table')."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why snapshot is being created",
            )

        class ValidateUserSpaceCodeInput(BaseModel):
            path: str = Field(
                default="dashboard/main.ts",
                description=("Relative path from workspace files root to validate."),
            )
            reason: str = Field(
                default="",
                description="Brief description of why code validation is needed",
            )

        class CaptureUserSpaceScreenshotInput(BaseModel):
            path: str = Field(
                default="",
                description=(
                    "Optional preview subpath to capture (for example: dashboard or reports/sales). "
                    "Default captures the preview root."
                ),
            )
            width: int = Field(
                default=1440,
                ge=320,
                le=IMAGE_PAYLOAD_LIMITS["max_width"],
                description=(
                    "Viewport width in pixels. Hard-capped for AI-friendly screenshot size."
                ),
            )
            height: int = Field(
                default=900,
                ge=240,
                le=IMAGE_PAYLOAD_LIMITS["max_height"],
                description=(
                    "Viewport height in pixels. Hard-capped for AI-friendly screenshot size."
                ),
            )
            full_page: bool = Field(
                default=True,
                description="Capture full page when true; viewport only when false.",
            )
            timeout_ms: int = Field(
                default=25000,
                ge=3000,
                le=120000,
                description="Navigation and screenshot timeout in milliseconds.",
            )
            wait_for_selector: str = Field(
                default="body",
                description=(
                    "Selector that should be visible before capture. "
                    "When capture_element is true, this selector is also used as the "
                    "element clipping target and must match exactly one visible element."
                ),
            )
            capture_element: bool = Field(
                default=False,
                description=(
                    "When true, clip screenshot to the unique visible element matched by "
                    "wait_for_selector. Capture fails if selector is missing or ambiguous."
                ),
            )
            clip_padding_px: int = Field(
                default=16,
                ge=0,
                le=256,
                description=(
                    "Optional padding around the captured element clip, in pixels. "
                    "Applies only when capture_element is true."
                ),
            )
            wait_after_load_ms: int = Field(
                default=1800,
                ge=0,
                le=15000,
                description="Additional post-render settle delay before screenshot capture (helps absorb HMR reloads).",
            )
            refresh_before_capture: bool = Field(
                default=True,
                description="Reload once after initial load to reduce stale render races.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why the screenshot is needed",
            )

        class RunTerminalCommandInput(BaseModel):
            command: str = Field(
                description=(
                    "Shell command to execute in the workspace runtime container. "
                    "Runs via sh -lc so login shell, PATH, and installed tools are available. "
                    "Chain commands with && or use pipes as needed."
                ),
            )
            timeout_seconds: int = Field(
                default=30,
                ge=1,
                le=120,
                description="Maximum execution time in seconds before the command is killed.",
            )
            cwd: str = Field(
                default=".",
                description=(
                    "Workspace-relative working directory. Defaults to workspace root. "
                    "Must stay within the workspace boundary."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why this command is being run",
            )

        def _append_sqlite_hint(message: str, include_sqlite: bool) -> str:
            if not include_sqlite:
                return message
            suffix = SQLITE_INCLUDE_MODE_HINT
            base = (message or "").strip()
            if not base:
                return suffix
            if suffix in base:
                return base
            return f"{base} {suffix}"

        def _build_userspace_tool_payload(
            *,
            tool_name: str,
            status: str,
            path: str | None = None,
            message: str | None = None,
            error: str | None = None,
            rejected: bool = False,
            persisted: bool = False,
            persisted_with_violations: bool = False,
            retryable: bool = True,
            failure_class: str | None = None,
            next_best_tool: str | None = None,
            action_required: str | None = None,
            diagnostics: dict[str, Any] | None = None,
            include_sqlite_hint: bool = False,
            contract_violations: list[str] | None = None,
            warnings: list[str] | None = None,
            **extra: Any,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "tool": tool_name,
                "status": status,
                "rejected": rejected,
                "persisted": persisted,
                "persisted_with_violations": persisted_with_violations,
                "retryable": retryable,
            }
            if path:
                payload["path"] = path
            if message:
                payload["message"] = (
                    _append_sqlite_hint(message, include_sqlite_hint)
                    if include_sqlite_hint
                    else message
                )
            if error:
                payload["error"] = (
                    _append_sqlite_hint(error, include_sqlite_hint)
                    if include_sqlite_hint
                    else error
                )
            resolved_failure_class = failure_class or self._classify_userspace_failure(
                message,
                error,
                contract_violations,
                warnings,
            )
            payload["failure_class"] = resolved_failure_class
            payload["next_best_tool"] = (
                next_best_tool
                or self._next_best_tool_for_failure(
                    resolved_failure_class,
                    fallback_tool="patch_userspace_file",
                )
            )
            if action_required:
                payload["action_required"] = (
                    _append_sqlite_hint(action_required, include_sqlite_hint)
                    if include_sqlite_hint
                    else action_required
                )
            if contract_violations:
                payload["contract_violations"] = contract_violations
            if warnings:
                payload["warnings"] = warnings
            if diagnostics:
                payload["diagnostics"] = diagnostics
            payload.update(extra)
            return payload

        def _render_userspace_tool_payload(
            *,
            tool_name: str,
            status: str,
            **kwargs: Any,
        ) -> str:
            return json.dumps(
                _build_userspace_tool_payload(
                    tool_name=tool_name,
                    status=status,
                    **kwargs,
                ),
                indent=2,
            )

        async def list_userspace_files(reason: str = "", **_: Any) -> str:
            del reason
            files = await userspace_service.list_workspace_files(
                workspace_id, user_id, include_dirs=True
            )
            file_items = [f.model_dump(mode="json") for f in files]
            return _render_userspace_tool_payload(
                tool_name="list_userspace_files",
                status="listed",
                message=f"Listed {len(file_items)} workspace paths.",
                persisted=False,
                retryable=True,
                failure_class="none",
                next_best_tool="read_userspace_file",
                files=file_items,
                count=len(file_items),
            )

        async def list_userspace_env_vars(reason: str = "", **_: Any) -> str:
            del reason
            env_vars = await userspace_service.list_workspace_env_var_summaries(
                workspace_id,
                user_id,
            )
            env_var_items = [item.model_dump(mode="json") for item in env_vars]
            return _render_userspace_tool_payload(
                tool_name="list_userspace_env_vars",
                status="listed",
                message=f"Listed {len(env_var_items)} workspace environment variables.",
                persisted=False,
                retryable=True,
                failure_class="none",
                next_best_tool="upsert_userspace_env_var",
                env_vars=env_var_items,
                count=len(env_var_items),
            )

        async def upsert_userspace_env_var(
            key: str,
            value: str | None = None,
            description: str | None = None,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            updated = await userspace_service.upsert_workspace_env_var(
                workspace_id,
                user_id,
                UpsertWorkspaceEnvVarRequest(
                    key=key,
                    value=value,
                    description=description,
                ),
            )
            normalized_key = (key or "").strip()
            return _render_userspace_tool_payload(
                tool_name="upsert_userspace_env_var",
                status="persisted",
                message=(
                    f"Environment variable {normalized_key} saved."
                    if value is not None
                    else f"Environment variable placeholder {normalized_key} saved."
                ),
                persisted=True,
                retryable=True,
                failure_class="none",
                next_best_tool="upsert_userspace_file",
                key=normalized_key,
                env_var=updated.model_dump(mode="json"),
            )

        def _compute_authoritative_entrypoint(
            file_paths: set[str],  # noqa: ARG001 – kept for call-site compat
        ) -> tuple[str | None, str]:
            ep_status = userspace_service.get_workspace_entrypoint_status(workspace_id)
            if ep_status.state == "valid":
                is_default = userspace_service.is_default_static_entrypoint(
                    workspace_id
                )
                if is_default:
                    return (
                        ".ragtime/runtime-entrypoint.json",
                        (
                            ".ragtime/runtime-entrypoint.json exists but uses the default "
                            "static server seed. Choose a framework that fits the user's "
                            "request and update the entrypoint."
                        ),
                    )
                framework_label = ep_status.framework or "custom"
                return (
                    ".ragtime/runtime-entrypoint.json",
                    (
                        f".ragtime/runtime-entrypoint.json is configured with "
                        f"framework={framework_label}. Runtime launch is locked in."
                    ),
                )
            if ep_status.state == "invalid":
                return (
                    ".ragtime/runtime-entrypoint.json",
                    f".ragtime/runtime-entrypoint.json is invalid: {ep_status.error}",
                )

            return (
                None,
                "Missing .ragtime/runtime-entrypoint.json runtime launch config.",
            )

        async def _get_workspace_structure() -> dict[str, Any]:
            files = await userspace_service.list_workspace_files(
                workspace_id, user_id, include_dirs=True
            )
            file_paths = {file.path for file in files}
            authoritative_entrypoint, entrypoint_reason = (
                _compute_authoritative_entrypoint(file_paths)
            )
            return {
                "files": files,
                "authoritative_entrypoint": authoritative_entrypoint,
                "entrypoint_reason": entrypoint_reason,
                "has_dashboard_entry": "dashboard/main.ts" in file_paths,
                "has_runtime_entrypoint": ".ragtime/runtime-entrypoint.json"
                in file_paths,
                "has_index_html": any(
                    PurePosixPath(path).name.lower() == "index.html"
                    for path in file_paths
                ),
            }

        def _is_index_html_path(path: str) -> bool:
            normalized = (path or "").strip().replace("\\", "/").lstrip("/")
            if not normalized:
                return False
            return PurePosixPath(normalized).name.lower() == "index.html"

        async def _build_live_data_contract_context(
            path: str,
            workspace: Any | None = None,
        ) -> dict[str, Any]:
            normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
            is_dashboard_entry = normalized_path.lower() == "dashboard/main.ts"

            ws = workspace
            if ws is None:
                ws = await userspace_service.get_workspace(workspace_id, user_id)

            selected_tool_ids = list(getattr(ws, "selected_tool_ids", []) or [])
            requires_contract = bool(selected_tool_ids) and is_dashboard_entry

            entry_file_state: dict[str, Any] = {
                "path": "dashboard/main.ts",
                "exists": False,
                "live_data_connections_count": 0,
                "live_data_checks_count": 0,
                "live_data_connection_component_ids": [],
                "live_data_check_component_ids": [],
            }

            try:
                entry_file = await userspace_service.get_workspace_file(
                    workspace_id,
                    "dashboard/main.ts",
                    user_id,
                )
            except HTTPException as exc:
                if getattr(exc, "status_code", None) != 404:
                    raise
            else:
                entry_connections = list(entry_file.live_data_connections or [])
                entry_checks = list(entry_file.live_data_checks or [])
                entry_file_state = {
                    "path": "dashboard/main.ts",
                    "exists": True,
                    "live_data_connections_count": len(entry_connections),
                    "live_data_checks_count": len(entry_checks),
                    "live_data_connection_component_ids": [
                        conn.component_id for conn in entry_connections
                    ],
                    "live_data_check_component_ids": [
                        check.component_id for check in entry_checks
                    ],
                }

            return {
                "path": normalized_path,
                "is_dashboard_entry": is_dashboard_entry,
                "workspace_has_selected_tools": bool(selected_tool_ids),
                "selected_tool_ids": selected_tool_ids,
                "requires_live_data_contract_for_path": requires_contract,
                "requirements_when_effective": [
                    "Provide live_data_connections with component_kind=tool_config, component_id, and request.",
                    "Provide live_data_checks with connection_check_passed=true and transformation_check_passed=true for each declared component_id.",
                    "Use only selected_tool_ids as component_id values.",
                    "Source code must structurally call context.components[componentId].execute() for live data wiring.",
                ],
                "entry_file_state": entry_file_state,
            }

        async def assay_userspace_code(
            max_files: int = 12,
            max_chars_per_file: int = 1600,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )

            structure = await _get_workspace_structure()
            files = structure["files"]
            has_dashboard_entry = bool(structure["has_dashboard_entry"])
            has_index_html = bool(structure["has_index_html"])
            authoritative_entrypoint = structure["authoritative_entrypoint"]
            entrypoint_reason = str(structure["entrypoint_reason"])
            contract_context_path = "dashboard/main.ts"
            live_data_contract_context = await _build_live_data_contract_context(
                contract_context_path
            )

            dashboard_paths = sorted(
                [file.path for file in files if file.path.startswith("dashboard/")]
            )
            non_dashboard_paths = sorted(
                [file.path for file in files if not file.path.startswith("dashboard/")]
            )

            prioritized_paths: list[str] = []
            if "dashboard/main.ts" in dashboard_paths:
                prioritized_paths.append("dashboard/main.ts")
                dashboard_paths.remove("dashboard/main.ts")

            prioritized_paths.extend(dashboard_paths)
            prioritized_paths.extend(non_dashboard_paths)

            selected_paths = prioritized_paths[:max_files]

            inspected: list[dict[str, Any]] = []
            for path in selected_paths:
                try:
                    file_data = await userspace_service.get_workspace_file(
                        workspace_id,
                        path,
                        user_id,
                        decode_errors="replace",
                    )
                except HTTPException:
                    continue
                content = file_data.content
                line_count = content.count("\n") + (1 if content else 0)
                inspected.append(
                    {
                        "path": file_data.path,
                        "artifact_type": file_data.artifact_type,
                        "content_chars": len(content),
                        "line_count": line_count,
                        "preview": content[:max_chars_per_file],
                    }
                )

            assay = {
                "workspace_id": workspace_id,
                "summary": {
                    "total_files": len(files),
                    "dashboard_file_count": len(
                        [file for file in files if file.path.startswith("dashboard/")]
                    ),
                    "has_dashboard_entry": has_dashboard_entry,
                    "inspected_file_count": len(inspected),
                },
                "structure": {
                    "authoritative_entrypoint": authoritative_entrypoint,
                    "entrypoint_reason": entrypoint_reason,
                    "has_dashboard_entry": has_dashboard_entry,
                    "has_runtime_entrypoint": bool(structure["has_runtime_entrypoint"]),
                    "has_index_html": has_index_html,
                },
                "live_data_contract": live_data_contract_context,
                "inspected_files": inspected,
                "next_step": (
                    "Read target files in full before overwrite, then update files and validate TypeScript."
                ),
                "editing_guidance": (
                    "When dashboard/main.ts exists, implement dashboard feature changes in dashboard/* files and avoid index.html edits for behavior changes."
                ),
            }
            return _render_userspace_tool_payload(
                tool_name="assay_userspace_code",
                status="assayed",
                message="Workspace assay completed.",
                persisted=False,
                retryable=True,
                failure_class="none",
                next_best_tool="read_userspace_file",
                workspace=assay,
                diagnostics={
                    "total_files": len(files),
                    "inspected_file_count": len(inspected),
                    "has_dashboard_entry": has_dashboard_entry,
                },
            )

        async def read_userspace_file(
            path: str,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
            search_query: Optional[str] = None,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
            try:
                file_data = await userspace_service.get_workspace_file(
                    workspace_id, normalized_path, user_id
                )
            except HTTPException as exc:
                if exc.status_code == 404:
                    return _render_userspace_tool_payload(
                        tool_name="read_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="path_invalid",
                        next_best_tool="list_userspace_files",
                        path=normalized_path,
                        error=f"File '{normalized_path}' does not exist in this workspace.",
                        action_required="Use list_userspace_files to see available files and retry with an exact path.",
                    )
                if exc.status_code == 415:
                    return _render_userspace_tool_payload(
                        tool_name="read_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=False,
                        failure_class="binary_file",
                        next_best_tool="list_userspace_files",
                        path=normalized_path,
                        error=f"File '{normalized_path}' is a binary file and cannot be read as text.",
                        action_required="Choose a text file path or use another tool better suited to the binary artifact.",
                    )
                raise
            payload = file_data.model_dump(mode="json")

            if payload.get("content") and (
                start_line is not None or end_line is not None or search_query
            ):
                lines = payload["content"].splitlines(keepends=True)
                total_lines = len(lines)

                if total_lines > 0:
                    s_line = start_line if start_line is not None else 1
                    e_line = end_line if end_line is not None else total_lines
                    s_line = max(1, min(s_line, total_lines))
                    e_line = max(1, max(s_line, min(e_line, total_lines)))

                    if search_query:
                        search_query_lower = search_query.lower()
                        context_padding = 5
                        match_indices = []
                        bounded_start = s_line - 1
                        bounded_end = e_line

                        for i in range(bounded_start, bounded_end):
                            if search_query_lower in lines[i].lower():
                                match_indices.append(i)

                        if not match_indices:
                            payload["content"] = (
                                f"No matches found for '{search_query}' in the specified range (lines {s_line}-{e_line})."
                            )
                            payload["_meta"] = {
                                "search_query": search_query,
                                "matches_found": 0,
                            }
                        else:
                            output_lines = []
                            last_end = -1
                            for idx in match_indices:
                                window_start = max(bounded_start, idx - context_padding)
                                window_end = min(bounded_end, idx + context_padding + 1)

                                if last_end != -1 and window_start > last_end:
                                    output_lines.append("... [omitted lines] ...\n")
                                else:
                                    window_start = max(window_start, last_end)

                                for i in range(window_start, window_end):
                                    output_lines.append(f"{i+1:5d} | {lines[i]}")
                                last_end = window_end

                            payload["content"] = "".join(output_lines)
                            payload["_meta"] = {
                                "search_query": search_query,
                                "matches_found": len(match_indices),
                                "total_lines": total_lines,
                            }
                    else:
                        output_lines = []
                        for i in range(s_line - 1, e_line):
                            output_lines.append(f"{i+1:5d} | {lines[i]}")
                        payload["content"] = "".join(output_lines)
                        payload["_meta"] = {
                            "start_line": s_line,
                            "end_line": e_line,
                            "total_lines": total_lines,
                        }

            structure = await _get_workspace_structure()
            if _is_index_html_path(normalized_path) and bool(
                structure["has_dashboard_entry"]
            ):
                payload["warning"] = (
                    "This workspace has dashboard/main.ts. For dashboard behavior changes, edit dashboard/* files (especially dashboard/main.ts and imported modules) instead of index.html."
                )
            warning = payload.pop("warning", None)
            meta = payload.pop("_meta", None)
            return _render_userspace_tool_payload(
                tool_name="read_userspace_file",
                status="read",
                path=normalized_path,
                message=f"Read {normalized_path}.",
                persisted=False,
                retryable=True,
                failure_class="none",
                next_best_tool="patch_userspace_file",
                file=payload,
                warnings=[warning] if warning else None,
                diagnostics=meta,
            )

        async def delete_userspace_file(path: str, reason: str = "", **_: Any) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
            normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
            if not normalized_path:
                raise ToolException("Invalid path: path is required.")
            try:
                await userspace_service.delete_workspace_file(
                    workspace_id,
                    normalized_path,
                    user_id,
                )
            except HTTPException as exc:
                if getattr(exc, "status_code", None) == 404:
                    return _render_userspace_tool_payload(
                        tool_name="delete_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="path_invalid",
                        next_best_tool="list_userspace_files",
                        path=normalized_path,
                        error=f"File not found: {normalized_path}.",
                        action_required="Check the exact path with list_userspace_files and retry delete_userspace_file.",
                    )
                return _render_userspace_tool_payload(
                    tool_name="delete_userspace_file",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class=self._classify_userspace_failure(
                        getattr(exc, "detail", exc)
                    ),
                    next_best_tool="list_userspace_files",
                    path=normalized_path,
                    error=f"Cannot delete {normalized_path}: {getattr(exc, 'detail', exc)}",
                    action_required="Inspect the file path and workspace state, then retry delete_userspace_file.",
                )
            await userspace_runtime_service.bump_workspace_generation(
                workspace_id, "file_delete", payload={"path": normalized_path}
            )
            return _render_userspace_tool_payload(
                tool_name="delete_userspace_file",
                status="persisted",
                path=normalized_path,
                message=f"Deleted {normalized_path}.",
                persisted=True,
                retryable=True,
                failure_class="none",
                next_best_tool="assay_userspace_code",
                deleted=True,
            )

        async def move_userspace_file(
            old_path: str,
            new_path: str,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
            normalized_old_path = (
                (old_path or "").strip().replace("\\", "/").lstrip("/")
            )
            normalized_new_path = (
                (new_path or "").strip().replace("\\", "/").lstrip("/")
            )
            if not normalized_old_path or not normalized_new_path:
                raise ToolException("Invalid path: old_path and new_path are required.")
            try:
                result = await userspace_service.move_workspace_file(
                    workspace_id,
                    normalized_old_path,
                    normalized_new_path,
                    user_id,
                )
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code == 404:
                    return _render_userspace_tool_payload(
                        tool_name="move_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="path_invalid",
                        next_best_tool="list_userspace_files",
                        path=normalized_old_path,
                        error=f"File not found: {normalized_old_path}.",
                        action_required="Check the source path with list_userspace_files and retry move_userspace_file.",
                    )
                if status_code == 409:
                    return _render_userspace_tool_payload(
                        tool_name="move_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="path_conflict",
                        next_best_tool="delete_userspace_file",
                        path=normalized_old_path,
                        error=f"Target already exists: {normalized_new_path}.",
                        action_required="Move to a different path or delete or rename the destination first.",
                        diagnostics={
                            "old_path": normalized_old_path,
                            "new_path": normalized_new_path,
                        },
                    )
                return _render_userspace_tool_payload(
                    tool_name="move_userspace_file",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class=self._classify_userspace_failure(
                        getattr(exc, "detail", exc)
                    ),
                    next_best_tool="list_userspace_files",
                    path=normalized_old_path,
                    error=(
                        "Cannot move file from "
                        f"{normalized_old_path} to {normalized_new_path}: "
                        f"{getattr(exc, 'detail', exc)}"
                    ),
                    action_required="Inspect both paths and retry move_userspace_file.",
                    diagnostics={
                        "old_path": normalized_old_path,
                        "new_path": normalized_new_path,
                    },
                )

            await userspace_runtime_service.bump_workspace_generation(
                workspace_id,
                "file_move",
                payload={
                    "old_path": normalized_old_path,
                    "new_path": normalized_new_path,
                },
            )
            return _render_userspace_tool_payload(
                tool_name="move_userspace_file",
                status="persisted",
                path=result["new_path"],
                message=(f"Moved {result['old_path']} to {result['new_path']}."),
                persisted=True,
                retryable=True,
                failure_class="none",
                next_best_tool="read_userspace_file",
                moved=True,
                old_path=result["old_path"],
                new_path=result["new_path"],
            )

        async def patch_userspace_file(
            path: str = "dashboard/main.ts",
            replacements: list[Any] | None = None,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )

            normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
            if not normalized_path:
                raise ToolException("Invalid path: path is required.")

            try:
                file_data = await userspace_service.get_workspace_file(
                    workspace_id, normalized_path, user_id
                )
            except HTTPException as exc:
                if getattr(exc, "status_code", None) == 404:
                    raise ToolException(
                        f"File not found: {normalized_path}. "
                        "Use upsert_userspace_file to create it first, or check the path."
                    ) from exc
                raise ToolException(
                    f"Cannot read {normalized_path}: {getattr(exc, 'detail', exc)}"
                ) from exc

            parsed_replacements: list[PatchUserSpaceFileInput.ReplacementInput] = []

            raw_replacements: Any = replacements
            if isinstance(raw_replacements, str):
                decoded: Any | None = None
                raw = raw_replacements.strip()
                if raw:
                    try:
                        decoded = json.loads(raw)
                    except Exception:
                        decoded = None

                    if isinstance(decoded, str):
                        nested_raw = decoded.strip()
                        if nested_raw:
                            try:
                                decoded = json.loads(nested_raw)
                            except Exception:
                                decoded = None

                raw_replacements = decoded

            if isinstance(raw_replacements, dict):
                raw_replacements = [raw_replacements]
            if raw_replacements is None:
                raw_replacements = []
            if not isinstance(raw_replacements, list):
                raise ToolException(
                    "Invalid replacements format. Provide a list of replacement objects or a JSON string that decodes to one."
                )

            for item in raw_replacements:
                payload = (
                    item.model_dump(mode="python")
                    if isinstance(item, BaseModel)
                    else item
                )
                try:
                    parsed_replacements.append(
                        PatchUserSpaceFileInput.ReplacementInput.model_validate(payload)
                    )
                except Exception as exc:
                    raise ToolException(
                        "Invalid replacement object. Each item must include old_text and new_text, with optional max_replacements and required."
                    ) from exc

            if not parsed_replacements:
                raise ToolException(
                    "No replacements provided. Supply at least one replacement operation."
                )

            updated_content = file_data.content
            applied: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []

            def _newline_style(text: str) -> str:
                has_crlf = "\r\n" in text
                has_lf = "\n" in text
                if has_crlf and has_lf:
                    # Mixed style means both CRLF and LF-only segments exist.
                    return "mixed"
                if has_crlf:
                    return "crlf"
                if has_lf:
                    return "lf"
                return "none"

            def _first_non_empty_line(text: str) -> str:
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        return stripped
                return ""

            def _last_non_empty_line(text: str) -> str:
                for line in reversed(text.splitlines()):
                    stripped = line.strip()
                    if stripped:
                        return stripped
                return ""

            for index, replacement in enumerate(parsed_replacements, start=1):
                old_text = replacement.old_text
                new_text = replacement.new_text
                max_replacements = replacement.max_replacements
                required = replacement.required

                if not old_text:
                    raise ToolException(
                        f"Replacement #{index} invalid: old_text must not be empty."
                    )

                found_count = updated_content.count(old_text)
                if found_count == 0:
                    if required:
                        old_prefix = _first_non_empty_line(old_text)[:160]
                        old_suffix = _last_non_empty_line(old_text)[:160]
                        normalized_old = old_text.replace("\r\n", "\n")
                        normalized_file = updated_content.replace("\r\n", "\n")
                        newline_normalized_match = (
                            normalized_file.count(normalized_old) > 0
                            if normalized_old
                            else False
                        )

                        prefix_matches = (
                            updated_content.count(old_prefix) if old_prefix else 0
                        )
                        suffix_matches = (
                            updated_content.count(old_suffix) if old_suffix else 0
                        )

                        file_excerpt = updated_content[:1200]
                        return _render_userspace_tool_payload(
                            tool_name="patch_userspace_file",
                            status="rejected_not_persisted",
                            rejected=True,
                            persisted=False,
                            retryable=True,
                            failure_class="patch_context_mismatch",
                            next_best_tool="read_userspace_file",
                            path=normalized_path,
                            message=f"Replacement #{index} failed: old_text not found.",
                            action_required=(
                                "Read the file again with read_userspace_file and patch using exact current text. "
                                "Avoid using terminal output from a different file view as the patch source."
                            ),
                            diagnostics={
                                "replacement_index": index,
                                "old_text_chars": len(old_text),
                                "new_text_chars": len(new_text),
                                "file_chars": len(updated_content),
                                "old_text_newline_style": _newline_style(old_text),
                                "file_newline_style": _newline_style(updated_content),
                                "newline_normalized_match": newline_normalized_match,
                                "old_text_prefix": old_prefix,
                                "old_text_suffix": old_suffix,
                                "prefix_occurrences_in_file": prefix_matches,
                                "suffix_occurrences_in_file": suffix_matches,
                                "file_excerpt_start": file_excerpt,
                            },
                            attempted_replacements=applied,
                            skipped=skipped,
                            updated=False,
                        )
                    skipped.append(
                        {
                            "index": index,
                            "reason": "old_text not found",
                        }
                    )
                    continue

                replace_count = min(found_count, max_replacements)
                updated_content = updated_content.replace(
                    old_text,
                    new_text,
                    replace_count,
                )
                applied.append(
                    {
                        "index": index,
                        "matched": found_count,
                        "replaced": replace_count,
                    }
                )

            if updated_content == file_data.content:
                return _render_userspace_tool_payload(
                    tool_name="patch_userspace_file",
                    status="no_changes",
                    path=normalized_path,
                    message="No content changes were applied.",
                    retryable=True,
                    failure_class="no_change",
                    next_best_tool="read_userspace_file",
                    applied=applied,
                    skipped=skipped,
                    updated=False,
                )

            normalized_lower_path = normalized_path.lower()
            patch_is_dashboard_entry = normalized_lower_path == "dashboard/main.ts"

            # Detect hardcoded data in entrypoint files.
            # Dashboard entry: hard reject.  Other entrypoints: soft warning.
            patch_mock_patterns = find_hardcoded_data_patterns(updated_content)
            patch_hardcoded_warning: str | None = None
            if patch_mock_patterns and patch_is_dashboard_entry:
                return _render_userspace_tool_payload(
                    tool_name="patch_userspace_file",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class="hardcoded_data_detected",
                    next_best_tool="patch_userspace_file",
                    path=normalized_path,
                    error=(
                        "LIVE DATA POLICY VIOLATION -- Hardcoded data patterns detected: "
                        + ", ".join(patch_mock_patterns)
                        + ". Replace static/mock data with live runtime data via context.components[componentId].execute()."
                    ),
                    action_required=(
                        "Replace mock/static dashboard data with live runtime data wiring, then retry patch_userspace_file."
                    ),
                    diagnostics={"patterns": patch_mock_patterns},
                )
            elif patch_mock_patterns and normalized_lower_path.endswith(
                _HARDCODED_DATA_SOURCE_EXTENSIONS
            ):
                # Check if this file is the runtime entrypoint and workspace has tools
                try:
                    patch_ws = await userspace_service.get_workspace(
                        workspace_id, user_id
                    )
                    if bool(patch_ws.selected_tool_ids):
                        patch_resolved_ep: str | None = None
                        try:
                            patch_ep_file = await userspace_service.get_workspace_file(
                                workspace_id,
                                ".ragtime/runtime-entrypoint.json",
                                user_id,
                            )
                            patch_cfg = json.loads(patch_ep_file.content or "{}")
                            if isinstance(patch_cfg, dict):
                                patch_resolved_ep = _resolve_entrypoint_source_file(
                                    patch_cfg
                                )
                        except Exception:
                            pass
                        is_entrypoint = (
                            patch_resolved_ep is not None
                            and normalized_lower_path == patch_resolved_ep.lower()
                        )
                        if is_entrypoint:
                            patch_hardcoded_warning = (
                                "Hardcoded data patterns detected: "
                                + ", ".join(patch_mock_patterns)
                                + ". When workspace has selected tools, entrypoint data "
                                "should be fetched from live sources, not embedded in source code."
                            )
                except Exception:
                    pass

            request_payload = UpsertWorkspaceFileRequest(
                content=updated_content,
                artifact_type=file_data.artifact_type,
                live_data_requested=bool(file_data.live_data_connections),
                live_data_connections=file_data.live_data_connections,
                live_data_checks=file_data.live_data_checks,
            )

            try:
                result = await userspace_service.upsert_workspace_file(
                    workspace_id,
                    normalized_path,
                    request_payload,
                    user_id,
                    skip_live_data_enforcement=True,
                )
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", None)
                detail_text = str(getattr(exc, "detail", exc))
                if status_code == 400:
                    return _render_userspace_tool_payload(
                        tool_name="patch_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class=self._classify_userspace_failure(detail_text),
                        path=normalized_path,
                        message=(
                            "Patch text replacements were computed locally but the file was NOT persisted."
                        ),
                        error=detail_text,
                        action_required=(
                            "Apply the required wiring or validation fixes from this response, then retry patch_userspace_file."
                        ),
                        attempted_replacements=applied,
                        skipped=skipped,
                        updated=False,
                    )
                raise

            typecheck: dict[str, Any] | None = None
            if is_userspace_typescript_path(normalized_path):
                typecheck = await validate_userspace_typescript_content(
                    updated_content,
                    normalized_path,
                )

            write_signature = hashlib.sha256(
                f"{normalized_path}\0{updated_content}".encode("utf-8")
            ).hexdigest()[:16]
            response_payload = _build_userspace_tool_payload(
                tool_name="patch_userspace_file",
                status=(
                    "persisted_with_violations"
                    if patch_hardcoded_warning
                    else "persisted"
                ),
                path=normalized_path,
                message=(
                    "Patch applied and persisted."
                    if not patch_hardcoded_warning
                    else "Patch applied and persisted, but the file still has live data policy violations."
                ),
                persisted=True,
                persisted_with_violations=bool(patch_hardcoded_warning),
                retryable=True,
                failure_class=(
                    "hardcoded_data_detected" if patch_hardcoded_warning else "none"
                ),
                next_best_tool=(
                    "patch_userspace_file"
                    if patch_hardcoded_warning
                    else "validate_userspace_code"
                ),
                action_required=(
                    "Run validate_userspace_code on the changed file."
                    if not patch_hardcoded_warning
                    else "Use patch_userspace_file to replace mock/static data with live data sources, then run validate_userspace_code."
                ),
                contract_violations=(
                    [patch_hardcoded_warning] if patch_hardcoded_warning else None
                ),
                file=result.model_dump(mode="json"),
                applied=applied,
                skipped=skipped,
                updated=True,
                write_signature=write_signature,
            )
            if typecheck is not None:
                response_payload["typescript_validation"] = typecheck
            await userspace_runtime_service.bump_workspace_generation(
                workspace_id, "file_patch", payload={"path": normalized_path}
            )
            return json.dumps(response_payload, indent=2)

        async def upsert_userspace_file(
            content: str,
            path: str = "",
            artifact_type: ArtifactType | None = "module_ts",
            live_data_requested: bool = False,
            live_data_connections: list[Any] | None = None,
            live_data_checks: list[Any] | None = None,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )

            path = (path or "").strip()
            if not path:
                raise ToolException(
                    "path is required. Provide a relative workspace file path "
                    "(for example: dashboard/main.ts)."
                )

            warnings: list[str] = []
            allowed_violations: list[str] = []
            hard_errors: list[str] = []
            contract_violations: list[str] = []
            lower_path = (path or "").lower()
            normalized_path = lower_path.replace("\\", "/")

            structure = await _get_workspace_structure()
            if _is_index_html_path(path) and bool(structure["has_dashboard_entry"]):
                warnings.append(
                    "This workspace uses dashboard/main.ts as the module entrypoint. "
                    "Prefer dashboard/* files for feature changes. "
                    "index.html is allowed for runtime scaffolding but should not contain application logic."
                )

            parsed_live_data_connections: list[UserSpaceLiveDataConnection] | None = (
                None
            )
            parsed_live_data_checks: list[UserSpaceLiveDataCheck] | None = None
            if live_data_connections is not None:
                parsed_live_data_connections = []
                for item in live_data_connections:
                    payload = (
                        item.model_dump(mode="python")
                        if isinstance(item, BaseModel)
                        else item
                    )
                    parsed_live_data_connections.append(
                        UserSpaceLiveDataConnection.model_validate(payload)
                    )

            if live_data_checks is not None:
                parsed_live_data_checks = []
                inferred_component_id: str | None = None
                if (
                    parsed_live_data_connections
                    and len(parsed_live_data_connections) == 1
                ):
                    inferred_component_id = parsed_live_data_connections[0].component_id
                for item in live_data_checks:
                    payload = (
                        item.model_dump(mode="python")
                        if isinstance(item, BaseModel)
                        else item
                    )
                    if not isinstance(payload, dict):
                        contract_violations.append(
                            "Invalid live_data_checks item: expected an object with component_id and validation flags."
                        )
                        continue

                    raw_component_id = (
                        str(payload.get("component_id") or "").strip()
                        or str(payload.get("source_tool_config_id") or "").strip()
                        or str(payload.get("tool_config_id") or "").strip()
                        or str(payload.get("id") or "").strip()
                    )
                    if not raw_component_id and inferred_component_id:
                        raw_component_id = inferred_component_id
                    if not raw_component_id:
                        contract_violations.append(
                            "Missing component_id in live_data_checks. Provide component_id explicitly, "
                            "or include exactly one live_data_connections entry so it can be inferred."
                        )
                        continue

                    payload["component_id"] = raw_component_id
                    parsed_live_data_checks.append(
                        UserSpaceLiveDataCheck.model_validate(payload)
                    )

            workspace = await userspace_service.get_workspace(workspace_id, user_id)
            ws_sqlite_include = workspace.sqlite_persistence_mode == "include"
            live_data_contract_context = await _build_live_data_contract_context(
                path,
                workspace=workspace,
            )
            allowed_component_ids = set(workspace.selected_tool_ids)
            if parsed_live_data_connections:
                for connection in parsed_live_data_connections:
                    if connection.component_id not in allowed_component_ids:
                        hard_errors.append(
                            "Invalid live_data_connections component_id: "
                            f"{connection.component_id}. It must match a tool selected for this workspace."
                        )

            if parsed_live_data_checks:
                for check in parsed_live_data_checks:
                    if check.component_id not in allowed_component_ids:
                        hard_errors.append(
                            "Invalid live_data_checks component_id: "
                            f"{check.component_id}. It must match a tool selected for this workspace."
                        )
                    if (
                        not check.connection_check_passed
                        or not check.transformation_check_passed
                    ):
                        contract_violations.append(
                            "live_data_checks must indicate successful connection and transformation for "
                            f"component_id={check.component_id}."
                        )

            # Live data contract only applies to the dashboard entry
            # module (dashboard/main.ts), not helper components under
            # dashboard/ or arbitrary .ts files elsewhere.
            is_dashboard_entry = normalized_path == "dashboard/main.ts"

            # Auto-require live data contract when workspace has selected
            # tools and the write targets the dashboard entry module.
            # Helper components under dashboard/ receive data as
            # parameters and do not need their own live data wiring.
            workspace_has_tools = bool(workspace.selected_tool_ids)
            effective_live_data_requested = live_data_requested or (
                workspace_has_tools and is_dashboard_entry
            )

            requires_live_data_contract = (
                effective_live_data_requested and is_dashboard_entry
            )

            # Detect hardcoded mock/sample data naming patterns.
            # Dashboard entry: hard policy violation (reject).
            # Other entrypoint files: soft contract violation (persist with warning).
            if is_dashboard_entry:
                mock_patterns = find_hardcoded_data_patterns(content)
                if mock_patterns:
                    hard_errors.append(
                        "Hardcoded data patterns detected: "
                        + ", ".join(mock_patterns)
                        + ". All dashboard data must be fetched at runtime via "
                        "context.components[componentId].execute()."
                    )
            elif workspace_has_tools and normalized_path.endswith(
                _HARDCODED_DATA_SOURCE_EXTENSIONS
            ):
                # Check if target is the runtime entrypoint
                resolved_ep: str | None = None
                try:
                    ep_file = await userspace_service.get_workspace_file(
                        workspace_id, ".ragtime/runtime-entrypoint.json", user_id
                    )
                    ep_cfg = json.loads(ep_file.content or "{}")
                    if isinstance(ep_cfg, dict):
                        resolved_ep = _resolve_entrypoint_source_file(ep_cfg)
                except Exception:
                    pass
                is_runtime_entrypoint = (
                    resolved_ep is not None and normalized_path == resolved_ep.lower()
                )
                if is_runtime_entrypoint:
                    ep_mock_patterns = find_hardcoded_data_patterns(content)
                    if ep_mock_patterns:
                        contract_violations.append(
                            "Hardcoded data patterns detected in entrypoint file: "
                            + ", ".join(ep_mock_patterns)
                            + ". When workspace has selected tools, entrypoint data "
                            "should be fetched from live sources, not embedded in source code."
                        )

            # No-tools conflict: warn when dashboard entry targets a
            # workspace without any selected tools for live data.
            if is_dashboard_entry and not workspace_has_tools:
                warnings.append(
                    "NO LIVE DATA TOOLS AVAILABLE: This workspace has no "
                    "selected tools. Dashboard data cannot be fetched from "
                    "live sources. Inform the user that tool configuration "
                    "is required before live data can be rendered."
                )

            if requires_live_data_contract and not parsed_live_data_connections:
                contract_violations.append(
                    "Missing required live_data_connections contract metadata for dashboard/main.ts. "
                    "Provide at least one connection with component_kind=tool_config, component_id, and request, or set live_data_requested=false for scaffolding."
                )

            if requires_live_data_contract:
                if not parsed_live_data_checks:
                    contract_violations.append(
                        "Missing required live_data_checks verification metadata for this module source write. "
                        "Provide checks with successful connection and transformation for each component_id."
                    )
                else:
                    connection_ids = {
                        connection.component_id
                        for connection in (parsed_live_data_connections or [])
                    }
                    verified_ids = {
                        check.component_id
                        for check in parsed_live_data_checks
                        if check.connection_check_passed
                        and check.transformation_check_passed
                    }
                    missing_ids = sorted(connection_ids - verified_ids)
                    if missing_ids:
                        contract_violations.append(
                            "Missing successful live_data_checks verification for component_id(s): "
                            + ", ".join(missing_ids)
                        )

            if requires_live_data_contract and parsed_live_data_connections:
                declared_connection_ids = {
                    connection.component_id
                    for connection in parsed_live_data_connections
                }
                missing_execution_proofs = userspace_service.verify_execution_proofs(
                    workspace_id,
                    declared_connection_ids,
                )
                if missing_execution_proofs:
                    proof_message = (
                        "No server-verified execution proof for component_id(s): "
                        + ", ".join(missing_execution_proofs)
                        + ". Execute a successful query via the selected workspace data tool before persisting live data connections."
                    )
                    self._record_userspace_failure(
                        workspace_id,
                        failure_class="execution_proof_missing",
                        summary=proof_message,
                        tool_name="upsert_userspace_file",
                    )
                    return _render_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="execution_proof_missing",
                        next_best_tool="assay_userspace_code",
                        path=path,
                        error=proof_message,
                        action_required=(
                            "Run the selected workspace query tool successfully for each declared component_id, then retry upsert_userspace_file."
                        ),
                        include_sqlite_hint=ws_sqlite_include,
                        diagnostics={
                            "missing_component_ids": missing_execution_proofs,
                        },
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                    )

            # AST-based structural validation: verify the module code
            # contains context.components[id].execute() call patterns.
            # This is deterministic and cannot be satisfied by fabricating
            # metadata alone -- the source code must structurally call
            # the live data execution API.
            if requires_live_data_contract and is_dashboard_entry:
                declared_ids = (
                    {c.component_id for c in parsed_live_data_connections}
                    if parsed_live_data_connections
                    else set()
                )
                binding = await validate_live_data_binding(
                    content,
                    path,
                    declared_component_ids=declared_ids or None,
                )
                if binding.get("validator_available", False):
                    has_execute = binding.get("has_execute_calls", False)
                    has_imports = binding.get("has_local_imports", False)
                    has_access = binding.get("has_context_components_access", False)
                    is_entry = normalized_path == "dashboard/main.ts"

                    # Entry file may delegate data fetching to imported
                    # sub-modules, so we accept local imports as evidence
                    # of deferred binding.  Non-entry files that declare
                    # live connections must call execute() directly.
                    if not has_execute and not (is_entry and has_imports):
                        contract_violations.append(
                            "Live data binding not found in module source. "
                            "Dashboard modules must call "
                            "context.components[componentId].execute() to "
                            "fetch data at runtime. Hardcoded/static data "
                            "is not permitted when workspace tools are "
                            "available."
                        )
                    elif not has_access and not has_imports:
                        contract_violations.append(
                            "Module source does not access "
                            "context.components anywhere. Dashboard "
                            "modules with live_data_connections must wire "
                            "data through "
                            "context.components[componentId].execute()."
                        )

                    ast_missing = binding.get("missing_component_ids", [])
                    if ast_missing and has_execute:
                        warnings.append(
                            "Declared component_ids not found in AST "
                            "execute() calls: "
                            + ", ".join(ast_missing)
                            + ". Verify these connections are used in "
                            "the code."
                        )

            if is_userspace_theme_audit_path(lower_path):
                hard_coded_hex = find_hard_coded_hex_colors(content)
                if hard_coded_hex:
                    sample = ", ".join(hard_coded_hex[:8])
                    warnings.append(
                        "Detected hard-coded hex color literals: "
                        f"{sample}. Prefer theme CSS tokens (e.g., var(--color-text-primary), var(--color-surface), var(--color-border))."
                    )

            typecheck: dict[str, Any] | None = None
            if is_userspace_typescript_path(lower_path):
                typecheck = await validate_userspace_typescript_content(content, path)
                if not typecheck.get("ok", False):
                    contract_errors = typecheck.get("contract_errors") or []
                    if contract_errors:
                        for err in contract_errors:
                            violation = "User Space runtime contract violation: " + str(
                                err
                            )
                            if violation not in allowed_violations:
                                allowed_violations.append(violation)
                    diagnostics = typecheck.get("errors") or []
                    if diagnostics:
                        warnings.append(
                            "TypeScript/runtime diagnostics detected. Run validate_userspace_code and fix reported errors before finalizing."
                        )

            if hard_errors:
                all_runtime_contract = all(
                    err.startswith("User Space runtime contract violation:")
                    for err in hard_errors
                )
                if all_runtime_contract:
                    detail = "USER SPACE RUNTIME CONTRACT VIOLATION -- " + " | ".join(
                        hard_errors
                    )
                else:
                    detail = "LIVE DATA POLICY VIOLATION -- " + " | ".join(hard_errors)
                if warnings:
                    detail += " [Warnings: " + "; ".join(warnings) + "]"
                failure_class = self._classify_userspace_failure(detail, hard_errors)
                self._record_userspace_failure(
                    workspace_id,
                    failure_class=failure_class,
                    summary=detail,
                    tool_name="upsert_userspace_file",
                )
                rejected_payload = _build_userspace_tool_payload(
                    tool_name="upsert_userspace_file",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class=failure_class,
                    path=path,
                    error=detail,
                    action_required=(
                        "Fix the hard policy violations listed above, then retry upsert_userspace_file."
                    ),
                    include_sqlite_hint=ws_sqlite_include,
                    diagnostics={"policy_violations": hard_errors},
                    contract_violations=contract_violations or None,
                    warnings=warnings or None,
                    live_data_contract=live_data_contract_context,
                    allowed_violations=allowed_violations or None,
                )
                if typecheck is not None:
                    rejected_payload["typescript_validation"] = typecheck
                return json.dumps(rejected_payload, indent=2)

            # Persist the file.  When contract_violations exist (but no
            # hard_errors), still write the file and tell the service
            # layer to skip live-data enforcement so the write succeeds.
            # The response will flag the violations so the agent can fix
            # them with a follow-up patch instead of regenerating the
            # entire content blob.
            skip_live_data = bool(contract_violations)

            try:
                result = await userspace_service.upsert_workspace_file(
                    workspace_id,
                    path,
                    UpsertWorkspaceFileRequest(
                        content=content,
                        artifact_type=artifact_type,
                        live_data_requested=live_data_requested,
                        live_data_connections=parsed_live_data_connections,
                        live_data_checks=parsed_live_data_checks,
                    ),
                    user_id,
                    skip_live_data_enforcement=skip_live_data,
                )
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", None)
                detail_text = str(getattr(exc, "detail", exc))
                lower_detail_text = detail_text.lower()
                if status_code == 400 and "invalid file path" in lower_detail_text:
                    policy_response_payload = _build_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="path_invalid",
                        next_best_tool="list_userspace_files",
                        path=path,
                        error=(
                            f"File not found: {path}. The requested path does not exist or is not accessible in this workspace."
                        ),
                        action_required=(
                            "Use list_userspace_files to choose an existing path, or provide a valid relative file path under the workspace files root."
                        ),
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                    )
                    if warnings:
                        policy_response_payload["warnings"] = warnings
                    if typecheck is not None:
                        policy_response_payload["typescript_validation"] = typecheck
                    return json.dumps(policy_response_payload, indent=2)
                if status_code == 400 and any(
                    marker in lower_detail_text
                    for marker in (
                        "invalid live_data_connections component_id",
                        "invalid live_data_checks component_id",
                    )
                ):
                    policy_response_payload = _build_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="invalid_component_id",
                        next_best_tool="assay_userspace_code",
                        path=path,
                        error=detail_text,
                        action_required=(
                            "Use only component_ids from tools selected for this workspace."
                        ),
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                    )
                    if warnings:
                        policy_response_payload["warnings"] = warnings
                    if typecheck is not None:
                        policy_response_payload["typescript_validation"] = typecheck
                    return json.dumps(policy_response_payload, indent=2)
                if status_code == 400 and detail_text.startswith(
                    "Entry-point wiring required:"
                ):
                    entrypoint_response_payload = _build_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="entrypoint_wiring_required",
                        next_best_tool="upsert_userspace_file",
                        path=path,
                        error=detail_text,
                        action_required=(
                            "Upsert dashboard/main.ts so it imports or composes the module, then retry upsert_userspace_file for the non-entry dashboard file."
                        ),
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                    )
                    if warnings:
                        entrypoint_response_payload["warnings"] = warnings
                    if typecheck is not None:
                        entrypoint_response_payload["typescript_validation"] = typecheck
                    return json.dumps(entrypoint_response_payload, indent=2)
                if (
                    status_code == 400
                    and "no server-verified execution proof" in lower_detail_text
                ):
                    execution_proof_response_payload = _build_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class="execution_proof_missing",
                        next_best_tool="assay_userspace_code",
                        path=path,
                        error=detail_text,
                        action_required=(
                            "Execute the declared component queries via the workspace tool or execute-component endpoint so the server can verify successful execution, then retry upsert_userspace_file."
                        ),
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                        contract_violations=contract_violations or None,
                    )
                    if warnings:
                        execution_proof_response_payload["warnings"] = warnings
                    if typecheck is not None:
                        execution_proof_response_payload["typescript_validation"] = (
                            typecheck
                        )
                    return json.dumps(execution_proof_response_payload, indent=2)
                if status_code == 400:
                    failure_class = self._classify_userspace_failure(detail_text)
                    policy_response_payload = _build_userspace_tool_payload(
                        tool_name="upsert_userspace_file",
                        status="rejected_not_persisted",
                        rejected=True,
                        persisted=False,
                        retryable=True,
                        failure_class=failure_class,
                        path=path,
                        error=detail_text,
                        action_required=(
                            "Resolve the validation error in this response and retry upsert_userspace_file."
                        ),
                        live_data_contract=live_data_contract_context,
                        warnings=warnings or None,
                        contract_violations=contract_violations or None,
                    )
                    if warnings:
                        policy_response_payload["warnings"] = warnings
                    if typecheck is not None:
                        policy_response_payload["typescript_validation"] = typecheck
                    return json.dumps(policy_response_payload, indent=2)
                raise

            write_signature = hashlib.sha256(
                f"{path}\0{content}".encode("utf-8")
            ).hexdigest()[:16]
            success_response_payload = _build_userspace_tool_payload(
                tool_name="upsert_userspace_file",
                status=(
                    "persisted_with_violations" if contract_violations else "persisted"
                ),
                path=path,
                message=(
                    "File persisted."
                    if not contract_violations
                    else "File persisted, but follow-up fixes are required before the build loop is complete."
                ),
                persisted=True,
                persisted_with_violations=bool(contract_violations),
                retryable=True,
                failure_class=(
                    self._classify_userspace_failure(contract_violations)
                    if contract_violations
                    else "none"
                ),
                next_best_tool=(
                    "patch_userspace_file"
                    if contract_violations
                    else "validate_userspace_code"
                ),
                action_required=(
                    "Run validate_userspace_code on every changed source file."
                    if not contract_violations
                    else "Use patch_userspace_file to fix the listed violations, then run validate_userspace_code."
                ),
                include_sqlite_hint=ws_sqlite_include and bool(contract_violations),
                contract_violations=contract_violations or None,
                warnings=warnings or None,
                file=result.model_dump(mode="json"),
                live_data_contract=live_data_contract_context,
                write_signature=write_signature,
            )
            if warnings:
                success_response_payload["warnings"] = warnings
            if allowed_violations:
                success_response_payload["allowed_violations"] = allowed_violations
                success_response_payload["created_with_violations"] = True
            if typecheck is not None:
                success_response_payload["typescript_validation"] = typecheck
            await userspace_runtime_service.bump_workspace_generation(
                workspace_id,
                "file_upsert",
                payload={"path": str(getattr(result, "path", path))},
            )
            return json.dumps(success_response_payload, indent=2)

        async def create_userspace_snapshot(
            message: str = "AI checkpoint",
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
            snapshot = await userspace_service.create_snapshot(
                workspace_id,
                user_id,
                message.strip() or "AI checkpoint",
            )
            self._mark_userspace_failures_resolved(
                workspace_id,
                "Snapshot created after successful validation loop.",
            )
            await userspace_runtime_service.bump_workspace_generation(
                workspace_id, "snapshot"
            )
            return _render_userspace_tool_payload(
                tool_name="create_userspace_snapshot",
                status="persisted",
                persisted=True,
                retryable=True,
                failure_class="none",
                next_best_tool="assay_userspace_code",
                message="Snapshot created.",
                snapshot=snapshot.model_dump(mode="json"),
            )

        async def validate_userspace_code(
            path: str = "dashboard/main.ts",
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
            normalized_start_path = (
                (path or "dashboard/main.ts").strip().replace("\\", "/").lstrip("/")
            )
            live_data_contract_context = await _build_live_data_contract_context(
                normalized_start_path
            )

            file_cache: dict[str, Any] = {}

            async def get_file(relative_path: str) -> Any | None:
                normalized = (
                    (relative_path or "").strip().replace("\\", "/").lstrip("/")
                )
                if not normalized:
                    return None
                if normalized in file_cache:
                    return file_cache[normalized]
                try:
                    file_data = await userspace_service.get_workspace_file(
                        workspace_id, normalized, user_id
                    )
                except HTTPException as exc:
                    if getattr(exc, "status_code", None) == 404:
                        return None
                    raise
                file_cache[normalized] = file_data
                return file_data

            async def resolve_local_import(
                current_path: str,
                specifier: str,
            ) -> str | None:
                spec = (specifier or "").strip()
                if not spec or not spec.startswith(("./", "../", "/")):
                    return None

                if spec.startswith("/"):
                    base = PurePosixPath(spec.lstrip("/"))
                else:
                    base = PurePosixPath(current_path).parent / PurePosixPath(spec)

                base_str = base.as_posix().lstrip("/")
                candidates: list[str] = []
                if is_userspace_module_source_path(base_str):
                    candidates.append(base_str)
                else:
                    for ext in USERSPACE_MODULE_SOURCE_EXTENSIONS:
                        candidates.append(f"{base_str}{ext}")
                    for ext in USERSPACE_MODULE_SOURCE_EXTENSIONS:
                        candidates.append(f"{base_str}/index{ext}")

                seen_candidates: set[str] = set()
                for candidate in candidates:
                    normalized_candidate = candidate.replace("\\", "/").lstrip("/")
                    if (
                        not normalized_candidate
                        or normalized_candidate in seen_candidates
                    ):
                        continue
                    seen_candidates.add(normalized_candidate)
                    if await get_file(normalized_candidate):
                        return normalized_candidate
                return None

            visited: set[str] = set()
            to_visit: list[str] = [normalized_start_path]
            file_results: dict[str, dict[str, Any]] = {}
            aggregate_errors: list[str] = []
            aggregate_contract_errors: list[str] = []
            aggregate_runtime_errors: list[str] = []
            aggregate_runtime_warnings: list[str] = []
            runtime_probe: dict[str, Any] = {
                "attempted": False,
                "devserver_running": None,
                "preview_status_code": None,
                "upstream_url": None,
            }

            def add_runtime_warning(message: str) -> None:
                text = (message or "").strip()
                if text and text not in aggregate_runtime_warnings:
                    aggregate_runtime_warnings.append(text)

            def add_runtime_error(message: str) -> None:
                text = (message or "").strip()
                if not text:
                    return
                if text not in aggregate_runtime_errors:
                    aggregate_runtime_errors.append(text)
                if text not in aggregate_errors:
                    aggregate_errors.append(text)

            while to_visit:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                file_data = await get_file(current)
                if not file_data:
                    missing_message = f"{current}: Local module import could not be resolved in workspace files."
                    if missing_message not in aggregate_errors:
                        aggregate_errors.append(missing_message)
                    continue

                result = await validate_userspace_typescript_content(
                    file_data.content,
                    current,
                )
                file_results[current] = result

                file_errors = result.get("errors") or []
                if isinstance(file_errors, list):
                    for err in file_errors:
                        if err not in aggregate_errors:
                            aggregate_errors.append(err)

                file_contract_errors = result.get("contract_errors") or []
                if isinstance(file_contract_errors, list):
                    for err in file_contract_errors:
                        if err not in aggregate_contract_errors:
                            aggregate_contract_errors.append(err)

                file_runtime_errors = result.get("runtime_errors") or []
                if isinstance(file_runtime_errors, list):
                    for err in file_runtime_errors:
                        if err not in aggregate_runtime_errors:
                            aggregate_runtime_errors.append(err)

                imports = _IMPORT_SPECIFIER_PATTERN.findall(file_data.content or "")
                for specifier in imports:
                    if not specifier.startswith(("./", "../", "/")):
                        continue
                    resolved = await resolve_local_import(current, specifier)
                    if resolved:
                        if resolved not in visited and resolved not in to_visit:
                            to_visit.append(resolved)
                    else:
                        unresolved_message = (
                            f"{current}: Unable to resolve local import '{specifier}'."
                        )
                        if unresolved_message not in aggregate_errors:
                            aggregate_errors.append(unresolved_message)

            should_check_runnable_entrypoint = normalized_start_path.startswith(
                "dashboard/"
            )
            if should_check_runnable_entrypoint:
                runnable_entrypoint_error = (
                    "No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json "
                    "with a command/cwd/framework."
                )
                runtime_config_file = await get_file(".ragtime/runtime-entrypoint.json")
                runtime_config_command_present = False
                if runtime_config_file is not None:
                    runtime_config_raw = runtime_config_file.content or ""
                    try:
                        runtime_config_payload = json.loads(runtime_config_raw)
                    except json.JSONDecodeError as exc:
                        add_runtime_warning(
                            "Runtime preflight: .ragtime/runtime-entrypoint.json is invalid JSON "
                            f"({exc.msg} at line {exc.lineno}). Runtime launch will fail until this file is fixed."
                        )
                    else:
                        if isinstance(runtime_config_payload, dict):
                            runtime_command = str(
                                runtime_config_payload.get("command") or ""
                            ).strip()
                            runtime_config_command_present = bool(runtime_command)
                            if runtime_config_command_present:
                                if shutil.which("sh") is None:
                                    add_runtime_warning(
                                        "Runtime preflight: shell executable 'sh' is unavailable, so .ragtime/runtime-entrypoint.json command launch may fail."
                                    )
                                try:
                                    command_tokens = shlex.split(runtime_command)
                                except ValueError as exc:
                                    add_runtime_warning(
                                        "Runtime preflight: .ragtime/runtime-entrypoint.json command parsing failed "
                                        f"({exc}). Launch may fail at runtime."
                                    )
                                else:
                                    if command_tokens:
                                        primary_command = command_tokens[0]
                                        if (
                                            "/" not in primary_command
                                            and shutil.which(primary_command) is None
                                        ):
                                            add_runtime_warning(
                                                "Runtime preflight: .ragtime/runtime-entrypoint.json command references "
                                                f"'{primary_command}', which is not found in PATH."
                                            )
                                    else:
                                        add_runtime_warning(
                                            "Runtime preflight: .ragtime/runtime-entrypoint.json command is empty after parsing."
                                        )
                        else:
                            add_runtime_warning(
                                "Runtime preflight: .ragtime/runtime-entrypoint.json should be a JSON object with command/cwd/framework keys."
                            )

                if not runtime_config_command_present:
                    add_runtime_error(runnable_entrypoint_error)

                # Check framework dependencies are declared in the
                # appropriate manifest (requirements.txt / package.json).
                if runtime_config_command_present:
                    entrypoint_framework = ""
                    if isinstance(runtime_config_payload, dict):  # type: ignore[possibly-undefined]
                        entrypoint_framework = (
                            str(runtime_config_payload.get("framework") or "")
                            .strip()
                            .lower()
                        )
                    dep_spec = FRAMEWORK_REQUIRED_PACKAGES.get(entrypoint_framework)
                    if dep_spec is not None:
                        manifest_name, required_pkgs = dep_spec
                        manifest_file = await get_file(manifest_name)
                        manifest_text = (
                            (manifest_file.content if manifest_file else None) or ""
                        ).lower()
                        missing = [
                            pkg
                            for pkg in required_pkgs
                            if pkg.lower() not in manifest_text
                        ]
                        if missing:
                            missing_list = ", ".join(missing)
                            if manifest_file is None:
                                add_runtime_error(
                                    f"Runtime preflight: entrypoint framework '{entrypoint_framework}' "
                                    f"requires [{missing_list}] but {manifest_name} does not exist. "
                                    f"Create {manifest_name} listing these dependencies so the "
                                    "runtime bootstrap can install them."
                                )
                            else:
                                add_runtime_error(
                                    f"Runtime preflight: entrypoint framework '{entrypoint_framework}' "
                                    f"requires [{missing_list}] but they are not listed in {manifest_name}. "
                                    f"Add them to {manifest_name} so the runtime bootstrap can install them."
                                )

            strict_frontend_candidate = is_userspace_strict_frontend_path(
                normalized_start_path
            )
            should_probe_runtime = (
                should_check_runnable_entrypoint or strict_frontend_candidate
            )

            if should_probe_runtime:
                runtime_probe["attempted"] = True

                async def _run_runtime_probe() -> None:
                    status = await userspace_runtime_service.get_devserver_status(
                        workspace_id,
                        user_id,
                    )
                    runtime_probe["devserver_running"] = bool(status.devserver_running)
                    if not status.devserver_running:
                        state = status.session_state or "unknown"
                        last_error = status.last_error or "unknown"
                        add_runtime_error(
                            "Runtime strict validation failed: devserver is not running "
                            f"(state={state}, last_error={last_error})."
                        )

                    upstream_url = await userspace_runtime_service.build_workspace_preview_upstream_url(
                        workspace_id,
                        user_id,
                        "",
                    )
                    runtime_probe["upstream_url"] = upstream_url

                    probe_headers: dict[str, str] = {}
                    worker_token = str(
                        getattr(settings, "userspace_runtime_worker_auth_token", "")
                    ).strip()
                    manager_token = str(
                        getattr(settings, "userspace_runtime_manager_auth_token", "")
                    ).strip()

                    if "/worker/" in upstream_url and worker_token:
                        probe_headers["Authorization"] = f"Bearer {worker_token}"
                    elif manager_token:
                        probe_headers["Authorization"] = f"Bearer {manager_token}"
                    elif worker_token:
                        probe_headers["Authorization"] = f"Bearer {worker_token}"

                    probe_timeout = httpx.Timeout(
                        connect=2.0, read=12.0, write=8.0, pool=4.0
                    )
                    async with httpx.AsyncClient(
                        timeout=probe_timeout, follow_redirects=False
                    ) as client:
                        response = await client.get(
                            upstream_url,
                            headers=probe_headers or None,
                        )

                    runtime_probe["preview_status_code"] = response.status_code
                    if response.status_code >= 400:
                        body_preview = (response.text or "")[:200].strip()
                        detail_suffix = f" Body: {body_preview}" if body_preview else ""
                        add_runtime_error(
                            "Runtime strict validation failed: preview upstream returned "
                            f"HTTP {response.status_code}.{detail_suffix}"
                        )
                    elif response.status_code == 200:
                        body_text = (response.text or "")[:2000]
                        if "<title>Directory listing for" in body_text:
                            runtime_probe["directory_listing_detected"] = True
                            add_runtime_error(
                                "Runtime validation failed: preview is returning a directory "
                                "listing instead of rendering the app. Update the runtime "
                                "entrypoint to a framework that builds and serves the code."
                            )
                        else:
                            # Run a Playwright content probe to detect white
                            # screens (JS fails to render any visible content).
                            # When the workspace has selected tools, inject a
                            # mock window.__ragtime_context so data-dependent
                            # rendering code (charts, tables) actually executes
                            # during the probe — exposing runtime DOM errors
                            # that would otherwise be hidden behind a "loading"
                            # placeholder that never resolves.
                            workspace_has_tools = bool(
                                live_data_contract_context.get(
                                    "workspace_has_selected_tools"
                                )
                            )
                            try:
                                probe_result = await userspace_runtime_service.probe_workspace_content(
                                    workspace_id,
                                    user_id,
                                    path="",
                                    timeout_ms=15000,
                                    wait_after_load_ms=2000,
                                    inject_mock_context=workspace_has_tools,
                                )
                                runtime_probe["content_probe"] = probe_result
                                body_text_length = int(
                                    probe_result.get("body_text_length", 0)
                                )
                                console_errors = normalize_runtime_console_errors(
                                    probe_result.get("console_errors", [])
                                )
                                has_error_indicator = bool(
                                    probe_result.get("has_error_indicator", False)
                                )
                                runtime_probe["console_error_count"] = len(
                                    console_errors
                                )
                                if console_errors:
                                    runtime_probe["console_errors"] = console_errors[:5]

                                serious_console_errors = [
                                    error
                                    for error in console_errors
                                    if is_serious_runtime_console_error(error)
                                ]
                                if serious_console_errors:
                                    sample_error = serious_console_errors[0][:220]
                                    guidance = explain_runtime_console_error(
                                        serious_console_errors[0]
                                    )
                                    detail_suffix = f" {guidance}" if guidance else ""
                                    add_runtime_error(
                                        "Runtime validation failed: browser console "
                                        "reported a JavaScript exception during preview. "
                                        f"Example: {sample_error}.{detail_suffix}"
                                    )

                                if body_text_length == 0:
                                    runtime_probe["blank_screen_detected"] = True
                                    error_detail = ""
                                    if console_errors:
                                        first_error = str(console_errors[0])[:150]
                                        error_detail = (
                                            f" Browser console error: {first_error}"
                                        )
                                    add_runtime_error(
                                        "Runtime validation failed: preview renders a blank page "
                                        "with no visible content. The app's JavaScript is likely "
                                        "failing to execute. Check the entrypoint, build output, "
                                        "and import paths."
                                        f"{error_detail}"
                                    )
                                elif has_error_indicator:
                                    runtime_probe["error_page_detected"] = True
                                    text_preview = str(
                                        probe_result.get("body_text_preview", "")
                                    )[:150]
                                    add_runtime_error(
                                        "Runtime validation failed: preview is "
                                        "rendering an error page. "
                                        f"Visible text: {text_preview}"
                                    )
                            except Exception:
                                # Content probe is best-effort; don't fail
                                # validation if Playwright is unavailable.
                                pass

                try:
                    await asyncio.wait_for(_run_runtime_probe(), timeout=55)
                except TimeoutError:
                    add_runtime_error(
                        "Runtime strict validation failed: runtime probe timed out after 55s."
                    )
                except HTTPException as exc:
                    detail_text = str(getattr(exc, "detail", exc)).strip() or str(exc)
                    add_runtime_error(
                        "Runtime strict validation failed: runtime session/preview setup failed. "
                        f"{detail_text}"
                    )
                except Exception as exc:
                    add_runtime_error(
                        "Runtime strict validation failed: runtime probe request failed. "
                        f"{exc}"
                    )

            # Validate index.html bootstrap pattern when present
            if should_check_runnable_entrypoint:
                index_html_file = await get_file("index.html")
                if index_html_file is not None:
                    html_content = getattr(index_html_file, "content", "") or ""
                    if "window.render" in html_content:
                        add_runtime_error(
                            "Bootstrap mismatch: index.html references window.render, which is "
                            "inaccessible when esbuild bundles in IIFE format (the default). "
                            'Use <script type="module"> with '
                            'import {{ render }} from "./dist/main.js" instead, and ensure '
                            "the esbuild command includes --format=esm."
                        )

            if aggregate_runtime_warnings:
                for warning in aggregate_runtime_warnings:
                    add_runtime_error(
                        f"Runtime strict validation warning treated as error: {warning}"
                    )

            # ── Hardcoded data scan across all workspace source files ──
            # Only runs when the workspace has selected tools.  Scans every
            # source file (py/ts/js/html) for mock/sample/static data patterns
            # and reports per-file violations as warnings.
            hardcoded_data_violations: dict[str, list[str]] = {}
            try:
                ws = await userspace_service.get_workspace(workspace_id, user_id)
                ws_has_tools = bool(ws.selected_tool_ids)
            except Exception:
                ws_has_tools = False

            if ws_has_tools:
                all_files = await userspace_service.list_workspace_files(
                    workspace_id, user_id
                )
                for ws_file in all_files:
                    fpath = (ws_file.path or "").strip()
                    if not fpath.lower().endswith(_HARDCODED_DATA_SOURCE_EXTENSIONS):
                        continue
                    # Skip .ragtime/ internal files
                    if fpath.startswith(".ragtime/"):
                        continue
                    # Read file content (use cache if available)
                    file_obj = await get_file(fpath)
                    if file_obj is None:
                        continue
                    file_content = getattr(file_obj, "content", None) or ""
                    if not file_content:
                        continue
                    patterns_found = find_hardcoded_data_patterns(file_content)
                    if patterns_found:
                        hardcoded_data_violations[fpath] = patterns_found

            overall_ok = (
                bool(file_results)
                and all(
                    bool((res or {}).get("ok", False)) for res in file_results.values()
                )
                and not aggregate_errors
            )

            diagnostic_groups = self._group_userspace_validation_diagnostics(
                errors=aggregate_errors,
                contract_errors=aggregate_contract_errors,
                runtime_errors=aggregate_runtime_errors,
                runtime_warnings=aggregate_runtime_warnings,
            )
            primary_failure_class = (
                "none"
                if overall_ok
                else self._classify_userspace_failure(
                    aggregate_runtime_errors,
                    aggregate_contract_errors,
                    aggregate_errors,
                )
            )

            root_artifact_type = None
            root_file = file_cache.get(normalized_start_path)
            if root_file is not None:
                root_artifact_type = getattr(root_file, "artifact_type", None)

            result = {
                "ok": overall_ok,
                "validator_available": (
                    all(
                        bool((res or {}).get("validator_available", False))
                        for res in file_results.values()
                    )
                    if file_results
                    else False
                ),
                "error_count": len(aggregate_errors),
                "errors": aggregate_errors,
                "runtime_errors": aggregate_runtime_errors,
                "runtime_error_count": len(aggregate_runtime_errors),
                "runtime_warnings": aggregate_runtime_warnings,
                "runtime_warning_count": len(aggregate_runtime_warnings),
                "runtime_probe": runtime_probe,
                "contract_errors": aggregate_contract_errors,
                "contract_error_count": len(aggregate_contract_errors),
                "validated_files": sorted(file_results.keys()),
                "file_results": file_results,
            }
            if hardcoded_data_violations:
                result["hardcoded_data_violations"] = hardcoded_data_violations
                result["hardcoded_data_violation_count"] = sum(
                    len(v) for v in hardcoded_data_violations.values()
                )
            result["diagnostics_by_category"] = diagnostic_groups

            if overall_ok:
                self._mark_userspace_failures_resolved(
                    workspace_id,
                    "Validation passed with no remaining errors.",
                )
            else:
                summary = (
                    aggregate_errors[0] if aggregate_errors else primary_failure_class
                )
                self._record_userspace_failure(
                    workspace_id,
                    failure_class=primary_failure_class,
                    summary=str(summary),
                    tool_name="validate_userspace_code",
                )

            response_payload = _build_userspace_tool_payload(
                tool_name="validate_userspace_code",
                status="validated" if overall_ok else "validation_failed",
                path=normalized_start_path,
                message=(
                    "Validation passed."
                    if overall_ok
                    else "Validation failed. Fix the reported diagnostics before finalizing."
                ),
                persisted=False,
                rejected=not overall_ok,
                retryable=True,
                failure_class=primary_failure_class,
                next_best_tool=(
                    "create_userspace_snapshot"
                    if overall_ok
                    else self._next_best_tool_for_failure(primary_failure_class)
                ),
                action_required=(
                    "Create a snapshot for this completed change loop."
                    if overall_ok
                    else "Fix the diagnostics in this response, then run validate_userspace_code again."
                ),
                diagnostics=diagnostic_groups,
                artifact_type=root_artifact_type,
                live_data_contract=live_data_contract_context,
                validation=result,
            )
            return json.dumps(response_payload, indent=2)

        async def capture_userspace_screenshot(
            path: str = "",
            width: int = 1440,
            height: int = 900,
            full_page: bool = True,
            timeout_ms: int = 25000,
            wait_for_selector: str = "body",
            capture_element: bool = False,
            clip_padding_px: int = 16,
            wait_after_load_ms: int = 1800,
            refresh_before_capture: bool = True,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            try:
                response_payload = (
                    await userspace_runtime_service.capture_workspace_screenshot(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        path=path,
                        width=width,
                        height=height,
                        full_page=full_page,
                        timeout_ms=timeout_ms,
                        wait_for_selector=wait_for_selector,
                        capture_element=capture_element,
                        clip_padding_px=clip_padding_px,
                        wait_after_load_ms=wait_after_load_ms,
                        refresh_before_capture=refresh_before_capture,
                    )
                )
            except HTTPException as exc:
                detail_text = str(getattr(exc, "detail", exc)).strip() or str(exc)
                return _render_userspace_tool_payload(
                    tool_name="capture_userspace_screenshot",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class=self._classify_userspace_failure(detail_text),
                    next_best_tool="run_terminal_command",
                    error=f"Runtime screenshot capture failed: {detail_text}",
                    action_required="Inspect the runtime state and retry capture_userspace_screenshot after the preview is reachable.",
                )
            except Exception as exc:
                return _render_userspace_tool_payload(
                    tool_name="capture_userspace_screenshot",
                    status="rejected_not_persisted",
                    rejected=True,
                    persisted=False,
                    retryable=True,
                    failure_class="runtime_capture_failed",
                    next_best_tool="run_terminal_command",
                    error=f"Screenshot capture failed unexpectedly: {exc}",
                    action_required="Inspect the runtime state and retry capture_userspace_screenshot.",
                )

            screenshot_path = str(response_payload.get("screenshot_path") or "").strip()
            if screenshot_path:
                screenshot_name = Path(screenshot_path).name
                if screenshot_name:
                    image_url = (
                        f"/indexes/userspace/runtime/workspaces/{workspace_id}/"
                        f"screenshots/{quote(screenshot_name)}"
                    )
                    response_payload["preview_image_url"] = image_url
            return _render_userspace_tool_payload(
                tool_name="capture_userspace_screenshot",
                status="captured",
                message="Runtime screenshot captured.",
                persisted=False,
                retryable=True,
                failure_class="none",
                next_best_tool="patch_userspace_file",
                screenshot=response_payload,
            )

        async def run_terminal_command(
            command: str,
            timeout_seconds: int = 30,
            cwd: str = ".",
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
            command = (command or "").strip()
            if not command:
                raise ToolException("command is required and must not be empty.")
            cwd_value = (cwd or ".").strip() or "."
            try:
                result = await userspace_runtime_service.exec_workspace_command(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    command=command,
                    timeout_seconds=timeout_seconds,
                    cwd=cwd_value if cwd_value != "." else None,
                )
            except HTTPException as exc:
                detail_text = str(getattr(exc, "detail", exc)).strip() or str(exc)
                return json.dumps(
                    {
                        "tool": "run_terminal_command",
                        "status": "rejected_not_persisted",
                        "cwd": cwd_value,
                        "command": command,
                        "timeout_seconds": timeout_seconds,
                        "error": f"Terminal command failed: {detail_text}",
                        "failure_class": self._classify_userspace_failure(detail_text),
                        "action_required": (
                            "Inspect the command, cwd, and runtime state before retrying. "
                            "If you need source diagnostics, prefer validate_userspace_code over shell-based lint or typecheck commands."
                        ),
                    },
                    indent=2,
                )

            exit_code = int(result.get("exit_code", 0) or 0)
            timed_out = bool(result.get("timed_out", False))
            command_failed = timed_out or exit_code != 0
            terminal_payload: dict[str, Any] = {
                "tool": "run_terminal_command",
                "status": (
                    "command_timed_out"
                    if timed_out
                    else "command_failed" if command_failed else "completed"
                ),
                "cwd": cwd_value,
                "exit_code": exit_code,
            }
            if timed_out:
                terminal_payload["timed_out"] = True
            if bool(result.get("truncated", False)):
                terminal_payload["truncated"] = True

            stdout = str(result.get("stdout") or "")
            stderr = str(result.get("stderr") or "")
            if stdout:
                terminal_payload["stdout"] = stdout
            if stderr:
                terminal_payload["stderr"] = stderr

            if command_failed:
                terminal_payload.update(
                    {
                        "command": command,
                        "timeout_seconds": timeout_seconds,
                        "error": "Terminal command finished with an error.",
                        "failure_class": "runtime_command_failed",
                        "action_required": (
                            "Inspect stdout and stderr in this response, adjust the workspace or command, then retry. "
                            "Use validate_userspace_code when you need code diagnostics rather than shell output."
                        ),
                    }
                )

            return json.dumps(terminal_payload, indent=2)

        return [
            _create_userspace_tool(
                coroutine=assay_userspace_code,
                name="assay_userspace_code",
                description=(
                    "Perform a focused assay of existing workspace code and structure before editing. "
                    "Use this first in each implementation loop to understand current state and identify target files."
                ),
                args_schema=AssayUserSpaceCodeInput,
            ),
            _create_userspace_tool(
                coroutine=list_userspace_files,
                name="list_userspace_files",
                description=(
                    "List files in the active User Space workspace. Use this before creating or updating files."
                ),
                args_schema=ListUserSpaceFilesInput,
            ),
            _create_userspace_tool(
                coroutine=list_userspace_env_vars,
                name="list_userspace_env_vars",
                description=(
                    "List configured workspace environment variable keys and has_value status (never returns secret values). "
                    "Use this to understand which keys are already configured or still placeholders."
                ),
                args_schema=ListUserSpaceEnvVarsInput,
            ),
            _create_userspace_tool(
                coroutine=upsert_userspace_env_var,
                name="upsert_userspace_env_var",
                description=(
                    "Create or update a workspace environment variable key. "
                    "Provide value to set a secret, or omit value to create a placeholder key that users can populate in the Environment Variables modal."
                ),
                args_schema=UpsertUserSpaceEnvVarInput,
            ),
            _create_userspace_tool(
                coroutine=read_userspace_file,
                name="read_userspace_file",
                description=(
                    "Read a file from the active User Space workspace by relative path."
                ),
                args_schema=ReadUserSpaceFileInput,
            ),
            _create_userspace_tool(
                coroutine=delete_userspace_file,
                name="delete_userspace_file",
                description=(
                    "Delete a file from the active User Space workspace by relative path. "
                    "Use to remove stale/conflicting files that block runtime builds."
                ),
                args_schema=DeleteUserSpaceFileInput,
            ),
            _create_userspace_tool(
                coroutine=move_userspace_file,
                name="move_userspace_file",
                description=(
                    "Move or rename a file in the active User Space workspace using relative paths. "
                    "Use this to reorganize files without rewriting content."
                ),
                args_schema=MoveUserSpaceFileInput,
            ),
            _create_userspace_tool(
                coroutine=upsert_userspace_file,
                name="upsert_userspace_file",
                description=(
                    "Create or update a file in the active User Space workspace. "
                    "Use when you need to add a new file or replace most of an existing file. "
                    "Do not use for small surgical edits when patch_userspace_file is sufficient. "
                    "For interactive reports, write TypeScript modules (artifact_type=module_ts). "
                    "Dashboard module writes in workspaces with selected tools automatically "
                    "require live_data_connections, live_data_checks, AND structurally verified "
                    "context.components[componentId].execute() calls in the source code (AST-checked). "
                    "Responses always include live_data_contract guidance with selected_tool_ids and required wiring context. "
                    "Policy violations raise tool errors that must be resolved before retrying. "
                    "Output may include CSS/theme warnings that must be fixed in follow-up edits."
                ),
                args_schema=UpsertUserSpaceFileInput,
            ),
            _create_userspace_tool(
                coroutine=patch_userspace_file,
                name="patch_userspace_file",
                description=(
                    "Apply targeted sed-style in-place replacements to an existing User Space file. "
                    "Use for surgical edits to avoid re-rendering full file content. "
                    "Do not use to create new files or replace large files wholesale. "
                    "Supports ordered exact old/new replacements with per-op required flags. "
                    "For reliable matches, source old_text from read_userspace_file output (not shell-derived views)."
                ),
                args_schema=PatchUserSpaceFileInput,
            ),
            _create_userspace_tool(
                coroutine=create_userspace_snapshot,
                name="create_userspace_snapshot",
                description=(
                    "Create a git-based checkpoint snapshot in the active User Space workspace. "
                    "Use this automatically at each completed user-requested change loop."
                ),
                args_schema=CreateUserSpaceSnapshotInput,
            ),
            _create_userspace_tool(
                coroutine=validate_userspace_code,
                name="validate_userspace_code",
                description=(
                    "Validate a workspace code file and return TypeScript/runtime diagnostics with file/line details. "
                    "Always includes live_data_contract guidance (selected_tool_ids, entrypoint metadata, and required live-data wiring rules). "
                    "Use after every write loop and before snapshot creation. Do not skip this even when the file persisted successfully. "
                    "Use after edits and before creating snapshots."
                ),
                args_schema=ValidateUserSpaceCodeInput,
            ),
            _create_userspace_tool(
                coroutine=capture_userspace_screenshot,
                name="capture_userspace_screenshot",
                description=(
                    "Capture a rendered screenshot of the live User Space preview using Playwright. "
                    "Waits for load/refresh before capture to reduce race conditions and helps diagnose runtime errors "
                    "(blank screen, crashes, broken layout). Saves PNG files under ./.data/_tmp/{workspace_id}/."
                ),
                args_schema=CaptureUserSpaceScreenshotInput,
            ),
            _create_userspace_tool(
                coroutine=run_terminal_command,
                name="run_terminal_command",
                description=(
                    "Execute a shell command in the workspace runtime container terminal. "
                    "Use for running migrations, installing packages, checking process status, "
                    "debugging build/runtime errors, or any CLI task. "
                    "Commands run via sh -lc in the workspace root with a configurable timeout (max 120s). "
                    "Returns exit code, stdout, and stderr."
                ),
                args_schema=RunTerminalCommandInput,
            ),
        ]

    @staticmethod
    def _clone_structured_tool(tool: Any, **overrides: Any) -> StructuredTool:
        """Clone StructuredTool while preserving tool/validation error handlers."""
        kwargs: dict[str, Any] = {
            "name": overrides.get("name", getattr(tool, "name", "")),
            "description": overrides.get(
                "description",
                getattr(tool, "description", ""),
            ),
            "func": overrides.get("func", getattr(tool, "func", None)),
            "coroutine": overrides.get(
                "coroutine",
                getattr(tool, "coroutine", None),
            ),
            "args_schema": overrides.get(
                "args_schema",
                getattr(tool, "args_schema", None),
            ),
            "return_direct": overrides.get(
                "return_direct",
                getattr(tool, "return_direct", False),
            ),
            "handle_tool_error": overrides.get(
                "handle_tool_error",
                getattr(tool, "handle_tool_error", False),
            ),
        }

        if "handle_validation_error" in overrides or hasattr(
            tool,
            "handle_validation_error",
        ):
            kwargs["handle_validation_error"] = overrides.get(
                "handle_validation_error",
                getattr(tool, "handle_validation_error", False),
            )

        try:
            return StructuredTool(**kwargs)
        except TypeError:
            kwargs.pop("handle_validation_error", None)
            return StructuredTool(**kwargs)

    def get_blocked_config_tool_names(
        self, allowed_tool_config_ids: list[str]
    ) -> set[str]:
        """Return generated tool names that should be blocked for current request context."""
        if not self._tool_configs:
            return set()

        allowed_ids = set(allowed_tool_config_ids)
        all_config_tool_names: set[str] = set()
        allowed_tool_names: set[str] = set()

        for config in self._tool_configs:
            tool_names = self._derive_config_tool_names(config)
            all_config_tool_names.update(tool_names)
            if (config.get("id") or "") in allowed_ids:
                allowed_tool_names.update(tool_names)

        return all_config_tool_names - allowed_tool_names

    def _map_runtime_tools_to_runnable_tool_config_ids(
        self,
        runtime_tools: list[Any],
    ) -> set[str]:
        """Map runtime tool names to runnable ToolConfig IDs for this request."""
        if not self._tool_configs or not runtime_tools:
            return set()

        runtime_tool_names = {
            getattr(tool, "name", "")
            for tool in runtime_tools
            if getattr(tool, "name", "")
        }
        if not runtime_tool_names:
            return set()

        runnable_ids: set[str] = set()
        for config in self._tool_configs:
            config_id = (config.get("id") or "").strip()
            if not config_id:
                continue
            config_tool_names = self._derive_config_tool_names(config)
            if runtime_tool_names.intersection(config_tool_names):
                runnable_ids.add(config_id)
        return runnable_ids

    async def _build_userspace_continuity_prompt(
        self,
        *,
        workspace_id: str,
        user_id: str,
        ep_status: Any,
        is_default_entrypoint: bool,
    ) -> str:
        """Build a compact continuity block for userspace requests."""
        ws_files = await userspace_service.list_workspace_files(
            workspace_id,
            user_id,
            include_dirs=False,
        )
        ws_file_paths = sorted(
            f.path for f in ws_files if not f.path.startswith(".ragtime/")
        )

        last_snapshot_msg: str | None = None
        try:
            snapshots = await userspace_service.list_snapshots(workspace_id, user_id)
            if snapshots:
                last_snapshot_msg = snapshots[0].message
        except Exception:
            # Snapshot history is optional prompt context; never fail request assembly.
            pass

        recent_failure_summaries = self._get_userspace_recent_failure_summaries(
            workspace_id
        )

        return build_workspace_continuity_context(
            file_count=len(ws_file_paths),
            key_files=ws_file_paths,
            framework=ep_status.framework if ep_status.state == "valid" else None,
            entrypoint_valid=ep_status.state == "valid" and not is_default_entrypoint,
            last_snapshot_message=last_snapshot_msg,
            recent_failure_summaries=recent_failure_summaries,
        )

    async def _build_request_runtime_context(
        self,
        *,
        is_ui: bool,
        executor: Optional[AgentExecutor],
        blocked_tool_names: Optional[set[str]],
        workspace_context: Optional[dict[str, str]],
        add_chat_visualization_prompt: bool,
    ) -> dict[str, Any]:
        """Build request-scoped runtime tools, mode, and prompt additions once."""
        t0 = time.monotonic()
        runtime_tools = list(getattr(executor, "tools", []) if executor else [])
        if blocked_tool_names:
            runtime_tools = [
                tool
                for tool in runtime_tools
                if getattr(tool, "name", "") not in blocked_tool_names
            ]

        mode = "chat"
        prompt_is_ui = is_ui
        allowed_tool_config_ids: list[str] | None = None
        prompt_additions = ""
        include_sqlite_persistence = False
        userspace_env_var_turn_hint = ""
        request_tool_state: dict[str, Any] = {
            "tool_calls": [],
            "signature_counts": {},
            "blocked_repeat_calls": 0,
            "max_iterations_reached": False,
            "internal_continue_attempts": 0,
            "internal_continue_stop_reason": "",
            "tool_free_synthesis_used": False,
        }

        workspace_id = (workspace_context or {}).get("workspace_id", "").strip()
        user_id = (workspace_context or {}).get("user_id", "").strip()
        has_workspace_context = bool(workspace_id and user_id)

        if has_workspace_context:
            workspace = await userspace_service.get_workspace(workspace_id, user_id)
            allowed_tool_config_ids = list(workspace.selected_tool_ids)

            # Expand group selections: add all enabled tools from selected groups
            if workspace.selected_tool_group_ids:
                group_tool_ids = await repository.get_tool_ids_for_groups(
                    workspace.selected_tool_group_ids
                )
                existing = set(allowed_tool_config_ids)
                for tid in group_tool_ids:
                    if tid not in existing:
                        allowed_tool_config_ids.append(tid)
                        existing.add(tid)

            include_sqlite_persistence = workspace.sqlite_persistence_mode == "include"

            (
                env_var_summaries,
                workspace_mounts,
                mountable_sources,
                object_storage_config,
            ) = await asyncio.gather(
                userspace_service.list_workspace_env_var_summaries(
                    workspace_id,
                    user_id,
                ),
                userspace_service.list_workspace_mounts(workspace_id, user_id),
                userspace_service.list_mountable_sources(workspace_id, user_id),
                userspace_service.get_workspace_object_storage_summary(
                    workspace_id,
                    user_id,
                ),
            )
            prompt_additions += self._build_userspace_env_var_prompt_fragment(
                env_var_summaries
            )
            prompt_additions += self._build_userspace_mount_prompt_fragment(
                mountable_sources,
                workspace_mounts,
            )
            prompt_additions += self._build_userspace_object_storage_prompt_fragment(
                object_storage_config
            )
            userspace_env_var_turn_hint = self._build_userspace_env_var_turn_hint(
                env_var_summaries
            )

            userspace_tools = await self._create_userspace_file_tools(
                workspace_id,
                user_id,
            )
            runtime_tools = [
                tool
                for tool in runtime_tools
                if getattr(tool, "name", "") not in {"create_chart", "create_datatable"}
            ]
            runtime_tools.extend(userspace_tools)
            runtime_tools = self._apply_mode_specific_tool_description_overrides(
                runtime_tools,
                mode="userspace",
            )
            runtime_tools = self._wrap_userspace_runtime_tools_for_execution_proofs(
                runtime_tools,
                workspace_id,
                allowed_tool_config_ids,
            )
            runtime_tools, request_tool_state = (
                self._wrap_runtime_tools_with_request_state(
                    runtime_tools,
                    mode="userspace",
                    workspace_id=workspace_id,
                )
            )

            mode = "userspace"
            prompt_is_ui = False

            # Dynamic entrypoint nudge: fetch status once and reuse for
            # both is_default check, nudge generation, and state summary.
            ep_status = userspace_service.get_workspace_entrypoint_status(workspace_id)
            is_default = userspace_service.is_default_static_entrypoint(
                workspace_id, status=ep_status
            )

            continuity_ctx = await self._build_userspace_continuity_prompt(
                workspace_id=workspace_id,
                user_id=user_id,
                ep_status=ep_status,
                is_default_entrypoint=is_default,
            )

            # Build prompt additions: visualization + userspace mode (with continuity context).
            prompt_additions += (
                UI_VISUALIZATION_USERSPACE_PROMPT
                + build_userspace_mode_prompt_addition(
                    include_sqlite_persistence=include_sqlite_persistence,
                    has_live_data_tools=bool(allowed_tool_config_ids),
                    workspace_continuity=continuity_ctx,
                )
            )

            # Cache nudge fragment by entrypoint state signature.
            nudge_cache_key = (
                "userspace_nudge",
                ep_status.state,
                is_default,
                ep_status.framework or "",
                ep_status.command or "",
                ep_status.cwd or ".",
            )
            nudge_fragment = self._request_prompt_cache.get(nudge_cache_key)
            if nudge_fragment is None:
                nudge_fragment = build_userspace_entrypoint_nudge(
                    ep_status, is_default_static=is_default
                )
                self._request_prompt_cache[nudge_cache_key] = nudge_fragment
            prompt_additions += nudge_fragment
        else:
            has_workspace_payload = workspace_context is not None
            has_inline_viz_tools = any(
                getattr(tool, "name", "") in {"create_chart", "create_datatable"}
                for tool in runtime_tools
            )
            if has_workspace_payload or has_inline_viz_tools:
                runtime_tools = self._apply_mode_specific_tool_description_overrides(
                    runtime_tools,
                    mode="chat",
                )
                if add_chat_visualization_prompt:
                    prompt_additions += UI_VISUALIZATION_CHAT_PROMPT

        elapsed_ms = (time.monotonic() - t0) * 1000
        prompt_bytes = len(prompt_additions.encode("utf-8", errors="replace"))
        logger.debug(
            "_build_request_runtime_context: mode=%s tools=%d prompt=%d bytes elapsed=%.1fms",
            mode,
            len(runtime_tools),
            prompt_bytes,
            elapsed_ms,
        )

        return {
            "mode": mode,
            "prompt_is_ui": prompt_is_ui,
            "allowed_tool_config_ids": allowed_tool_config_ids,
            "runtime_tools": runtime_tools,
            "prompt_additions": prompt_additions,
            "include_sqlite_persistence": include_sqlite_persistence,
            "userspace_env_var_turn_hint": userspace_env_var_turn_hint,
            "request_tool_state": request_tool_state,
            "workspace_id": workspace_id or None,
        }

    def _build_request_system_prompt(
        self,
        *,
        is_ui: bool,
        mode: str,
        allowed_tool_config_ids: list[str] | None,
        runtime_tools: list[Any] | None = None,
    ) -> str:
        """Build request-scoped system prompt with optional tool visibility filtering."""
        if allowed_tool_config_ids is None and runtime_tools is None:
            if mode == "userspace":
                # Userspace mode appends its own prompt additions per request, so
                # keep this branch aligned with chat-only prebuilt prompts.
                return self._system_prompt
            return self._system_prompt_ui if is_ui else self._system_prompt

        tool_configs = self._tool_configs or []
        if allowed_tool_config_ids is None:
            candidate_tool_configs = list(tool_configs)
        else:
            allowed_ids = set(allowed_tool_config_ids)
            candidate_tool_configs = [
                config
                for config in tool_configs
                if (config.get("id") or "") in allowed_ids
            ]

        unavailable_tool_configs: list[dict] = []
        filtered_tool_configs = candidate_tool_configs
        if runtime_tools is not None:
            runnable_ids = self._map_runtime_tools_to_runnable_tool_config_ids(
                runtime_tools
            )
            filtered_tool_configs = [
                config
                for config in candidate_tool_configs
                if (config.get("id") or "") in runnable_ids
            ]
            unavailable_tool_configs = [
                config
                for config in candidate_tool_configs
                if (config.get("id") or "") not in runnable_ids
            ]

        cache_key = (
            "request_system_prompt",
            bool(is_ui),
            mode,
            tuple(sorted((config.get("id") or "") for config in filtered_tool_configs)),
            tuple(
                sorted((config.get("id") or "") for config in unavailable_tool_configs)
            ),
            bool(
                self._app_settings
                and self._app_settings.get("tool_output_mode", "default") == "auto"
            ),
        )
        cached_prompt = self._request_prompt_cache.get(cache_key)
        if cached_prompt is not None:
            return cached_prompt

        index_prompt_section = build_index_system_prompt(self._index_metadata or [])
        tool_prompt_section = build_tool_system_prompt(
            filtered_tool_configs,
            unavailable_tool_configs=unavailable_tool_configs,
        )

        base_prompt = (
            BASE_USERSPACE_SYSTEM_PROMPT
            if mode == "userspace"
            else BASE_CHAT_SYSTEM_PROMPT
        )
        prompt = base_prompt + index_prompt_section + tool_prompt_section
        if is_ui:
            prompt += UI_VISUALIZATION_COMMON_PROMPT
            if (
                self._app_settings
                and self._app_settings.get("tool_output_mode", "default") == "auto"
            ):
                prompt += TOOL_OUTPUT_VISIBILITY_PROMPT
        self._request_prompt_cache[cache_key] = prompt
        return prompt

    def _build_runtime_executor(
        self,
        tools: list[Any],
        system_prompt: str,
        llm: Optional[Any] = None,
        turn_system_content: Optional[str] = None,
    ) -> Optional[AgentExecutor]:
        """Build a lightweight executor for request-scoped tool filtering."""
        runtime_llm = llm or self.llm
        if runtime_llm is None or not tools:
            return None

        prompt_system = system_prompt
        include_ai_turn_reminder = bool(turn_system_content)
        if turn_system_content and self._uses_copilot_responses_instructions(
            runtime_llm
        ):
            prompt_system = f"{system_prompt}\n\n{turn_system_content}"
            include_ai_turn_reminder = False

        messages: list[Any] = [
            ("system", escape_prompt_template_braces(prompt_system)),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
        ]
        if include_ai_turn_reminder:
            messages.append(("ai", escape_prompt_template_braces(turn_system_content)))
        messages.extend(
            [
                MessagesPlaceholder(variable_name="user_input"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        prompt = ChatPromptTemplate.from_messages(messages)

        agent = self._create_thinking_aware_agent(
            runtime_llm,
            tools,
            prompt,
            message_formatter=self._format_intermediate_steps_for_agent,
        )

        max_iterations = 15
        if self._app_settings:
            try:
                max_iterations = int(self._app_settings.get("max_iterations", 15))
            except (TypeError, ValueError):
                max_iterations = 15

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=max(1, max_iterations),
            return_intermediate_steps=True,
        )

    # ------------------------------------------------------------------
    # Thinking-model tool-call support
    # ------------------------------------------------------------------

    def _is_thinking_ollama(self, llm: Any) -> bool:
        """Return True when *llm* is a ChatOllama instance with thinking/reasoning enabled."""
        return isinstance(llm, ChatOllama) and getattr(llm, "reasoning", None) is True

    @staticmethod
    def _uses_copilot_responses_instructions(llm: Any) -> bool:
        """Return True for Copilot models using the Responses API path.

        GPT-5.4 reasoning summaries are suppressed when per-turn reminders are
        sent as assistant turns. For this path, we fold those reminders into the
        system/instructions channel instead.
        """
        return isinstance(llm, _CopilotChatOpenAI) and bool(
            getattr(llm, "use_responses_api", False)
        )

    @staticmethod
    def _parse_tool_calls_from_thinking(thinking_text: str) -> list[dict]:
        """Parse pseudo tool-call blocks that Ollama thinking models emit
        inside ``<think>`` / reasoning text instead of the structured
        ``tool_calls`` channel.

        Supported formats
        -----------------
        XML-style (qwen family)::

            <tool_call>
            <function=tool_name>
            <parameter=key>value</parameter>
            </function>
            </tool_call>

        JSON-style::

            <tool_call>
            {"name": "tool_name", "arguments": {"key": "value"}}
            </tool_call>

        Returns a list of dicts compatible with ``AIMessage.tool_calls``.
        """
        tool_calls: list[dict] = []
        blocks = re.findall(r"<tool_call>(.*?)</tool_call>", thinking_text, re.DOTALL)
        for idx, block in enumerate(blocks):
            block = block.strip()

            # --- JSON format ---
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append(
                        {
                            "name": data["name"],
                            "args": data.get("arguments", data.get("args", {})),
                            "id": f"think_tc_{idx}_{os.urandom(4).hex()}",
                            "type": "tool_call",
                        }
                    )
                    continue
            except (json.JSONDecodeError, ValueError):
                pass

            # --- XML format (<function=name> ... </function>) ---
            func_match = re.search(r"<function=(\w+)>", block)
            if func_match:
                func_name = func_match.group(1)
                params: dict[str, str] = {}
                for pm in re.finditer(
                    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL
                ):
                    params[pm.group(1)] = pm.group(2).strip()
                tool_calls.append(
                    {
                        "name": func_name,
                        "args": params,
                        "id": f"think_tc_{idx}_{os.urandom(4).hex()}",
                        "type": "tool_call",
                    }
                )
        return tool_calls

    @staticmethod
    def _promote_thinking_tool_calls(message: AIMessage) -> AIMessage:
        """Post-process an AIMessage: if structured ``tool_calls`` is empty
        but the reasoning/thinking text contains pseudo tool-call blocks,
        parse them and promote to real ``tool_calls`` so the agent executor
        can dispatch them.
        """
        if message.tool_calls:
            return message  # Model already emitted structured calls

        # Gather thinking text from all known locations
        thinking = (
            RAGComponents._extract_reasoning_text_from_payload(
                message.additional_kwargs.get("reasoning")
            )
            or message.additional_kwargs.get("reasoning_content", "")
            or message.additional_kwargs.get("reasoning_text", "")
            or message.additional_kwargs.get("reasoning_summary_text", "")
            or message.additional_kwargs.get("thinking", "")
            or ""
        )
        # Also check structured content blocks (Anthropic style)
        if not thinking and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") in (
                    "thinking",
                    "reasoning",
                ):
                    thinking += block.get("text", "") + "\n"

        if not thinking:
            return message

        parsed = RAGComponents._parse_tool_calls_from_thinking(thinking)
        if not parsed:
            return message

        logger.debug(
            "Extracted %d tool call(s) from thinking text: %s",
            len(parsed),
            [tc["name"] for tc in parsed],
        )
        message.tool_calls = parsed
        return message

    def _create_thinking_aware_agent(
        self,
        llm: Any,
        tools: list[Any],
        prompt: ChatPromptTemplate,
        message_formatter: Any = None,
    ) -> Any:
        """Build a tool-calling agent chain.

        For Ollama thinking models, inserts a post-LLM step that extracts
        pseudo tool calls from reasoning text and promotes them to structured
        ``tool_calls`` on the AIMessage, allowing the standard output parser
        to dispatch them.

        For all other LLMs, delegates to ``create_tool_calling_agent``
        unchanged.
        """
        formatter = message_formatter or format_to_tool_messages

        # Always build a custom chain that promotes tool calls found in
        # thinking/reasoning text.  This covers every provider — Ollama,
        # Anthropic (extended thinking), and OpenAI (reasoning models).
        llm_with_tools = llm.bind_tools(tools)
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: formatter(x["intermediate_steps"])
            )
            | prompt
            | llm_with_tools
            | RunnableLambda(self._promote_thinking_tool_calls)
            | ToolsAgentOutputParser()
        )
        return agent

    def _parse_provider_scoped_model(
        self,
        conversation_model: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse optional provider-scoped model format: provider::model_id."""
        if not conversation_model:
            return None, None

        model = conversation_model.strip()
        if not model:
            return None, None

        if "::" not in model:
            return None, model

        prefix, _, remainder = model.partition("::")
        if prefix in {"github_copilot", "github_models"}:
            remainder = remainder.lstrip("/")
        if (
            prefix
            in {
                "openai",
                "anthropic",
                "ollama",
                "github_copilot",
                "github_models",
            }
            and remainder
        ):
            return prefix, remainder

        return None, model

    async def _resolve_chat_request_max_tokens(self, provider: str, model: str) -> int:
        """Resolve chat-specific max_tokens, capped to selected model limits when known."""
        assert self._app_settings is not None

        resolved = await self._resolve_llm_max_tokens(provider, model)

        if provider == "ollama":
            base_url = self._app_settings.get(
                "llm_ollama_base_url",
                self._app_settings.get("ollama_base_url", "http://localhost:11434"),
            )
            model_limit = await get_model_context_length(model, base_url)
            if model_limit and resolved > model_limit:
                logger.debug(
                    f"Capping chat max_tokens for model {model}: {resolved} -> {model_limit}"
                )
                return model_limit
            return resolved

        model_limit = await get_output_limit(model)
        if model_limit and resolved > model_limit:
            logger.debug(
                f"Capping chat max_tokens for model {model}: {resolved} -> {model_limit}"
            )
            return model_limit

        return resolved

    async def _ensure_copilot_llm_fresh(self) -> None:
        """Transparently refresh the cached LLM when the Copilot HMAC token has changed.

        Called before returning ``self.llm`` for github_copilot (OAuth) requests.
        If the stored token differs from what was baked into the LLM at construction
        time (e.g. proactive refresh or user re-authorization), the LLM is rebuilt
        in-place so subsequent requests use the valid credential.
        """
        if not self._app_settings or not self.llm:
            return
        provider = str(self._app_settings.get("llm_provider", "openai")).lower()
        if provider != "github_copilot":
            return
        fresh_token = await ensure_copilot_token_fresh()
        if not fresh_token:
            # Token is gone/expired and couldn't be refreshed.
            # Clear the cached LLM so callers get a clear error.
            self.llm = None
            self._copilot_llm_token = None
            return

        if fresh_token != self._copilot_llm_token:
            model = str(self._app_settings.get("llm_model", "")).strip()
            max_tokens = await self._resolve_llm_max_tokens(provider, model)
            new_llm = await self._build_llm(provider, model, max_tokens)
            if new_llm is not None:
                self.llm = new_llm
                self._copilot_llm_token = fresh_token
                logger.debug("Refreshed cached Copilot LLM with updated token")

    async def _get_request_scoped_llm(
        self, conversation_model: Optional[str]
    ) -> Optional[Any]:
        """Resolve a request-scoped LLM honoring the conversation model override."""
        provider_override, model_id = self._parse_provider_scoped_model(
            conversation_model
        )

        if not model_id:
            await self._ensure_copilot_llm_fresh()
            return self.llm

        if not self._app_settings:
            return self.llm

        configured_model = str(self._app_settings.get("llm_model", "")).strip()
        configured_provider = str(
            self._app_settings.get("llm_provider", "openai")
        ).lower()

        if (
            configured_model
            and model_id == configured_model
            and self.llm is not None
            and (provider_override is None or provider_override == configured_provider)
        ):
            await self._ensure_copilot_llm_fresh()
            return self.llm

        provider = provider_override or configured_provider

        # Backward-compatible provider alias handling for provider-scoped
        # conversation/default models. GitHub Copilot serves OpenAI-family
        # model IDs, so legacy values like "openai::gpt-5.1-codex-mini"
        # should still route through the configured Copilot provider.
        if provider_override == "openai" and configured_provider in {
            "github_copilot",
            "github_models",
        }:
            provider = configured_provider

        max_tokens = await self._resolve_chat_request_max_tokens(provider, model_id)
        return await self._build_llm(provider, model_id, max_tokens)

    def _content_to_text_for_token_estimate(self, content: Any) -> str:
        """Convert message/tool content into plain text for token estimate math."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        chunks.append(str(item.get("text") or ""))
                    elif item_type == "image_url":
                        chunks.append("[image]")
                    elif item_type == "file":
                        chunks.append("[file]")
                    else:
                        chunks.append(str(item))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return str(content.get("text") or "")
            return str(content)
        return str(content)

    async def _build_context_headroom_prompt(
        self,
        *,
        chat_history: list[Any],
        user_content: Any,
        model_id: Optional[str] = None,
    ) -> str:
        """Build a request-scoped context headroom advisory for the model."""
        effective_model_id = model_id or (
            (self._app_settings or {}).get("llm_model", "gpt-4-turbo")
            if self._app_settings
            else "gpt-4-turbo"
        )
        try:
            provider = str(
                (self._app_settings or {}).get("llm_provider", "openai")
            ).lower()
            if provider == "ollama":
                # Use Ollama API as single source of truth for context window
                base_url = (self._app_settings or {}).get(
                    "llm_ollama_base_url",
                    (self._app_settings or {}).get(
                        "ollama_base_url", "http://localhost:11434"
                    ),
                )
                detected = await get_model_context_length(effective_model_id, base_url)
                context_limit = max(1, detected or 8192)
            else:
                # OpenAI/Anthropic: use LiteLLM dataset
                context_limit = max(1, int(await get_context_limit(effective_model_id)))
        except Exception:
            context_limit = 8192

        estimated_tokens = 0
        for message in chat_history:
            content = getattr(message, "content", message)
            estimated_tokens += (
                len(self._content_to_text_for_token_estimate(content)) // 4
            )

        estimated_tokens += (
            len(self._content_to_text_for_token_estimate(user_content)) // 4
        )

        usage_percent = int((estimated_tokens / context_limit) * 100)
        headroom_tokens = max(0, context_limit - estimated_tokens)
        risk_level = (
            "high"
            if usage_percent >= 85
            else "medium" if usage_percent >= 70 else "low"
        )

        return (
            "\n\n## CONTEXT HEADROOM ASSAY\n"
            f"- Estimated conversation usage: {estimated_tokens} / {context_limit} tokens (~{usage_percent}%)\n"
            f"- Estimated headroom: {headroom_tokens} tokens\n"
            f"- Risk level: {risk_level}\n"
            "- Keep responses concise when risk is medium/high and avoid unnecessary tool churn.\n"
            "- For implementation tasks, prioritize minimal edits that complete the request in one loop.\n"
        )

    async def process_query(
        self,
        user_message: Union[str, Any],
        chat_history: Optional[List[Any]] = None,
        blocked_tool_names: Optional[set[str]] = None,
        workspace_context: Optional[dict[str, str]] = None,
        conversation_model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_task_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ) -> str:
        """
        Process a user query through the RAG pipeline (non-streaming).

        Args:
            user_message: The user's question (string or Message object with multimodal content)
            chat_history: Previous messages in the conversation.

        Returns:
            The assistant's response.
        """
        if chat_history is None:
            chat_history = []

        # Convert to LangChain format (preserves multimodal content)
        langchain_content = await self._convert_message_to_langchain_async(user_message)

        try:
            executor = self.agent_executor
            request_llm = await self._get_request_scoped_llm(conversation_model)
            _, request_model_id = self._parse_provider_scoped_model(conversation_model)
            t_ctx = time.monotonic()
            request_context = await self._build_request_runtime_context(
                is_ui=False,
                executor=executor,
                blocked_tool_names=blocked_tool_names,
                workspace_context=workspace_context,
                add_chat_visualization_prompt=True,
            )
            system_prompt = self._build_request_system_prompt(
                is_ui=request_context["prompt_is_ui"],
                mode=request_context["mode"],
                allowed_tool_config_ids=request_context["allowed_tool_config_ids"],
                runtime_tools=request_context["runtime_tools"],
            )
            system_prompt += request_context["prompt_additions"]
            runtime_tools = request_context["runtime_tools"]
            request_tool_state = request_context["request_tool_state"]
            tool_scope_prompt = ""

            # Build per-turn system content (reminders + context headroom)
            turn_system_content = self._build_turn_reminder_text(
                request_context["mode"],
                include_sqlite_persistence=request_context[
                    "include_sqlite_persistence"
                ],
                userspace_env_var_turn_hint=request_context[
                    "userspace_env_var_turn_hint"
                ],
            )
            turn_system_content += await self._build_context_headroom_prompt(
                chat_history=chat_history,
                user_content=langchain_content,
                model_id=request_model_id,
            )

            if runtime_tools:
                tool_scope_prompt = self._build_request_tool_scope_prompt(
                    runtime_tools,
                    mode=request_context["mode"],
                )
                scoped_prompt = system_prompt + tool_scope_prompt
                executor = self._build_runtime_executor(
                    runtime_tools,
                    scoped_prompt,
                    llm=request_llm,
                    turn_system_content=turn_system_content,
                )
            elif executor:
                # Default executor exists but needs turn system content injected
                scoped_prompt = system_prompt
                executor = self._build_runtime_executor(
                    executor.tools,
                    scoped_prompt,
                    llm=request_llm,
                    turn_system_content=turn_system_content,
                )

            ctx_ms = (time.monotonic() - t_ctx) * 1000
            prompt_chars = len(system_prompt)
            logger.debug(
                "process_query: mode=%s prompt_assembly=%.1fms prompt_chars=%d",
                request_context["mode"],
                ctx_ms,
                prompt_chars,
            )

            if executor:
                provider_messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": self._serialize_prompt_content(
                            system_prompt + tool_scope_prompt
                        ),
                    },
                ]
                provider_messages.extend(
                    self._serialize_base_message(message) for message in chat_history
                )
                provider_messages.append(
                    {
                        "role": "assistant",
                        "content": turn_system_content,
                    }
                )
                provider_messages.append(
                    {
                        "role": "user",
                        "content": self._serialize_prompt_content(langchain_content),
                    }
                )
                provider_name = (
                    request_model_id
                    and self._parse_provider_scoped_model(conversation_model)[0]
                ) or str(
                    (self._app_settings or {}).get("llm_provider", "openai")
                ).lower()
                effective_model = request_model_id or str(
                    (self._app_settings or {}).get("llm_model", "")
                )
                # Use agent with tools
                result = await executor.ainvoke(
                    {
                        "input": langchain_content,
                        "user_input": [HumanMessage(content=langchain_content)],
                        "chat_history": chat_history,
                    }
                )
                output = result.get("output", "I couldn't generate a response.")
                # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                if isinstance(output, list):
                    output = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in output
                    )
                lowered_output = str(output).lower()
                request_tool_state["max_iterations_reached"] = bool(
                    "iteration limit" in lowered_output
                    or "max iterations" in lowered_output
                )
                if request_tool_state["max_iterations_reached"] and request_context.get(
                    "workspace_id"
                ):
                    self._record_userspace_failure(
                        request_context["workspace_id"],
                        failure_class="max_iterations_reached",
                        summary="Agent hit the max iteration limit before completing the userspace build loop.",
                        tool_name="agent_executor",
                    )
                debug_metadata = self._build_request_debug_metadata(
                    mode=request_context["mode"],
                    request_tool_state=request_tool_state,
                    workspace_id=request_context.get("workspace_id"),
                )
                await self._persist_provider_prompt_debug_record(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    chat_task_id=chat_task_id,
                    provider=provider_name,
                    model=effective_model,
                    mode=request_context["mode"],
                    request_kind="agent_executor",
                    system_prompt=system_prompt,
                    rendered_user_input=langchain_content,
                    chat_history=chat_history,
                    provider_messages=provider_messages,
                    tool_scope_prompt=tool_scope_prompt,
                    prompt_additions=request_context["prompt_additions"],
                    turn_reminders=turn_system_content,
                    debug_metadata=debug_metadata,
                    message_index=message_index,
                )
                return output
            else:
                # Direct LLM call without tools - use multimodal content
                if request_llm is None:
                    return (
                        "Error: No LLM configured. Please configure an LLM in Settings."
                    )

                direct_system_prompt = system_prompt
                include_ai_turn_reminder = True
                if self._uses_copilot_responses_instructions(request_llm):
                    direct_system_prompt = f"{system_prompt}\n\n{turn_system_content}"
                    include_ai_turn_reminder = False

                messages: List[BaseMessage] = [
                    SystemMessage(content=direct_system_prompt)
                ]
                messages.extend(chat_history)
                if include_ai_turn_reminder:
                    messages.append(AIMessage(content=turn_system_content))
                messages.append(HumanMessage(content=langchain_content))
                provider_name = (
                    self._parse_provider_scoped_model(conversation_model)[0]
                    or str(
                        (self._app_settings or {}).get("llm_provider", "openai")
                    ).lower()
                )
                effective_model = request_model_id or str(
                    (self._app_settings or {}).get("llm_model", "")
                )
                response = await request_llm.ainvoke(messages)
                content = response.content
                debug_metadata = self._build_request_debug_metadata(
                    mode=request_context["mode"],
                    request_tool_state=request_tool_state,
                    workspace_id=request_context.get("workspace_id"),
                )
                await self._persist_provider_prompt_debug_record(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    chat_task_id=chat_task_id,
                    provider=provider_name,
                    model=effective_model,
                    mode=request_context["mode"],
                    request_kind="direct_llm",
                    system_prompt=system_prompt,
                    rendered_user_input=langchain_content,
                    chat_history=chat_history,
                    provider_messages=[
                        self._serialize_base_message(message) for message in messages
                    ],
                    tool_scope_prompt="",
                    prompt_additions=request_context["prompt_additions"],
                    turn_reminders=turn_system_content,
                    debug_metadata=debug_metadata,
                    message_index=message_index,
                )
                return content if isinstance(content, str) else str(content)

        except Exception as e:
            logger.exception("Error processing query")
            return f"I encountered an error processing your request: {str(e)}"

    async def process_query_stream(
        self,
        user_message: Union[str, Any],
        chat_history: Optional[List[Any]] = None,
        is_ui: bool = False,
        blocked_tool_names: Optional[set[str]] = None,
        workspace_context: Optional[dict[str, str]] = None,
        conversation_model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_task_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ):
        """
        Process a user query with true token-by-token streaming.

        For agent with tools: executes tool calls first, then streams the final response.
        For direct LLM: streams tokens directly from the LLM.

        Args:
            user_message: The user's question (string or Message object with multimodal content)
            chat_history: Previous messages in the conversation.
            is_ui: If True, use the UI agent with chart tool and enhanced prompt.
                   If False (default), use the standard agent for API/MCP.

        Yields:
            dict or str: Structured events for tool calls, or text tokens for content.
            - Tool start: {"type": "tool_start", "tool": "tool_name", "input": {...}}
            - Tool end: {"type": "tool_end", "tool": "tool_name", "output": "..."}
            - Content: str (individual tokens/chunks)
            - Max iterations: {"type": "max_iterations_reached"}
        """
        if chat_history is None:
            chat_history = []

        # Convert to LangChain format (preserves multimodal content)
        langchain_content = await self._convert_message_to_langchain_async(user_message)

        # Select the appropriate agent executor
        executor = self.agent_executor_ui if is_ui else self.agent_executor
        request_llm = await self._get_request_scoped_llm(conversation_model)
        _, request_model_id = self._parse_provider_scoped_model(conversation_model)
        t_ctx = time.monotonic()
        request_context = await self._build_request_runtime_context(
            is_ui=is_ui,
            executor=executor,
            blocked_tool_names=blocked_tool_names,
            workspace_context=workspace_context,
            add_chat_visualization_prompt=is_ui,
        )
        system_prompt = self._build_request_system_prompt(
            is_ui=request_context["prompt_is_ui"],
            mode=request_context["mode"],
            allowed_tool_config_ids=request_context["allowed_tool_config_ids"],
            runtime_tools=request_context["runtime_tools"],
        )
        system_prompt += request_context["prompt_additions"]
        runtime_tools = request_context["runtime_tools"]
        request_tool_state = request_context["request_tool_state"]
        tool_scope_prompt = ""

        # Build per-turn system content (reminders + context headroom)
        turn_system_content = self._build_turn_reminder_text(
            request_context["mode"],
            include_sqlite_persistence=request_context["include_sqlite_persistence"],
            userspace_env_var_turn_hint=request_context["userspace_env_var_turn_hint"],
        )
        turn_system_content += await self._build_context_headroom_prompt(
            chat_history=chat_history,
            user_content=langchain_content,
            model_id=request_model_id,
        )

        if runtime_tools:
            tool_scope_prompt = self._build_request_tool_scope_prompt(
                runtime_tools,
                mode=request_context["mode"],
            )
            scoped_prompt = system_prompt + tool_scope_prompt
            executor = self._build_runtime_executor(
                runtime_tools,
                scoped_prompt,
                llm=request_llm,
                turn_system_content=turn_system_content,
            )
        elif executor:
            # Default executor exists but needs turn system content injected
            scoped_prompt = system_prompt
            executor = self._build_runtime_executor(
                executor.tools,
                scoped_prompt,
                llm=request_llm,
                turn_system_content=turn_system_content,
            )

        ctx_ms = (time.monotonic() - t_ctx) * 1000
        prompt_chars = len(system_prompt)
        logger.debug(
            "process_query_stream: mode=%s prompt_assembly=%.1fms prompt_chars=%d",
            request_context["mode"],
            ctx_ms,
            prompt_chars,
        )

        try:
            if executor:
                # Agent with tools: use astream_events for true streaming
                # Strip images from input - tool-calling agents resend the full input
                # on each iteration (tool call -> response -> tool call...) which
                # quickly exhausts rate limits. Images are replaced with [image attached].
                if self._has_image_content(langchain_content):
                    logger.info(
                        "Streaming request contains image_url content; preserving images "
                        "for first-party vision handling"
                    )
                    agent_input = langchain_content
                else:
                    agent_input = self._strip_images_from_content(langchain_content)
                provider_name = (
                    self._parse_provider_scoped_model(conversation_model)[0]
                    or str(
                        (self._app_settings or {}).get("llm_provider", "openai")
                    ).lower()
                )
                effective_model = request_model_id or str(
                    (self._app_settings or {}).get("llm_model", "")
                )
                stream_provider_messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": self._serialize_prompt_content(
                            system_prompt + tool_scope_prompt
                        ),
                    },
                ]
                stream_provider_messages.extend(
                    self._serialize_base_message(message) for message in chat_history
                )
                stream_provider_messages.append(
                    {
                        "role": "assistant",
                        "content": turn_system_content,
                    }
                )
                stream_provider_messages.append(
                    {
                        "role": "user",
                        "content": self._serialize_prompt_content(agent_input),
                    }
                )

                # Track tool runs to avoid duplicates from nested events
                request_tool_state["internal_continue_attempts"] = 0
                attempt_chat_history = list(chat_history)
                attempt_original_input = agent_input
                attempt_history_has_original_input = False
                attempt_input = agent_input
                attempt_number = 0
                any_tool_activity = False

                try:
                    while True:
                        active_tool_runs: set[str] = set()
                        streamed_content_by_chat_run: dict[str, str] = {}
                        streamed_reasoning_by_chat_run: dict[str, str] = {}
                        _generating_tool_lines: dict[str, int] = {}
                        _generating_tool_names: dict[str, str] = {}
                        _tool_start_times: dict[str, tuple[float, str]] = {}
                        _tool_start_payloads: dict[str, dict[str, Any]] = {}
                        attempt_emitted_content = False
                        attempt_had_tool_activity = False
                        attempt_intermediate_steps: list[Any] = []
                        attempt_replayed_tool_messages: list[BaseMessage] = []

                        async for event in executor.astream_events(
                            {
                                "input": attempt_input,
                                "user_input": [HumanMessage(content=attempt_input)],
                                "chat_history": attempt_chat_history,
                            },
                            version="v2",
                        ):
                            kind = event.get("event", "")
                            run_id = event.get("run_id", "")

                            if kind == "on_tool_start":
                                if run_id in active_tool_runs:
                                    continue
                                active_tool_runs.add(run_id)
                                attempt_had_tool_activity = True
                                any_tool_activity = True

                                tool_name = event.get("name", "unknown")
                                tool_input = event.get("data", {}).get("input", {})
                                connection_meta = self._get_tool_connection_metadata(
                                    tool_name
                                )
                                _tool_start_payloads[run_id] = {
                                    "tool": tool_name,
                                    "input": tool_input,
                                }
                                _tool_start_times[run_id] = (
                                    time.monotonic(),
                                    tool_name,
                                )
                                logger.debug(
                                    f"Tool started: {tool_name} (run_id={run_id[:8]})"
                                )
                                yield {
                                    "type": "tool_start",
                                    "tool": tool_name,
                                    "input": tool_input,
                                    "connection": connection_meta,
                                    "run_id": run_id,
                                }

                            elif kind == "on_tool_end":
                                if run_id not in active_tool_runs:
                                    continue
                                active_tool_runs.discard(run_id)

                                tool_name = event.get("name", "unknown")
                                tool_output = event.get("data", {}).get("output", "")
                                connection_meta = self._get_tool_connection_metadata(
                                    tool_name
                                )
                                start_payload = _tool_start_payloads.pop(run_id, None)

                                start_info = _tool_start_times.pop(run_id, None)
                                if start_info:
                                    elapsed = time.monotonic() - start_info[0]
                                    if elapsed > 10:
                                        logger.warning(
                                            f"Slow tool execution: {tool_name} took {elapsed:.1f}s (run_id={run_id[:8]})"
                                        )
                                    else:
                                        logger.debug(
                                            f"Tool completed: {tool_name} in {elapsed:.1f}s (run_id={run_id[:8]})"
                                        )

                                ui_tools = {"create_chart", "create_datatable"}
                                display_output = tool_output
                                if (
                                    isinstance(display_output, str)
                                    and len(display_output) > 2000
                                    and tool_name not in ui_tools
                                ):
                                    display_output = (
                                        display_output[:2000] + "... (truncated)"
                                    )
                                tool_args = (
                                    start_payload.get("input") if start_payload else {}
                                )
                                if not isinstance(tool_args, dict):
                                    tool_args = {"input": tool_args}
                                tool_call_id = (
                                    run_id
                                    or f"stream_tool_{len(attempt_replayed_tool_messages)}"
                                )
                                attempt_replayed_tool_messages.extend(
                                    [
                                        AIMessage(
                                            content="",
                                            tool_calls=[
                                                {
                                                    "name": (
                                                        start_payload.get(
                                                            "tool", tool_name
                                                        )
                                                        if start_payload
                                                        else tool_name
                                                    ),
                                                    "args": tool_args,
                                                    "id": tool_call_id,
                                                }
                                            ],
                                        ),
                                        ToolMessage(
                                            content=str(tool_output or ""),
                                            tool_call_id=tool_call_id,
                                        ),
                                    ]
                                )
                                yield {
                                    "type": "tool_end",
                                    "tool": tool_name,
                                    "output": display_output,
                                    "connection": connection_meta,
                                    "run_id": run_id,
                                }

                            elif kind == "on_tool_error":
                                if run_id not in active_tool_runs:
                                    continue
                                active_tool_runs.discard(run_id)
                                attempt_had_tool_activity = True
                                any_tool_activity = True

                                tool_name = event.get("name", "unknown")
                                error_data = event.get("data", {})
                                error_output = (
                                    str(error_data.get("error", error_data)).strip()
                                    or "Tool execution failed"
                                )
                                connection_meta = self._get_tool_connection_metadata(
                                    tool_name
                                )
                                start_payload = _tool_start_payloads.pop(run_id, None)

                                start_info = _tool_start_times.pop(run_id, None)
                                if start_info:
                                    elapsed = time.monotonic() - start_info[0]
                                    logger.warning(
                                        f"Tool error: {tool_name} failed after {elapsed:.1f}s (run_id={run_id[:8]}): {error_output[:200]}"
                                    )

                                tool_args = (
                                    start_payload.get("input") if start_payload else {}
                                )
                                if not isinstance(tool_args, dict):
                                    tool_args = {"input": tool_args}
                                tool_call_id = (
                                    run_id
                                    or f"stream_tool_{len(attempt_replayed_tool_messages)}"
                                )
                                attempt_replayed_tool_messages.extend(
                                    [
                                        AIMessage(
                                            content="",
                                            tool_calls=[
                                                {
                                                    "name": (
                                                        start_payload.get(
                                                            "tool", tool_name
                                                        )
                                                        if start_payload
                                                        else tool_name
                                                    ),
                                                    "args": tool_args,
                                                    "id": tool_call_id,
                                                }
                                            ],
                                        ),
                                        ToolMessage(
                                            content=f"Error: {error_output}",
                                            tool_call_id=tool_call_id,
                                        ),
                                    ]
                                )

                                yield {
                                    "type": "tool_end",
                                    "tool": tool_name,
                                    "output": f"Error: {error_output}",
                                    "connection": connection_meta,
                                    "run_id": run_id,
                                }

                            elif kind == "on_chat_model_stream":
                                chunk = event.get("data", {}).get("chunk")
                                if chunk:
                                    tool_call_chunks = getattr(
                                        chunk, "tool_call_chunks", None
                                    )
                                    if tool_call_chunks:
                                        for tc_chunk in tool_call_chunks:
                                            tc_id = (
                                                tc_chunk.get("id")
                                                or tc_chunk.get("index")
                                                if isinstance(tc_chunk, dict)
                                                else getattr(tc_chunk, "id", None)
                                                or getattr(tc_chunk, "index", None)
                                            )
                                            tc_name = (
                                                tc_chunk.get("name")
                                                if isinstance(tc_chunk, dict)
                                                else getattr(tc_chunk, "name", None)
                                            )
                                            tc_args = (
                                                tc_chunk.get("args", "")
                                                if isinstance(tc_chunk, dict)
                                                else getattr(tc_chunk, "args", "")
                                            ) or ""
                                            tc_key = (
                                                str(tc_id)
                                                if tc_id is not None
                                                else run_id
                                            )
                                            if tc_name:
                                                _generating_tool_names[tc_key] = tc_name
                                            if tc_args and tc_key:
                                                prev = _generating_tool_lines.get(
                                                    tc_key, 0
                                                )
                                                new_lines = tc_args.count("\n")
                                                if new_lines:
                                                    total = prev + new_lines
                                                    _generating_tool_lines[tc_key] = (
                                                        total
                                                    )
                                                    yield {
                                                        "type": "tool_generating",
                                                        "tool": _generating_tool_names.get(
                                                            tc_key, ""
                                                        ),
                                                        "lines": total,
                                                    }

                                    reasoning_text = (
                                        self._extract_reasoning_from_stream_chunk(chunk)
                                    )
                                    if reasoning_text:
                                        if run_id:
                                            streamed_reasoning_by_chat_run[run_id] = (
                                                streamed_reasoning_by_chat_run.get(
                                                    run_id, ""
                                                )
                                                + reasoning_text
                                            )
                                        yield {
                                            "type": "reasoning",
                                            "content": reasoning_text,
                                        }

                                    if hasattr(chunk, "content") and chunk.content:
                                        content = (
                                            self._extract_text_from_stream_content(
                                                chunk.content
                                            )
                                        )
                                        if content:
                                            if content.strip():
                                                attempt_emitted_content = True
                                            if run_id:
                                                streamed_content_by_chat_run[run_id] = (
                                                    streamed_content_by_chat_run.get(
                                                        run_id, ""
                                                    )
                                                    + content
                                                )
                                            yield content

                            elif kind == "on_chat_model_end":
                                output = event.get("data", {}).get("output")
                                final_reasoning = (
                                    self._extract_reasoning_from_chat_model_output(
                                        output
                                    )
                                )
                                emitted_reasoning = streamed_reasoning_by_chat_run.pop(
                                    run_id, ""
                                )
                                if final_reasoning:
                                    reasoning_suffix = self._compute_missing_suffix(
                                        emitted_reasoning, final_reasoning
                                    )
                                    if reasoning_suffix:
                                        yield {
                                            "type": "reasoning",
                                            "content": reasoning_suffix,
                                        }
                                final_text = self._extract_text_from_chat_model_output(
                                    output
                                )
                                emitted_text = streamed_content_by_chat_run.get(
                                    run_id, ""
                                )
                                missing_suffix = self._compute_missing_suffix(
                                    emitted_text, final_text
                                )
                                if missing_suffix:
                                    if missing_suffix.strip():
                                        attempt_emitted_content = True
                                    if run_id:
                                        streamed_content_by_chat_run[run_id] = (
                                            emitted_text + missing_suffix
                                        )
                                    yield missing_suffix

                            elif kind == "on_chain_end":
                                output = event.get("data", {}).get("output", {})

                                if isinstance(output, dict):
                                    intermediate_steps = output.get(
                                        "intermediate_steps"
                                    )
                                    if isinstance(intermediate_steps, list):
                                        attempt_intermediate_steps = intermediate_steps
                                        if intermediate_steps:
                                            attempt_had_tool_activity = True
                                            any_tool_activity = True

                                    agent_output = output.get("output", "")
                                    if (
                                        "iteration limit" in str(agent_output).lower()
                                        or "max iterations" in str(agent_output).lower()
                                    ):
                                        request_tool_state["max_iterations_reached"] = (
                                            True
                                        )
                                        yield {"type": "max_iterations_reached"}
                                    return_values = output.get("return_values", {})
                                    if isinstance(return_values, dict):
                                        rv_output = return_values.get("output", "")
                                        if "iteration limit" in str(rv_output).lower():
                                            request_tool_state[
                                                "max_iterations_reached"
                                            ] = True
                                            yield {"type": "max_iterations_reached"}

                        should_internal_continue = (
                            not request_tool_state.get("max_iterations_reached", False)
                            and not attempt_emitted_content
                            and attempt_had_tool_activity
                            and bool(
                                attempt_intermediate_steps
                                or attempt_replayed_tool_messages
                            )
                            and attempt_number < MAX_INTERNAL_AGENT_CONTINUATIONS
                        )
                        if not should_internal_continue:
                            stop_reason = ""
                            if request_tool_state.get("max_iterations_reached", False):
                                stop_reason = "max_iterations_reached"
                            elif attempt_emitted_content:
                                stop_reason = "content_emitted"
                            elif not attempt_had_tool_activity:
                                stop_reason = "no_tool_activity_this_attempt"
                            elif not (
                                attempt_intermediate_steps
                                or attempt_replayed_tool_messages
                            ):
                                stop_reason = "no_replay_context"
                            elif attempt_number >= MAX_INTERNAL_AGENT_CONTINUATIONS:
                                stop_reason = "continuation_limit_reached"
                            request_tool_state["internal_continue_stop_reason"] = (
                                stop_reason
                            )
                            if not attempt_emitted_content and any_tool_activity:
                                logger.info(
                                    "Internal continuation stopped: reason=%s max_iter=%s emitted=%s "
                                    "this_attempt_tools=%s steps=%d replayed=%d attempt=%d/%d",
                                    stop_reason,
                                    request_tool_state.get(
                                        "max_iterations_reached", False
                                    ),
                                    attempt_emitted_content,
                                    attempt_had_tool_activity,
                                    len(attempt_intermediate_steps),
                                    len(attempt_replayed_tool_messages),
                                    attempt_number,
                                    MAX_INTERNAL_AGENT_CONTINUATIONS,
                                )
                            break

                        attempt_chat_history = list(attempt_chat_history)
                        if not attempt_history_has_original_input:
                            attempt_chat_history.append(
                                HumanMessage(content=attempt_original_input)
                            )
                            attempt_history_has_original_input = True
                        if attempt_intermediate_steps:
                            attempt_chat_history.extend(
                                self._format_intermediate_steps_for_agent(
                                    attempt_intermediate_steps
                                )
                            )
                        else:
                            attempt_chat_history.extend(attempt_replayed_tool_messages)
                        attempt_input = INTERNAL_AGENT_CONTINUATION_PROMPT
                        attempt_number += 1
                        request_tool_state["internal_continue_attempts"] = (
                            attempt_number
                        )
                        logger.info(
                            "Agent stream attempt %d ended after tool activity without final text; continuing internally",
                            attempt_number,
                        )

                    # ── Tool-free synthesis fallback ──
                    # If all agent continuation attempts are exhausted and
                    # we still have no user-visible text, make one final
                    # direct LLM call WITHOUT tools to guarantee output.
                    if (
                        not attempt_emitted_content
                        and any_tool_activity
                        and not request_tool_state.get("max_iterations_reached", False)
                        and request_llm is not None
                    ):
                        request_tool_state["tool_free_synthesis_used"] = True
                        logger.info(
                            "All %d agent continuation attempt(s) exhausted without "
                            "final text; falling back to tool-free LLM synthesis",
                            attempt_number,
                        )
                        synthesis_messages: list[BaseMessage] = [
                            SystemMessage(content=system_prompt + tool_scope_prompt),
                        ]
                        synthesis_messages.extend(attempt_chat_history)
                        if not attempt_history_has_original_input:
                            synthesis_messages.append(
                                HumanMessage(content=attempt_original_input)
                            )
                        synthesis_tool_context = self._build_internal_synthesis_tool_context(
                            intermediate_steps=attempt_intermediate_steps,
                            replay_messages=attempt_replayed_tool_messages,
                        )
                        if synthesis_tool_context is not None:
                            synthesis_messages.append(synthesis_tool_context)
                        synthesis_messages.append(
                            SystemMessage(content=INTERNAL_AGENT_FINAL_SYNTHESIS_PROMPT)
                        )
                        try:
                            async for chunk in request_llm.astream(synthesis_messages):
                                reasoning_text = (
                                    self._extract_reasoning_from_stream_chunk(chunk)
                                )
                                if reasoning_text:
                                    yield {
                                        "type": "reasoning",
                                        "content": reasoning_text,
                                    }
                                if hasattr(chunk, "content") and chunk.content:
                                    content = self._extract_text_from_stream_content(
                                        chunk.content
                                    )
                                    if content:
                                        attempt_emitted_content = True
                                        yield content
                        except Exception:
                            logger.warning(
                                "Tool-free synthesis LLM call failed",
                                exc_info=True,
                            )
                        else:
                            await self._persist_provider_prompt_debug_record(
                                conversation_id=conversation_id,
                                user_id=user_id,
                                chat_task_id=chat_task_id,
                                provider=provider_name,
                                model=effective_model,
                                mode=request_context["mode"],
                                request_kind="internal_final_synthesis",
                                system_prompt=system_prompt,
                                rendered_user_input=INTERNAL_AGENT_FINAL_SYNTHESIS_PROMPT,
                                chat_history=attempt_chat_history,
                                provider_messages=[
                                    self._serialize_base_message(message)
                                    for message in synthesis_messages
                                ],
                                tool_scope_prompt=tool_scope_prompt,
                                prompt_additions=request_context["prompt_additions"],
                                turn_reminders=turn_system_content,
                                debug_metadata=self._build_request_debug_metadata(
                                    mode=request_context["mode"],
                                    request_tool_state=request_tool_state,
                                    workspace_id=request_context.get("workspace_id"),
                                ),
                                message_index=message_index,
                            )
                finally:
                    if request_tool_state.get(
                        "max_iterations_reached"
                    ) and request_context.get("workspace_id"):
                        self._record_userspace_failure(
                            request_context["workspace_id"],
                            failure_class="max_iterations_reached",
                            summary="Agent hit the max iteration limit before completing the userspace build loop.",
                            tool_name="agent_executor",
                        )
                    debug_metadata = self._build_request_debug_metadata(
                        mode=request_context["mode"],
                        request_tool_state=request_tool_state,
                        workspace_id=request_context.get("workspace_id"),
                    )
                    await self._persist_provider_prompt_debug_record(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        chat_task_id=chat_task_id,
                        provider=provider_name,
                        model=effective_model,
                        mode=request_context["mode"],
                        request_kind="agent_executor",
                        system_prompt=system_prompt,
                        rendered_user_input=agent_input,
                        chat_history=chat_history,
                        provider_messages=stream_provider_messages,
                        tool_scope_prompt=tool_scope_prompt,
                        prompt_additions=request_context["prompt_additions"],
                        turn_reminders=turn_system_content,
                        debug_metadata=debug_metadata,
                        message_index=message_index,
                    )
            else:
                # Direct LLM streaming without tools - use multimodal content
                if request_llm is None:
                    yield "Error: No LLM configured. Please configure an LLM in Settings."
                    return

                direct_system_prompt = system_prompt
                include_ai_turn_reminder = True
                if self._uses_copilot_responses_instructions(request_llm):
                    direct_system_prompt = f"{system_prompt}\n\n{turn_system_content}"
                    include_ai_turn_reminder = False

                messages: List[BaseMessage] = [
                    SystemMessage(content=direct_system_prompt)
                ]
                messages.extend(chat_history)
                if include_ai_turn_reminder:
                    messages.append(AIMessage(content=turn_system_content))
                messages.append(HumanMessage(content=langchain_content))

                provider_name = (
                    self._parse_provider_scoped_model(conversation_model)[0]
                    or str(
                        (self._app_settings or {}).get("llm_provider", "openai")
                    ).lower()
                )
                effective_model = request_model_id or str(
                    (self._app_settings or {}).get("llm_model", "")
                )
                try:
                    async for chunk in request_llm.astream(messages):
                        reasoning_text = self._extract_reasoning_from_stream_chunk(
                            chunk
                        )
                        if reasoning_text:
                            yield {"type": "reasoning", "content": reasoning_text}

                        if hasattr(chunk, "content") and chunk.content:
                            content = self._extract_text_from_stream_content(
                                chunk.content
                            )
                            if content:
                                yield content
                finally:
                    debug_metadata = self._build_request_debug_metadata(
                        mode=request_context["mode"],
                        request_tool_state=request_tool_state,
                        workspace_id=request_context.get("workspace_id"),
                    )
                    await self._persist_provider_prompt_debug_record(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        chat_task_id=chat_task_id,
                        provider=provider_name,
                        model=effective_model,
                        mode=request_context["mode"],
                        request_kind="direct_llm",
                        system_prompt=system_prompt,
                        rendered_user_input=langchain_content,
                        chat_history=chat_history,
                        provider_messages=[
                            self._serialize_base_message(message)
                            for message in messages
                        ],
                        tool_scope_prompt="",
                        prompt_additions=request_context["prompt_additions"],
                        turn_reminders=turn_system_content,
                        debug_metadata=debug_metadata,
                        message_index=message_index,
                    )

        except Exception as e:
            logger.exception("Error in streaming query")
            yield f"I encountered an error processing your request: {str(e)}"


# Global RAG components instance
rag = RAGComponents()
rag = RAGComponents()
