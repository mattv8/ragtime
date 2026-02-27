"""
RAG Components - FAISS Vector Store and LangChain Agent setup.
"""

import asyncio
import fnmatch
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional, Union

import httpx
from fastapi import HTTPException
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool, ToolException
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.core.file_constants import (
    USERSPACE_MODULE_SOURCE_EXTENSIONS,
    USERSPACE_STRICT_FRONTEND_EXTENSIONS,
    USERSPACE_THEME_AUDIT_EXTENSIONS,
    USERSPACE_TYPESCRIPT_EXTENSIONS,
)
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import get_context_limit, get_output_limit
from ragtime.core.ollama import get_model_context_length
from ragtime.core.security import (
    _SSH_ENV_VAR_RE,
    sanitize_output,
    validate_odoo_code,
    validate_sql_query,
    validate_ssh_command,
)
from ragtime.core.sql_utils import add_table_metadata_to_psql_output
from ragtime.core.ssh import (
    SSHConfig,
    SSHTunnel,
    build_ssh_tunnel_config,
    execute_ssh_command,
    expand_env_vars_via_ssh,
    ssh_tunnel_config_from_dict,
)
from ragtime.core.tokenization import truncate_to_token_budget
from ragtime.indexer.pdm_service import pdm_indexer, search_pdm_index
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import schema_indexer, search_schema_index
from ragtime.indexer.vector_backends import FaissBackend, get_faiss_backend
from ragtime.tools import get_all_tools, get_enabled_tools
from ragtime.tools.chart import (
    CHAT_CHART_DESCRIPTION_SUFFIX,
    USERSPACE_CHART_DESCRIPTION_SUFFIX,
    create_chart_tool,
)
from ragtime.tools.datatable import (
    CHAT_DATATABLE_DESCRIPTION_SUFFIX,
    USERSPACE_DATATABLE_DESCRIPTION_SUFFIX,
    create_datatable_tool,
)
from ragtime.tools.filesystem_indexer import search_filesystem_index
from ragtime.tools.git_history import (
    _is_shallow_repository,
    create_aggregate_git_history_tool,
    create_per_index_git_history_tool,
)
from ragtime.tools.mssql import create_mssql_tool
from ragtime.tools.mysql import create_mysql_tool
from ragtime.tools.odoo_shell import filter_odoo_output
from ragtime.userspace.models import (
    ArtifactType,
    UpsertWorkspaceFileRequest,
    UserSpaceLiveDataCheck,
    UserSpaceLiveDataConnection,
)
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

# Maximum timeout for any tool execution (5 minutes)
# AI can request up to this limit; configured per-tool timeout is the default
MAX_TOOL_TIMEOUT_SECONDS = 300

# User Space screenshot caps to keep vision payloads useful but bounded.
MAX_USERSPACE_SCREENSHOT_WIDTH = 1600
MAX_USERSPACE_SCREENSHOT_HEIGHT = 1200
MAX_USERSPACE_SCREENSHOT_PIXELS = 1_440_000


def resolve_effective_timeout(requested_timeout: int, timeout_max_seconds: int) -> int:
    """Resolve runtime timeout using per-tool max (0 = unlimited)."""
    requested = max(0, int(requested_timeout))
    max_timeout = max(0, int(timeout_max_seconds))

    if max_timeout == 0:
        return requested
    return min(requested, max_timeout)


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

    # Create a new tool with the same config but wrapped functions
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        func=truncating_func if original_func else None,
        coroutine=truncating_coroutine if original_coroutine else None,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        handle_tool_error=tool.handle_tool_error,
    )


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

    return f"[{tool_name}] Input: {input_summary} -> Output: {obs_summary}"


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


# =============================================================================
# SYSTEM PROMPT COMPONENTS
# =============================================================================
# The system prompt is composed of these sections in order:
# 1. BASE_SYSTEM_PROMPT - Role, capabilities, tool selection strategy, response format
# 2. build_index_system_prompt() - Dynamic: lists available knowledge indexes
# 3. build_tool_system_prompt() - Dynamic: lists available query/action tools
# 4. UI_VISUALIZATION_COMMON_PROMPT - (UI only) Visualization guidance shared across modes
# 5. UI_VISUALIZATION_CHAT_PROMPT / UI_VISUALIZATION_USERSPACE_PROMPT - Mode-specific visualization guidance (added per request context)
# 6. TOOL_OUTPUT_VISIBILITY_PROMPT - (Conditional) When tool_output_mode is 'auto'
# =============================================================================

BASE_SYSTEM_PROMPT = """You are a technical assistant with access to indexed documentation and live system connections.

## CAPABILITIES

You have two types of tools:

1. **Knowledge Search** - Query indexed documentation, code, configs, and technical docs
2. **System Tools** - Execute queries/commands against live databases and systems

## TOOL SELECTION

Choose the right tool based on what information you need:

| Need | Tool Type | Examples |
|------|-----------|----------|
| Schema, structure, or how things work | Knowledge Search | "What fields does orders table have?", "How is authentication implemented?" |
| Current data values or state | System Tools | "How many active users?", "Show recent orders" |
| Implementation details, business logic | Knowledge Search | "What validation runs on checkout?", "How are prices calculated?" |
| Reports, aggregations, specific records | System Tools | "Revenue this month", "Find user with email X" |

**Workflow for complex questions:**
1. If unsure about schema/structure, search knowledge first
2. Use system tools when you know the target structure
3. If a query fails (unknown column/table), search knowledge to find correct names

## GUIDELINES

- Read each tool's description to understand what system it connects to
- Use LIMIT clauses in SQL queries
- You may call knowledge search multiple times with different queries
- Combine tools as needed: search docs -> query data -> refine

## RESPONSE FORMAT

- Lead with the answer, not implementation details
- Use tables or lists for structured data
- Show queries/code only if explicitly requested
- Be concise but thorough"""

# Reminder prepended to user message - adjacent placement is more effective than distant system prompt
# Contains the critical anti-hallucination instructions that need to be fresh in context
TOOL_USAGE_REMINDER = """[CRITICAL: Use the tool calling API. Do NOT write fake tool invocations or results as text. If you catch yourself writing "(I used X..." or "Result:", STOP - you are hallucinating. Make actual tool calls and wait for real responses.]

"""

USERSPACE_TURN_REMINDER = """[USER SPACE TURN CHECKLIST: Before finalizing, run validate_userspace_code on EVERY changed .ts/.tsx file (including dashboard/main.ts) and fix all reported errors. Persist implementation changes via userspace file tools (not chat-only prose). Treat any tool result with rejected=true as a failed step (even if replacement counts are shown), and fix/retry before claiming success. Never use hardcoded/mock/sample/static data in dashboard modules; wire live data via context.components[componentId].execute() with required metadata/checks. After each completed change loop, call create_userspace_snapshot with a concise completion message.]

"""


# Visualization guidance is layered by context: common + chat + userspace
UI_VISUALIZATION_COMMON_PROMPT = """

## DATA VISUALIZATION

You have visualization tools for rich, interactive displays.

### Tools

- **create_chart** - Chart.js visualizations (bar, line, pie, doughnut)
- **create_datatable** - Interactive DataTables with sorting/search/pagination
- Runtime note: use built-in visualization runtime support; do not add external CDN/npm script imports for chart/table libraries in generated User Space modules.

### When to Use Each

| Data Type | Tool |
|-----------|------|
| Numeric comparisons | create_chart (bar) |
| Time series, trends | create_chart (line) |
| Parts of whole, distribution | create_chart (pie/doughnut) - max 7 segments |
| Any tabular data | create_datatable |
| Aggregations with raw data | BOTH: chart for viz + datatable for details |

### Chart Type Guide

- **Bar**: Category comparisons (by region, status, type)
- **Line**: Sequential/time data (daily, monthly, trends)
- **Pie/Doughnut**: Proportions, market share (<7 categories)
"""


UI_VISUALIZATION_CHAT_PROMPT = """

### Chat Mode Rules

1. **NEVER use markdown tables** - Always use create_datatable instead
2. **Pass data explicitly** - Visualization tools cannot see prior outputs implicitly; include values in each call
3. **Visualize proactively** - Don't wait to be asked; render charts and tables automatically
"""


USERSPACE_SHARED_LIVE_DATA_GUARDRAILS = """

- **NEVER embed hardcoded, mock, sample, or static data arrays in module source.** The system performs AST analysis on TypeScript source and rejects writes that lack structural `context.components[componentId].execute()` binding.
- Chat query tools may enforce `LIMIT` for safe exploration, but persisted `live_data_connections.request` payloads do not require `LIMIT` unless intentionally desired.
- If live wiring is blocked by missing context, persist a scaffold with `execute()` call sites and state the blocker. Do NOT substitute mock data.
- If no tools are selected for the workspace, report the conflict and request tool configuration before proceeding. Do NOT fabricate data.
"""


UI_VISUALIZATION_USERSPACE_PROMPT = f"""

## DATA VISUALIZATION

All charts, tables, and visualizations must be built directly in TypeScript module source
files and persisted via `upsert_userspace_file`.

### Chart.js

Chart.js is preloaded in the User Space preview runtime.

- Use `new (window as any).Chart(canvas, config)` in module code.
- Supported types: bar, line, pie, doughnut, radar, polarArea, scatter, bubble.
- The runtime automatically applies theme colors (text, ticks, grid, legend, title). Do NOT set those manually.
- Only set data-specific colors: dataset `backgroundColor`, `borderColor`, etc.

| Data Type | Chart Type |
|-----------|------------|
| Numeric comparisons | bar |
| Time series, trends | line |
| Parts of whole, distribution | pie/doughnut (max 7 segments) |

### Tables

- Use standard DOM APIs to create `<table>` elements with sorting/filtering as needed.
- Style with theme CSS tokens (`var(--color-text-primary)`, `var(--color-border)`, etc.).
- Do NOT use markdown tables in chat responses for User Space -- build proper DOM tables in modules.

### Live Data Wiring

1. Dashboard module writes in workspaces with selected tools automatically require `live_data_connections`, `live_data_checks`, AND structurally verified `context.components[componentId].execute()` calls in the source code.
2. Chart axes/labels and series data must be sourced from runtime live data payloads via `context.components[componentId].execute()`.
3. Provide `live_data_checks` proving successful connection and transformation before persisting.
{USERSPACE_SHARED_LIVE_DATA_GUARDRAILS}
"""


# Conditional: tool output visibility control (when tool_output_mode is 'auto')
TOOL_OUTPUT_VISIBILITY_PROMPT = """

## TOOL OUTPUT VISIBILITY

Tool output visibility is set to "auto". You control whether tool details (queries, code, raw results) appear in the response.

**Show output when:**
- User asks to see the query/code
- Debugging or educational context
- Raw data provides useful context

**Hide output when:**
- User wants a concise answer
- Tool calls are routine lookups
- Implementation details would clutter the response

Default to hiding unless the user benefits from seeing technical details.
"""


_HEX_COLOR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})(?![A-Za-z0-9_])"
)
_IMPORT_SPECIFIER_PATTERN = re.compile(
    r"^\s*import(?:\s+type)?(?:[\s\w{},*]+from\s*)?[\"']([^\"']+)[\"']",
    re.MULTILINE,
)
_JSX_TAG_PATTERN = re.compile(r"<\s*[A-Za-z][A-Za-z0-9_:\-]*(?:\s|>|/)")

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
    violations: list[str] = []
    normalized_path = (file_path or "").strip()

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

    # JSX in .ts files creates parse errors in TypeScript validator path.
    if normalized_path.endswith(".ts") and _JSX_TAG_PATTERN.search(content or ""):
        violations.append(
            "JSX-like syntax detected in a .ts file. Move JSX to .tsx modules and keep dashboard/main.ts as a thin composition entrypoint."
        )

    return violations


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

    node_script = r"""
const fs = require('fs');
const path = require('path');

const input = fs.readFileSync(0, 'utf8');
const filePath = process.argv[1] || 'dashboard/main.ts';

let ts;
try {
  const tsModulePath = path.join('/ragtime/ragtime/frontend/node_modules/typescript');
  ts = require(tsModulePath);
} catch (err) {
  console.log(JSON.stringify({
    ok: false,
    validator_available: false,
    message: 'TypeScript compiler unavailable for AST analysis',
    has_execute_calls: false,
    found_component_ids: [],
    has_local_imports: false,
    has_context_components_access: false,
  }));
  process.exit(0);
}

const scriptKind = filePath.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
const sourceFile = ts.createSourceFile(filePath, input, ts.ScriptTarget.ES2020, true, scriptKind);

const foundComponentIds = [];
const foundIdSet = new Set();
let hasAnyExecuteCall = false;
let hasLocalImports = false;
let hasContextComponentsAccess = false;

function visit(node) {
  // Detect local workspace imports (static)
  if (ts.isImportDeclaration(node) &&
      node.moduleSpecifier &&
      ts.isStringLiteral(node.moduleSpecifier)) {
    const spec = node.moduleSpecifier.text;
    if (spec.startsWith('./') || spec.startsWith('../')) {
      hasLocalImports = true;
    }
  }

  // Detect dynamic imports: import('./...')
  if (ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      node.arguments.length > 0 &&
      ts.isStringLiteral(node.arguments[0])) {
    const spec = node.arguments[0].text;
    if (spec.startsWith('./') || spec.startsWith('../')) {
      hasLocalImports = true;
    }
  }

  // Detect context.components property access
  if (ts.isPropertyAccessExpression(node) &&
      node.name.text === 'components') {
    const obj = node.expression;
    if (ts.isIdentifier(obj) && obj.text === 'context') {
      hasContextComponentsAccess = true;
    }
  }

  // Detect context.components[X].execute() call expressions
  if (ts.isCallExpression(node)) {
    const expr = node.expression;
    if (ts.isPropertyAccessExpression(expr) && expr.name.text === 'execute') {
      const obj = expr.expression;

      // Pattern: context.components["id"].execute()
      if (ts.isElementAccessExpression(obj)) {
        const container = obj.expression;
        if (ts.isPropertyAccessExpression(container) &&
            container.name.text === 'components') {
          const root = container.expression;
          if (ts.isIdentifier(root) && root.text === 'context') {
            hasAnyExecuteCall = true;
            hasContextComponentsAccess = true;
            const arg = obj.argumentExpression;
            if (ts.isStringLiteral(arg) && !foundIdSet.has(arg.text)) {
              foundIdSet.add(arg.text);
              foundComponentIds.push(arg.text);
            }
          }
        }
      }

      // Pattern: context.components.propName.execute()
      if (ts.isPropertyAccessExpression(obj)) {
        const container = obj.expression;
        if (ts.isPropertyAccessExpression(container) &&
            container.name.text === 'components') {
          const root = container.expression;
          if (ts.isIdentifier(root) && root.text === 'context') {
            hasAnyExecuteCall = true;
            hasContextComponentsAccess = true;
            const propName = obj.name.text;
            if (!foundIdSet.has(propName)) {
              foundIdSet.add(propName);
              foundComponentIds.push(propName);
            }
          }
        }
      }
    }
  }

  ts.forEachChild(node, visit);
}

visit(sourceFile);

console.log(JSON.stringify({
  ok: true,
  validator_available: true,
  has_execute_calls: hasAnyExecuteCall,
  found_component_ids: foundComponentIds,
  has_local_imports: hasLocalImports,
  has_context_components_access: hasContextComponentsAccess,
}));
"""

    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            "-e",
            node_script,
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

    node_script = r"""
const fs = require('fs');
const path = require('path');

const input = fs.readFileSync(0, 'utf8');
const filePath = process.argv[1] || 'dashboard/main.ts';

let ts;
try {
  const tsModulePath = path.join('/ragtime/ragtime/frontend/node_modules/typescript');
  ts = require(tsModulePath);
} catch (err) {
    console.log(JSON.stringify({
        ok: false,
        validator_available: false,
        message: 'TypeScript validator unavailable in runtime',
        errors: [],
    }));
  process.exit(0);
}

    const scriptKind = filePath.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
    const sourceFile = ts.createSourceFile(filePath, input, ts.ScriptTarget.ES2020, true, scriptKind);

const result = ts.transpileModule(input, {
  fileName: filePath,
  reportDiagnostics: true,
  compilerOptions: {
    module: ts.ModuleKind.ES2020,
    target: ts.ScriptTarget.ES2020,
    isolatedModules: true,
    jsx: ts.JsxEmit.ReactJSX,
  },
});

const diagnostics = (result.diagnostics || []).filter((d) => d.category === ts.DiagnosticCategory.Error);
const errors = diagnostics.map((d) => {
  const message = ts.flattenDiagnosticMessageText(d.messageText, '\n');
  if (!d.file || d.start === undefined) return message;
  const pos = d.file.getLineAndCharacterOfPosition(d.start);
  return `${d.file.fileName}:${pos.line + 1}:${pos.character + 1} ${message}`;
});

function isExecuteCall(node) {
    return ts.isCallExpression(node)
        && ts.isPropertyAccessExpression(node.expression)
        && node.expression.name.text === 'execute';
}

const asyncExecuteVars = new Set();
function collectExecuteAssignments(node) {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer) {
        if (isExecuteCall(node.initializer)) {
            asyncExecuteVars.add(node.name.text);
        }
    }
    if (ts.isBinaryExpression(node)
            && node.operatorToken.kind === ts.SyntaxKind.EqualsToken
            && ts.isIdentifier(node.left)
            && isExecuteCall(node.right)) {
        asyncExecuteVars.add(node.left.text);
    }
    ts.forEachChild(node, collectExecuteAssignments);
}

collectExecuteAssignments(sourceFile);

const runtimeErrors = [];
function addRuntimeError(node, message) {
    const pos = sourceFile.getLineAndCharacterOfPosition(node.getStart());
    runtimeErrors.push(`${filePath}:${pos.line + 1}:${pos.character + 1} ${message}`);
}

function inspectRuntimeMisuse(node) {
    if (ts.isPropertyAccessExpression(node)
            && ts.isIdentifier(node.expression)
            && asyncExecuteVars.has(node.expression.text)) {
        const prop = node.name.text;
        if (!['then', 'catch', 'finally'].includes(prop)) {
            addRuntimeError(
                node,
                `Potential async execute() misuse: ${node.expression.text}.${prop} is accessed synchronously after execute(). Await execute() or use .then(...) before reading properties.`
            );
        }
    }

    if (ts.isPropertyAccessExpression(node) && isExecuteCall(node.expression)) {
        const prop = node.name.text;
        if (!['then', 'catch', 'finally'].includes(prop)) {
            addRuntimeError(
                node,
                `Potential async execute() misuse: execute().${prop} is accessed synchronously. Await execute() or use .then(...) before reading properties.`
            );
        }
    }

    ts.forEachChild(node, inspectRuntimeMisuse);
}

inspectRuntimeMisuse(sourceFile);

const mergedErrors = [...errors];
for (const runtimeError of runtimeErrors) {
    if (!mergedErrors.includes(runtimeError)) {
        mergedErrors.push(runtimeError);
    }
}

console.log(JSON.stringify({
    ok: mergedErrors.length === 0,
  validator_available: true,
    error_count: mergedErrors.length,
    errors: mergedErrors,
    runtime_errors: runtimeErrors,
    runtime_error_count: runtimeErrors.length,
}));
"""

    try:
        process = await asyncio.create_subprocess_exec(
            "node",
            "-e",
            node_script,
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


USERSPACE_MODE_PROMPT_ADDITION = f"""

## USER SPACE MODE

You are operating in User Space mode for a persistent workspace artifact workflow.

### Persistence rules

- Persist implementation as files, not only chat text.
- Prefer TypeScript modules for interactive reports and dashboards.
- Build one cohesive frontend app with `dashboard/main.ts` as the fixed entry artifact.
- If `dashboard/main.ts` exists, treat it as the authoritative app entrypoint for feature work. Do not implement dashboard behavior changes in `index.html` unless the user explicitly asks to edit `index.html`.
- For any request to create/build/update a report, dashboard, or frontend, you MUST write/update workspace files via `upsert_userspace_file` before finalizing.
- Do not end with chat-only guidance when the user asked for implementation; persist a runnable scaffold first, then describe blockers.
- Do not keep all logic in `dashboard/main.ts` once complexity grows.
- Split concerns into stable subpaths under `dashboard/` (for example: `dashboard/components/*`, `dashboard/data/*`, `dashboard/charts/*`, `dashboard/styles/*`).
- Keep `dashboard/main.ts` as a thin composition entrypoint that wires imports, layout, and bootstrapping.
- Do not treat each file as an independent rendered page; compose routes, breadcrumbs, and shell navigation inside the single app rooted at `dashboard/main.ts`.
- When adding files, preserve a clear module boundary and reusable naming conventions.

### Runtime contract (must follow)

- User Space preview runs in a Node.js-managed devserver/runtime session (not browser `srcDoc` execution).
- The workspace must have a runnable web entrypoint, resolved in this order: `.ragtime/runtime-entrypoint.json` (`command`, optional `cwd`, optional `framework`), `package.json` with `dev` script, Python app (`manage.py`, `main.py`, or `app.py`), or `index.html` fallback.
- If a runnable web entrypoint is missing, preview fails with: `No runnable web entrypoint found. Add .ragtime/runtime-entrypoint.json with a command/cwd or provide package.json dev script, Python app.py/main.py, or index.html.`
- For module-style dashboard artifacts, keep `dashboard/main.ts` present as the thin composition entrypoint for dashboard modules.
- npm dependencies are allowed when explicitly declared in `package.json`; do not assume globally preloaded libraries.
- Do not inject CDN scripts for runtime dependencies in generated modules.
- The runtime automatically applies theme-matched text, tick, grid, legend, and title colors to every Chart.js instance. Do NOT set `color`, `ticks.color`, `grid.color`, `labels.color`, or `title.color` in chart options; the runtime handles them. Only set data-specific colors (dataset `backgroundColor`, `borderColor`, etc.).
- Do not inject DataTables CDN scripts in generated User Space modules.
- Prefer local workspace modules (`./` or `../`) for internal code organization.
- If JSX is used, keep it out of `dashboard/main.ts`; maintain `dashboard/main.ts` as a valid TypeScript render entrypoint.

### Data wiring rules

- Use real tool outputs as the source of truth for rendered data.
- Persistent User Space dashboards must be live-wired via `context.components[componentId].execute()`.
- When the workspace has selected tools, dashboard module writes (`dashboard/*` or `artifact_type=module_ts`) automatically require `live_data_connections`, `live_data_checks`, and verified `execute()` call sites in the source code.
- Each `live_data_connections` entry must include at least `component_kind=tool_config`, `component_id`, and `request`.
- Include `live_data_checks` for each connection with `connection_check_passed=true` and `transformation_check_passed=true`.
- Never invent or guess `component_id` values. Use only IDs from ACTIVE TOOL CONNECTIONS FOR THIS REQUEST.
- Do not persist dashboard files without connection metadata, even when scaffolding.
- Data connections are internal components, abstracted from end users.
- These components map to admin-configured tools from Settings.
- Persist the connection request (query/command payload + component reference), then read/fetch through `context.components` at render/runtime.
{USERSPACE_SHARED_LIVE_DATA_GUARDRAILS.strip()}
- When creating chart/datatable payloads for reusable artifacts, include `data_connection` as a component reference:
    - `component_kind`: `tool_config`
    - `component_id`: admin-configured tool config ID
    - `component_name`: optional friendly name
    - `component_type`: optional tool type
    - `request`: query/command payload used for refresh
    - `refresh_interval_seconds`: optional refresh cadence
- Do not expose credentials, hostnames, or connection internals in user-facing narrative.

### File tool workflow

- Start by running `assay_userspace_code` to understand current workspace structure and implementation status before editing.
- If assay is unavailable for any reason, fallback to `list_userspace_files` and targeted `read_userspace_file` calls.
- Read any target file before overwriting it.
- For focused, minimal edits, prefer `patch_userspace_file` with explicit old/new snippets instead of re-rendering entire files.
- Then create/update files with full content via User Space file tools.
- For implementation requests, never finish with only prose: ensure at least one artifact file write occurred in the current turn.
- Before declaring "done" or finalizing, you MUST run `validate_userspace_code` on EVERY changed `.ts`/`.tsx` file (including `dashboard/main.ts`) and fix all reported errors first.
- On every completed user-requested change loop, call `create_userspace_snapshot` immediately with a concise completion message.

### Theme + CSS rules

- Style rendered modules to match the active app theme.
- Prefer CSS variables/tokens over hard-coded values, especially for colors, spacing, and radii.
- If module code injects CSS dynamically, keep the same token-based approach so dark/light mode stays consistent.
- Do not introduce custom font stacks, raw hex palettes, or fixed theme-specific colors unless explicitly requested.
- Available CSS variable tokens (always use `var(--token-name)` syntax):

  | Category       | Tokens |
  |----------------|--------|
  | Backgrounds    | `--color-bg-primary`, `--color-bg-secondary`, `--color-bg-tertiary` |
  | Surfaces       | `--color-surface`, `--color-surface-hover`, `--color-surface-active` |
  | Text           | `--color-text-primary`, `--color-text-secondary`, `--color-text-muted`, `--color-text-inverse` |
  | Brand/Accent   | `--color-primary`, `--color-primary-hover`, `--color-primary-light`, `--color-primary-border`, `--color-accent`, `--color-accent-hover`, `--color-accent-light` |
  | Semantic       | `--color-success`, `--color-success-light`, `--color-success-border`, `--color-error`, `--color-error-light`, `--color-error-border`, `--color-warning`, `--color-warning-light`, `--color-warning-border`, `--color-info`, `--color-info-light`, `--color-info-border` |
  | Borders        | `--color-border`, `--color-border-strong` |
  | Input          | `--color-input-bg`, `--color-input-border`, `--color-input-focus` |
  | Shadows        | `--shadow-sm`, `--shadow-md`, `--shadow-lg`, `--shadow-xl` |
  | Spacing        | `--space-xs` (4px), `--space-sm` (8px), `--space-md` (16px), `--space-lg` (24px), `--space-xl` (32px), `--space-2xl` (48px) |
  | Radius         | `--radius-sm` (4px), `--radius-md` (8px), `--radius-lg` (12px), `--radius-xl` (16px), `--radius-full` (pill) |
  | Typography     | `--font-sans`, `--font-mono`, `--text-xs` .. `--text-2xl`, `--leading-tight` / `--leading-normal` / `--leading-relaxed` |
  | Transitions    | `--transition-fast` (150ms), `--transition-normal` (200ms), `--transition-slow` (300ms) |
"""


def build_index_system_prompt(index_metadata: List[dict]) -> str:
    """
    Build system prompt section describing available knowledge indexes.

    Args:
        index_metadata: List of index metadata dictionaries from database.

    Returns:
        System prompt section with index descriptions.
    """
    if not index_metadata:
        return """

## KNOWLEDGE INDEXES

No indexes loaded. Knowledge search unavailable - use system tools only.
"""

    # Filter to only enabled indexes
    enabled_indexes = [idx for idx in index_metadata if idx.get("enabled", True)]
    if not enabled_indexes:
        return """

## KNOWLEDGE INDEXES

All indexes disabled. Enable them in the Indexes tab to search documentation.
"""

    index_lines = []
    for idx in enabled_indexes:
        name = idx.get("name", "Unnamed")
        description = idx.get("description", "")
        doc_count = idx.get("document_count", 0)
        chunk_count = idx.get("chunk_count", 0)
        source_type = idx.get("source_type", "unknown")

        line = f"- **{name}** ({source_type}, {doc_count} files, {chunk_count} chunks)"
        if description:
            line += f": {description}"
        index_lines.append(line)

    return f"""

## KNOWLEDGE INDEXES

Available for search:
{chr(10).join(index_lines)}

These are searched automatically. Use results to understand schemas, business logic, and implementation details before querying live systems.
"""


def build_tool_system_prompt(
    tool_configs: List[dict],
    unavailable_tool_configs: Optional[List[dict]] = None,
) -> str:
    """
    Build system prompt section describing available system tools.

    Args:
        tool_configs: List of tool configuration dictionaries.

    Returns:
        System prompt section with tool descriptions.
    """
    if not tool_configs:
        prompt = """

## SYSTEM TOOLS

No tools configured. Answer from indexed documentation only.
"""
        if unavailable_tool_configs:
            unavailable_lines = []
            for config in unavailable_tool_configs[:8]:
                name = config.get("name", "Unnamed")
                tool_type = config.get("tool_type", "unknown")
                unavailable_lines.append(f"- **{name}** [{tool_type}]")
            remaining = len(unavailable_tool_configs) - len(unavailable_lines)
            if remaining > 0:
                unavailable_lines.append(f"- ... and {remaining} more")

            prompt += (
                "\n\n## CONFIGURED BUT UNAVAILABLE IN THIS REQUEST\n\n"
                + "\n".join(unavailable_lines)
                + "\n"
            )
        return prompt

    # Group tools by type for cleaner organization
    type_labels = {
        "postgres": "PostgreSQL",
        "mssql": "SQL Server",
        "mysql": "MySQL",
        "odoo_shell": "Odoo ORM",
        "ssh_shell": "SSH Shell",
        "filesystem_indexer": "File Search",
    }

    tool_lines = []
    for config in tool_configs:
        tool_type = config.get("tool_type", "unknown")
        name = config.get("name", "Unnamed")
        description = config.get("description", "")

        type_label = type_labels.get(tool_type, tool_type)
        line = f"- **{name}** [{type_label}]"
        if description:
            line += f": {description}"
        tool_lines.append(line)

    prompt = f"""

## SYSTEM TOOLS

{chr(10).join(tool_lines)}

Each tool connects to a different system. Read the description to choose the correct one.
"""
    if unavailable_tool_configs:
        unavailable_lines = []
        for config in unavailable_tool_configs[:8]:
            name = config.get("name", "Unnamed")
            tool_type = config.get("tool_type", "unknown")
            unavailable_lines.append(f"- **{name}** [{tool_type}]")
        remaining = len(unavailable_tool_configs) - len(unavailable_lines)
        if remaining > 0:
            unavailable_lines.append(f"- ... and {remaining} more")

        prompt += (
            "\n\n## CONFIGURED BUT UNAVAILABLE IN THIS REQUEST\n\n"
            + "\n".join(unavailable_lines)
            + "\n"
        )

    return prompt


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
        self._system_prompt: str = BASE_SYSTEM_PROMPT
        self._system_prompt_ui: str = BASE_SYSTEM_PROMPT  # Includes UI additions
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._init_in_progress: bool = False
        self._embedding_model: Optional[Any] = None  # Cached for background loading
        # Detailed index loading tracking
        self._index_details: dict[str, dict] = (
            {}
        )  # name -> {status, size_mb, chunk_count, load_time, error}
        self._loading_index: Optional[str] = None  # Currently loading index name
        self._faiss_loading_task: Optional[asyncio.Task] = None  # prevent GC
        # Token optimization settings
        self._scratchpad_window_size: int = 6  # Default, updated from settings
        # Request-scoped prompt fragments cache
        self._request_prompt_cache: dict[tuple[Any, ...], str] = {}

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
            BASE_SYSTEM_PROMPT + index_prompt_section + tool_prompt_section
        )

        # UI system prompt (includes common visualization instructions)
        self._system_prompt_ui = (
            BASE_SYSTEM_PROMPT
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

        if provider_normalized == "ollama":
            try:
                base_url = self._app_settings.get(
                    "llm_ollama_base_url",
                    self._app_settings.get("ollama_base_url", "http://localhost:11434"),
                )
                return ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=0,
                    num_predict=max_tokens,
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
                )
            except ImportError:
                logger.warning("langchain-anthropic not installed")
                return None

        api_key = self._app_settings.get("openai_api_key", "")
        if not api_key:
            logger.warning("OpenAI selected but no API key configured")
            return None

        return ChatOpenAI(
            model=model,
            temperature=0,
            streaming=True,
            api_key=api_key,
            max_tokens=max_tokens,
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
                model=model, base_url=base_url, num_gpu=-1, keep_alive=-1
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
            fetch_k = max(search_k * 4, 20)  # At least 4x k or 20 candidates
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
                num_steps = len(intermediate_steps)

                if window_size <= 0 or num_steps <= window_size:
                    result = format_to_tool_messages(intermediate_steps)
                    if num_steps > 0:
                        total_chars = sum(len(str(m.content)) for m in result)
                        logger.debug(
                            f"Scratchpad: {num_steps} steps, {len(result)} messages, "
                            f"{total_chars:,} chars (no compression needed)"
                        )
                    return result

                # Format all steps first
                all_msgs = format_to_tool_messages(intermediate_steps)
                full_chars = sum(len(str(m.content)) for m in all_msgs)

                # Compress old tool results (messages beyond recent window)
                # Each step produces 2 messages: AIMessage + ToolMessage
                # So window_size steps = window_size * 2 messages from the end
                recent_msg_count = window_size * 2
                old_msg_count = len(all_msgs) - recent_msg_count

                if old_msg_count > 0:
                    compressed_msgs = []
                    for i, msg in enumerate(all_msgs):
                        if i < old_msg_count and isinstance(msg, ToolMessage):
                            # Compress old tool results to brief summary
                            content = str(msg.content)
                            if len(content) > 200:
                                summary = content[:150] + "... [truncated]"
                            else:
                                summary = content
                            # Preserve tool_call_id for Anthropic pairing
                            compressed_msgs.append(
                                ToolMessage(
                                    content=summary,
                                    tool_call_id=msg.tool_call_id,
                                )
                            )
                        else:
                            compressed_msgs.append(msg)
                    all_msgs = compressed_msgs

                compressed_chars = sum(len(str(m.content)) for m in all_msgs)
                reduction = (
                    100 * (1 - compressed_chars / full_chars) if full_chars > 0 else 0
                )

                if old_msg_count > 0:
                    logger.info(
                        f"Scratchpad compression: {num_steps} steps | "
                        f"{old_msg_count // 2} old results compressed | "
                        f"{full_chars:,} -> {compressed_chars:,} chars ({reduction:.1f}% reduction)"
                    )

                return all_msgs

            return windowed_format

        # Create the message formatter (or None for default behavior)
        message_formatter = (
            create_windowed_formatter(scratchpad_window_size)
            if scratchpad_window_size > 0
            else None  # Use default formatter when disabled
        )

        if scratchpad_window_size > 0:
            logger.info(
                f"Using windowed scratchpad formatter (window={scratchpad_window_size})"
            )

        # Create standard agent (for API/MCP)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if tools:
            # Use create_tool_calling_agent which works with both OpenAI and Anthropic
            # Pass custom message_formatter for scratchpad window compression
            agent = create_tool_calling_agent(
                self.llm,
                tools,
                prompt,
                message_formatter=message_formatter or format_to_tool_messages,
            )
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=settings.debug_mode,
            )
        else:
            self.agent_executor = None

        # Create UI agent (with visualization tools and UI prompt)
        # Note: create_chart_tool and create_datatable_tool are NOT wrapped with
        # truncation because their JSON output must be complete for rendering
        ui_tools = tools + [create_chart_tool, create_datatable_tool]

        prompt_ui = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt_ui),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if ui_tools:
            # Pass same message_formatter for consistent behavior
            agent_ui = create_tool_calling_agent(
                self.llm,
                ui_tools,
                prompt_ui,
                message_formatter=message_formatter or format_to_tool_messages,
            )
            self.agent_executor_ui = AgentExecutor(
                agent=agent_ui,
                tools=ui_tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=settings.debug_mode,
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

        async def _build_single_tool(config: dict) -> List[Any]:
            """Build tool(s) from a single config entry with timeout."""
            tool_type = config.get("tool_type")
            raw_name = (config.get("name", "") or "").strip()
            tool_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
            tool_id = config.get("id") or ""
            result_tools = []

            try:
                if tool_type == "postgres":
                    tool = await self._create_postgres_tool(config, tool_name, tool_id)
                    schema_tool = await self._create_schema_search_tool(
                        config, tool_name, tool_id
                    )
                    if schema_tool:
                        result_tools.append(schema_tool)
                elif tool_type == "mssql":
                    tool = await self._create_mssql_tool(config, tool_name, tool_id)
                    schema_tool = await self._create_schema_search_tool(
                        config, tool_name, tool_id
                    )
                    if schema_tool:
                        result_tools.append(schema_tool)
                elif tool_type == "mysql":
                    tool = await self._create_mysql_tool(config, tool_name, tool_id)
                    schema_tool = await self._create_schema_search_tool(
                        config, tool_name, tool_id
                    )
                    if schema_tool:
                        result_tools.append(schema_tool)
                elif tool_type == "odoo_shell":
                    tool = await self._create_odoo_tool(config, tool_name, tool_id)
                elif tool_type == "ssh_shell":
                    tool = await self._create_ssh_tool(config, tool_name, tool_id)
                elif tool_type == "filesystem_indexer":
                    tool = await self._create_filesystem_tool(
                        config, tool_name, tool_id
                    )
                elif tool_type == "solidworks_pdm":
                    tool = await self._create_pdm_search_tool(
                        config, tool_name, tool_id
                    )
                else:
                    logger.warning(f"Unknown tool type: {tool_type}")
                    return []

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
            results = []
            errors = []

            # Clamp parameters
            k = max(1, min(50, k))
            max_chars_per_result = max(0, min(10000, max_chars_per_result))

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

            for name, db in dbs_to_search.items():
                try:
                    logger.debug(
                        f"Searching index '{name}' with query: {query[:50]}..., k={k}"
                    )
                    # Use MMR or similarity search based on settings
                    if use_mmr:
                        fetch_k = max(k * 4, 20)  # Get more candidates for diversity
                        docs = db.max_marginal_relevance_search(
                            query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
                        )
                    else:
                        docs = db.similarity_search(query, k=k)

                    logger.debug(f"Index '{name}' returned {len(docs)} documents")
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        content = doc.page_content
                        # Apply truncation if max_chars_per_result > 0
                        if (
                            max_chars_per_result > 0
                            and len(content) > max_chars_per_result
                        ):
                            content = content[:max_chars_per_result] + "... (truncated)"
                        results.append(f"[{name}] {source}:\n{content}")
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error searching {name}: {e}", exc_info=True)
                    # Detect Ollama connectivity issues
                    if (
                        "ollama" in error_msg.lower()
                        or "failed to connect" in error_msg.lower()
                    ):
                        errors.append(
                            f"[{name}] Embedding service unavailable - Cannot connect to Ollama. "
                            "Check that Ollama is running and the URL in Settings is accessible from the server "
                            "(use 'host.docker.internal' instead of 'localhost' when running in Docker)."
                        )
                    else:
                        errors.append(f"[{name}] Search error: {error_msg}")

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

        return StructuredTool.from_function(
            func=search_knowledge,
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

        read_file_tool = StructuredTool.from_function(
            func=read_file_from_index,
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

        list_files_tool = StructuredTool.from_function(
            func=list_files_in_index,
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
                    results = []

                    # Clamp parameters
                    k = max(1, min(50, k))
                    max_chars_per_result = max(0, min(10000, max_chars_per_result))

                    logger.debug(
                        f"search_{idx_name} called with query='{query[:50]}...', k={k}, max_chars={max_chars_per_result}"
                    )

                    try:
                        # Use MMR or similarity search based on settings
                        if use_mmr_:
                            fetch_k = max(k * 4, 20)
                            docs = idx_db.max_marginal_relevance_search(
                                query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda_
                            )
                        else:
                            docs = idx_db.similarity_search(query, k=k)

                        logger.debug(
                            f"Index '{idx_name}' returned {len(docs)} documents"
                        )
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            content = doc.page_content
                            # Apply truncation if max_chars_per_result > 0
                            if (
                                max_chars_per_result > 0
                                and len(content) > max_chars_per_result
                            ):
                                content = (
                                    content[:max_chars_per_result] + "... (truncated)"
                                )
                            results.append(f"{source}:\n{content}")
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(
                            f"Error searching {idx_name}: {e}", exc_info=True
                        )
                        if (
                            "ollama" in error_msg.lower()
                            or "failed to connect" in error_msg.lower()
                        ):
                            return (
                                "Embedding service unavailable - Cannot connect to Ollama. "
                                "Check that Ollama is running and accessible."
                            )
                        return f"Search error: {error_msg}"

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

            tool = StructuredTool.from_function(
                func=make_search_func(index_name, db, use_mmr, mmr_lambda, default_k),
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

        # Add timeout field dynamically to avoid scope issues
        PostgresInput.model_fields["timeout"] = Field(
            default=_default_timeout,
            ge=0,
            le=86400,
            description=(
                f"Query timeout in seconds (default: {_default_timeout}, "
                f"max: {'unlimited' if timeout_max_seconds == 0 else timeout_max_seconds}). "
                "Use 0 for no timeout."
            ),
        )
        PostgresInput.model_rebuild()

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

        # Add timeout field dynamically to avoid scope issues
        OdooInput.model_fields["timeout"] = Field(
            default=_default_timeout,
            ge=0,
            le=86400,
            description=(
                f"Execution timeout in seconds (default: {_default_timeout}, "
                f"max: {'unlimited' if timeout_max_seconds == 0 else timeout_max_seconds}). "
                "Use 0 for no timeout."
            ),
        )
        OdooInput.model_rebuild()

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

        # Add timeout field dynamically to avoid scope issues
        SSHInput.model_fields["timeout"] = Field(
            default=_default_timeout,
            ge=0,
            le=86400,
            description=(
                f"Command timeout in seconds (default: {_default_timeout}, "
                f"max: {'unlimited' if timeout_max_seconds == 0 else timeout_max_seconds}). "
                "Use 0 for no timeout."
            ),
        )
        SSHInput.model_rebuild()

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
        reminder_text = TOOL_USAGE_REMINDER
        if mode == "userspace":
            reminder_text += USERSPACE_TURN_REMINDER

        if isinstance(content, str):
            return f"{reminder_text}{content}"

        if isinstance(content, list):
            # Prepend reminder as first text block
            reminder_block = {"type": "text", "text": reminder_text}
            return [reminder_block] + content

        # Fallback
        return f"{reminder_text}{content}"

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
                        langchain_content.append(item)  # Pass through
                elif hasattr(item, "type"):
                    # Pydantic model
                    if item.type == "text":
                        langchain_content.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        langchain_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": item.image_url.url},
                            }
                        )
            return langchain_content if langchain_content else ""

        # Fallback: convert to string
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
                return {
                    "tool_config_id": tool_id,
                    "tool_config_name": tool_name,
                    "tool_type": tool_type,
                }
        return None

    def _build_request_tool_scope_prompt(
        self,
        tools: list[Any],
        mode: str = "chat",
    ) -> str:
        """Build a request-scoped prompt section listing active tool connections."""
        if not tools:
            return ""

        lines: list[str] = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")
            if not tool_name:
                continue

            connection_meta = self._get_tool_connection_metadata(tool_name)
            if connection_meta:
                line = (
                    f"- `{tool_name}` -> {connection_meta.get('tool_config_name') or tool_name} "
                    f"(id={connection_meta.get('tool_config_id')}, type={connection_meta.get('tool_type')})"
                )
            else:
                line = f"- `{tool_name}`"
            lines.append(line)

        if not lines:
            return ""

        prompt = (
            "\n\n## ACTIVE TOOL CONNECTIONS FOR THIS REQUEST\n\n"
            + "Use only these active tool connections in this request context.\n"
        )
        if mode == "userspace":
            prompt += "When creating reusable dashboards/charts/tables, preserve the tool name and input as the stable data connection reference.\n"

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
                    StructuredTool(
                        name=tool.name,
                        description=getattr(tool, "description", ""),
                        func=original_func,
                        coroutine=proofing_coroutine,
                        args_schema=getattr(tool, "args_schema", None),
                        return_direct=getattr(tool, "return_direct", False),
                        handle_tool_error=getattr(tool, "handle_tool_error", False),
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
                StructuredTool(
                    name=tool.name,
                    description=(
                        description.rstrip() + "\n" + description_suffix.strip()
                    ),
                    func=getattr(tool, "func", None),
                    coroutine=getattr(tool, "coroutine", None),
                    args_schema=getattr(tool, "args_schema", None),
                    return_direct=getattr(tool, "return_direct", False),
                    handle_tool_error=getattr(tool, "handle_tool_error", False),
                )
            )

        return overridden_tools

    async def _create_userspace_file_tools(
        self,
        workspace_id: str,
        user_id: str,
    ) -> list[StructuredTool]:
        """Create request-scoped User Space file tools for agentic artifact editing."""

        class ListUserSpaceFilesInput(BaseModel):
            reason: str = Field(
                default="",
                description="Brief description of why files are being listed",
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
            reason: str = Field(
                default="",
                description="Brief description of why this file is being read",
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
                component_id: str = Field(
                    description="Tool config ID verified during live data check.",
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
                default="dashboard/main.ts",
                description=(
                    "Relative path from the workspace files root to create/update. "
                    "Default entry file: dashboard/main.ts"
                ),
            )
            content: str = Field(description="Full UTF-8 file content to write")
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
            replacements: list[ReplacementInput] = Field(
                default_factory=list,
                min_length=1,
                max_length=50,
                description="Ordered replacement operations to apply.",
            )
            reason: str = Field(
                default="",
                description="Brief description of why this patch is being applied",
            )

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
                le=MAX_USERSPACE_SCREENSHOT_WIDTH,
                description=(
                    "Viewport width in pixels. Hard-capped for AI-friendly screenshot size."
                ),
            )
            height: int = Field(
                default=900,
                ge=240,
                le=MAX_USERSPACE_SCREENSHOT_HEIGHT,
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
                    "Use a stable app-root selector when possible."
                ),
            )
            wait_after_load_ms: int = Field(
                default=900,
                ge=0,
                le=15000,
                description="Additional post-render delay before screenshot capture.",
            )
            refresh_before_capture: bool = Field(
                default=True,
                description="Reload once after initial load to reduce stale render races.",
            )
            filename: str | None = Field(
                default=None,
                description=(
                    "Optional output filename ('.png' added automatically if missing). "
                    "Stored under ./.data/_tmp/{workspace_id}/."
                ),
            )
            reason: str = Field(
                default="",
                description="Brief description of why the screenshot is needed",
            )

        async def list_userspace_files(reason: str = "", **_: Any) -> str:
            del reason
            files = await userspace_service.list_workspace_files(workspace_id, user_id)
            return json.dumps([f.model_dump(mode="json") for f in files], indent=2)

        def _compute_workspace_mode(
            file_paths: set[str],
        ) -> tuple[str, str, str | None]:
            if "dashboard/main.ts" in file_paths:
                return (
                    "module_dashboard",
                    "dashboard/main.ts is present and is the authoritative module entrypoint.",
                    "dashboard/main.ts",
                )

            if ".ragtime/runtime-entrypoint.json" in file_paths:
                return (
                    "custom_entrypoint",
                    ".ragtime/runtime-entrypoint.json exists and can override runtime launch behavior.",
                    ".ragtime/runtime-entrypoint.json",
                )

            if "package.json" in file_paths:
                return (
                    "node_app",
                    "package.json exists; runtime typically resolves launch via npm dev script.",
                    "package.json",
                )

            for python_entry in ("manage.py", "main.py", "app.py"):
                if python_entry in file_paths:
                    return (
                        "python_app",
                        f"{python_entry} exists and can serve as a Python runtime entrypoint.",
                        python_entry,
                    )

            if "index.html" in file_paths:
                return (
                    "static_html",
                    "index.html exists without dashboard/main.ts.",
                    "index.html",
                )

            return (
                "unknown",
                "No recognized web entrypoint files detected in workspace.",
                None,
            )

        async def _get_workspace_structure() -> dict[str, Any]:
            files = await userspace_service.list_workspace_files(workspace_id, user_id)
            file_paths = {file.path for file in files}
            workspace_mode, workspace_mode_reason, authoritative_entrypoint = (
                _compute_workspace_mode(file_paths)
            )
            return {
                "files": files,
                "workspace_mode": workspace_mode,
                "workspace_mode_reason": workspace_mode_reason,
                "authoritative_entrypoint": authoritative_entrypoint,
                "has_dashboard_entry": "dashboard/main.ts" in file_paths,
                "has_index_html": "index.html" in file_paths,
            }

        def _is_index_html_path(path: str) -> bool:
            normalized = (path or "").strip().replace("\\", "/").lstrip("/")
            return normalized.lower() == "index.html"

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
            workspace_mode = str(structure["workspace_mode"])
            workspace_mode_reason = str(structure["workspace_mode_reason"])
            authoritative_entrypoint = structure["authoritative_entrypoint"]

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
                file_data = await userspace_service.get_workspace_file(
                    workspace_id, path, user_id
                )
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
                    "workspace_mode": workspace_mode,
                    "workspace_mode_reason": workspace_mode_reason,
                    "authoritative_entrypoint": authoritative_entrypoint,
                    "has_dashboard_entry": has_dashboard_entry,
                    "has_index_html": has_index_html,
                },
                "inspected_files": inspected,
                "next_step": (
                    "Read target files in full before overwrite, then update files and validate TypeScript."
                ),
                "editing_guidance": (
                    "When dashboard/main.ts exists, implement dashboard feature changes in dashboard/* files and avoid index.html edits for behavior changes."
                ),
            }
            return json.dumps(assay, indent=2)

        async def read_userspace_file(path: str, reason: str = "", **_: Any) -> str:
            del reason
            normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
            file_data = await userspace_service.get_workspace_file(
                workspace_id, normalized_path, user_id
            )
            payload = file_data.model_dump(mode="json")
            structure = await _get_workspace_structure()
            if (
                _is_index_html_path(normalized_path)
                and structure["workspace_mode"] == "module_dashboard"
            ):
                payload["warning"] = (
                    "This workspace has dashboard/main.ts. For dashboard behavior changes, edit dashboard/* files (especially dashboard/main.ts and imported modules) instead of index.html."
                )
            return json.dumps(payload, indent=2)

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

            structure = await _get_workspace_structure()
            if (
                _is_index_html_path(normalized_path)
                and structure["workspace_mode"] == "module_dashboard"
            ):
                raise ToolException(
                    "STRUCTURE GUARDRAIL: This workspace uses dashboard/main.ts as the module entrypoint. "
                    "Patch dashboard/* files for feature or behavior updates. index.html edits are blocked to avoid no-op changes."
                )

            file_data = await userspace_service.get_workspace_file(
                workspace_id, normalized_path, user_id
            )

            parsed_replacements: list[PatchUserSpaceFileInput.ReplacementInput] = []
            for item in replacements or []:
                payload = (
                    item.model_dump(mode="python")
                    if isinstance(item, BaseModel)
                    else item
                )
                parsed_replacements.append(
                    PatchUserSpaceFileInput.ReplacementInput.model_validate(payload)
                )

            if not parsed_replacements:
                raise ToolException(
                    "No replacements provided. Supply at least one replacement operation."
                )

            updated_content = file_data.content
            applied: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []

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
                        raise ToolException(
                            f"Replacement #{index} failed: old_text not found in {normalized_path}."
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
                return json.dumps(
                    {
                        "path": normalized_path,
                        "updated": False,
                        "message": "No content changes were applied.",
                        "applied": applied,
                        "skipped": skipped,
                    },
                    indent=2,
                )

            normalized_lower_path = normalized_path.lower()
            patch_is_module_source = is_userspace_module_source_path(
                normalized_lower_path
            )
            patch_is_dashboard_module = patch_is_module_source and (
                normalized_lower_path.startswith("dashboard/")
                or file_data.artifact_type == "module_ts"
            )
            if patch_is_dashboard_module:
                mock_patterns = find_hardcoded_data_patterns(updated_content)
                if mock_patterns:
                    raise ToolException(
                        "LIVE DATA POLICY VIOLATION -- Hardcoded data patterns detected: "
                        + ", ".join(mock_patterns)
                        + ". Replace static/mock data with live runtime data via "
                        "context.components[componentId].execute()."
                    )

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
                )
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", None)
                detail_text = str(getattr(exc, "detail", exc))
                if status_code == 400:
                    return json.dumps(
                        {
                            "rejected": True,
                            "status": "rejected_not_persisted",
                            "updated": False,
                            "persisted": False,
                            "message": (
                                "Patch text replacements were computed locally but the file was NOT persisted."
                            ),
                            "error": detail_text,
                            "action_required": (
                                "Apply the required wiring/contract fixes, then retry patch_userspace_file."
                            ),
                            "path": normalized_path,
                            "attempted_replacements": applied,
                            "skipped": skipped,
                        },
                        indent=2,
                    )
                raise

            typecheck: dict[str, Any] | None = None
            if is_userspace_typescript_path(normalized_path):
                typecheck = await validate_userspace_typescript_content(
                    updated_content,
                    normalized_path,
                )

            response_payload: dict[str, Any] = {
                "updated": True,
                "path": normalized_path,
                "file": result.model_dump(mode="json"),
                "applied": applied,
                "skipped": skipped,
            }
            if typecheck is not None:
                response_payload["typescript_validation"] = typecheck
            return json.dumps(response_payload, indent=2)

        async def upsert_userspace_file(
            path: str,
            content: str,
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

            warnings: list[str] = []
            allowed_violations: list[str] = []
            hard_errors: list[str] = []
            lower_path = (path or "").lower()
            normalized_path = lower_path.replace("\\", "/")

            structure = await _get_workspace_structure()
            if (
                _is_index_html_path(path)
                and structure["workspace_mode"] == "module_dashboard"
            ):
                raise ToolException(
                    "STRUCTURE GUARDRAIL: This workspace uses dashboard/main.ts as the module entrypoint. "
                    "Use upsert_userspace_file on dashboard/* files for feature changes; index.html writes are blocked to prevent ineffective updates."
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
                for item in live_data_checks:
                    payload = (
                        item.model_dump(mode="python")
                        if isinstance(item, BaseModel)
                        else item
                    )
                    parsed_live_data_checks.append(
                        UserSpaceLiveDataCheck.model_validate(payload)
                    )

            workspace = await userspace_service.get_workspace(workspace_id, user_id)
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
                        hard_errors.append(
                            "live_data_checks must indicate successful connection and transformation for "
                            f"component_id={check.component_id}."
                        )

            is_module_source = is_userspace_module_source_path(normalized_path)
            is_dashboard_module = is_module_source and (
                normalized_path.startswith("dashboard/") or artifact_type == "module_ts"
            )

            # Auto-require live data contract when workspace has selected
            # tools and the write targets a dashboard module.  The agent
            # should not be able to bypass the contract by leaving
            # live_data_requested=false while embedding hardcoded data.
            workspace_has_tools = bool(workspace.selected_tool_ids)
            effective_live_data_requested = live_data_requested or (
                workspace_has_tools and is_dashboard_module
            )

            requires_live_data_contract = (
                effective_live_data_requested and is_dashboard_module
            )

            # Detect hardcoded mock/sample data naming patterns.
            # Hardcoded data is always a hard policy violation.
            if is_dashboard_module:
                mock_patterns = find_hardcoded_data_patterns(content)
                if mock_patterns:
                    hard_errors.append(
                        "Hardcoded data patterns detected: "
                        + ", ".join(mock_patterns)
                        + ". All dashboard data must be fetched at runtime via "
                        "context.components[componentId].execute()."
                    )

            # No-tools conflict: warn when dashboard module targets a
            # workspace without any selected tools for live data.
            if is_dashboard_module and not workspace_has_tools:
                warnings.append(
                    "NO LIVE DATA TOOLS AVAILABLE: This workspace has no "
                    "selected tools. Dashboard data cannot be fetched from "
                    "live sources. Inform the user that tool configuration "
                    "is required before live data can be rendered."
                )

            if requires_live_data_contract and not parsed_live_data_connections:
                hard_errors.append(
                    "Missing required live_data_connections contract metadata for this module source write. "
                    "Provide at least one connection with component_kind=tool_config, component_id, and request, or set live_data_requested=false for scaffolding."
                )

            if requires_live_data_contract:
                if not parsed_live_data_checks:
                    hard_errors.append(
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
                        hard_errors.append(
                            "Missing successful live_data_checks verification for component_id(s): "
                            + ", ".join(missing_ids)
                        )

            # AST-based structural validation: verify the module code
            # contains context.components[id].execute() call patterns.
            # This is deterministic and cannot be satisfied by fabricating
            # metadata alone -- the source code must structurally call
            # the live data execution API.
            if requires_live_data_contract and is_dashboard_module:
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
                        hard_errors.append(
                            "Live data binding not found in module source. "
                            "Dashboard modules must call "
                            "context.components[componentId].execute() to "
                            "fetch data at runtime. Hardcoded/static data "
                            "is not permitted when workspace tools are "
                            "available."
                        )
                    elif not has_access and not has_imports:
                        hard_errors.append(
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
                raise ToolException(detail)

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
                )
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", None)
                detail_text = str(getattr(exc, "detail", exc))
                if (
                    status_code == 400
                    and "No server-verified execution proof for component_id(s):"
                    in detail_text
                ):
                    response_payload: dict[str, Any] = {
                        "rejected": True,
                        "error": detail_text,
                        "action_required": (
                            "Run a successful live-data query for each declared component_id "
                            "(workspace tool or execute-component endpoint), then retry upsert_userspace_file."
                        ),
                    }
                    if warnings:
                        response_payload["warnings"] = warnings
                    if allowed_violations:
                        response_payload["allowed_violations"] = allowed_violations
                    if typecheck is not None:
                        response_payload["typescript_validation"] = typecheck
                    return json.dumps(response_payload, indent=2)
                if status_code == 400 and detail_text.startswith(
                    "Entry-point wiring required:"
                ):
                    entrypoint_response_payload: dict[str, Any] = {
                        "rejected": True,
                        "error": detail_text,
                        "action_required": (
                            "Upsert dashboard/main.ts so it imports/composes the module, "
                            "then retry upsert_userspace_file for the non-entry dashboard file."
                        ),
                    }
                    if warnings:
                        entrypoint_response_payload["warnings"] = warnings
                    if allowed_violations:
                        entrypoint_response_payload["allowed_violations"] = (
                            allowed_violations
                        )
                    if typecheck is not None:
                        entrypoint_response_payload["typescript_validation"] = typecheck
                    return json.dumps(entrypoint_response_payload, indent=2)
                raise

            success_response_payload: dict[str, Any] = {
                "file": result.model_dump(mode="json"),
            }
            if warnings:
                success_response_payload["warnings"] = warnings
            if allowed_violations:
                success_response_payload["allowed_violations"] = allowed_violations
                success_response_payload["created_with_violations"] = True
            if typecheck is not None:
                success_response_payload["typescript_validation"] = typecheck
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
            return json.dumps(snapshot.model_dump(mode="json"), indent=2)

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
                    "with a command/cwd or provide package.json dev script, Python app.py/main.py, or index.html."
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
                            f"({exc.msg} at line {exc.lineno}). The runtime will ignore it and use fallback entrypoint detection."
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

                has_package_json = await get_file("package.json") is not None
                has_index_html = await get_file("index.html") is not None
                package_json_has_dev_script = False
                if has_package_json:
                    package_file = await get_file("package.json")
                    try:
                        package_payload = json.loads(
                            (getattr(package_file, "content", "") or "")
                        )
                    except json.JSONDecodeError as exc:
                        add_runtime_warning(
                            "Runtime preflight: package.json is invalid JSON "
                            f"({exc.msg} at line {exc.lineno}). npm-based launch may fail."
                        )
                    else:
                        scripts = (
                            package_payload.get("scripts", {})
                            if isinstance(package_payload, dict)
                            else {}
                        )
                        if isinstance(scripts, dict):
                            package_json_has_dev_script = bool(
                                str(scripts.get("dev") or "").strip()
                            )
                        if not package_json_has_dev_script:
                            add_runtime_warning(
                                "Runtime preflight: package.json is present but scripts.dev is missing; npm run dev will fail unless .ragtime/runtime-entrypoint.json overrides launch."
                            )

                npm_available = shutil.which("npm") is not None
                python_available = shutil.which("python3") is not None
                manage_py = await get_file("manage.py") is not None
                main_py = await get_file("main.py") is not None
                app_py = await get_file("app.py") is not None
                has_python_entrypoint = manage_py or main_py or app_py

                if has_package_json and not npm_available:
                    add_runtime_warning(
                        "Runtime preflight: npm is not available in this environment; Node devserver launch may fail."
                    )
                if has_python_entrypoint and not python_available:
                    add_runtime_warning(
                        "Runtime preflight: python3 is not available in this environment; Python/static fallback launch may fail."
                    )

                has_effective_runtime_candidate = (
                    runtime_config_command_present
                    or (has_package_json and npm_available)
                    or (has_python_entrypoint and python_available)
                    or (has_index_html and python_available)
                )

                if not has_effective_runtime_candidate:
                    add_runtime_error(runnable_entrypoint_error)

            strict_frontend_candidate = is_userspace_strict_frontend_path(
                normalized_start_path
            )
            should_probe_runtime = (
                should_check_runnable_entrypoint or strict_frontend_candidate
            )

            if should_probe_runtime:
                runtime_probe["attempted"] = True
                try:
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

                    timeout = httpx.Timeout(connect=2.0, read=12.0, write=8.0, pool=4.0)
                    async with httpx.AsyncClient(
                        timeout=timeout, follow_redirects=False
                    ) as client:
                        response = await client.get(upstream_url)

                    runtime_probe["preview_status_code"] = response.status_code
                    if response.status_code >= 400:
                        body_preview = (response.text or "")[:200].strip()
                        detail_suffix = f" Body: {body_preview}" if body_preview else ""
                        add_runtime_error(
                            "Runtime strict validation failed: preview upstream returned "
                            f"HTTP {response.status_code}.{detail_suffix}"
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

            if aggregate_runtime_warnings:
                for warning in aggregate_runtime_warnings:
                    add_runtime_error(
                        f"Runtime strict validation warning treated as error: {warning}"
                    )

            overall_ok = (
                bool(file_results)
                and all(
                    bool((res or {}).get("ok", False)) for res in file_results.values()
                )
                and not aggregate_errors
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
            response_payload = {
                "path": normalized_start_path,
                "artifact_type": root_artifact_type,
                "validation": result,
            }
            return json.dumps(response_payload, indent=2)

        async def capture_userspace_screenshot(
            path: str = "",
            width: int = 1440,
            height: int = 900,
            full_page: bool = True,
            timeout_ms: int = 25000,
            wait_for_selector: str = "body",
            wait_after_load_ms: int = 900,
            refresh_before_capture: bool = True,
            filename: str | None = None,
            reason: str = "",
            **_: Any,
        ) -> str:
            del reason
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "viewer"
            )

            requested_width = max(320, int(width))
            requested_height = max(240, int(height))
            width = min(requested_width, MAX_USERSPACE_SCREENSHOT_WIDTH)
            height = min(requested_height, MAX_USERSPACE_SCREENSHOT_HEIGHT)

            requested_pixels = width * height
            if requested_pixels > MAX_USERSPACE_SCREENSHOT_PIXELS:
                scale = (MAX_USERSPACE_SCREENSHOT_PIXELS / requested_pixels) ** 0.5
                width = max(320, int(width * scale))
                height = max(240, int(height * scale))

            normalized_preview_path = (path or "").strip().lstrip("/")
            upstream_url = (
                await userspace_runtime_service.build_workspace_preview_upstream_url(
                    workspace_id,
                    user_id,
                    normalized_preview_path,
                )
            )
            cache_busted_url = (
                f"{upstream_url}{'&' if '?' in upstream_url else '?'}"
                f"_ragtime_screenshot_ts={int(time.time() * 1000)}"
            )

            index_data_root = Path(getattr(settings, "index_data_path", ".data"))
            output_dir = index_data_root / "_tmp" / workspace_id
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            if filename and str(filename).strip():
                candidate = str(filename).strip().replace("\\", "/").split("/")[-1]
            else:
                path_slug = (
                    normalized_preview_path.replace("/", "_").replace(" ", "_")
                    or "root"
                )
                candidate = f"preview_{path_slug}_{timestamp}.png"

            safe_candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)[:200]
            if not safe_candidate:
                safe_candidate = f"preview_{timestamp}.png"
            if not safe_candidate.lower().endswith(".png"):
                safe_candidate += ".png"

            output_path = output_dir / safe_candidate

            selector_value = (wait_for_selector or "").strip()
            node_script = r"""
const targetUrl = process.argv[1];
const outputPath = process.argv[2];
const viewportWidth = Number(process.argv[3] || 1440);
const viewportHeight = Number(process.argv[4] || 900);
const captureFullPage = (process.argv[5] || 'true') === 'true';
const timeoutMs = Number(process.argv[6] || 25000);
const waitForSelector = process.argv[7] || '';
const waitAfterLoadMs = Number(process.argv[8] || 900);
const refreshBeforeCapture = (process.argv[9] || 'true') === 'true';
const maxPixels = Number(process.argv[10] || 1440000);

let playwright;
try {
    playwright = require('/ragtime/ragtime/frontend/node_modules/playwright');
} catch (_) {
    playwright = require('playwright');
}

async function run() {
    const browser = await playwright.chromium.launch({
        headless: true,
        args: ['--disable-dev-shm-usage'],
    });

    try {
        const context = await browser.newContext({
            viewport: { width: viewportWidth, height: viewportHeight },
            deviceScaleFactor: 1,
        });
        const page = await context.newPage();
        page.setDefaultTimeout(timeoutMs);

        const initialResponse = await page.goto(targetUrl, {
            waitUntil: 'domcontentloaded',
            timeout: timeoutMs,
        });

        await page.waitForLoadState('networkidle', {
            timeout: Math.min(timeoutMs, 8000),
        }).catch(() => null);

        if (refreshBeforeCapture) {
            await page.reload({
                waitUntil: 'domcontentloaded',
                timeout: timeoutMs,
            });
            await page.waitForLoadState('networkidle', {
                timeout: Math.min(timeoutMs, 8000),
            }).catch(() => null);
        }

        if (waitForSelector) {
            await page.waitForSelector(waitForSelector, {
                state: 'visible',
                timeout: Math.min(timeoutMs, 10000),
            }).catch(() => null);
        }

        if (waitAfterLoadMs > 0) {
            await page.waitForTimeout(waitAfterLoadMs);
        }

        const screenshotOptions = {
            path: outputPath,
            animations: 'disabled',
        };

        let effectiveWidth = viewportWidth;
        let effectiveHeight = viewportHeight;
        let effectiveFullPage = captureFullPage;

        if (captureFullPage) {
            const fullHeight = await page.evaluate(() => {
                const bodyHeight = document.body ? document.body.scrollHeight : 0;
                const docHeight = document.documentElement
                    ? document.documentElement.scrollHeight
                    : 0;
                return Math.max(bodyHeight, docHeight, window.innerHeight || 0);
            });

            if (viewportWidth * fullHeight <= maxPixels) {
                screenshotOptions.fullPage = true;
                effectiveHeight = fullHeight;
            } else {
                effectiveFullPage = false;
                const clipHeight = Math.max(240, Math.floor(maxPixels / Math.max(1, viewportWidth)));
                screenshotOptions.clip = {
                    x: 0,
                    y: 0,
                    width: viewportWidth,
                    height: clipHeight,
                };
                effectiveHeight = clipHeight;
            }
        } else if (viewportWidth * viewportHeight > maxPixels) {
            const scale = Math.sqrt(maxPixels / (viewportWidth * viewportHeight));
            const clipWidth = Math.max(320, Math.floor(viewportWidth * scale));
            const clipHeight = Math.max(240, Math.floor(viewportHeight * scale));
            screenshotOptions.clip = {
                x: 0,
                y: 0,
                width: clipWidth,
                height: clipHeight,
            };
            effectiveWidth = clipWidth;
            effectiveHeight = clipHeight;
        }

        await page.screenshot(screenshotOptions);

        const title = await page.title().catch(() => '');
        const htmlLength = await page
            .content()
            .then((html) => html.length)
            .catch(() => null);

        const output = {
            ok: true,
            status_code: initialResponse ? initialResponse.status() : null,
            title,
            html_length: htmlLength,
            output_path: outputPath,
            screenshot_url: targetUrl,
            effective_width: effectiveWidth,
            effective_height: effectiveHeight,
            effective_full_page: effectiveFullPage,
        };
        process.stdout.write(JSON.stringify(output));
    } finally {
        await browser.close();
    }
}

run().catch((error) => {
    const message = error && error.message ? error.message : String(error);
    process.stderr.write(message);
    process.exit(1);
});
"""

            process = await asyncio.create_subprocess_exec(
                "node",
                "-e",
                node_script,
                cache_busted_url,
                str(output_path),
                str(width),
                str(height),
                "true" if full_page else "false",
                str(timeout_ms),
                selector_value,
                str(wait_after_load_ms),
                "true" if refresh_before_capture else "false",
                str(MAX_USERSPACE_SCREENSHOT_PIXELS),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if process.returncode != 0:
                raise ToolException(
                    "Screenshot capture failed. "
                    "Ensure Playwright + Chromium are installed in the container. "
                    f"Details: {stderr_text or stdout_text or 'unknown error'}"
                )

            parsed: dict[str, Any]
            try:
                parsed = json.loads(stdout_text) if stdout_text else {}
            except json.JSONDecodeError:
                parsed = {}

            if not output_path.exists():
                raise ToolException(
                    "Screenshot capture reported success but no file was written. "
                    f"Output path: {output_path}"
                )

            response_payload = {
                "ok": True,
                "workspace_id": workspace_id,
                "preview_path": normalized_preview_path,
                "screenshot_path": str(output_path),
                "screenshot_size_bytes": output_path.stat().st_size,
                "render": {
                    "requested_width": requested_width,
                    "requested_height": requested_height,
                    "width": width,
                    "height": height,
                    "full_page": full_page,
                    "max_pixels": MAX_USERSPACE_SCREENSHOT_PIXELS,
                    "wait_for_selector": selector_value or None,
                    "wait_after_load_ms": wait_after_load_ms,
                    "refresh_before_capture": refresh_before_capture,
                },
                "probe": parsed,
            }
            return json.dumps(response_payload, indent=2)

        return [
            StructuredTool.from_function(
                coroutine=assay_userspace_code,
                name="assay_userspace_code",
                description=(
                    "Perform a focused assay of existing workspace code and structure before editing. "
                    "Use this first in each implementation loop to understand current state and identify target files."
                ),
                args_schema=AssayUserSpaceCodeInput,
            ),
            StructuredTool.from_function(
                coroutine=list_userspace_files,
                name="list_userspace_files",
                description=(
                    "List files in the active User Space workspace. Use this before creating or updating files."
                ),
                args_schema=ListUserSpaceFilesInput,
            ),
            StructuredTool.from_function(
                coroutine=read_userspace_file,
                name="read_userspace_file",
                description=(
                    "Read a file from the active User Space workspace by relative path."
                ),
                args_schema=ReadUserSpaceFileInput,
            ),
            StructuredTool.from_function(
                coroutine=upsert_userspace_file,
                name="upsert_userspace_file",
                description=(
                    "Create or update a file in the active User Space workspace. "
                    "For interactive reports, write TypeScript modules (artifact_type=module_ts). "
                    "Dashboard module writes in workspaces with selected tools automatically "
                    "require live_data_connections, live_data_checks, AND structurally verified "
                    "context.components[componentId].execute() calls in the source code (AST-checked). "
                    "Policy violations raise tool errors that must be resolved before retrying. "
                    "Output may include CSS/theme warnings that must be fixed in follow-up edits."
                ),
                args_schema=UpsertUserSpaceFileInput,
                handle_tool_error=True,
            ),
            StructuredTool.from_function(
                coroutine=patch_userspace_file,
                name="patch_userspace_file",
                description=(
                    "Apply targeted sed-style in-place replacements to an existing User Space file. "
                    "Use for surgical edits to avoid re-rendering full file content. "
                    "Supports ordered exact old/new replacements with per-op required flags."
                ),
                args_schema=PatchUserSpaceFileInput,
                handle_tool_error=True,
            ),
            StructuredTool.from_function(
                coroutine=create_userspace_snapshot,
                name="create_userspace_snapshot",
                description=(
                    "Create a git-based checkpoint snapshot in the active User Space workspace. "
                    "Use this automatically at each completed user-requested change loop."
                ),
                args_schema=CreateUserSpaceSnapshotInput,
            ),
            StructuredTool.from_function(
                coroutine=validate_userspace_code,
                name="validate_userspace_code",
                description=(
                    "Validate a workspace code file and return TypeScript/runtime diagnostics with file/line details. "
                    "Use after edits and before creating snapshots."
                ),
                args_schema=ValidateUserSpaceCodeInput,
            ),
            StructuredTool.from_function(
                coroutine=capture_userspace_screenshot,
                name="capture_userspace_screenshot",
                description=(
                    "Capture a rendered screenshot of the live User Space preview using Playwright. "
                    "Waits for load/refresh before capture to reduce race conditions and helps diagnose runtime errors "
                    "(blank screen, crashes, broken layout). Saves PNG files under ./.data/_tmp/{workspace_id}/."
                ),
                args_schema=CaptureUserSpaceScreenshotInput,
                handle_tool_error=True,
            ),
        ]

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

        workspace_id = (workspace_context or {}).get("workspace_id", "").strip()
        user_id = (workspace_context or {}).get("user_id", "").strip()
        has_workspace_context = bool(workspace_id and user_id)

        if has_workspace_context:
            workspace = await userspace_service.get_workspace(workspace_id, user_id)
            allowed_tool_config_ids = workspace.selected_tool_ids

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

            mode = "userspace"
            prompt_is_ui = False
            prompt_additions += UI_VISUALIZATION_USERSPACE_PROMPT
            prompt_additions += USERSPACE_MODE_PROMPT_ADDITION
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

        return {
            "mode": mode,
            "prompt_is_ui": prompt_is_ui,
            "allowed_tool_config_ids": allowed_tool_config_ids,
            "runtime_tools": runtime_tools,
            "prompt_additions": prompt_additions,
        }

    def _build_request_system_prompt(
        self,
        *,
        is_ui: bool,
        allowed_tool_config_ids: list[str] | None,
        runtime_tools: list[Any] | None = None,
    ) -> str:
        """Build request-scoped system prompt with optional tool visibility filtering."""
        if allowed_tool_config_ids is None and runtime_tools is None:
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

        prompt = BASE_SYSTEM_PROMPT + index_prompt_section + tool_prompt_section
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
    ) -> Optional[AgentExecutor]:
        """Build a lightweight executor for request-scoped tool filtering."""
        runtime_llm = llm or self.llm
        if runtime_llm is None or not tools:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(
            runtime_llm,
            tools,
            prompt,
            message_formatter=format_to_tool_messages,
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
            verbose=settings.debug_mode,
            handle_parsing_errors=True,
            max_iterations=max(1, max_iterations),
            return_intermediate_steps=settings.debug_mode,
        )

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
        if prefix in {"openai", "anthropic", "ollama"} and remainder:
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

    async def _get_request_scoped_llm(
        self, conversation_model: Optional[str]
    ) -> Optional[Any]:
        """Resolve a request-scoped LLM honoring the conversation model override."""
        provider_override, model_id = self._parse_provider_scoped_model(
            conversation_model
        )

        if not model_id:
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
            return self.llm

        provider = provider_override or configured_provider
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
        langchain_content = self._convert_message_to_langchain(user_message)

        try:
            executor = self.agent_executor
            request_llm = await self._get_request_scoped_llm(conversation_model)
            _, request_model_id = self._parse_provider_scoped_model(conversation_model)
            request_context = await self._build_request_runtime_context(
                is_ui=False,
                executor=executor,
                blocked_tool_names=blocked_tool_names,
                workspace_context=workspace_context,
                add_chat_visualization_prompt=True,
            )
            # Prepend per-turn reminder (adjacent to current message for effectiveness)
            augmented_content = self._prepend_reminder_to_content(
                langchain_content,
                mode=request_context["mode"],
            )
            system_prompt = self._build_request_system_prompt(
                is_ui=request_context["prompt_is_ui"],
                allowed_tool_config_ids=request_context["allowed_tool_config_ids"],
                runtime_tools=request_context["runtime_tools"],
            )
            system_prompt += await self._build_context_headroom_prompt(
                chat_history=chat_history,
                user_content=augmented_content,
                model_id=request_model_id,
            )
            system_prompt += request_context["prompt_additions"]
            runtime_tools = request_context["runtime_tools"]

            if runtime_tools:
                scoped_prompt = system_prompt + self._build_request_tool_scope_prompt(
                    runtime_tools,
                    mode=request_context["mode"],
                )
                executor = self._build_runtime_executor(
                    runtime_tools,
                    scoped_prompt,
                    llm=request_llm,
                )

            if executor:
                # Use agent with tools
                result = await executor.ainvoke(
                    {"input": augmented_content, "chat_history": chat_history}
                )
                output = result.get("output", "I couldn't generate a response.")
                # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                if isinstance(output, list):
                    output = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in output
                    )
                return output
            else:
                # Direct LLM call without tools - use multimodal content
                if request_llm is None:
                    return (
                        "Error: No LLM configured. Please configure an LLM in Settings."
                    )

                messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
                messages.extend(chat_history)
                messages.append(HumanMessage(content=augmented_content))
                response = await request_llm.ainvoke(messages)
                content = response.content
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
        langchain_content = self._convert_message_to_langchain(user_message)

        # Select the appropriate agent executor
        executor = self.agent_executor_ui if is_ui else self.agent_executor
        request_llm = await self._get_request_scoped_llm(conversation_model)
        _, request_model_id = self._parse_provider_scoped_model(conversation_model)
        request_context = await self._build_request_runtime_context(
            is_ui=is_ui,
            executor=executor,
            blocked_tool_names=blocked_tool_names,
            workspace_context=workspace_context,
            add_chat_visualization_prompt=is_ui,
        )
        system_prompt = self._build_request_system_prompt(
            is_ui=request_context["prompt_is_ui"],
            allowed_tool_config_ids=request_context["allowed_tool_config_ids"],
            runtime_tools=request_context["runtime_tools"],
        )
        system_prompt += await self._build_context_headroom_prompt(
            chat_history=chat_history,
            user_content=langchain_content,
            model_id=request_model_id,
        )
        system_prompt += request_context["prompt_additions"]
        runtime_tools = request_context["runtime_tools"]

        if runtime_tools:
            scoped_prompt = system_prompt + self._build_request_tool_scope_prompt(
                runtime_tools,
                mode=request_context["mode"],
            )
            executor = self._build_runtime_executor(
                runtime_tools,
                scoped_prompt,
                llm=request_llm,
            )

        try:
            if executor:
                # Agent with tools: use astream_events for true streaming
                # Strip images from input - tool-calling agents resend the full input
                # on each iteration (tool call -> response -> tool call...) which
                # quickly exhausts rate limits. Images are replaced with [image attached].
                stripped_content = self._strip_images_from_content(langchain_content)

                # Prepend tool usage reminder (adjacent to current message for effectiveness)
                agent_input = self._prepend_reminder_to_content(
                    stripped_content,
                    mode=request_context["mode"],
                )

                # Track tool runs to avoid duplicates from nested events
                active_tool_runs: set[str] = set()

                async for event in executor.astream_events(
                    {"input": agent_input, "chat_history": chat_history},
                    version="v2",
                ):
                    kind = event.get("event", "")
                    run_id = event.get("run_id", "")

                    # Emit tool start events - only for new tool runs
                    if kind == "on_tool_start":
                        # Skip if we've already seen this tool run
                        if run_id in active_tool_runs:
                            continue
                        active_tool_runs.add(run_id)

                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", {})
                        connection_meta = self._get_tool_connection_metadata(tool_name)
                        yield {
                            "type": "tool_start",
                            "tool": tool_name,
                            "input": tool_input,
                            "connection": connection_meta,
                            "run_id": run_id,  # Include run_id for matching with tool_end
                        }

                    # Emit tool end events - only for runs we started
                    elif kind == "on_tool_end":
                        # Skip if we didn't track this tool run starting
                        if run_id not in active_tool_runs:
                            continue
                        active_tool_runs.discard(run_id)

                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        connection_meta = self._get_tool_connection_metadata(tool_name)
                        # Don't truncate UI visualization tools - their JSON must be complete
                        # Truncate other long outputs for display
                        ui_tools = {"create_chart", "create_datatable"}
                        if (
                            isinstance(tool_output, str)
                            and len(tool_output) > 2000
                            and tool_name not in ui_tools
                        ):
                            tool_output = tool_output[:2000] + "... (truncated)"
                        yield {
                            "type": "tool_end",
                            "tool": tool_name,
                            "output": tool_output,
                            "connection": connection_meta,
                            "run_id": run_id,  # Include run_id for matching with tool_start
                        }

                    # Stream tokens from the chat model
                    elif kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                            if isinstance(content, list):
                                content = "".join(
                                    (
                                        block.get("text", "")
                                        if isinstance(block, dict)
                                        else str(block)
                                    )
                                    for block in content
                                )
                            if content:
                                yield content

                    # Detect when agent executor finishes - check for max iterations
                    elif kind == "on_chain_end":
                        # Check if this is the AgentExecutor finishing
                        output = event.get("data", {}).get("output", {})

                        # AgentExecutor sets "Agent stopped due to iteration limit" in output
                        if isinstance(output, dict):
                            agent_output = output.get("output", "")
                            if (
                                "iteration limit" in str(agent_output).lower()
                                or "max iterations" in str(agent_output).lower()
                            ):
                                yield {"type": "max_iterations_reached"}
                            # Also check return_values for the same message
                            return_values = output.get("return_values", {})
                            if isinstance(return_values, dict):
                                rv_output = return_values.get("output", "")
                                if "iteration limit" in str(rv_output).lower():
                                    yield {"type": "max_iterations_reached"}
            else:
                # Direct LLM streaming without tools - use multimodal content
                if request_llm is None:
                    yield "Error: No LLM configured. Please configure an LLM in Settings."
                    return

                messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
                messages.extend(chat_history)
                # Use langchain_content (can be string or multimodal list)
                messages.append(HumanMessage(content=langchain_content))

                # Use astream for true token-by-token streaming
                async for chunk in request_llm.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                        if isinstance(content, list):
                            content = "".join(
                                (
                                    block.get("text", "")
                                    if isinstance(block, dict)
                                    else str(block)
                                )
                                for block in content
                            )
                        if content:
                            yield content

        except Exception as e:
            logger.exception("Error in streaming query")
            yield f"I encountered an error processing your request: {str(e)}"


# Global RAG components instance
rag = RAGComponents()
