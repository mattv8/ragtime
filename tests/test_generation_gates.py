"""Regression inventory for trusted generation-policy choke points."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CALL_NAMES = {"ainvoke", "astream", "astream_events"}

# Each exemption is a concrete, reviewed non-generative execution boundary.
# Do not add module-wide exemptions: a new model call in any module must fail.
NON_GENERATIVE_CALL_EXEMPTIONS = {
    ("ragtime/indexer/routes.py", "_invoke_retry_terminal_tool_with_http_timeout", "tool.ainvoke"),
    ("ragtime/indexer/visualization_retry.py", "_rerun_source_query", "tool.ainvoke"),
    ("ragtime/mcp/tools.py", "executor", "tool.ainvoke"),  # MCP configured-tool execution, not an LLM
    ("ragtime/userspace/service.py", "_invoke_runtime_bridge_tool", "runtime_tool.ainvoke"),  # runtime bridge tool dispatch
    ("ragtime/rag/components.py", "invoke", "self.ainvoke"),
    ("ragtime/rag/components.py", "_get_context_from_retrievers_async", "retriever.ainvoke"),
    # Index-description generation belongs to indexing, not Chat or User Space.
    ("ragtime/indexer/service.py", "generate_index_description", "llm.ainvoke"),
}

# The classifier has no agent/tools path and only returns the strict verdict schema.
SECURITY_CLASSIFICATION_PROVIDER_BOUNDARIES = {
    ("ragtime/content_protection/provider.py", "classify"),
    ("ragtime/content_protection/provider.py", "_run"),
}


def _function_calls(path: str, function_name: str) -> set[str]:
    """Return qualified call names inside one concrete function implementation."""
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name)
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


class _ModelCallVisitor(ast.NodeVisitor):
    """Collect model-shaped calls with their innermost function scope."""

    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.function_stack and isinstance(node.func, ast.Attribute) and node.func.attr in MODEL_CALL_NAMES:
            self.calls.append((self.function_stack[-1], ast.unparse(node.func)))
        self.generic_visit(node)


def _all_first_party_model_calls() -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    for source_path in sorted((ROOT / "ragtime").rglob("*.py")):
        relative_path = source_path.relative_to(ROOT).as_posix()
        source = source_path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # The local verifier is Python 3.9 while application sources may
            # use newer syntax. Never silently skip a model-shaped call.
            if re.search(r"\.(?:ainvoke|astream|astream_events)\s*\(", source):
                raise AssertionError(f"Cannot inventory model call in {relative_path} with this Python version")
            continue
        visitor = _ModelCallVisitor()
        visitor.visit(tree)
        calls.extend((relative_path, function_name, expression) for function_name, expression in visitor.calls)
    return calls


class GenerationGateInventoryTests(unittest.TestCase):
    def test_user_message_and_queued_task_gates_are_explicit(self) -> None:
        self.assertIn(
            "require_generation",
            _function_calls("ragtime/indexer/routes.py", "_validate_generation_ready_after_user_message"),
        )
        self.assertIn("require_generation", _function_calls("ragtime/indexer/background_tasks.py", "run"))
        self.assertIn("generation_context", _function_calls("ragtime/indexer/background_tasks.py", "run_with_policy_context"))

    def test_provider_and_auxiliary_generation_boundaries_are_gated(self) -> None:
        inventory = {
            ("ragtime/indexer/title_generation.py", "_generate_title"),
            ("ragtime/indexer/visualization_retry.py", "_repair_with_ai"),
            ("ragtime/rag/components.py", "process_query"),
            ("ragtime/rag/components.py", "process_query_stream"),
            ("ragtime/rag/components.py", "_run_nonstream_tool_skill_stage_loop"),
            ("ragtime/rag/components.py", "summarize_for_compaction"),
            ("ragtime/rag/components.py", "_stream_llm_chunks_with_transient_retries"),
            ("ragtime/rag/components.py", "_describe_image_part_for_compaction"),
        }
        for path, function_name in inventory:
            with self.subTest(path=path, function=function_name):
                self.assertIn(
                    "require_generation",
                    _function_calls(path, function_name),
                )

    def test_v1_binds_its_trusted_independent_surface_before_provider_execution(self) -> None:
        self.assertIn("generation_context", _function_calls("ragtime/api/routes.py", "chat_completions"))

    def test_security_classifier_is_the_only_provider_gate_exception(self) -> None:
        self.assertEqual(
            SECURITY_CLASSIFICATION_PROVIDER_BOUNDARIES,
            {
                ("ragtime/content_protection/provider.py", "classify"),
                ("ragtime/content_protection/provider.py", "_run"),
            },
        )

    def test_ast_inventory_classifies_every_model_boundary(self) -> None:
        """Every first-party model call is gated or explicitly proven non-generative."""
        calls = _all_first_party_model_calls()
        self.assertGreater(len(calls), 0)
        for path, function_name, expression in calls:
            if (path, function_name) in SECURITY_CLASSIFICATION_PROVIDER_BOUNDARIES:
                continue
            if (path, function_name, expression) in NON_GENERATIVE_CALL_EXEMPTIONS:
                continue
            with self.subTest(path=path, function=function_name, expression=expression):
                self.assertIn(
                    "require_generation",
                    _function_calls(path, function_name),
                    msg=f"Unclassified model boundary: {path}:{function_name}:{expression}",
                )

    def test_inventory_visitor_detects_arbitrary_new_module_call_shape(self) -> None:
        visitor = _ModelCallVisitor()
        visitor.visit(ast.parse("async def generate():\n    await llm.ainvoke([])\n"))
        self.assertEqual(visitor.calls, [("generate", "llm.ainvoke")])

    def test_external_agent_access_stays_available_but_build_entrypoint_is_userspace_gated(self) -> None:
        self.assertNotIn("require_userspace_generation", _function_calls("ragtime/userspace/agent_access.py", "resolve_agent_access_token"))
        self.assertIn("require_userspace_generation", _function_calls("ragtime/userspace/build_task_service.py", "start_build_task"))
