#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import shutil
import subprocess
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class FileReport:
    path: Path
    inline_count: int
    hoisted_count: int
    inline_promoted_count: int
    inline_kept_count: int
    skipped_guarded: int
    tail_duplicates_removed: int
    rewritten: bool


@dataclass
class ModuleIndex:
    module_for_path: dict[Path, str]
    path_for_module: dict[str, Path]
    module_names: set[str]
    local_import_graph: dict[str, set[str]]


_DUPLICATE_TAIL_LINE_RE = re.compile(r"(?P<line>[^\n]*\S[^\n]*)(?:\n(?P=line)){1,2}\n?\Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote safe inline Python imports, then delegate import organization to Ruff.")
    parser.add_argument(
        "root",
        nargs="?",
        default="ragtime",
        help="Directory to scan (defaults to 'ragtime').",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in-place. Defaults to a dry-run.",
    )
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Fail if any safe-to-hoist inline imports are found.",
    )
    parser.add_argument(
        "--promote-inline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Promote inline imports by default, with safety checks. Use --no-promote-inline to only report them."),
    )
    parser.add_argument(
        "--protect-local-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Keep inline local imports when promoting them would create/activate a local module cycle. Disable with --no-protect-local-cycles."),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Path '{root}' does not exist.")

    python_files = sorted(p for p in root.rglob("*.py") if p.is_file())
    module_index = build_module_index(root, python_files)
    reports: list[FileReport] = []

    for path in python_files:
        try:
            report = process_file(
                path,
                apply_changes=args.apply,
                promote_inline=args.promote_inline,
                protect_local_cycles=args.protect_local_cycles,
                module_index=module_index,
            )
            if report:
                reports.append(report)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"Error processing {path}: {e}")
            raise SystemExit(1) from e
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not reports:
        print("No files needed updates.")
        return

    total_inline = sum(r.inline_count for r in reports)
    total_hoisted = sum(r.hoisted_count for r in reports)
    total_inline_promoted = sum(r.inline_promoted_count for r in reports)
    total_inline_kept = sum(r.inline_kept_count for r in reports)
    total_skipped = sum(r.skipped_guarded for r in reports)
    total_tail_duplicates_removed = sum(r.tail_duplicates_removed for r in reports)
    rewritten = sum(1 for r in reports if r.rewritten)

    for report in reports:
        status = "UPDATED" if report.rewritten else "DRY-RUN"
        print(
            f"[{status}] {report.path}: inline detected={report.inline_count}, "
            f"inline promoted={report.inline_promoted_count}, inline kept={report.inline_kept_count}, "
            f"hoisted import lines={report.hoisted_count}, skipped guarded={report.skipped_guarded}, "
            f"tail duplicates removed={report.tail_duplicates_removed}."
        )

    mode = "applied" if args.apply else "dry-run"
    print(
        f"\nSummary ({mode}): Processed {len(reports)} files ({rewritten} rewritten). "
        f"Detected {total_inline} inline imports. "
        f"Promoted {total_inline_promoted}, kept {total_inline_kept}, "
        f"skipped guarded {total_skipped}, hoisted {total_hoisted} import lines, "
        f"removed {total_tail_duplicates_removed} duplicate tail line(s)."
    )

    if args.check:
        if total_inline_promoted > 0:
            print(
                "\nInline import check failed: one or more files contain safe-to-hoist inline imports. "
                "Run `python3 docker/scripts/fix_inline_imports.py <root> --apply` for each reported root, "
                "then commit the resulting changes."
            )
            raise SystemExit(1)
        print("Inline import check passed: no safe-to-hoist inline imports found.")
    elif not args.apply:
        print("Run again with --apply to rewrite files.")


def process_file(
    path: Path,
    apply_changes: bool,
    promote_inline: bool,
    protect_local_cycles: bool,
    module_index: ModuleIndex,
) -> FileReport | None:
    original_text = path.read_text(encoding="utf-8")
    text, tail_duplicates_removed = collapse_duplicate_tail_lines(original_text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        if tail_duplicates_removed == 0:
            return None

        rewritten = False
        if apply_changes and text != original_text:
            path.write_text(text, encoding="utf-8")
            rewritten = True

        return FileReport(
            path=path,
            inline_count=0,
            hoisted_count=0,
            inline_promoted_count=0,
            inline_kept_count=0,
            skipped_guarded=0,
            tail_duplicates_removed=tail_duplicates_removed,
            rewritten=rewritten,
        )

    parent_map = build_parent_map(tree)
    lines = text.splitlines(keepends=True)
    current_module = module_index.module_for_path.get(path)

    # Identify inline imports to hoist. Ruff owns the final import ordering.
    top_level_sources = {source for stmt in tree.body if isinstance(stmt, (ast.Import, ast.ImportFrom)) if (source := normalize_block(lines, stmt))}
    hoisted_sources: list[str] = []
    seen_hoisted_sources: set[str] = set()
    removal_spans: List[Tuple[int, int]] = []
    skipped_count = 0
    inline_found_count = 0
    inline_promoted_count = 0
    inline_kept_count = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Check if guarded (inside try/except or if block)
            if is_guarded(node, parent_map):
                skipped_count += 1
                continue

            source = normalize_block(lines, node)
            if not source:
                continue

            inside_def = is_inside_definition(node, parent_map)
            if not inside_def:
                continue

            inline_found_count += 1
            if not should_promote_inline(
                node,
                parent_map,
                promote_inline=promote_inline,
                protect_local_cycles=protect_local_cycles,
                current_module=current_module,
                module_index=module_index,
            ):
                inline_kept_count += 1
                continue
            inline_promoted_count += 1

            if source not in top_level_sources and source not in seen_hoisted_sources:
                hoisted_sources.append(source)
                seen_hoisted_sources.add(source)

            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None:
                end = end if end is not None else start
                removal_spans.append((start - 1, end))

    if not removal_spans and tail_duplicates_removed == 0:
        return None

    new_text = text
    if removal_spans:
        updated_lines = remove_spans(lines, removal_spans)

        real_insert_idx = find_insertion_idx_in_lines(updated_lines)
        while real_insert_idx < len(updated_lines) and not updated_lines[real_insert_idx].strip():
            real_insert_idx += 1

        new_import_block = [f"{source}\n" for source in hoisted_sources]
        if new_import_block and real_insert_idx < len(updated_lines) and updated_lines[real_insert_idx].strip():
            new_import_block.append("\n")

        final_lines = updated_lines[:real_insert_idx] + new_import_block + updated_lines[real_insert_idx:]
        new_text = "".join(final_lines)

    rewritten = False
    if apply_changes and new_text != original_text:
        path.write_text(new_text, encoding="utf-8")
        if removal_spans:
            run_ruff_on_file(path)
        rewritten = path.read_text(encoding="utf-8") != original_text

    return FileReport(
        path=path,
        inline_count=inline_found_count,
        hoisted_count=len(hoisted_sources),
        inline_promoted_count=inline_promoted_count,
        inline_kept_count=inline_kept_count,
        skipped_guarded=skipped_count,
        tail_duplicates_removed=tail_duplicates_removed,
        rewritten=rewritten,
    )


def should_promote_inline(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    *,
    promote_inline: bool,
    protect_local_cycles: bool,
    current_module: str | None,
    module_index: ModuleIndex,
) -> bool:
    if not promote_inline:
        return False

    # Never hoist imports from class bodies; they often intentionally control symbol scope.
    if is_inside_class_definition(node, parent_map):
        return False

    # Protect likely cycle-breaker imports: local inline imports that would
    # participate in a local import cycle if promoted to module top-level.
    if protect_local_cycles and current_module:
        local_targets = resolve_local_targets(node, current_module, module_index)
        for target in local_targets:
            if has_path(module_index.local_import_graph, target, current_module):
                return False

    return True


def build_module_index(root: Path, python_files: list[Path]) -> ModuleIndex:
    module_for_path: dict[Path, str] = {}
    path_for_module: dict[str, Path] = {}

    for path in python_files:
        module = module_name_from_path(root, path)
        if not module:
            continue
        module_for_path[path] = module
        path_for_module[module] = path

    graph: dict[str, set[str]] = {module: set() for module in path_for_module.keys()}
    module_names = set(path_for_module.keys())

    for path, module in module_for_path.items():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            for target in resolve_local_targets(
                stmt,
                module,
                ModuleIndex(
                    module_for_path=module_for_path,
                    path_for_module=path_for_module,
                    module_names=module_names,
                    local_import_graph={},
                ),
                module_names,
            ):
                graph[module].add(target)

    return ModuleIndex(
        module_for_path=module_for_path,
        path_for_module=path_for_module,
        module_names=module_names,
        local_import_graph=graph,
    )


def module_name_from_path(root: Path, path: Path) -> str:
    """Return an importable-looking module path for *path*.

    For package files, prefer the fully qualified package name by walking
    parent directories that contain ``__init__.py``. This makes cycle
    detection line up with real imports like ``ragtime.userspace.service``.

    For non-package scripts, fall back to a stable name relative to the scan
    root so they can still participate in de-dup/sort bookkeeping.
    """
    if path.suffix != ".py":
        return ""

    stem = path.stem
    package_parts: list[str] = []
    current_dir = path.parent
    while (current_dir / "__init__.py").is_file():
        package_parts.insert(0, current_dir.name)
        current_dir = current_dir.parent

    if package_parts:
        if stem != "__init__":
            package_parts.append(stem)
        return ".".join(package_parts)

    try:
        rel = path.relative_to(root)
    except ValueError:
        return stem

    parts = list(rel.parts)
    if not parts:
        return stem
    filename = parts[-1]
    file_stem = filename[:-3]
    if file_stem == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = file_stem
    cleaned = [part for part in parts if part]
    return ".".join(cleaned) if cleaned else stem


def resolve_local_targets(
    node: ast.AST,
    current_module: str,
    module_index: ModuleIndex,
    module_names: set[str] | None = None,
) -> set[str]:
    names = module_names or set(module_index.path_for_module.keys())
    targets: set[str] = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            imported = alias.name
            target = resolve_known_module(imported, names)
            if target:
                targets.add(target)
    elif isinstance(node, ast.ImportFrom):
        level = node.level or 0
        module_part = node.module or ""
        base = resolve_relative_base(current_module, level)
        if base is None:
            return targets
        if module_part:
            imported = module_part if level == 0 else (f"{base}.{module_part}" if base else module_part)
            target = resolve_known_module(imported, names)
            if target:
                targets.add(target)
        else:
            # from . import name
            for alias in node.names:
                imported = f"{base}.{alias.name}" if base else alias.name
                target = resolve_known_module(imported, names)
                if target:
                    targets.add(target)
    return targets


def resolve_relative_base(current_module: str, level: int) -> str | None:
    if level == 0:
        return ""
    package_parts = current_module.split(".")[:-1]
    if level - 1 > len(package_parts):
        return None
    remaining = package_parts[: len(package_parts) - (level - 1)]
    return ".".join(remaining)


def resolve_known_module(module: str, module_names: set[str]) -> str | None:
    if module in module_names:
        return module
    candidate = f"{module}.__init__"
    if candidate in module_names:
        return candidate
    prefix = f"{module}."
    for name in module_names:
        if name.startswith(prefix):
            return module
    return None


def has_path(graph: dict[str, set[str]], start: str, target: str) -> bool:
    if start == target:
        return True
    seen: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for nxt in graph.get(current, set()):
            if nxt == target:
                return True
            if nxt not in seen:
                queue.append(nxt)
    return False


def run_ruff_on_file(path: Path) -> None:
    ruff_command = get_ruff_command()
    subprocess.run([*ruff_command, "check", "--select", "I", "--fix", str(path)], check=True)
    subprocess.run([*ruff_command, "format", str(path)], check=True)


def get_ruff_command() -> list[str]:
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]

    ruff_executable = shutil.which("ruff")
    if ruff_executable:
        return [ruff_executable]

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    for venv_name in (".venv", ".venv-py312"):
        candidate = repo_root / venv_name / scripts_dir / "ruff"
        if candidate.is_file():
            return [str(candidate)]

    raise RuntimeError("Ruff is required when inline imports are promoted. Install the test dependencies or put `ruff` on PATH.")


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def is_inside_definition(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    current = parent_map.get(node)
    while current:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        current = parent_map.get(current)
    return False


def is_inside_class_definition(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    current = parent_map.get(node)
    while current:
        if isinstance(current, ast.ClassDef):
            return True
        current = parent_map.get(current)
    return False


def is_guarded(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    current = parent_map.get(node)
    while current:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(current, ast.Try):
            return True
        if isinstance(current, ast.If):
            return True
        current = parent_map.get(current)
    return False


def normalize_block(lines: Sequence[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None:
        return ""
    end = end if end is not None else start
    segment = lines[start - 1 : end]
    block = "".join(segment)
    dedented = textwrap.dedent(block)
    return dedented.strip()


def remove_spans(lines: Sequence[str], spans: Sequence[Tuple[int, int]]) -> list[str]:
    # spans are 0-based start, 1-based end logic from ast line numbers conversion
    # AST lineno is 1-based. Our tuples are (start-1, end).
    keep = [True] * len(lines)
    for start, end in spans:
        # Remove the import lines
        for i in range(start, end):
            if i < len(keep):
                keep[i] = False

        # Aggressively remove ONE trailing blank line if it exists.
        # This fixes the "gap" left when an import that was separated from code
        # by a blank line is removed.
        if end < len(lines) and lines[end].strip() == "":
            keep[end] = False

    return [line for i, line in enumerate(lines) if keep[i]]


def collapse_duplicate_tail_lines(text: str) -> tuple[str, int]:
    match = _DUPLICATE_TAIL_LINE_RE.search(text)
    if not match:
        return text, 0

    block = match.group(0)
    line = match.group("line")
    duplicates_removed = len(block.rstrip("\n").split("\n")) - 1
    cleaned = text[: match.start()] + line
    if block.endswith("\n"):
        cleaned += "\n"
    return cleaned, duplicates_removed


def find_insertion_idx_in_lines(lines: List[str]) -> int:
    idx = 0

    # Skip shebang/comments/blank lines at top.
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("#"):
            idx += 1
            continue
        break

    # Detect module docstring in cleaned content.
    if idx < len(lines):
        header_text = "".join(lines)
        try:
            tree = ast.parse(header_text)
            if tree.body:
                first = tree.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    end_lineno = first.end_lineno if first.end_lineno else 0
                    return end_lineno
        except SyntaxError:
            pass

    # Fallback: after shebang/comments section.
    return idx


if __name__ == "__main__":
    main()
