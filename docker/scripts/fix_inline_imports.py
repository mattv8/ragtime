#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

# Attempt to get stdlib modules for the running python version
try:
    STDLIB_MODULES = sys.stdlib_module_names
except AttributeError:
    # Fallback for older python (though environment is 3.10+)
    STDLIB_MODULES = frozenset(
        {
            "abc",
            "argparse",
            "ast",
            "asyncio",
            "base64",
            "collections",
            "concurrent",
            "contextlib",
            "copy",
            "csv",
            "dataclasses",
            "datetime",
            "decimal",
            "difflib",
            "email",
            "enum",
            "functools",
            "hashlib",
            "hmac",
            "html",
            "http",
            "importlib",
            "inspect",
            "io",
            "json",
            "logging",
            "math",
            "mimetypes",
            "multiprocessing",
            "os",
            "pathlib",
            "pickle",
            "pkgutil",
            "platform",
            "pprint",
            "random",
            "re",
            "shlex",
            "shutil",
            "signal",
            "socket",
            "sqlite3",
            "ssl",
            "stat",
            "string",
            "subprocess",
            "sys",
            "tempfile",
            "textwrap",
            "threading",
            "time",
            "token",
            "tokenize",
            "traceback",
            "types",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "warnings",
            "weakref",
            "xml",
            "zipfile",
            "zoneinfo",
        }
    )


@dataclass
class FileReport:
    path: Path
    inline_count: int
    promoted_count: int
    inline_promoted_count: int
    inline_kept_count: int
    skipped_guarded: int
    rewritten: bool


@dataclass
class ImportItem:
    node: ast.AST
    category: int  # 0=Future, 1=StdLib, 2=ThirdParty, 3=Local
    import_type: int  # 0=import, 1=from
    source: str
    module_name: str


@dataclass
class ModuleIndex:
    module_for_path: dict[Path, str]
    path_for_module: dict[str, Path]
    local_import_graph: dict[str, set[str]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote inline Python imports and organize top-level imports."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="ragtime",
        help="Directory to scan (defaults to 'ragtime').",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in-place. Defaults to a dry-run.",
    )
    parser.add_argument(
        "--promote-inline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Promote inline imports by default, with safety checks. "
            "Use --no-promote-inline to only normalize top-level imports."
        ),
    )
    parser.add_argument(
        "--protect-local-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep inline local imports when promoting them would create/activate a local module cycle. "
            "Disable with --no-protect-local-cycles."
        ),
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
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not reports:
        print("No files needed updates.")
        return

    total_inline = sum(r.inline_count for r in reports)
    total_promoted = sum(r.promoted_count for r in reports)
    total_inline_promoted = sum(r.inline_promoted_count for r in reports)
    total_inline_kept = sum(r.inline_kept_count for r in reports)
    total_skipped = sum(r.skipped_guarded for r in reports)
    rewritten = sum(1 for r in reports if r.rewritten)

    for report in reports:
        status = "UPDATED" if report.rewritten else "DRY-RUN"
        print(
            f"[{status}] {report.path}: inline detected={report.inline_count}, "
            f"inline promoted={report.inline_promoted_count}, inline kept={report.inline_kept_count}, "
            f"final import lines={report.promoted_count}, skipped guarded={report.skipped_guarded}."
        )

    mode = "applied" if args.apply else "dry-run"
    print(
        f"\nSummary ({mode}): Processed {len(reports)} files ({rewritten} rewritten). "
        f"Detected {total_inline} inline imports. "
        f"Promoted {total_inline_promoted}, kept {total_inline_kept}, "
        f"skipped guarded {total_skipped}, organized into {total_promoted} top-level import lines."
    )
    if not args.apply:
        print("Run again with --apply to rewrite files.")


def process_file(
    path: Path,
    apply_changes: bool,
    promote_inline: bool,
    protect_local_cycles: bool,
    module_index: ModuleIndex,
) -> FileReport | None:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    parent_map = build_parent_map(tree)
    lines = text.splitlines(keepends=True)

    # Identify imports to normalize. Top-level imports are always normalized.
    # Inline imports are only promoted when explicitly enabled and safe.
    movable_items: List[ImportItem] = []
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

            inside_def = is_inside_definition(node, parent_map)
            if inside_def:
                inline_found_count += 1
                if not should_promote_inline(
                    node,
                    parent_map,
                    promote_inline=promote_inline,
                    protect_local_cycles=protect_local_cycles,
                    current_path=path,
                    module_index=module_index,
                ):
                    inline_kept_count += 1
                    continue
                inline_promoted_count += 1

            # Prepare ImportItem
            source = normalize_block(lines, node)
            if not source:
                continue

            cat = categorize_import(node)
            imp_type = 0 if isinstance(node, ast.Import) else 1
            mod_name = get_module_name(node)

            movable_items.append(
                ImportItem(
                    node=node,
                    category=cat,
                    import_type=imp_type,
                    source=source,
                    module_name=mod_name,
                )
            )

            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None:
                end = end if end is not None else start
                removal_spans.append((start - 1, end))

    if not movable_items:
        return None

    # Remove the old import lines
    updated_lines = remove_spans(lines, removal_spans)

    # Deduplicate and Sort
    # Deduplicate by source string
    seen_sources = set()
    unique_items = []
    for item in movable_items:
        if item.source not in seen_sources:
            unique_items.append(item)
            seen_sources.add(item.source)

    # Sort: Category -> ImportType -> ModuleName -> Source (for stability)
    unique_items.sort(
        key=lambda x: (x.category, x.import_type, x.module_name, x.source)
    )

    # Build text blocks with spacing
    new_import_block: list[str] = []
    last_category = -1

    for item in unique_items:
        if last_category != -1 and item.category != last_category:
            new_import_block.append("\n")  # Blank line between groups
        new_import_block.append(f"{item.source}\n")
        last_category = item.category

    real_insert_idx = find_insertion_idx_in_lines(updated_lines)

    # If the insertion point points to blank lines, skip over them so they appear
    # BEFORE the imports (preserving separation from docstring/header).
    while (
        real_insert_idx < len(updated_lines)
        and not updated_lines[real_insert_idx].strip()
    ):
        real_insert_idx += 1

    final_lines = (
        updated_lines[:real_insert_idx]
        + new_import_block
        + updated_lines[real_insert_idx:]
    )

    # Ensure reasonable blank lines around block
    if real_insert_idx < len(updated_lines) and updated_lines[real_insert_idx].strip():
        final_lines.insert(real_insert_idx + len(new_import_block), "\n")

    new_text = "".join(final_lines)

    rewritten = False
    if apply_changes and new_text != text:
        path.write_text(new_text, encoding="utf-8")
        rewritten = True

    return FileReport(
        path=path,
        inline_count=inline_found_count,
        promoted_count=len(unique_items),
        inline_promoted_count=inline_promoted_count,
        inline_kept_count=inline_kept_count,
        skipped_guarded=skipped_count,
        rewritten=rewritten,
    )


def should_promote_inline(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    *,
    promote_inline: bool,
    protect_local_cycles: bool,
    current_path: Path,
    module_index: ModuleIndex,
) -> bool:
    if not promote_inline:
        return False

    # Never hoist imports from class bodies; they often intentionally control symbol scope.
    if is_inside_class_definition(node, parent_map):
        return False

    # Protect likely cycle-breaker imports: local inline imports that would
    # participate in a local import cycle if promoted to module top-level.
    if protect_local_cycles and categorize_import(node) == 3:
        current_module = module_index.module_for_path.get(current_path)
        if not current_module:
            return False
        local_targets = resolve_local_targets(node, current_module, module_index)
        if not local_targets:
            return False
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
                ModuleIndex(module_for_path, path_for_module, {}),
                module_names,
            ):
                graph[module].add(target)

    return ModuleIndex(
        module_for_path=module_for_path,
        path_for_module=path_for_module,
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
            imported = (
                module_part
                if level == 0
                else (f"{base}.{module_part}" if base else module_part)
            )
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


def categorize_import(node: ast.AST) -> int:
    # 0=Future, 1=StdLib, 2=ThirdParty, 3=Local
    name = get_module_name(node)

    if name == "__future__":
        return 0

    base_module = name.split(".")[0]

    if base_module in STDLIB_MODULES:
        return 1

    if name.startswith("."):
        return 3

    if base_module == "ragtime":
        return 3

    # Heuristic: if it's not stdlib and not ragged, it's 3rd party
    return 2


def get_module_name(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return node.module
        # Relative import e.g. "from . import x"
        return "." * node.level
    elif isinstance(node, ast.Import):
        # Sort by first alias
        if node.names:
            return node.names[0].name
    return ""


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


def is_inside_class_definition(
    node: ast.AST, parent_map: dict[ast.AST, ast.AST]
) -> bool:
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
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    end_lineno = first.end_lineno if first.end_lineno else 0
                    return end_lineno
        except SyntaxError:
            pass

    # Fallback: after shebang/comments section.
    return idx


if __name__ == "__main__":
    main()
