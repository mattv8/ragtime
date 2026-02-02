#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

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
    skipped_guarded: int
    rewritten: bool


@dataclass
class ImportItem:
    node: ast.AST
    category: int  # 0=Future, 1=StdLib, 2=ThirdParty, 3=Local
    import_type: int  # 0=import, 1=from
    source: str
    module_name: str


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
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Path '{root}' does not exist.")

    python_files = sorted(p for p in root.rglob("*.py") if p.is_file())
    reports: list[FileReport] = []

    for path in python_files:
        try:
            report = process_file(path, apply_changes=args.apply)
            if report:
                reports.append(report)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not reports:
        print("No files needed updates.")
        return

    total_inline = sum(r.inline_count for r in reports)
    total_promoted = sum(r.promoted_count for r in reports)
    total_skipped = sum(r.skipped_guarded for r in reports)
    rewritten = sum(1 for r in reports if r.rewritten)

    for report in reports:
        status = "UPDATED" if report.rewritten else "DRY-RUN"
        print(
            f"[{status}] {report.path}: merged {report.inline_count} inline + top-level imports "
            f"(total {report.promoted_count} lines, skipped {report.skipped_guarded} guarded)."
        )

    mode = "applied" if args.apply else "dry-run"
    print(
        f"\nSummary ({mode}): Processed {len(reports)} files ({rewritten} rewritten). "
        f"Detected {total_inline} inline/movable imports. "
        f"Organized into {total_promoted} top-level import lines."
    )
    if not args.apply:
        print("Run again with --apply to rewrite files.")


def process_file(path: Path, apply_changes: bool) -> FileReport | None:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    parent_map = build_parent_map(tree)
    lines = text.splitlines(keepends=True)

    # Identify all movable imports (inline + existing top-level)
    movable_items: List[ImportItem] = []
    removal_spans: List[Tuple[int, int]] = []
    skipped_count = 0
    inline_found_count = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Check if guarded (inside try/except or if block)
            if is_guarded(node, parent_map):
                skipped_count += 1
                continue

            # Identify if inline
            if is_inside_definition(node, parent_map):
                inline_found_count += 1

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

    if not movable_items and inline_found_count == 0:
        return None

    # If we only have top-level imports and they are already sorted/clean, we might not need to touch the file.
    # But for now, we assume if we found ANY movable imports, we process the file to enforce the sorting.
    # To avoid noise, we could check if rewriting actually changes anything.

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
    new_import_block = []
    last_category = -1

    for item in unique_items:
        if last_category != -1 and item.category != last_category:
            new_import_block.append("\n")  # Blank line between groups
        new_import_block.append(f"{item.source}\n")
        last_category = item.category

    # Insert at correct location
    insert_idx = find_insertion_line(tree)

    # If the removal made the line count shorter,  (based on old tree)
    # might effectively point to garbage if we strictly index into .
    # However,  removes lines but preserves validity of indices for *surviving* lines
    # relative to the structure (top is still top).
    # Actually,  returns a line number from the *original* file.
    # Since we removed lines *before* insertion point (if any top-level imports were before,
    # e.g. future imports we are moving), we need to be careful?

    # Wait:  looks for Docstrings and logic *before* imports.
    # Standard top-level imports usually come *after* insertion point.
    # If we remove them,  shrinks.
    # But  is just a list of strings now.

    # The header (docstring) is preserved in .
    # We just need to find where the header ends in .
    # Since we removed content, the line number  from ORIGINAL tree is risky
    # if we removed lines *before* it.
    # But imports generally don't appear before module docstring.
    # However,  does.
    # If we move ,  (which skips future imports in calculation)
    # refers to line AFTER future imports.
    # If we remove future imports from , the  is now too large
    # relative to the content (by the number of removed future lines).

    # Better strategy: logic to re-find insertion point on .
    # Since imports are gone, the insertion point is:
    # After the shebang/encoding.
    # After the docstring.
    # Before the first real code.
    # Since we removed all imports, "first real code" is the first non-blank line
    # that survived.

    real_insert_idx = find_insertion_idx_in_lines(updated_lines)

    # If the insertion point points to blank lines, skip over them so they appear 
    # BEFORE the imports (preserving separation from docstring/header).
    while real_insert_idx < len(updated_lines) and not updated_lines[real_insert_idx].strip():
        real_insert_idx += 1

    final_lines = (
        updated_lines[:real_insert_idx]
        + new_import_block
        + updated_lines[real_insert_idx:]
    )

    # Ensure reasonable blank lines around block
    # Check lines before and after
    # (implied by just pasting it in,  might need leading/trailing newlines
    # if surrounding context is tight, but usually 1 blank line is standard/PEP8).
    # We added blank lines *between* groups.
    # We might want a blank line *after* the whole block if code follows.
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
        skipped_guarded=skipped_count,
        rewritten=rewritten,
    )


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


def find_insertion_line(tree: ast.AST) -> int:
    # Just for reference if valid
    return 0


def find_insertion_idx_in_lines(lines: List[str]) -> int:
    # Skip shebangs and comments at top
    # But wait, we removed imports.
    # So we just want to skip:
    # 1. Shebang
    # 2. Encoding
    # 3. Module Docstring (quoted string)
    # The rest should be imports (now removed) or code.

    idx = 0

    # 1. Skip hashbang/comments at very top
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("#"):
            idx += 1
            continue
        break

    # 2. Check for docstring
    # Naive check: if line starts with quotes """ or '''
    # This is tricky without AST.
    # But we can try to re-parse the *header* part.

    # Simpler:
    # Use AST of the *original* file to find END of docstring.
    # Convert that line number to index in *updated_lines*?
    # No, line numbers shifted.

    # Let's try to detect docstring in updated_lines.
    if idx < len(lines):
        header_text = "".join(lines)
        try:
            # Parse the cleaned text to find the first statement
            tree = ast.parse(header_text)
            if tree.body:
                first = tree.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    # It's a docstring
                    end_lineno = first.end_lineno if first.end_lineno else 0
                    # Return line after docstring
                    return end_lineno
        except SyntaxError:
            pass

    # Fallback: index 0 or after shebang
    # If we found shebang/comments loop above, idx is start of code.
    return idx


if __name__ == "__main__":
    main()
