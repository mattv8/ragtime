"""
Tree-sitter based extraction of code metadata (imports, definitions).

This module provides robust, non-heuristic extraction of imports and top-level
definitions from source code using Tree-sitter parsers. It replaces the
fragile regex/string parsing used previously.

Supported languages include:
- Python
- JavaScript / TypeScript
- Go
- Rust
- Java
- PHP
"""

from typing import Dict, List

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

from ragtime.core.file_constants import LANG_MAPPING
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Tree-sitter queries for extracting metadata
# format: language -> { type -> query_string }
QUERIES: Dict[str, Dict[str, str]] = {
    "python": {
        "imports": """
            (import_statement) @import
            (import_from_statement) @import
        """,
        "definitions": """
            (function_definition name: (identifier) @name)
            (class_definition name: (identifier) @name)
        """,
    },
    "javascript": {
        "imports": """
            (import_statement) @import
        """,
        "definitions": """
            (function_declaration name: (identifier) @name)
            (class_declaration name: (identifier) @name)
        """,
    },
    "typescript": {
        "imports": """
            (import_statement) @import
        """,
        "definitions": """
            (function_declaration name: (identifier) @name)
            (class_declaration name: (type_identifier) @name)
            (interface_declaration name: (type_identifier) @name)
            (type_alias_declaration name: (type_identifier) @name)
        """,
    },
    "tsx": {
        "imports": """
            (import_statement) @import
        """,
        "definitions": """
            (function_declaration name: (identifier) @name)
            (class_declaration name: (type_identifier) @name)
            (interface_declaration name: (type_identifier) @name)
        """,
    },
    "go": {
        "imports": """
            (import_declaration) @import
        """,
        "definitions": """
            (function_declaration name: (identifier) @name)
            (type_declaration (type_spec name: (type_identifier) @name))
        """,
    },
    "rust": {
        "imports": """
            (use_declaration) @import
        """,
        "definitions": """
            (function_item name: (identifier) @name)
            (struct_item name: (type_identifier) @name)
            (enum_item name: (type_identifier) @name)
            (trait_item name: (type_identifier) @name)
            (impl_item type: (type_identifier) @name)
        """,
    },
    "java": {
        "imports": """
            (import_declaration) @import
        """,
        "definitions": """
            (class_declaration name: (identifier) @name)
            (interface_declaration name: (identifier) @name)
            (enum_declaration name: (identifier) @name)
            (method_declaration name: (identifier) @name)
        """,
    },
    "php": {
        "imports": """
            (namespace_use_declaration) @import
        """,
        "definitions": """
            (function_definition name: (name) @name)
            (class_declaration name: (name) @name)
            (interface_declaration name: (name) @name)
            (trait_declaration name: (name) @name)
        """,
    },
    "ruby": {
        "imports": """
            (call) @import
        """,
        "definitions": """
            (method name: (identifier) @name)
            (class name: (constant) @name)
            (module name: (constant) @name)
        """,
    },
}


def _get_node_text(node, source_bytes: bytes) -> str:
    """Helper to get text from a node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def extract_metadata(text: str, file_ext: str) -> tuple[List[str], List[str]]:
    """
    Extract imports and definitions from source code using Tree-sitter.

    Args:
        text: Source code content
        file_ext: File extension (e.g., '.py')

    Returns:
        Tuple of (imports list, definitions list)
    """
    lang_name = LANG_MAPPING.get(file_ext.lower())
    if not lang_name:
        return [], []

    try:
        # Get parser and language
        parser = get_parser(lang_name)
        language = get_language(lang_name)
    except Exception as e:
        logger.debug(f"Failed to load tree-sitter parser for {lang_name}: {e}")
        return [], []

    try:
        source_bytes = text.encode("utf-8")
        tree = parser.parse(source_bytes)
        root_node = tree.root_node

        imports = []
        definitions = []

        # Get queries for this language
        lang_queries = QUERIES.get(lang_name, {})

        # Extract Imports using QueryCursor (tree-sitter 0.25+ API)
        if "imports" in lang_queries:
            try:
                query = Query(language, lang_queries["imports"])
                cursor = QueryCursor(query)
                matches = cursor.matches(root_node)
                seen_imports = set()
                for _pattern_idx, captures in matches:
                    for capture_name, nodes in captures.items():
                        if capture_name == "import":
                            for node in nodes:
                                import_text = _get_node_text(node, source_bytes).strip()
                                # Normalize whitespace
                                import_text = " ".join(import_text.split())
                                if import_text and import_text not in seen_imports:
                                    imports.append(import_text)
                                    seen_imports.add(import_text)
            except Exception as e:
                logger.debug(f"Error extracting imports for {lang_name}: {e}")

        # Extract Definitions using QueryCursor
        if "definitions" in lang_queries:
            try:
                query = Query(language, lang_queries["definitions"])
                cursor = QueryCursor(query)
                matches = cursor.matches(root_node)
                seen_defs = set()
                for _pattern_idx, captures in matches:
                    for capture_name, nodes in captures.items():
                        if capture_name == "name":
                            for node in nodes:
                                # For definitions, we captured the 'name' node
                                # Extract the first line of the parent definition
                                parent = node.parent
                                if parent:
                                    parent_text = _get_node_text(parent, source_bytes)
                                    sig_line = parent_text.split("\n")[0].strip()
                                    if len(sig_line) > 100:
                                        sig_line = sig_line[:97] + "..."

                                    if sig_line and sig_line not in seen_defs:
                                        definitions.append(sig_line)
                                        seen_defs.add(sig_line)
            except Exception as e:
                logger.debug(f"Error extracting definitions for {lang_name}: {e}")

        return imports, definitions

    except Exception as e:
        logger.warning(f"Tree-sitter extraction failed for {file_ext}: {e}")
        return [], []
