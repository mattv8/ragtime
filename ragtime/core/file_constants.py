"""
Centralized file type constants for indexing operations.

These constants are shared between:
- Document indexer (FAISS) for archive/git repo analysis
- Filesystem indexer for directory scanning
- Analysis endpoints for pre-indexing estimates
"""

# =============================================================================
# UNPARSEABLE BINARY EXTENSIONS
# =============================================================================
# Truly binary files that cannot be parsed as text by any handler.
# These are ALWAYS skipped - no user override possible.
UNPARSEABLE_BINARY_EXTENSIONS: set[str] = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    # Raw camera formats (can be processed with OCR if rawpy is installed)
    ".cr2",  # Canon RAW 2
    ".cr3",  # Canon RAW 3
    ".nef",  # Nikon Electronic Format
    ".arw",  # Sony Alpha RAW
    ".dng",  # Adobe Digital Negative
    ".orf",  # Olympus RAW
    ".rw2",  # Panasonic RAW
    ".pef",  # Pentax Electronic File
    ".raf",  # Fujifilm RAW
    ".srw",  # Samsung RAW
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".xz",
    ".lz",
    ".lzma",
    # Executables and compiled
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".msi",
    ".dmg",
    ".app",
    # Compiled bytecode
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".obj",
    ".a",
    ".lib",
    # Media
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".flac",
    ".ogg",
    ".webm",
    ".mkv",
    ".m4a",
    ".m4v",
    # Databases
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mdb",
    ".accdb",
    # Development artifacts
    ".map",  # Source maps
    ".wasm",  # WebAssembly
    # Serialized data (binary)
    ".pickle",
    ".pkl",
    ".npy",
    ".npz",
    ".pt",  # PyTorch
    ".pth",  # PyTorch
    ".h5",
    ".hdf5",
    ".parquet",
    ".avro",
}

# =============================================================================
# PARSEABLE DOCUMENT EXTENSIONS
# =============================================================================
# Documents that require special parsers (PDF, Office, OpenDocument, etc.).
# Both the filesystem indexer and git/upload indexer can parse these using
# document_parser.py extractors.
PARSEABLE_DOCUMENT_EXTENSIONS: set[str] = {
    # Office documents
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # OpenDocument
    ".odt",
    ".ods",
    ".odp",
    # Rich text
    ".rtf",
    # Ebooks
    ".epub",
    # Email
    ".eml",
    ".msg",
}

# =============================================================================
# RAW CAMERA EXTENSIONS
# =============================================================================
# Raw camera formats that require rawpy for processing.
# These need special handling to convert to standard image formats.
RAW_CAMERA_EXTENSIONS: set[str] = {
    ".cr2",  # Canon RAW 2
    ".cr3",  # Canon RAW 3
    ".nef",  # Nikon Electronic Format
    ".arw",  # Sony Alpha RAW
    ".dng",  # Adobe Digital Negative
    ".orf",  # Olympus RAW
    ".rw2",  # Panasonic RAW
    ".pef",  # Pentax Electronic File
    ".raf",  # Fujifilm RAW
    ".srw",  # Samsung RAW
}

# =============================================================================
# OCR-CAPABLE EXTENSIONS
# =============================================================================
# Image formats that can have text extracted via OCR (when enabled).
# These are in UNPARSEABLE_BINARY_EXTENSIONS but can be processed with OCR.
# Includes standard image formats and raw camera formats (requires rawpy).
OCR_EXTENSIONS: set[str] = {
    # Standard image formats
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
} | RAW_CAMERA_EXTENSIONS  # Include raw camera formats

# =============================================================================
# NEVER SUGGEST EXCLUDE EXTENSIONS
# =============================================================================
# Extensions that should NEVER be suggested for exclusion by LLM or heuristics.
# These are valuable formats that all indexers can parse and are useful for RAG.
# Includes plain text formats and parseable document formats.
NEVER_SUGGEST_EXCLUDE_EXTENSIONS: set[str] = {
    # Plain text formats - universally readable and valuable
    ".txt",
    ".md",
    ".rst",
    ".csv",
    # Office documents - parsed by document_parser.py
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # OpenDocument formats
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    # Ebooks
    ".epub",
    # Email
    ".eml",
    ".msg",
}

# =============================================================================
# LEGACY ALIAS (deprecated - use specific constants above)
# =============================================================================
# For backward compatibility, BINARY_EXTENSIONS includes both unparseable and
# parseable documents. New code should use the specific constants.
BINARY_EXTENSIONS: set[str] = (
    UNPARSEABLE_BINARY_EXTENSIONS | PARSEABLE_DOCUMENT_EXTENSIONS
)

# =============================================================================
# LOW VALUE EXTENSIONS
# =============================================================================
# Files that are technically text but rarely useful for RAG search.
# User-controlled via suggestions (not auto-skipped).
LOW_VALUE_EXTENSIONS: set[str] = {
    ".lock",  # Lock files (package-lock.json, yarn.lock, etc.)
}

# Patterns indicating minified/generated content that's not useful for RAG
# These files are text but contain compressed/obfuscated code
MINIFIED_PATTERNS: list[str] = [
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.bundle.css",
    "*-min.js",
    "*-min.css",
    "*.chunk.js",
    "*.chunk.css",
    "*.min.map",
    "*.bundle.map",
]

# Common directories that should be excluded from indexing
DEFAULT_EXCLUDE_DIRS: list[str] = [
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/venv/**",
    "**/.venv/**",
    "**/.git/**",
    "**/dist/**",
    "**/build/**",
    "**/.next/**",
    "**/.nuxt/**",
    "**/coverage/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.tox/**",
    "**/vendor/**",
    "**/.cache/**",
]

# Default file patterns for indexing (common source/doc files)
DEFAULT_FILE_PATTERNS: list[str] = [
    "**/*.py",
    "**/*.md",
    "**/*.rst",
    "**/*.txt",
    "**/*.xml",
    "**/*.json",
    "**/*.yaml",
    "**/*.yml",
    "**/*.toml",
    "**/*.ini",
    "**/*.cfg",
    "**/*.js",
    "**/*.ts",
    "**/*.jsx",
    "**/*.tsx",
    "**/*.html",
    "**/*.css",
    "**/*.scss",
    "**/*.sass",
    "**/*.less",
    "**/*.sql",
    "**/*.sh",
    "**/*.bash",
    "**/*.zsh",
    "**/*.ps1",
    "**/*.go",
    "**/*.rs",
    "**/*.java",
    "**/*.kt",
    "**/*.scala",
    "**/*.c",
    "**/*.cpp",
    "**/*.h",
    "**/*.hpp",
    "**/*.cs",
    "**/*.rb",
    "**/*.php",
    "**/*.swift",
    "**/*.r",
    "**/*.R",
    "**/*.jl",
    "**/*.lua",
    "**/*.pl",
    "**/*.pm",
    "**/*.ex",
    "**/*.exs",
    "**/*.erl",
    "**/*.hrl",
    "**/*.elm",
    "**/*.vue",
    "**/*.svelte",
]


# =============================================================================
# CODE EXTENSIONS
# =============================================================================
# Language-specific extensions for code analysis and chunking.
PYTHON_EXTENSIONS: set[str] = {".py", ".pyi", ".pyx"}
JS_EXTENSIONS: set[str] = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
GO_EXTENSIONS: set[str] = {".go"}
RUST_EXTENSIONS: set[str] = {".rs"}
JAVA_EXTENSIONS: set[str] = {".java", ".kt", ".kts"}

# =============================================================================
# ALL DOCUMENT EXTENSIONS
# =============================================================================
# Comprehensive list of all supported document extensions including
# parseable documents, plain text, and code files.
DOCUMENT_EXTENSIONS: set[str] = (
    PARSEABLE_DOCUMENT_EXTENSIONS
    | PYTHON_EXTENSIONS
    | JS_EXTENSIONS
    | GO_EXTENSIONS
    | RUST_EXTENSIONS
    | JAVA_EXTENSIONS
    | {
        # Plain text
        ".txt",
        ".md",
        ".rst",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".csv",
        ".tsv",
        # Other Code
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
    }
)


# =============================================================================
# TREE-SITTER LANGUAGE MAPPING
# =============================================================================
# Unified mapping for both file extensions AND Magika content types to tree-sitter.
# This is the SINGLE SOURCE OF TRUTH for language mappings.
#
# Keys can be:
#   - File extensions (e.g., ".py", ".tsx")
#   - Filenames without extensions (e.g., "makefile", "dockerfile")
#   - Magika content types (e.g., "shell", "jinja", "objectivec")
#
# Values:
#   - tree-sitter language name (e.g., "python", "bash")
#   - None = use RecursiveChunker (plain text, no AST benefit)
#
# Note: 59+ Magika types auto-map because names match tree-sitter exactly
# (python, javascript, rust, go, etc.). Only add entries here for:
#   1. Non-obvious file extensions
#   2. Magika types that need translation to different tree-sitter names
#   3. Content that should use RecursiveChunker (None)
LANG_MAPPING: dict[str, str | None] = {
    # =========================================================================
    # FILE EXTENSIONS (non-obvious mappings only)
    # =========================================================================
    # Python variants
    ".pyi": "python",
    ".pyx": "python",
    # JavaScript variants
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "tsx",
    # C/C++ ambiguous headers
    ".h": "c",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hh": "cpp",
    # C#
    ".cs": "csharp",
    # Objective-C
    ".m": "objc",
    ".mm": "objc",
    # Web aliases
    ".htm": "html",
    # Template languages -> HTML
    ".j2": "html",
    ".jinja": "html",
    ".jinja2": "html",
    ".ejs": "html",
    ".hbs": "html",
    ".mustache": "html",
    ".njk": "html",
    ".liquid": "html",
    ".erb": "embeddedtemplate",
    # Config files
    ".yml": "yaml",
    ".conf": "ini",
    ".cfg": "ini",
    ".env": "properties",
    ".properties": "properties",
    ".editorconfig": "ini",
    # Shell variants
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".crontab": "bash",
    ".bashrc": "bash",
    ".zshrc": "bash",
    ".profile": "bash",
    # Build files
    ".mk": "make",
    ".dockerfile": "dockerfile",
    # Data formats
    ".jsonl": "json",
    ".ndjson": "json",
    # Documentation
    ".md": "markdown",
    ".markdown": "markdown",
    ".tex": "latex",
    ".bib": "bibtex",
    # Plain text -> RecursiveChunker
    ".txt": None,
    ".text": None,
    ".log": None,
    ".csv": None,
    ".tsv": None,
    # =========================================================================
    # FILENAMES (for extensionless files like Makefile)
    # =========================================================================
    "makefile": "make",
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
    # =========================================================================
    # MAGIKA CONTENT TYPES (when name differs from tree-sitter)
    # =========================================================================
    # Plain text / extracted content
    "txt": None,
    "unknown": None,
    "empty": None,
    # Binary formats that got parsed to text
    "pdf": None,
    "doc": None,
    "docx": None,
    "xls": None,
    "xlsx": None,
    "ppt": None,
    "pptx": None,
    "rtf": None,
    "jpeg": None,
    "png": None,
    "gif": None,
    "webp": None,
    "svg": None,
    # Shell (Magika: shell -> tree-sitter: bash)
    "shell": "bash",
    "batch": "bash",
    "awk": "bash",
    # C family
    "cs": "csharp",
    "objectivec": "objc",
    # Template languages
    "jinja": "html",
    "handlebars": "html",
    "erb": "embeddedtemplate",
    # Config/data
    "htaccess": "ini",
    "ignorefile": "gitignore",
    "gitmodules": "ini",
    "gemfile": "ruby",
    "gemspec": "ruby",
    "ipynb": "json",
    "jsonl": "json",
    "textproto": "proto",
    # Markup
    "sgml": "xml",
    "rdf": "xml",
    # Lisp
    "lisp": "commonlisp",
    # Build systems
    "makefile": "make",
    "bazel": "starlark",
    "gradle": "groovy",
    "bib": "bibtex",
}


# =============================================================================
# EMBEDDING TOKENIZER SAFETY MARGINS
# =============================================================================
# tiktoken (cl100k_base) is used to count tokens, but embedding models may use
# different tokenizers (e.g., BERT WordPiece for nomic-embed-text).
# BERT tokenizers typically produce MORE tokens than tiktoken for the same text
# because they have smaller vocabularies (~30k vs ~100k tokens).
# We use aggressive safety margins to account for this mismatch when truncating.

# Ollama models typically use BERT-based tokenizers (WordPiece)
# tiktoken tends to undercount by ~20-30% compared to BERT tokenizers
EMBEDDING_SAFETY_MARGIN_OLLAMA: float = 0.70

# OpenAI and other providers use tokenizers closer to tiktoken
EMBEDDING_SAFETY_MARGIN_DEFAULT: float = 0.90


def get_embedding_safety_margin(provider: str) -> float:
    """Get the appropriate safety margin for truncation based on embedding provider.

    Args:
        provider: The embedding provider name (e.g., 'ollama', 'openai', 'anthropic')

    Returns:
        Safety margin multiplier (0.0-1.0) to apply when truncating content
    """
    if provider.lower() == "ollama":
        return EMBEDDING_SAFETY_MARGIN_OLLAMA
    return EMBEDDING_SAFETY_MARGIN_DEFAULT


def is_binary_extension(ext: str) -> bool:
    """Check if a file extension indicates a truly unparseable binary file."""
    return ext.lower() in UNPARSEABLE_BINARY_EXTENSIONS


def is_parseable_document(ext: str) -> bool:
    """Check if a file extension is a document that requires special parsers."""
    return ext.lower() in PARSEABLE_DOCUMENT_EXTENSIONS


def should_exclude_minified(filename: str) -> bool:
    """Check if a filename matches minified/bundle patterns."""
    import fnmatch

    for pattern in MINIFIED_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False
