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
# Map extensions to tree-sitter language names
# Note: These must match names supported by tree-sitter-language-pack
EXTENSION_TO_LANG: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # PHP
    ".php": "php",
    # Ruby
    ".rb": "ruby",
    # C/C++
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",  # heuristic, could be cpp
    ".hpp": "cpp",
    ".cc": "cpp",
    # Web
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    # Template languages -> parse as HTML (preserves tag structure)
    ".j2": "html",  # Jinja2 templates
    ".jinja": "html",
    ".jinja2": "html",
    ".twig": "twig",  # Twig templates (has native support)
    ".ejs": "html",  # EJS templates
    ".hbs": "html",  # Handlebars templates
    ".mustache": "html",  # Mustache templates
    ".njk": "html",  # Nunjucks templates
    ".liquid": "html",  # Liquid templates
    # Data/Config
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".sql": "sql",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
}


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
