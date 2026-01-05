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
# Documents that require special parsers (PDF, Office, OpenDocument).
# - Filesystem indexer: CAN parse these (has document_parser.py)
# - Git/Upload indexer: CANNOT parse these (uses TextLoader only)
#
# For git/upload indexer, these are suggested for exclusion with a warning,
# but the user can override if they want to try (will likely produce garbage).
PARSEABLE_DOCUMENT_EXTENSIONS: set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
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
