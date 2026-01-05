"""
LLM-powered exclusion suggestions for file indexing.

Uses structured output from OpenAI or Anthropic to intelligently
suggest file exclusion patterns based on discovered file extensions
and their statistics (file count, size, estimated chunks).
"""

from typing import Optional

from pydantic import BaseModel, Field

from ragtime.core.app_settings import get_app_settings
from ragtime.core.file_constants import (
    MINIFIED_PATTERNS,
    NEVER_SUGGEST_EXCLUDE_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
)
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


def _filter_bad_exclusion_patterns(patterns: list[str]) -> list[str]:
    """
    Filter out exclusion patterns that would exclude valuable file types.

    This is a safeguard to catch bad suggestions from LLM or heuristics.

    Args:
        patterns: List of glob patterns like "**/*.txt" or "**/docs/**"

    Returns:
        Filtered list with valuable file type patterns removed
    """
    filtered = []
    for pattern in patterns:
        lower_pattern = pattern.lower()

        # Never exclude valuable file types (text files and parseable documents)
        is_valuable = any(
            lower_pattern.endswith(ext) or lower_pattern.endswith(f'{ext}"')
            for ext in NEVER_SUGGEST_EXCLUDE_EXTENSIONS
        )
        if is_valuable:
            logger.debug(f"Filtering out valuable file exclusion pattern: {pattern}")
            continue

        filtered.append(pattern)

    return filtered


class ExclusionSuggestion(BaseModel):
    """A single exclusion pattern suggestion with reasoning."""

    pattern: str = Field(
        description="Glob pattern to exclude, e.g. '**/*.pyc' or '**/node_modules/**'"
    )
    reason: str = Field(description="Brief explanation of why this should be excluded")
    category: str = Field(
        description="Category of exclusion: 'binary', 'documents', 'generated', 'dependencies', 'cache', 'build', 'media', 'other'"
    )


class ExclusionAnalysis(BaseModel):
    """Structured response from LLM analysis of file extensions."""

    suggested_patterns: list[ExclusionSuggestion] = Field(
        description="List of glob patterns to exclude from indexing"
    )
    reasoning_summary: str = Field(
        description="Brief summary explaining the overall exclusion strategy"
    )


def _format_file_stats_for_prompt(ext_stats: dict[str, dict]) -> str:
    """
    Format file type statistics into a readable table for the LLM prompt.

    Args:
        ext_stats: Dict mapping extension to stats dict with:
            - file_count: number of files
            - total_size: total bytes
            - estimated_chunks: estimated embedding chunks

    Returns:
        Formatted string table of file types sorted by estimated chunks descending.
    """
    if not ext_stats:
        return "No file statistics available."

    # Sort by estimated chunks descending (most impactful first)
    sorted_stats = sorted(
        ext_stats.items(),
        key=lambda x: x[1].get("estimated_chunks", 0),
        reverse=True,
    )

    lines = ["Extension | Files | Size (KB) | Est. Chunks"]
    lines.append("----------|-------|-----------|------------")

    for ext, stats in sorted_stats:
        file_count = stats.get("file_count", 0)
        total_kb = stats.get("total_size", 0) / 1024
        est_chunks = stats.get("estimated_chunks", 0)
        lines.append(f"{ext} | {file_count} | {total_kb:.1f} | {est_chunks}")

    return "\n".join(lines)


def _build_exclusion_prompt(
    repo_name: str,
    ext_stats: dict[str, dict],
) -> str:
    """Build the exclusion analysis prompt with file statistics."""
    stats_table = _format_file_stats_for_prompt(ext_stats)

    # Unified prompt - all indexers now support document parsing
    return f"""Analyze these file types found in "{repo_name}" and suggest which should be excluded from a search index.

CONTEXT: This indexer can parse documents including PDF, Word, Excel, PowerPoint, and OpenDocument formats.

FILE TYPE STATISTICS (sorted by estimated chunks - higher = more impact on index size):

{stats_table}

EXCLUSION CATEGORIES:

1. ALWAYS EXCLUDE (binary files that cannot be meaningfully indexed):
   - Images: .png, .jpg, .gif, .ico, .svg, .webp
   - Fonts: .woff, .ttf, .eot
   - Compiled code: .pyc, .class, .o, .exe, .dll, .so
   - Archives: .zip, .tar, .gz, .7z, .rar
   - Media: .mp3, .mp4, .wav, .avi

2. DO NOT EXCLUDE (valuable documents that CAN be parsed):
   - Office documents: .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx
   - OpenDocument: .odt, .ods, .odp
   - Text formats: .txt, .md, .rst, .csv, .rtf
   - Source code: .py, .js, .ts, .java, .go, .rs, .c, .cpp, etc.
   - Config files: .json, .yaml, .yml, .toml, .xml, .ini

3. SUGGEST EXCLUDING (generated/low-value content):
   - Minified: *.min.js, *.min.css, *.bundle.js, *.chunk.js
   - Lock files: package-lock.json, yarn.lock, Gemfile.lock
   - Build outputs: dist/, build/, __pycache__/
   - Dependencies: node_modules/, vendor/, .venv/
   - Temporary files: *.tmp, *.bak, *~

Only suggest exclusions for file types that are actually in the statistics above.
Return patterns in glob format like "**/*.ext" for extensions or "**/dirname/**" for directories.
DO NOT suggest excluding document types (.pdf, .docx, .xlsx, etc.) - this indexer can parse them."""


async def _get_llm_exclusions_openai(
    ext_stats: dict[str, dict],
    repo_name: str,
    api_key: str,
    model: str,
) -> Optional[ExclusionAnalysis]:
    """Get exclusion suggestions using OpenAI's structured output."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)

        prompt = _build_exclusion_prompt(repo_name, ext_stats)

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing code repositories and determining which files should be excluded from search indexing. Respond only with the structured output.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "exclusion_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "suggested_patterns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {
                                            "type": "string",
                                            "description": "Glob pattern to exclude",
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Brief explanation",
                                        },
                                        "category": {
                                            "type": "string",
                                            "enum": [
                                                "binary",
                                                "documents",
                                                "generated",
                                                "dependencies",
                                                "cache",
                                                "build",
                                                "media",
                                                "other",
                                            ],
                                        },
                                    },
                                    "required": ["pattern", "reason", "category"],
                                    "additionalProperties": False,
                                },
                            },
                            "reasoning_summary": {
                                "type": "string",
                                "description": "Brief summary of exclusion strategy",
                            },
                        },
                        "required": ["suggested_patterns", "reasoning_summary"],
                        "additionalProperties": False,
                    },
                },
            },
            max_tokens=2000,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        if content:
            import json

            data = json.loads(content)
            return ExclusionAnalysis(**data)
        return None

    except Exception as e:
        logger.warning(f"OpenAI exclusion analysis failed: {e}")
        return None


async def _get_llm_exclusions_anthropic(
    ext_stats: dict[str, dict],
    repo_name: str,
    api_key: str,
    model: str,
) -> Optional[ExclusionAnalysis]:
    """Get exclusion suggestions using Anthropic's tool use for structured output."""
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)

        prompt = _build_exclusion_prompt(repo_name, ext_stats)
        prompt += "\n\nYou MUST call the analyze_exclusions tool with your suggestions."

        # Use tool calling for structured output (more reliable than beta.messages.parse)
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.1,
            tools=[
                {
                    "name": "analyze_exclusions",
                    "description": "Submit the exclusion analysis results",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "suggested_patterns": {
                                "type": "array",
                                "description": "List of exclusion patterns",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {
                                            "type": "string",
                                            "description": "Glob pattern to exclude, e.g. '**/*.pyc'",
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Brief explanation of why this should be excluded",
                                        },
                                        "category": {
                                            "type": "string",
                                            "description": "Category of exclusion",
                                            "enum": [
                                                "binary",
                                                "documents",
                                                "generated",
                                                "dependencies",
                                                "cache",
                                                "build",
                                                "media",
                                                "other",
                                            ],
                                        },
                                    },
                                    "required": ["pattern", "reason", "category"],
                                },
                            },
                            "reasoning_summary": {
                                "type": "string",
                                "description": "Brief summary explaining the overall exclusion strategy",
                            },
                        },
                        "required": ["suggested_patterns", "reasoning_summary"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "analyze_exclusions"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract tool use result
        for block in response.content:
            if block.type == "tool_use" and block.name == "analyze_exclusions":
                # block.input is a dict from the tool use response
                input_data = dict(block.input)  # Explicit cast to dict
                # Parse nested suggestion objects
                raw_patterns = input_data.get("suggested_patterns", [])
                patterns = [
                    ExclusionSuggestion(**p)
                    for p in (raw_patterns if isinstance(raw_patterns, list) else [])
                ]
                return ExclusionAnalysis(
                    suggested_patterns=patterns,
                    reasoning_summary=str(input_data.get("reasoning_summary", "")),
                )

        return None

    except Exception as e:
        logger.warning(f"Anthropic exclusion analysis failed: {e}")
        return None


def _get_heuristic_exclusions(ext_stats: dict[str, dict]) -> list[str]:
    """
    Fallback heuristic-based exclusion suggestions.

    Used when no LLM API key is available or LLM call fails.

    Args:
        ext_stats: Dict mapping extension to stats dict with file_count, total_size, estimated_chunks
    """
    suggested = []

    extensions = set(ext_stats.keys())

    # Suggest excluding truly unparseable binary files
    unparseable_found = [
        ext for ext in extensions if ext in UNPARSEABLE_BINARY_EXTENSIONS
    ]
    for ext in unparseable_found:
        suggested.append(f"**/*{ext}")

    # Note: Parseable documents (.pdf, .docx, etc.) are NOT suggested for exclusion
    # since all indexers can now parse them using document_parser.py

    # Check for minified patterns (these are pattern-based, not just extensions)
    # Add common minified patterns if we see related extensions
    if any(ext in extensions for ext in [".js", ".css"]):
        suggested.extend(MINIFIED_PATTERNS[:4])  # Add main minified patterns

    # Final safeguard: filter out any patterns that would exclude valuable files
    return _filter_bad_exclusion_patterns(list(set(suggested)))


async def get_smart_exclusion_suggestions(
    ext_stats: dict[str, dict],
    repo_name: str = "repository",
) -> tuple[list[str], bool]:
    """
    Get intelligent exclusion suggestions for the given file extension statistics.

    Attempts to use LLM for smart analysis, falls back to heuristics.

    Args:
        ext_stats: Dict mapping extension to stats dict with file_count, total_size, estimated_chunks
        repo_name: Name of the repository for context

    Returns:
        Tuple of (list of exclusion patterns, whether LLM was used)
    """
    if not ext_stats:
        return [], False

    # Get settings to check for LLM API keys
    settings = await get_app_settings()

    llm_provider = settings.get("llm_provider", "")
    openai_key = settings.get("openai_api_key", "")
    anthropic_key = settings.get("anthropic_api_key", "")
    llm_model = settings.get("llm_model", "")

    result: Optional[ExclusionAnalysis] = None

    # Try OpenAI if configured
    if llm_provider == "openai" and openai_key:
        # Use a fast, capable model for this task
        analysis_model = llm_model if llm_model.startswith("gpt-4") else "gpt-4o-mini"
        result = await _get_llm_exclusions_openai(
            ext_stats, repo_name, openai_key, analysis_model
        )
        if result:
            patterns = [s.pattern for s in result.suggested_patterns]
            # Filter out any bad suggestions the LLM might have made
            patterns = _filter_bad_exclusion_patterns(patterns)
            logger.info(
                f"LLM (OpenAI) suggested {len(patterns)} exclusion patterns: {result.reasoning_summary}"
            )
            return patterns, True

    # Try Anthropic if configured
    if llm_provider == "anthropic" and anthropic_key:
        # Use haiku for fast, cheap analysis
        analysis_model = (
            "claude-3-5-haiku-latest"
            if "haiku" in llm_model.lower() or "sonnet" in llm_model.lower()
            else llm_model
        )
        result = await _get_llm_exclusions_anthropic(
            ext_stats, repo_name, anthropic_key, analysis_model
        )
        if result:
            patterns = [s.pattern for s in result.suggested_patterns]
            # Filter out any bad suggestions the LLM might have made
            patterns = _filter_bad_exclusion_patterns(patterns)
            logger.info(
                f"LLM (Anthropic) suggested {len(patterns)} exclusion patterns: {result.reasoning_summary}"
            )
            return patterns, True

    # Fallback to heuristics
    patterns = _get_heuristic_exclusions(ext_stats)
    logger.debug(f"Using heuristic exclusions: {len(patterns)} patterns")
    return patterns, False
