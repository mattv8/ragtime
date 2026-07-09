"""Shared LLM-facing tool description helpers."""

WRITE_ACCESS_ENABLED_SENTENCE = "Write access is enabled for this request."
READ_ONLY_ACCESS_SENTENCE = "Read-only access: write operations are disabled for this request."
LEGACY_READ_ONLY_ACCESS_SENTENCE = "Read-only mode: write operations are blocked."


def format_tool_write_access_sentence(allow_write: bool) -> str:
    """Return the prompt-visible write policy sentence for a configured tool."""
    return WRITE_ACCESS_ENABLED_SENTENCE if allow_write else READ_ONLY_ACCESS_SENTENCE


def extract_tool_write_access_sentence(description: str) -> str | None:
    """Return the known write-policy sentence from a tool description, if present."""
    if not isinstance(description, str):
        return None
    for sentence in (
        WRITE_ACCESS_ENABLED_SENTENCE,
        READ_ONLY_ACCESS_SENTENCE,
        LEGACY_READ_ONLY_ACCESS_SENTENCE,
    ):
        if sentence in description:
            return sentence
    return None
