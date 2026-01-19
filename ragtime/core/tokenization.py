"""
Token counting utilities for chunking and context budget management.

Uses tiktoken for accurate token counting with OpenAI models, with
a fallback character-based estimation for other models.
"""

import functools
from typing import Callable, Optional

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Cache for tiktoken encoders (expensive to create)
_encoder_cache: dict[str, object] = {}


def _get_tiktoken_encoder(model: str = "cl100k_base"):
    """Get or create a tiktoken encoder."""
    if model in _encoder_cache:
        return _encoder_cache[model]

    try:
        import tiktoken

        # Use cl100k_base as default - works for GPT-4, text-embedding-3-*, etc.
        encoder = tiktoken.get_encoding(model)
        _encoder_cache[model] = encoder
        return encoder
    except Exception as e:
        logger.debug(f"Failed to load tiktoken encoder {model}: {e}")
        return None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Tiktoken encoding name (default: cl100k_base for GPT-4/embeddings)

    Returns:
        Token count, or character-based estimate if tiktoken unavailable
    """
    encoder = _get_tiktoken_encoder(model)
    if encoder:
        return len(encoder.encode(text))

    # Fallback: rough estimate of 4 chars per token
    return len(text) // 4


def get_token_length_function(model: str = "cl100k_base") -> Callable[[str], int]:
    """
    Get a length function that counts tokens instead of characters.

    This can be passed to RecursiveCharacterTextSplitter's length_function
    parameter for token-aware chunking.

    Args:
        model: Tiktoken encoding name

    Returns:
        A function that takes text and returns token count
    """
    encoder = _get_tiktoken_encoder(model)

    if encoder:

        def token_len(text: str) -> int:
            return len(encoder.encode(text))

        return token_len

    # Fallback to character-based estimate
    logger.warning(
        "tiktoken not available, using character-based token estimation (4 chars/token)"
    )

    def char_estimate(text: str) -> int:
        return len(text) // 4

    return char_estimate


def truncate_to_token_budget(
    texts: list[str],
    max_tokens: int,
    model: str = "cl100k_base",
    separator: str = "\n\n---\n\n",
) -> tuple[str, int]:
    """
    Truncate a list of texts to fit within a token budget.

    Joins texts with separator and truncates to fit max_tokens.
    Returns as much content as possible while staying under budget.

    Args:
        texts: List of text chunks to join and truncate
        max_tokens: Maximum total tokens allowed
        model: Tiktoken encoding name
        separator: Separator to use between texts

    Returns:
        Tuple of (truncated combined text, actual token count)
    """
    if not texts:
        return "", 0

    encoder = _get_tiktoken_encoder(model)
    if not encoder:
        # Fallback: estimate and truncate by characters
        char_budget = max_tokens * 4
        separator_chars = len(separator)
        result_parts = []
        total_chars = 0

        for text in texts:
            needed = len(text) + (separator_chars if result_parts else 0)
            if total_chars + needed > char_budget:
                # Try to fit partial text
                remaining = (
                    char_budget - total_chars - (separator_chars if result_parts else 0)
                )
                if remaining > 100:  # Only include if meaningful amount left
                    result_parts.append(text[:remaining] + "...")
                break
            result_parts.append(text)
            total_chars += needed

        combined = separator.join(result_parts)
        return combined, len(combined) // 4

    # Token-accurate truncation
    separator_tokens = len(encoder.encode(separator))
    result_parts = []
    total_tokens = 0

    for text in texts:
        text_tokens = len(encoder.encode(text))
        sep_cost = separator_tokens if result_parts else 0

        if total_tokens + text_tokens + sep_cost > max_tokens:
            # Try to fit partial text
            remaining_budget = (
                max_tokens - total_tokens - sep_cost - 3
            )  # reserve for "..."
            if remaining_budget > 50:  # Only include if meaningful budget left
                # Binary search for truncation point
                tokens = encoder.encode(text)
                if len(tokens) > remaining_budget:
                    truncated = encoder.decode(tokens[:remaining_budget]) + "..."
                    result_parts.append(truncated)
                    total_tokens += remaining_budget + 1
            break

        result_parts.append(text)
        total_tokens += text_tokens + sep_cost

    combined = separator.join(result_parts)
    return combined, total_tokens


def estimate_context_tokens(
    chunks: list[str],
    separator: str = "\n\n---\n\n",
    model: str = "cl100k_base",
) -> int:
    """
    Estimate total tokens for a list of chunks when joined.

    Args:
        chunks: List of text chunks
        separator: Separator between chunks
        model: Tiktoken encoding name

    Returns:
        Estimated total token count
    """
    if not chunks:
        return 0

    encoder = _get_tiktoken_encoder(model)
    if not encoder:
        total_chars = sum(len(c) for c in chunks) + len(separator) * (len(chunks) - 1)
        return total_chars // 4

    total = 0
    sep_tokens = len(encoder.encode(separator))

    for i, chunk in enumerate(chunks):
        total += len(encoder.encode(chunk))
        if i < len(chunks) - 1:
            total += sep_tokens

    return total
