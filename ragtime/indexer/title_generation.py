"""Generate and apply chat titles using the LLM."""

import asyncio
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ragtime.core.event_bus import task_event_bus
from ragtime.core.logging import get_logger
from ragtime.core.ollama import KEEP_ALIVE, NUM_GPU
from ragtime.indexer.repository import repository
from ragtime.rag import rag

logger = get_logger(__name__)

# Short, LLM-friendly instruction for concise titles
_TITLE_PROMPT = (
    "You create short, specific chat titles (maximum 6 words) that summarize the user's question. "
    'Avoid quotes or trailing punctuation. Respond with JSON: {"title": "..."}.'
)


class TitleResponse(BaseModel):
    title: str


def _extract_text(message: str) -> str:
    """Extract plain text from a message payload that may contain multimodal parts."""
    try:
        data = json.loads(message)
    except Exception:
        return message

    if isinstance(data, list):
        parts: list[str] = []
        for item in data:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        if parts:
            return " ".join(parts)

    return message


def _clean_title(raw: str) -> str:
    title = raw.strip().strip('"').strip("'")
    title = title.replace("\n", " ")
    title = re.sub(r"\s+", " ", title).strip()
    if title.endswith("."):
        title = title[:-1].rstrip()
    if len(title) > 80:
        title = title[:80].rstrip()
    return title


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    stripped = text.strip()
    # Match ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return stripped


def _extract_json_title(content: str) -> Optional[str]:
    """Try to extract a title from JSON in LLM output, handling code fences."""
    # Strip markdown code fences first
    cleaned = _strip_code_fences(content)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "title" in parsed:
            return str(parsed["title"])
    except Exception:
        pass
    # Last resort: try to find a JSON object embedded in the text
    m = re.search(r'\{[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*\}', content)
    if m:
        return m.group(1)
    return None


def _is_ollama_llm(llm: object) -> bool:
    """Check if the LLM instance is a ChatOllama model."""
    try:
        from langchain_ollama import ChatOllama

        return isinstance(llm, ChatOllama)
    except ImportError:
        return False


async def _generate_title(question_text: str) -> Optional[str]:
    if not question_text.strip():
        return None

    llm = getattr(rag, "llm", None)
    if not llm or not getattr(rag, "is_ready", False):
        logger.debug("LLM not ready for title generation")
        return None

    messages = [
        SystemMessage(content=_TITLE_PROMPT),
        HumanMessage(content=question_text.strip()),
    ]

    # Ollama needs special handling: use format="json" directly with reasoning
    # disabled so thinking tokens don't leak into the output, and the model
    # is constrained to produce valid JSON without needing tool-calling support.
    if _is_ollama_llm(llm):
        try:
            from langchain_ollama import ChatOllama

            title_llm = ChatOllama(
                model=llm.model,  # type: ignore[attr-defined]
                base_url=llm.base_url,  # type: ignore[attr-defined]
                temperature=0,
                format="json",
                num_gpu=getattr(llm, "num_gpu", NUM_GPU),  # type: ignore[attr-defined]
                keep_alive=getattr(llm, "keep_alive", KEEP_ALIVE),  # type: ignore[attr-defined]
                reasoning=False,
            )
            response = await title_llm.ainvoke(messages)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            parsed_title = _extract_json_title(str(content))
            if parsed_title:
                title = _clean_title(parsed_title)
                return title or None
            # JSON parse failed; use raw content as last resort
            title = _clean_title(str(content))
            return title or None
        except Exception as exc:
            logger.warning("Ollama title generation failed: %s", exc)
            return None

    # Prefer native structured output for OpenAI/Anthropic
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(TitleResponse)
            result = await structured_llm.ainvoke(messages)
            title = _clean_title(result.title)
            return title or None
        except Exception as exc:
            logger.warning("Structured title generation failed, falling back: %s", exc)

    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning("Failed to generate chat title: %s", exc)
        return None

    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    # Attempt to parse JSON output (handling code fences)
    parsed_title = _extract_json_title(str(content))
    raw_title = parsed_title or str(content)
    title = _clean_title(raw_title)
    return title or None


async def update_conversation_title_from_question(
    conversation_id: str, user_message: str
) -> None:
    """Update the conversation title if it is still the default."""
    conv = await repository.get_conversation(conversation_id)
    if not conv or conv.title != "New Chat":
        return

    question_text = _extract_text(user_message).strip()
    if not question_text:
        return

    title = await _generate_title(question_text)
    if not title:
        # Fallback to truncated question if LLM is unavailable
        title = question_text[:50] + ("..." if len(question_text) > 50 else "")

    try:
        await repository.update_conversation_title(conversation_id, title)
        await task_event_bus.publish(
            f"conversation:{conversation_id}", {"type": "title_update", "title": title}
        )
    except Exception as exc:
        logger.warning("Could not update conversation title: %s", exc)


# Strong references for fire-and-forget tasks so they are not garbage-collected.
_background_tasks: set[asyncio.Task] = set()


def schedule_title_generation(conversation_id: str, user_message: str) -> None:
    """Fire-and-forget task to generate a chat title."""

    async def _runner() -> None:
        try:
            await update_conversation_title_from_question(conversation_id, user_message)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Title generation task failed for %s: %s", conversation_id, exc
            )

    task = asyncio.create_task(_runner())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
