"""LLM-facing guidance for the SolidWorks tool, read from ``guidance/*.md``.

Topic ids map to fixed file names in a table; caller input is never joined into a path.
"""

from functools import lru_cache
from pathlib import Path

GUIDE_TOPICS: tuple[dict[str, str], ...] = (
    {
        "id": "overview",
        "title": "Overview and first steps",
        "when_to_read": "Read first: what the tool does, the call shape, error codes and units.",
    },
    {
        "id": "documents",
        "title": "Documents, properties and mass",
        "when_to_read": "Open or list documents; read summaries, configurations, custom properties, equations or mass properties.",
    },
    {
        "id": "features-and-sketches",
        "title": "Feature tree, sketches and dimensions",
        "when_to_read": "Inspect the feature tree, rebuild errors, sketch geometry, constraints or driving dimensions.",
    },
    {
        "id": "assemblies",
        "title": "Assemblies, components and mates",
        "when_to_read": "Inspect an assembly: components, their positions, fixed or suppressed state, and mates.",
    },
    {
        "id": "paging",
        "title": "Paging, snapshots and large results",
        "when_to_read": "A list result has next_offset, a snapshot, or result_too_large; or you need to limit output size.",
    },
    {
        "id": "pdm",
        "title": "PDM vault: search, live state, get latest, check-out and check-in",
        "when_to_read": "The tool is linked to a PDM vault and you need to find vault files, see who has them checked out, get the latest copy, or add, check out or check in files.",
    },
    {
        "id": "studying-models",
        "title": "Learn how existing models are built",
        "when_to_read": "Before building or changing a model: find similar existing models in the vault and read how their features, parameters, configurations and mates are set up.",
    },
)

_GUIDANCE_DIR = Path(__file__).parent / "guidance"
_TOPIC_FILES: dict[str, str] = {topic["id"]: f"{topic['id']}.md" for topic in GUIDE_TOPICS}
DEFAULT_TOPIC = "overview"


def _normalise_topic(topic: str | None) -> str:
    if topic is None:
        return DEFAULT_TOPIC
    normalised = topic.strip().lower()
    return normalised or DEFAULT_TOPIC


@lru_cache(maxsize=None)
def _read_topic_file(topic_id: str) -> str:
    return (_GUIDANCE_DIR / _TOPIC_FILES[topic_id]).read_text(encoding="utf-8")


def read_guide(topic: str | None) -> dict:
    """Return the topic list, the resolved topic id and its markdown.

    Raises ValueError for an unknown topic; the message lists the valid ids.
    """
    topic_id = _normalise_topic(topic)
    if topic_id not in _TOPIC_FILES:
        valid = ", ".join(_TOPIC_FILES)
        raise ValueError(f"Unknown guide topic '{topic}'. Valid topics: {valid}.")
    return {
        "topics": [dict(entry) for entry in GUIDE_TOPICS],
        "topic": topic_id,
        "markdown": _read_topic_file(topic_id),
    }
