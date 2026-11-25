"""MCP content for successful SolidWorks artifact results."""

import asyncio
import base64
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from mcp.types import ImageContent, ResourceLink
from pydantic import AnyUrl

from ragtime.config.settings import settings
from ragtime.solidworks.artifacts import ARTIFACT_URL_PREFIX, ArtifactNotFound, load_artifact

MAX_IMAGE_BYTES = 1536 * 1024
_ID = re.compile(r"^[0-9a-f]{32}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _artifact(result: str) -> dict | None:
    try:
        payload = json.loads(result)
        if payload.get("ok") is not True or payload.get("operation") not in {"model.preview", "model.export"}:
            return None
        artifact = payload["result"]["artifact"]
        return artifact if isinstance(artifact, dict) else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _preview_id(artifact: dict) -> str | None:
    url = artifact.get("preview_image_url")
    if not isinstance(url, str):
        return None
    match = re.fullmatch(re.escape(ARTIFACT_URL_PREFIX) + r"/([0-9a-f]{32})/content", url)
    return match.group(1) if match else None


def _public_origin() -> str | None:
    configured = str(settings.external_base_url or "").strip()
    try:
        parsed = urlsplit(configured)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        ):
            parsed.port
            return str(AnyUrl(f"{parsed.scheme}://{parsed.netloc}")).rstrip("/")
    except ValueError:
        pass
    return None


def _link(artifact: dict, origin: str | None) -> ResourceLink | None:
    artifact_id = artifact.get("id")
    if not origin or not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id):
        return None
    if artifact.get("page_url") != f"/?artifact={artifact_id}":
        return None
    return ResourceLink(
        type="resource_link",
        uri=AnyUrl(f"{origin}/?artifact={artifact_id}"),
        name=artifact["file_name"],
        mimeType="text/html",
        description=f"{artifact['file_name']} ({artifact.get('format', 'unknown')}). Ragtime sign-in is required.",
    )


def _read_image(path: Path) -> bytes:
    with path.open("rb") as file:
        return file.read(MAX_IMAGE_BYTES + 1)


async def build_artifact_content(result: str) -> list[ImageContent | ResourceLink]:
    artifact = _artifact(result)
    if artifact is None:
        return []
    content: list[ImageContent | ResourceLink] = []
    preview_id = _preview_id(artifact)
    if preview_id:
        try:
            meta, path = await load_artifact(preview_id)
            if not meta.get("expired") and meta.get("format") == "png":
                data = await asyncio.to_thread(_read_image, path)
                if len(data) <= MAX_IMAGE_BYTES and data.startswith(_PNG_SIGNATURE):
                    content.append(ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mimeType="image/png"))
        except (ArtifactNotFound, OSError):
            pass
    link = _link(artifact, _public_origin())
    if link:
        content.append(link)
    return content
