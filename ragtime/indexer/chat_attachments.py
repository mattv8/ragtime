"""Chat attachment upload, caching, and preprocessing helpers."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from ragtime.config import settings
from ragtime.core.file_constants import (
    OCR_EXTENSIONS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
)
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import get_context_limit
from ragtime.core.tokenization import count_tokens, truncate_to_token_budget
from ragtime.indexer.chunking import chunk_documents_parallel
from ragtime.indexer.document_parser import extract_text_from_file_async

logger = get_logger(__name__)

CHAT_ATTACHMENT_DIR = Path(settings.index_data_path) / "_tmp" / "chat_attachments"
CHAT_ATTACHMENT_TTL = timedelta(days=7)
CHAT_ATTACHMENT_MAX_FILE_SIZE = 20 * 1024 * 1024
CHAT_ATTACHMENT_CHUNK_SIZE = 800
CHAT_ATTACHMENT_CHUNK_OVERLAP = 80
CHAT_ATTACHMENT_MIN_BUDGET_TOKENS = 2048
CHAT_ATTACHMENT_MAX_BUDGET_TOKENS = 12000
CHAT_ATTACHMENT_SOURCE = "chat_upload"
_METADATA_FILE = "metadata.json"
_CHUNKS_FILE = "chunks.json"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_chat_attachment_dir() -> Path:
    CHAT_ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_ATTACHMENT_DIR


def _sanitize_filename(filename: str) -> str:
    candidate = Path(filename or "attachment").name.strip() or "attachment"
    sanitized = _SAFE_FILENAME_RE.sub("_", candidate)
    return sanitized[:255] or "attachment"


def _attachment_dir(attachment_id: str) -> Path:
    return ensure_chat_attachment_dir() / attachment_id


def _metadata_path(attachment_id: str) -> Path:
    return _attachment_dir(attachment_id) / _METADATA_FILE


def _chunks_path(attachment_id: str) -> Path:
    return _attachment_dir(attachment_id) / _CHUNKS_FILE


def _normalize_suffix(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported_chat_attachment(
    filename: str, mime_type: Optional[str] = None
) -> bool:
    suffix = _normalize_suffix(filename)
    normalized_mime = (mime_type or "").strip().lower()

    if normalized_mime.startswith("image/") or suffix in OCR_EXTENSIONS:
        return False

    if suffix in PARSEABLE_DOCUMENT_EXTENSIONS:
        return True

    if suffix and suffix in UNPARSEABLE_BINARY_EXTENSIONS:
        return False

    if normalized_mime.startswith("text/"):
        return True

    return True


async def store_chat_attachment_upload(
    file: UploadFile,
    conversation_id: str,
    user_id: str,
    workspace_id: Optional[str] = None,
) -> dict[str, Any]:
    filename = _sanitize_filename(file.filename or "attachment")
    mime_type = (file.content_type or "application/octet-stream").strip()

    if not is_supported_chat_attachment(filename, mime_type):
        raise HTTPException(
            status_code=400, detail="File type is not supported in chat attachments"
        )

    payload = await file.read()
    size_bytes = len(payload)
    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Attachment is empty")
    if size_bytes > CHAT_ATTACHMENT_MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Attachment exceeds the 20MB limit")

    attachment_id = uuid.uuid4().hex
    expires_at = _now_utc() + CHAT_ATTACHMENT_TTL
    suffix = _normalize_suffix(filename)
    stored_name = f"source{suffix}" if suffix else "source"
    attachment_dir = _attachment_dir(attachment_id)
    attachment_dir.mkdir(parents=True, exist_ok=False)
    stored_path = attachment_dir / stored_name
    metadata = {
        "attachment_id": attachment_id,
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "created_at": _now_utc().isoformat(),
        "expires_at": expires_at.isoformat(),
        "conversation_id": conversation_id,
        "user_id": user_id,
        "workspace_id": workspace_id,
        "stored_name": stored_name,
        "source": CHAT_ATTACHMENT_SOURCE,
    }

    await asyncio.to_thread(stored_path.write_bytes, payload)
    await asyncio.to_thread(
        _metadata_path(attachment_id).write_text, json.dumps(metadata), "utf-8"
    )
    logger.info(
        "Stored chat attachment %s for conversation=%s filename=%s size=%d",
        attachment_id,
        conversation_id,
        filename,
        size_bytes,
    )
    return metadata


def _load_metadata(attachment_id: str) -> Optional[dict[str, Any]]:
    path = _metadata_path(attachment_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to read chat attachment metadata %s: %s", attachment_id, exc
        )
        return None


def _validate_attachment_owner(
    metadata: dict[str, Any],
    conversation_id: Optional[str],
    user_id: Optional[str],
    workspace_id: Optional[str],
) -> bool:
    if conversation_id and metadata.get("conversation_id") not in {
        None,
        conversation_id,
    }:
        return False
    if user_id and metadata.get("user_id") not in {None, user_id}:
        return False
    if workspace_id and metadata.get("workspace_id") not in {None, workspace_id}:
        return False
    return True


def resolve_chat_attachment(
    attachment_id: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> tuple[dict[str, Any], Path]:
    metadata = _load_metadata(attachment_id)
    if not metadata:
        raise FileNotFoundError(f"Chat attachment {attachment_id} was not found")

    expires_at_raw = metadata.get("expires_at")
    expires_at = None
    if isinstance(expires_at_raw, str):
        try:
            expires_at = datetime.fromisoformat(expires_at_raw)
        except ValueError:
            expires_at = None
    if expires_at and expires_at <= _now_utc():
        raise FileNotFoundError(f"Chat attachment {attachment_id} has expired")

    if not _validate_attachment_owner(metadata, conversation_id, user_id, workspace_id):
        raise PermissionError(
            f"Chat attachment {attachment_id} is not available in this conversation"
        )

    stored_name = str(metadata.get("stored_name") or "source")
    path = _attachment_dir(attachment_id) / stored_name
    if not path.exists():
        raise FileNotFoundError(f"Chat attachment {attachment_id} payload is missing")
    return metadata, path


async def cleanup_expired_chat_attachments(now: Optional[datetime] = None) -> int:
    current = now or _now_utc()
    if not CHAT_ATTACHMENT_DIR.exists():
        return 0

    removed = 0
    for candidate in CHAT_ATTACHMENT_DIR.iterdir():
        if not candidate.is_dir():
            continue
        metadata = _load_metadata(candidate.name)
        if metadata is None:
            await asyncio.to_thread(shutil.rmtree, candidate, True)
            removed += 1
            continue
        expires_at_raw = metadata.get("expires_at")
        try:
            expires_at = datetime.fromisoformat(str(expires_at_raw))
        except Exception:
            expires_at = current - timedelta(seconds=1)
        if expires_at <= current:
            await asyncio.to_thread(shutil.rmtree, candidate, True)
            removed += 1

    if removed:
        logger.info("Removed %d expired chat attachment(s)", removed)
    return removed


async def _load_cached_chunks(attachment_id: str) -> Optional[list[str]]:
    cache_path = _chunks_path(attachment_id)
    if not cache_path.exists():
        return None
    try:
        payload = await asyncio.to_thread(cache_path.read_text, "utf-8")
        data = json.loads(payload)
        if isinstance(data, list):
            return [str(item) for item in data if str(item).strip()]
    except Exception as exc:
        logger.warning(
            "Failed to load cached chat attachment chunks %s: %s", attachment_id, exc
        )
    return None


async def _write_cached_chunks(attachment_id: str, chunks: list[str]) -> None:
    cache_path = _chunks_path(attachment_id)
    try:
        await asyncio.to_thread(cache_path.write_text, json.dumps(chunks), "utf-8")
    except Exception as exc:
        logger.warning(
            "Failed to cache chat attachment chunks %s: %s", attachment_id, exc
        )


async def _extract_attachment_chunks(
    metadata: dict[str, Any], payload_path: Path
) -> list[str]:
    attachment_id = str(metadata.get("attachment_id") or "")
    cached = await _load_cached_chunks(attachment_id)
    if cached is not None:
        return cached

    extracted = await extract_text_from_file_async(payload_path)
    if not extracted.strip():
        return []

    documents = await chunk_documents_parallel(
        [
            Document(
                page_content=extracted,
                metadata={"source": metadata.get("filename") or payload_path.name},
            )
        ],
        chunk_size=CHAT_ATTACHMENT_CHUNK_SIZE,
        chunk_overlap=CHAT_ATTACHMENT_CHUNK_OVERLAP,
        use_tokens=True,
        batch_size=1,
    )
    chunk_texts = [
        doc.page_content.strip() for doc in documents if doc.page_content.strip()
    ]
    if not chunk_texts:
        chunk_texts = [extracted.strip()]

    await _write_cached_chunks(attachment_id, chunk_texts)
    return chunk_texts


async def get_chat_attachment_budget_tokens(model_id: Optional[str]) -> int:
    effective_model = (model_id or "").strip()
    context_limit = 8192
    if effective_model:
        try:
            context_limit = max(1, int(await get_context_limit(effective_model)))
        except Exception:
            logger.debug(
                "Falling back to default chat attachment context budget for model %s",
                effective_model,
            )
    return min(
        CHAT_ATTACHMENT_MAX_BUDGET_TOKENS,
        max(CHAT_ATTACHMENT_MIN_BUDGET_TOKENS, context_limit // 4),
    )


def _format_chunk_block(
    filename: str, chunk_text: str, chunk_index: int, total_chunks: int
) -> str:
    return (
        f"--- Attached file: {filename} ---\n"
        f"Chunk {chunk_index}/{total_chunks}\n"
        f"{chunk_text.strip()}"
    )


async def preprocess_chat_attachment_content_parts(
    content: Any,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> tuple[Any, Optional[dict[str, int]]]:
    if not isinstance(content, list):
        return content, None

    has_chat_attachment = any(
        isinstance(part, dict)
        and part.get("type") == "file"
        and part.get("attachment_id")
        and part.get("attachment_source") == CHAT_ATTACHMENT_SOURCE
        for part in content
    )
    if not has_chat_attachment:
        return content, None

    remaining_budget = await get_chat_attachment_budget_tokens(model_id)
    transformed: list[Any] = []
    stats = {
        "file_count": 0,
        "cached_chunk_count": 0,
        "included_chunk_count": 0,
        "extracted_chars": 0,
        "used_tokens": 0,
        "omitted_chunk_count": 0,
    }

    for part in content:
        if not (
            isinstance(part, dict)
            and part.get("type") == "file"
            and part.get("attachment_id")
            and part.get("attachment_source") == CHAT_ATTACHMENT_SOURCE
        ):
            transformed.append(part)
            continue

        stats["file_count"] += 1
        attachment_id = str(part.get("attachment_id"))
        filename = str(part.get("filename") or "attachment")

        if remaining_budget <= 0:
            transformed.append(
                {
                    "type": "text",
                    "text": f'[Attached file "{filename}" omitted because the attachment context budget was exhausted.]',
                }
            )
            continue

        try:
            metadata, payload_path = resolve_chat_attachment(
                attachment_id,
                conversation_id=conversation_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            chunks = await _extract_attachment_chunks(metadata, payload_path)
            stats["cached_chunk_count"] += len(chunks)
            stats["extracted_chars"] += sum(len(chunk) for chunk in chunks)
            if not chunks:
                transformed.append(
                    {
                        "type": "text",
                        "text": f'[Attached file "{filename}" could not be parsed into readable text.]',
                    }
                )
                continue

            total_chunks = len(chunks)
            selected_blocks: list[str] = []
            included_chunks = 0
            for index, chunk_text in enumerate(chunks, start=1):
                block = _format_chunk_block(filename, chunk_text, index, total_chunks)
                block_tokens = count_tokens(block)
                if selected_blocks and block_tokens > remaining_budget:
                    break
                if not selected_blocks and block_tokens > remaining_budget:
                    truncated, used_tokens = truncate_to_token_budget(
                        [block], remaining_budget
                    )
                    if truncated.strip():
                        selected_blocks.append(truncated)
                        remaining_budget = max(0, remaining_budget - used_tokens)
                        stats["used_tokens"] += used_tokens
                        included_chunks = 1
                    break

                selected_blocks.append(block)
                remaining_budget = max(0, remaining_budget - block_tokens)
                stats["used_tokens"] += block_tokens
                included_chunks += 1

            omitted_chunks = max(0, total_chunks - included_chunks)
            stats["included_chunk_count"] += included_chunks
            stats["omitted_chunk_count"] += omitted_chunks

            if omitted_chunks > 0:
                note = (
                    f'[Omitted {omitted_chunks} additional chunk(s) from "{filename}" '
                    f"to stay within the attachment context budget.]"
                )
                note_tokens = count_tokens(note)
                if note_tokens <= remaining_budget:
                    selected_blocks.append(note)
                    remaining_budget -= note_tokens
                    stats["used_tokens"] += note_tokens

            transformed.append({"type": "text", "text": "\n\n".join(selected_blocks)})
        except PermissionError:
            transformed.append(
                {
                    "type": "text",
                    "text": f'[Attached file "{filename}" is not available in this conversation.]',
                }
            )
        except FileNotFoundError:
            transformed.append(
                {
                    "type": "text",
                    "text": f'[Attached file "{filename}" is no longer available and could not be used as context.]',
                }
            )
        except Exception as exc:
            logger.exception(
                "Failed to preprocess chat attachment %s: %s", attachment_id, exc
            )
            transformed.append(
                {
                    "type": "text",
                    "text": f'[Attached file "{filename}" could not be processed for context.]',
                }
            )

    logger.info(
        "Preprocessed chat attachments files=%d chunks=%d included=%d omitted=%d chars=%d tokens=%d",
        stats["file_count"],
        stats["cached_chunk_count"],
        stats["included_chunk_count"],
        stats["omitted_chunk_count"],
        stats["extracted_chars"],
        stats["used_tokens"],
    )
    return transformed, stats
