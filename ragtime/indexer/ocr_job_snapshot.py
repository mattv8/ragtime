"""Durable, credential-free OCR target snapshots for resumable index jobs."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ragtime.core.ocr_target import OcrTarget

_SNAPSHOT_NAME = "ocr-target.json"
_CONFIG_KEYS = {"ocr_mode", "ocr_provider", "ocr_vision_model"}


class OcrJobSnapshotError(ValueError):
    """A durable OCR job snapshot is absent, unsafe, or malformed."""


@dataclass(frozen=True)
class OcrJobSnapshot:
    """Non-secret OCR settings frozen before an index job starts extraction."""

    target: OcrTarget
    pipeline_fingerprint: str

    @property
    def config_overrides(self) -> dict[str, str]:
        return {
            "ocr_mode": "vision",
            "ocr_provider": self.target.provider,
            "ocr_vision_model": self.target.model,
        }


def _job_root(root: Path, job_id: str, *, create: bool) -> Path:
    if not job_id or Path(job_id).name != job_id:
        raise OcrJobSnapshotError("OCR snapshot job id must be a single path component")
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    job_root = resolved_root / job_id
    if job_root.is_symlink():
        raise OcrJobSnapshotError("OCR snapshot job directory must not be a symlink")
    if create:
        job_root.mkdir(exist_ok=True)
    elif not job_root.exists():
        return job_root
    if not job_root.is_dir() or job_root.resolve().parent != resolved_root:
        raise OcrJobSnapshotError("OCR snapshot job directory escapes its root")
    return job_root


def _validate_target(payload: Any) -> OcrTarget:
    if not isinstance(payload, dict) or set(payload) != {"provider", "model", "base_url"}:
        raise OcrJobSnapshotError("OCR snapshot target is malformed")
    provider = payload["provider"]
    model = payload["model"]
    base_url = payload["base_url"]
    if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
        raise OcrJobSnapshotError("OCR snapshot target must include provider and model")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise OcrJobSnapshotError("OCR snapshot endpoint is malformed")
        parsed = urlsplit(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise OcrJobSnapshotError("OCR snapshot endpoint must not contain secrets")
    return OcrTarget(provider=provider, model=model, base_url=base_url)


def _decode_snapshot(path: Path) -> OcrJobSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OcrJobSnapshotError("OCR job snapshot cannot be read") from exc
    if not isinstance(payload, dict) or set(payload) != {"target", "config", "pipeline_fingerprint"}:
        raise OcrJobSnapshotError("OCR job snapshot is malformed")
    target = _validate_target(payload["target"])
    config = payload["config"]
    if (
        not isinstance(config, dict)
        or set(config) != _CONFIG_KEYS
        or config
        != {
            "ocr_mode": "vision",
            "ocr_provider": target.provider,
            "ocr_vision_model": target.model,
        }
    ):
        raise OcrJobSnapshotError("OCR job snapshot config does not match its target")
    fingerprint = payload["pipeline_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise OcrJobSnapshotError("OCR job snapshot pipeline fingerprint is malformed")
    return OcrJobSnapshot(target=target, pipeline_fingerprint=fingerprint)


def load_ocr_job_snapshot(root: Path, job_id: str) -> OcrJobSnapshot | None:
    """Load a job's frozen OCR target, rejecting corrupt state rather than re-resolving it."""
    job_root = _job_root(root, job_id, create=False)
    path = job_root / _SNAPSHOT_NAME
    if path.is_symlink():
        raise OcrJobSnapshotError("OCR job snapshot must not be a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise OcrJobSnapshotError("OCR job snapshot is not a regular file")
    return _decode_snapshot(path)


def create_ocr_job_snapshot(root: Path, job_id: str, target: OcrTarget, pipeline_fingerprint: str) -> OcrJobSnapshot:
    """Atomically persist a frozen target before any extraction can begin."""
    if not isinstance(pipeline_fingerprint, str) or not pipeline_fingerprint:
        raise OcrJobSnapshotError("OCR pipeline fingerprint is required")
    job_root = _job_root(root, job_id, create=True)
    path = job_root / _SNAPSHOT_NAME
    if path.exists() or path.is_symlink():
        existing = load_ocr_job_snapshot(root, job_id)
        if existing is None:
            raise OcrJobSnapshotError("OCR job snapshot disappeared while opening it")
        return existing
    snapshot = OcrJobSnapshot(target=target, pipeline_fingerprint=pipeline_fingerprint)
    payload = {
        "target": target.to_snapshot(),
        "config": snapshot.config_overrides,
        "pipeline_fingerprint": pipeline_fingerprint,
    }
    temporary = job_root / f".{_SNAPSHOT_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot
