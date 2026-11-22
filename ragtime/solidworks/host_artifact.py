"""Locate the packaged SolidWorks host artifact."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ragtime.config import settings

HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class HostArtifact:
    """Validated metadata and absolute path of the host zip."""

    host_version: str
    protocol: int
    file_name: str
    sha256: str
    size_bytes: int
    path: Path


def get_host_artifact() -> HostArtifact | None:
    """Read the manifest without hashing the artifact on each request."""
    try:
        root = Path(settings.solidworks_host_artifact_dir).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        version = manifest.get("host_version")
        protocol = manifest.get("protocol")
        filename = manifest.get("file")
        sha256 = manifest.get("sha256")
        size = manifest.get("size_bytes")
        if (
            not isinstance(version, str)
            or not version.strip()
            or type(protocol) is not int
            or not isinstance(filename, str)
            or filename in {"", ".", ".."}
            or "/" in filename
            or "\\" in filename
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or type(size) is not int
            or size < 0
        ):
            return None
        path = (root / filename).resolve()
        if not path.is_file() or not path.is_relative_to(root) or path.stat().st_size != size:
            return None
        return HostArtifact(version, protocol, filename, sha256, size, path)
    except (OSError, ValueError, UnicodeError):
        return None


def verify_host_artifact(artifact: HostArtifact) -> bool:
    """Hash the artifact file in chunks and compare it with the manifest SHA-256."""
    digest = hashlib.sha256()
    try:
        with artifact.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == artifact.sha256
