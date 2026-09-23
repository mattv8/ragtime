#!/usr/bin/env python3
"""Install a checksum-pinned Restic release without shelling out."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

RESTIC_VERSION = "0.19.1"
RESTIC_SHA256_BY_ARCH = {
    "amd64": "f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c",
    "arm64": "a5f64aaab53d51e311fa3829124c5b703f2d14cf187d8640b6be3b2b49376465",
}
RESTIC_INSTALL_PATH = Path("/opt/ragtime-backup/bin/restic")


def restic_download_url(arch: str) -> str:
    return f"https://github.com/restic/restic/releases/download/v{RESTIC_VERSION}/restic_{RESTIC_VERSION}_linux_{arch}.bz2"


def install_restic(
    arch: str,
    destination: Path = RESTIC_INSTALL_PATH,
    *,
    archive_url: str | None = None,
) -> None:
    """Download, verify, and atomically install the pinned binary."""
    try:
        expected_sha256 = RESTIC_SHA256_BY_ARCH[arch]
    except KeyError as error:
        raise ValueError(f"Unsupported Restic architecture: {arch}") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    binary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".restic-", suffix=".bz2", delete=False) as archive_file:
            archive_path = Path(archive_file.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(archive_url or restic_download_url(arch)) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    archive_file.write(chunk)

        if digest.hexdigest() != expected_sha256:
            raise ValueError("Restic archive checksum verification failed")

        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".restic-", delete=False) as binary_file:
            binary_path = Path(binary_file.name)
            with bz2.open(archive_path, "rb") as archive:
                while chunk := archive.read(1024 * 1024):
                    binary_file.write(chunk)
            binary_file.flush()
            os.fsync(binary_file.fileno())
        os.chmod(binary_path, 0o755)
        os.replace(binary_path, destination)
        binary_path = None
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        if binary_path is not None:
            binary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--destination", type=Path, default=RESTIC_INSTALL_PATH)
    args = parser.parse_args()
    try:
        install_restic(args.arch, args.destination)
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Failed to install Restic: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
