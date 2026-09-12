#!/usr/bin/env python3
"""Extract the linux/amd64 SPDX document from ``buildx imagetools inspect`` JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PLATFORM = "linux/amd64"


def _spdx_document(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("imagetools output must be an object")
    # ``imagetools inspect --format '{{json .SBOM}}'`` emits this mapping directly.
    # Accepting the optional outer wrapper keeps the helper useful for saved full inspect output.
    sbom = payload.get("SBOM") if "SBOM" in payload else payload
    if not isinstance(sbom, dict):
        raise ValueError("imagetools output has no SBOM object")

    # buildx emits either a direct SPDX object or one keyed by platform.
    candidate = sbom.get("SPDX")
    if candidate is None:
        platform_entry = sbom.get(PLATFORM)
        if not isinstance(platform_entry, dict):
            raise ValueError(f"imagetools output has no SBOM for {PLATFORM}")
        candidate = platform_entry.get("SPDX")
    if not isinstance(candidate, dict):
        raise ValueError("SBOM SPDX document is missing or malformed")
    if not isinstance(candidate.get("spdxVersion"), str) or not candidate["spdxVersion"].startswith("SPDX-"):
        raise ValueError("SBOM SPDX version is missing or malformed")
    if candidate.get("SPDXID") != "SPDXRef-DOCUMENT":
        raise ValueError("SBOM SPDX document is incomplete")
    return candidate


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} INPUT OUTPUT", file=sys.stderr)
        return 2
    try:
        payload = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        document = _spdx_document(payload)
        Path(argv[2]).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Cannot extract BuildKit SPDX SBOM: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
