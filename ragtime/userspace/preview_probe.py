from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

PREVIEW_HOST_PROBE_LABEL = "preview-probe"
PREVIEW_HOST_PROBE_PATH = "/__ragtime/preview-host-probe"
PREVIEW_HOST_PROBE_HEADER = "X-Ragtime-Preview-Probe"
PREVIEW_HOST_PROBE_VALUE = "ok"


def build_preview_probe_url(preview_origin: str) -> str:
    parsed = urlsplit(str(preview_origin or "").strip())
    hostname = str(parsed.hostname or "").strip().strip(".")
    if "." in hostname:
        probe_host = f"{PREVIEW_HOST_PROBE_LABEL}.{hostname.split('.', 1)[1]}"
    else:
        probe_host = hostname
    port = parsed.port
    use_default_port = (parsed.scheme == "https" and port in {None, 443}) or (
        parsed.scheme == "http" and port in {None, 80}
    )
    netloc = probe_host if use_default_port else f"{probe_host}:{port}"
    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            PREVIEW_HOST_PROBE_PATH,
            "",
            "",
        )
    )


def is_preview_probe_response(
    status_code: int,
    headers: Mapping[str, str],
) -> bool:
    header_value = str(
        headers.get(PREVIEW_HOST_PROBE_HEADER)
        or headers.get(PREVIEW_HOST_PROBE_HEADER.lower())
        or ""
    ).strip()
    return status_code == 204 and header_value.lower() == PREVIEW_HOST_PROBE_VALUE
