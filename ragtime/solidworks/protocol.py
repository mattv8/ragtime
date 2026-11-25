"""SolidWorks host wire protocol."""

import base64
import json
import re
import uuid

PROTOCOL_VERSION = 1
MAX_ENCODED_REQUEST_LENGTH = 8000
# The host accepts up to CallRequestArgument.MaxLength (dotnet Core); keep that at least this length.
MAX_ENCODED_SCRIPT_REQUEST_LENGTH = 32000


class SolidworksProtocolError(ValueError):
    """A request or response does not meet the host protocol."""


def build_request(operation: str, arguments: dict | None, context: dict) -> dict:
    """Build a request with an independent correlation ID."""
    return {
        "protocol": PROTOCOL_VERSION,
        "id": str(uuid.uuid4()),
        "operation": operation,
        "arguments": arguments if arguments is not None else {},
        "context": context,
    }


def encode_request(request: dict) -> str:
    """Encode a request for the host command line."""
    try:
        payload = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SolidworksProtocolError("Request is not JSON serializable") from exc
    encoded = base64.b64encode(payload).decode("ascii")
    max_length = MAX_ENCODED_SCRIPT_REQUEST_LENGTH if request.get("operation") == "script.run" else MAX_ENCODED_REQUEST_LENGTH
    if len(encoded) > max_length or not re.fullmatch(r"[A-Za-z0-9+/=]+", encoded):
        raise SolidworksProtocolError("Encoded request is invalid or too large")
    return encoded


def parse_response(stdout: str) -> dict:
    """Return the last response envelope in stdout, ignoring shell noise."""
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            response = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(response, dict)
            and response.get("protocol") == PROTOCOL_VERSION
            and isinstance(response.get("id"), str)
            and isinstance(response.get("operation"), str)
            and isinstance(response.get("ok"), bool)
            and ("result" in response if response["ok"] else isinstance(response.get("error"), dict))
        ):
            return response
    raise SolidworksProtocolError("No response envelope in host output")


def error_response(operation: str, code: str, message: str, details: dict | None = None) -> dict:
    """Construct a protocol error without including transport diagnostics."""
    return {
        "protocol": PROTOCOL_VERSION,
        "id": str(uuid.uuid4()),
        "operation": operation,
        "ok": False,
        "error": {"code": code, "message": message, "details": details or {}},
        "host": {
            "host_version": None,
            "solidworks_revision": None,
            "solidworks_version": None,
            "instance_id": None,
        },
        "elapsed_ms": 0,
    }
