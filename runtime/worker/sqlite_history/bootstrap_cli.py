"""Operator CLI for the durable SQLite-history Restic bootstrap API."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO
from uuid import UUID, uuid4

_DEFAULT_URL = "http://127.0.0.1:8090"
_TERMINAL_FAILURES = frozenset({"blocked", "interrupted", "cancelled", "failed"})
_TERMINAL_STATUSES = _TERMINAL_FAILURES | {"completed"}


@dataclass(frozen=True)
class HttpTransport:
    """Minimal injectable HTTP boundary used by the command and its tests."""

    request: Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


class HttpResponseError(RuntimeError):
    """A response reached the runtime, so acceptance is definitive."""

    def __init__(self, status: int) -> None:
        super().__init__(f"runtime returned HTTP {status}")
        self.status = status


def _stdlib_request(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory or run the durable SQLite-history Restic bootstrap.",
        epilog=(
            "Examples:\n"
            "  %(prog)s --workspace-id workspace-a\n"
            "  %(prog)s --apply --user-id local:admin --workspace-id workspace-a\n"
            "  %(prog)s --resume 123e4567-e89b-12d3-a456-426614174000 --retry-failed\n"
            "  %(prog)s --status 123e4567-e89b-12d3-a456-426614174000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace-id", action="append", default=[], help="Restrict inventory or apply to this workspace (repeatable).")
    parser.add_argument("--apply", action="store_true", help="Accept a new durable bootstrap run.")
    parser.add_argument("--user-id", help="Actor recorded for --apply.")
    parser.add_argument("--run-id", help="UUID used for --apply idempotency.")
    parser.add_argument("--resume", metavar="UUID", help="Resume an existing bootstrap run.")
    parser.add_argument("--retry-failed", action="store_true", help="Permit failed child workspace operations to be retried with a new child ID.")
    parser.add_argument("--status", metavar="UUID", help="Read one bootstrap run status.")
    parser.add_argument("--cancel", metavar="UUID", help="Request cooperative cancellation for a bootstrap run.")
    parser.add_argument("--no-wait", action="store_true", help="Return immediately after apply or resume acceptance.")
    parser.add_argument("--url", default=_DEFAULT_URL, help=f"Runtime API URL (default: {_DEFAULT_URL}).")
    parser.add_argument("--timeout-seconds", type=float, default=3600, help="Maximum status-observation time (default: 3600).")
    return parser


def _validated_uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID") from error


def _validated_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--url must be an absolute http(s) URL without credentials, query, or fragment")
    if parsed.scheme == "http":
        try:
            addresses = [ipaddress.ip_address(parsed.hostname)]
        except ValueError:
            try:
                addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)]
            except (OSError, ValueError) as error:
                raise ValueError("plain HTTP hostname could not be resolved to a loopback address") from error
        if not addresses or not all(address.is_loopback for address in addresses):
            raise ValueError("plain HTTP is permitted only for a loopback runtime service")
        address = next((item for item in addresses if item.version == 4), addresses[0])
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    return value.rstrip("/")


def _endpoint(base_url: str, path: str, query: Sequence[tuple[str, str]] = ()) -> str:
    suffix = urllib.parse.urlencode(query)
    return f"{base_url}{path}" + (f"?{suffix}" if suffix else "")


def _decode_response(status: int, raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"runtime returned HTTP {status} with a non-JSON response") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"runtime returned HTTP {status} with an invalid JSON response")
    if not 200 <= status < 300:
        raise HttpResponseError(status)
    return value


def _request(
    transport: HttpTransport,
    method: str,
    url: str,
    token: str | None,
    timeout: float,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    status, raw = transport.request(method, url, headers, body, timeout)
    return _decode_response(status, raw)


def _emit(stream: TextIO, value: Mapping[str, object]) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    stream.flush()


def _safe_error(stream: TextIO, message: str) -> None:
    stream.write(f"bootstrap-cli: {message}\n")
    stream.flush()


def _exit_for_status(result: Mapping[str, object]) -> int:
    status = result.get("status")
    return 1 if status in _TERMINAL_FAILURES else 0


def _observe(
    transport: HttpTransport,
    base_url: str,
    token: str | None,
    run_id: str,
    timeout_seconds: float,
    stdout: TextIO,
    stderr: TextIO,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    started = monotonic()
    delay = 1.0
    try:
        while True:
            remaining = timeout_seconds - (monotonic() - started)
            if remaining <= 0:
                _emit(stdout, {"error": "Timed out while observing run", "run_id": run_id, "status": "interrupted"})
                return 1
            result = _request(transport, "GET", _endpoint(base_url, f"/sqlite-history/bootstrap/{run_id}"), token, min(30.0, remaining))
            status = result.get("status")
            if status in _TERMINAL_STATUSES:
                _emit(stdout, result)
                return _exit_for_status(result)
            _safe_error(stderr, f"run {run_id} is {status or 'unknown'}; observing")
            remaining = timeout_seconds - (monotonic() - started)
            if remaining <= 0:
                continue
            sleep(min(delay, remaining))
            delay = min(delay * 2, 30.0)
    except KeyboardInterrupt:
        _emit(stdout, {"error": "Observation interrupted; runtime work continues", "run_id": run_id, "status": "interrupted"})
        return 1
    except (OSError, RuntimeError, urllib.error.URLError):
        _emit(stdout, {"error": "Unable to observe run", "run_id": run_id, "status": "interrupted"})
        return 1


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: HttpTransport | object | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run the command. Injection points intentionally keep this module stdlib-contained."""
    parser = _parser()
    args = parser.parse_args(argv)
    out: TextIO = stdout or sys.stdout
    err: TextIO = stderr or sys.stderr
    environment = os.environ if environ is None else environ
    request_transport = transport or HttpTransport(_stdlib_request)
    if not isinstance(request_transport, HttpTransport):
        request_transport = HttpTransport(request_transport.request)  # type: ignore[attr-defined]

    modes = sum(bool(value) for value in (args.apply, args.resume, args.status, args.cancel))
    if (
        modes > 1
        or (args.retry_failed and not args.resume)
        or (args.no_wait and not (args.apply or args.resume))
        or (args.user_id and not args.apply)
        or (args.run_id and not args.apply)
    ):
        parser.error("select exactly one action; --retry-failed requires --resume, --no-wait applies to --apply/--resume, and --user-id requires --apply")
    if args.apply and not args.user_id:
        parser.error("--apply requires --user-id")
    if args.workspace_id and (args.resume or args.status or args.cancel):
        parser.error("--workspace-id is valid only for inventory and --apply")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive finite number")
    try:
        base_url = _validated_base_url(args.url)
        for label, value in (("--run-id", args.run_id), ("--resume", args.resume), ("--status", args.status), ("--cancel", args.cancel)):
            if value:
                _validated_uuid(value, label)
    except ValueError as error:
        parser.error(str(error))
    token = (environment.get("RUNTIME_AUTH_TOKEN") or environment.get("RUNTIME_MANAGER_AUTH_TOKEN") or "").strip() or None

    try:
        if not modes:
            result = _request(
                request_transport,
                "GET",
                _endpoint(base_url, "/sqlite-history/bootstrap/inventory", [("workspace_id", workspace_id) for workspace_id in args.workspace_id]),
                token,
                args.timeout_seconds,
            )
            _emit(out, result)
            return 0

        if args.apply:
            run_id = str(UUID(args.run_id)) if args.run_id else str(uuid4())
            payload: dict[str, object] = {"run_id": run_id, "user_id": args.user_id, "workspace_ids": args.workspace_id or None}
            _emit(out, {"event": "submitting", "run_id": run_id})
            try:
                result = _request(request_transport, "POST", _endpoint(base_url, "/sqlite-history/bootstrap"), token, args.timeout_seconds, payload)
            except HttpResponseError as error:
                if 400 <= error.status < 500:
                    _emit(
                        out,
                        {
                            "instruction": "Acceptance was definitively rejected by the runtime; do not retry this run ID.",
                            "run_id": run_id,
                            "status": "rejected",
                            "http_status": error.status,
                        },
                    )
                else:
                    _emit(out, {"instruction": "Acceptance is unknown; observe this same run ID with --status.", "run_id": run_id, "status": "unknown"})
                return 1
            except (OSError, RuntimeError, urllib.error.URLError):
                _emit(out, {"instruction": "Acceptance is unknown; observe this same run ID with --status.", "run_id": run_id, "status": "unknown"})
                return 1
            _emit(out, result)
            if args.no_wait:
                return _exit_for_status(result)
            return _observe(request_transport, base_url, token, run_id, args.timeout_seconds, out, err, monotonic, sleep)

        if args.resume:
            run_id = str(UUID(args.resume))
            result = _request(
                request_transport,
                "POST",
                _endpoint(base_url, f"/sqlite-history/bootstrap/{run_id}/resume"),
                token,
                min(30.0, args.timeout_seconds),
                {"retry_failed": args.retry_failed},
            )
            _emit(out, result)
            if args.no_wait:
                return _exit_for_status(result)
            return _observe(request_transport, base_url, token, run_id, args.timeout_seconds, out, err, monotonic, sleep)

        run_id = str(UUID(args.status or args.cancel))
        if args.status:
            result = _request(request_transport, "GET", _endpoint(base_url, f"/sqlite-history/bootstrap/{run_id}"), token, min(30.0, args.timeout_seconds))
        else:
            result = _request(
                request_transport, "POST", _endpoint(base_url, f"/sqlite-history/bootstrap/{run_id}/cancel"), token, min(30.0, args.timeout_seconds), {}
            )
        _emit(out, result)
        return _exit_for_status(result)
    except (OSError, RuntimeError, urllib.error.URLError):
        _safe_error(err, "Runtime request failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
