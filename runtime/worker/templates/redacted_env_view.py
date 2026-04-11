#!/usr/bin/env python3
"""Redacted env/printenv wrapper for workspace sandbox shells.

Deployed into the sandbox at /tmp/.ragtime-agent-shell/redacted_env_view.py
and invoked via wrapper scripts in /tmp/.ragtime-agent-shell/bin/{env,printenv}.

This file is a template managed by runtime/worker/service.py.
Do not edit the deployed copy; edit this source file instead.

Constants below must stay in sync with service.py module-level values:
  _AGENT_SHELL_INTERNAL_ENV_KEYS, _RAGTIME_REDACTED_ENV_FILE_VAR,
  _RAGTIME_REDACTED_ENV_SENTINEL_SET, _RAGTIME_REDACTED_ENV_SENTINEL_MISSING.
"""

import json
import os
import sys

# -- Constants (kept in sync with runtime/worker/service.py) --

METADATA_ENV_VAR = "RAGTIME_REDACTED_ENV_FILE"
INTERNAL_KEYS = ("RAGTIME_REDACTED_ENV_FILE",)
SET_FALLBACK = "*****"
MISSING_FALLBACK = "__RAGTIME_SECRET_MISSING__"
SYSTEM_ENV = "/usr/bin/env"
SYSTEM_PRINTENV = "/usr/bin/printenv"


def load_redacted_items():
    """Read per-key redacted sentinel values from the workspace-env metadata file."""
    metadata_path = os.environ.get(METADATA_ENV_VAR, "")
    if not metadata_path:
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    items = {}
    for item in payload.get("items", []):
        key = str(item.get("key", "") or "").strip()
        if not key:
            continue
        has_value = bool(item.get("has_value", False))
        sentinel = str(item.get("sentinel", "") or "")
        if not sentinel:
            sentinel = SET_FALLBACK if has_value else MISSING_FALLBACK
        items[key] = sentinel
    return items


def actual_env_items(redacted):
    """Return real env vars, excluding internal keys and redacted secret keys."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in INTERNAL_KEYS and key not in redacted
    }


def emit_listing(entries, null_sep=False):
    separator = "\0" if null_sep else "\n"
    payload = separator.join(entries)
    if payload:
        sys.stdout.write(payload)
        if not null_sep:
            sys.stdout.write("\n")


def run_printenv(args):
    if "--help" in args or "--version" in args:
        os.execv(SYSTEM_PRINTENV, [SYSTEM_PRINTENV, *args])
    null_sep = False
    if args and args[0] in ("-0", "--null"):
        null_sep = True
        args = args[1:]
    redacted = load_redacted_items()
    actual = actual_env_items(redacted)
    if args:
        found_all = True
        outputs = []
        for key in args:
            if key in redacted:
                outputs.append(redacted[key])
                continue
            if key in actual:
                outputs.append(actual[key])
                continue
            found_all = False
        emit_listing(outputs, null_sep=null_sep)
        raise SystemExit(0 if found_all else 1)
    entries = [f"{key}={value}" for key, value in actual.items()]
    for key in sorted(redacted):
        entries.append(f"{key}={redacted[key]}")
    emit_listing(entries, null_sep=null_sep)


def run_env(args):
    if args and any(arg not in ("-0", "--null") for arg in args):
        os.execv(SYSTEM_ENV, [SYSTEM_ENV, *args])
    null_sep = bool(args and args[0] in ("-0", "--null"))
    redacted = load_redacted_items()
    actual = actual_env_items(redacted)
    entries = [f"{key}={value}" for key, value in actual.items()]
    for key in sorted(redacted):
        entries.append(f"{key}={redacted[key]}")
    emit_listing(entries, null_sep=null_sep)


def main():
    invoked_as = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    if invoked_as == "env":
        run_env(args)
        return
    run_printenv(args)


if __name__ == "__main__":
    main()
