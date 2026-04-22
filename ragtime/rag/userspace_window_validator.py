"""Cross-origin ``window.parent`` / ``top`` / ``opener`` access detector.

Centralises the static-analysis rules that `validate_userspace_runtime_contract`
(in :mod:`ragtime.rag.components`) runs against userspace-authored
``dashboard/*.ts(x)`` files.

The preview iframe is always served on a dedicated subdomain origin
(``<workspace_id>.<host>``) while the parent UI lives on the root host, so any
property read on ``window.parent`` / ``window.top`` / ``window.opener`` throws
``SecurityError: Blocked a frame ... from accessing a cross-origin frame``
synchronously. The only cross-origin-safe APIs are ``postMessage(...)`` and
identity comparisons against ``window`` itself (``=== window``, ``!== window``).

This module has no third-party dependencies so it can be imported directly from
tests without dragging in langchain / PIL / etc. from ``components.py``.
"""

from __future__ import annotations

import re
import time

# Direct property access: ``window.parent.foo`` / ``window.parent[key]`` /
# ``opener.foo`` / ``parent[key]``. These throw immediately.
_CROSS_ORIGIN_WINDOW_ACCESS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_$.])(?:window\s*\.\s*)?(parent|top|opener)"
    r"\s*"
    r"(?:"
    r"\[[^\]]+\]"  # subscript access: parent[key]
    r"|"
    r"\.\s*(?!postMessage\b)[A-Za-z_$][\w$]*"  # .prop access except postMessage
    r")"
)

# Handle leak: ``window.parent`` / ``window.top`` / ``window.opener`` mentioned
# in any way other than ``.postMessage(...)`` or an identity comparison. Catches
# ``windows.push(window.parent as CandidateWindow)`` and ``const p = window.parent;``
# — both legal JS, but the variable ``p`` will throw ``SecurityError`` the moment
# its properties are read downstream.
_CROSS_ORIGIN_WINDOW_HANDLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_$.])window\s*\.\s*(parent|top|opener)\b"
    r"(?!"
    r"\s*\.\s*postMessage\b"
    r"|\s*(?:!==|===|==|!=)\s*(?:window\b|self\b|globalThis\b)"
    r")"
)

# Declaration that aliases a window handle, e.g.:
#   const runtimeWindow = window as RuntimeWindowWithContext;
#   const w = (window);
#   const g = globalThis;
#   const w2 = runtimeWindow;
# The trailing negative lookahead ``(?!\s*\.)`` prevents false positives on
# property reads such as ``const loc = window.location``.
_WINDOW_ALIAS_DECL_PATTERN = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"\s*(?::[^=;\n]+)?"
    r"=\s*(?:<[^>]+>\s*)?\(?\s*([A-Za-z_$][\w$]*)\b"
    r"(?!\s*\.)"
)

# Destructured parent/top/opener handles: ``const { parent } = window``.
_WINDOW_DESTRUCTURE_PATTERN = re.compile(
    r"\b(?:const|let|var)\s*\{\s*([^}]+?)\s*\}"
    r"\s*(?::[^=;\n]+)?"
    r"=\s*(?:<[^>]+>\s*)?\(?\s*([A-Za-z_$][\w$]*)\b"
)

# Inline TypeScript cast followed by parent/top/opener access:
#   (window as RuntimeWindow).parent.foo
#   (<RuntimeWindow>window).opener[key]
_CROSS_ORIGIN_CAST_ACCESS_PATTERN = re.compile(
    r"\(\s*(?:<[^>]+>\s*)?([A-Za-z_$][\w$]*)\s+as\s+[^)]+\)"
    r"\s*\.\s*(parent|top|opener)\b"
    r"(?!\s*\.\s*postMessage\b)"
    r"|"
    r"\(\s*<[^>]+>\s*([A-Za-z_$][\w$]*)\s*\)"
    r"\s*\.\s*(parent|top|opener)\b"
    r"(?!\s*\.\s*postMessage\b)"
)

# Built-in global identifiers that always reference the current window object.
# ``self`` is intentionally excluded because user code frequently rebinds it
# (e.g. ``const self = this``), which would produce false positives.
_WINDOW_SEED_ALIASES: frozenset[str] = frozenset({"window", "globalThis"})

_MAX_FINDINGS = 8
_SNIPPET_MAX_LEN = 120
_MAX_ALIAS_SCAN = 256
_VALIDATION_TIMEOUT_SECONDS = 0.35


def _collect_window_aliases(content: str) -> set[str]:
    """Resolve the transitive set of identifiers bound to a window handle.

    Handles alias-of-alias chains such as::

        const runtimeWindow = window as RuntimeWindowWithContext;
        const w = runtimeWindow;

    Both ``runtimeWindow`` and ``w`` end up in the returned set so that
    downstream regex scans can flag ``w.parent.foo`` alongside the original
    ``runtimeWindow.parent.foo``.
    """
    aliases: set[str] = set(_WINDOW_SEED_ALIASES)
    if not content:
        return aliases
    # Iteratively widen the alias set until it stabilises. Each pass adds any
    # ``const X = <known-alias>`` declaration encountered. Bounded by a sane
    # iteration cap to guarantee termination on pathological input.
    for _ in range(8):
        discovered: set[str] = set()
        for match in _WINDOW_ALIAS_DECL_PATTERN.finditer(content):
            lhs = (match.group(1) or "").strip()
            rhs = (match.group(2) or "").strip()
            if lhs and rhs in aliases and lhs not in aliases:
                discovered.add(lhs)
        if not discovered:
            break
        aliases.update(discovered)
    return aliases


def find_cross_origin_window_access(
    content: str,
    *,
    timeout_seconds: float = _VALIDATION_TIMEOUT_SECONDS,
    max_alias_scan: int = _MAX_ALIAS_SCAN,
) -> list[str]:
    """Detect property reads on ``window.parent`` / ``window.top`` / ``window.opener``.

    Preview iframes are always cross-origin with the parent UI, so any such
    read throws ``SecurityError`` at runtime. Returns a de-duplicated list of
    the offending snippets (trimmed to a reasonable length for display).
    """
    if not content:
        return []
    deadline = time.perf_counter() + max(0.0, timeout_seconds)

    def expired() -> bool:
        # Best-effort timeout guard for pathological generated sources.
        # The validator returns whatever findings it has collected so far.
        return timeout_seconds > 0 and time.perf_counter() >= deadline

    findings: list[str] = []
    seen: set[str] = set()

    def add(snippet: str) -> None:
        normalized = snippet.strip()[:_SNIPPET_MAX_LEN]
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        findings.append(normalized)

    for match in _CROSS_ORIGIN_WINDOW_ACCESS_PATTERN.finditer(content):
        if expired():
            return findings
        if len(findings) >= _MAX_FINDINGS:
            return findings
        add(match.group(0))

    for match in _CROSS_ORIGIN_WINDOW_HANDLE_PATTERN.finditer(content):
        if expired():
            return findings
        if len(findings) >= _MAX_FINDINGS:
            return findings
        add(match.group(0))

    # Inline TS casts: ``(window as Foo).parent`` / ``(<Foo>window).opener``.
    for match in _CROSS_ORIGIN_CAST_ACCESS_PATTERN.finditer(content):
        if expired():
            return findings
        if len(findings) >= _MAX_FINDINGS:
            return findings
        cast_target = (match.group(1) or match.group(3) or "").strip()
        if cast_target and cast_target not in _WINDOW_SEED_ALIASES:
            # Only flag casts of known window-like globals; arbitrary casts
            # such as ``(foo as Bar).parent`` are not necessarily unsafe.
            continue
        add(match.group(0))

    window_aliases = sorted(_collect_window_aliases(content), key=len, reverse=True)

    # Destructured parent/top/opener handles: ``const { parent } = window``.
    for match in _WINDOW_DESTRUCTURE_PATTERN.finditer(content):
        if expired():
            return findings
        if len(findings) >= _MAX_FINDINGS:
            return findings
        names_blob = match.group(1) or ""
        rhs = (match.group(2) or "").strip()
        if rhs not in window_aliases:
            continue
        # Split destructuring names on commas, strip rename aliases
        # (``parent: p``) and default values (``parent = null``).
        extracted: list[str] = []
        for raw in names_blob.split(","):
            token = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
            if token in {"parent", "top", "opener"}:
                extracted.append(token)
        if extracted:
            add(match.group(0))

    for alias in window_aliases[:max_alias_scan]:
        if expired():
            return findings
        # ``window`` is covered by the dedicated patterns above; every other
        # alias (including seed globals like ``globalThis`` and any
        # user-declared alias) still needs a per-alias scan.
        if alias == "window":
            continue
        escaped_alias = re.escape(alias)
        alias_access_pattern = re.compile(
            rf"(?<![A-Za-z0-9_$.]){escaped_alias}\s*\.\s*(parent|top|opener)"
            rf"\s*(?:\[[^\]]+\]|\.\s*(?!postMessage\b)[A-Za-z_$][\w$]*)"
        )
        alias_handle_pattern = re.compile(
            rf"(?<![A-Za-z0-9_$.]){escaped_alias}\s*\.\s*(parent|top|opener)\b"
            rf"(?!"
            rf"\s*\.\s*postMessage\b"
            rf"|\s*(?:!==|===|==|!=)\s*(?:{escaped_alias}\b|window\b|self\b|globalThis\b)"
            rf")"
        )

        for match in alias_access_pattern.finditer(content):
            if expired():
                return findings
            if len(findings) >= _MAX_FINDINGS:
                return findings
            add(match.group(0))

        for match in alias_handle_pattern.finditer(content):
            if expired():
                return findings
            if len(findings) >= _MAX_FINDINGS:
                return findings
            add(match.group(0))

    return findings


__all__ = ["find_cross_origin_window_access"]
