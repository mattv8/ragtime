from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TEMPLATES_DIR = Path(__file__).with_name("templates")
_TSX_TEMPLATE_PREFIX = "export default String.raw`"
_TSX_TEMPLATE_SUFFIX = "`;"


@lru_cache(maxsize=None)
def _load_tsx_html_template(template_name: str) -> str:
    source = (_TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    start = source.find(_TSX_TEMPLATE_PREFIX)
    end = source.rfind(_TSX_TEMPLATE_SUFFIX)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Invalid TSX HTML template: {template_name}")
    return source[start + len(_TSX_TEMPLATE_PREFIX) : end]


def _render_tsx_html_template(template_name: str, replacements: dict[str, str]) -> str:
    rendered = _load_tsx_html_template(template_name)
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _escape_text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _escape_attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _status_class_suffix(value: Any) -> str:
    raw = str(value or "unavailable")
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw)
    return normalized or "unavailable"


def render_browser_auth_start_page_html(
    *,
    method_label: str,
    methods: list[dict[str, Any]],
    surface_value: str,
    auth_method_key: str | None,
    return_to: str,
    username_value: str,
    password_value: str,
    parent_origin: str | None = None,
    error: str | None = None,
) -> str:
    method_items = "".join(
        "<li>"
        f'<span class="dot status-{_status_class_suffix(method.get("status"))}"></span>'
        f"{_escape_text(method.get('label') or method.get('key') or 'Auth')}"
        f"<small>{_escape_text(method.get('detail'))}</small>"
        "</li>"
        for method in methods
        if bool(method.get("configured"))
    )
    error_html = f'<div class="error">{_escape_text(error)}</div>' if error else ""
    return _render_tsx_html_template(
        "browser_auth_start_page.tsx",
        {
            "__RAGTIME_METHOD_LABEL__": _escape_text(method_label),
            "__RAGTIME_ERROR_HTML__": error_html,
            "__RAGTIME_SURFACE_VALUE__": _escape_attr(surface_value),
            "__RAGTIME_AUTH_METHOD_KEY__": _escape_attr(auth_method_key),
            "__RAGTIME_RETURN_TO__": _escape_attr(return_to),
            "__RAGTIME_USERNAME_VALUE__": _escape_attr(username_value),
            "__RAGTIME_PASSWORD_VALUE__": _escape_attr(password_value),
            "__RAGTIME_PARENT_ORIGIN_JSON__": json.dumps(str(parent_origin or "").strip()),
            "__RAGTIME_METHOD_ITEMS_HTML__": method_items,
        },
    )


def render_preview_host_unreachable_page_html(
    *,
    workspace_id: str,
    preview_origin: str,
    warning: Any,
) -> str:
    preview_host = str(getattr(warning, "preview_host", "") or "")
    base_domain = str(getattr(warning, "resolved_base_domain", "") or "")
    return _render_tsx_html_template(
        "preview_host_unreachable_page.tsx",
        {
            "__RAGTIME_PREVIEW_HOST__": _escape_text(preview_host),
            "__RAGTIME_WORKSPACE_ID__": _escape_text(workspace_id),
            "__RAGTIME_PREVIEW_ORIGIN__": _escape_text(preview_origin),
            "__RAGTIME_PREVIEW_BASE_DOMAIN__": _escape_text(base_domain),
        },
    )


def render_share_unlock_prompt_html(
    *,
    title: str,
    form_action: str,
    subtitle: str | None = None,
    owner_label: str | None = None,
    error: str | None = None,
    next_target: str | None = None,
) -> str:
    subtitle_block = f'<p class="subtitle">{_escape_text(subtitle)}</p>' if subtitle else ""
    owner_block = f'<p class="owner">{_escape_text(owner_label)}</p>' if owner_label else ""
    error_block = f'<p class="error">{_escape_text(error)}</p>' if error else ""
    next_block = f'<input type="hidden" name="next" value="{_escape_attr(next_target)}">' if next_target else ""
    return _render_tsx_html_template(
        "share_unlock_prompt.tsx",
        {
            "__RAGTIME_TITLE__": _escape_text(title),
            "__RAGTIME_FORM_ACTION__": _escape_attr(form_action),
            "__RAGTIME_SUBTITLE_BLOCK__": subtitle_block,
            "__RAGTIME_OWNER_BLOCK__": owner_block,
            "__RAGTIME_ERROR_BLOCK__": error_block,
            "__RAGTIME_NEXT_BLOCK__": next_block,
        },
    )
