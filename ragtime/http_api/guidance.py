from ragtime.http_api.models import HttpApiAuthMode, HttpApiConnectionConfig


def build_http_api_headers_description(config: HttpApiConnectionConfig) -> str:
    names = config.approved_request_headers
    if not names:
        return "No per-request headers are approved; omit headers."
    return f"Only these per-request headers are approved: {', '.join(names)}. Omit all other headers."


def build_http_api_request_guidance(config: HttpApiConnectionConfig) -> str:
    parts: list[str] = []
    if config.auth_mode != HttpApiAuthMode.NONE:
        parts.append("Authentication is applied automatically. Use resource paths; do not call login or token endpoints.")
    parts.append(build_http_api_headers_description(config))
    return " ".join(parts)
