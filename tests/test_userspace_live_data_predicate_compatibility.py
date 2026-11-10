import sys
import types

import pytest


def _call_requires_live_data_contract(
    relative_path: str,
    live_data_requested: bool,
    workspace_has_tools: bool,
) -> bool:
    fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")

    async def ensure_copilot_token_fresh(*_args: object, **_kwargs: object) -> None:
        return None

    fake_copilot_auth.ensure_copilot_token_fresh = ensure_copilot_token_fresh
    sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

    # Load after the Copilot stub, with components before service to avoid import cycles.
    import ragtime.rag.components as rag_components  # inline-import: keep
    from ragtime.userspace.service import _requires_live_data_contract as predicate  # inline-import: keep

    assert rag_components
    return predicate(relative_path, live_data_requested, workspace_has_tools)


@pytest.mark.parametrize(
    ("relative_path", "live_data_requested", "workspace_has_tools", "expected"),
    [
        ("dashboard/main.ts", False, False, False),
        ("dashboard/main.ts", False, True, False),
        ("dashboard/main.ts", True, False, True),
        ("dashboard/main.ts", True, True, True),
        ("dashboard/Main.TS", True, True, True),
        ("dashboard\\main.ts", True, True, True),
        ("dashboard/widget.tsx", True, False, True),
        ("app.js", True, True, True),
        ("app.jsx", True, True, True),
        ("app.mjs", True, True, True),
        ("app.cjs", True, True, True),
        ("app.mts", True, True, True),
        ("app.cts", True, True, True),
        ("notes.txt", True, True, False),
        ("", True, True, False),
    ],
)
def test_requires_live_data_contract_compatibility(
    relative_path: str,
    live_data_requested: bool,
    workspace_has_tools: bool,
    expected: bool,
) -> None:
    assert _call_requires_live_data_contract(relative_path, live_data_requested, workspace_has_tools) is expected
