"""Shared User Space preview iframe sandbox configuration helpers."""


from typing import TypedDict

class UserSpacePreviewSandboxFlagOption(TypedDict):
    """Canonical metadata for a single iframe sandbox capability."""

    value: str
    label: str
    description: str


USERSPACE_PREVIEW_SANDBOX_FLAG_OPTIONS: tuple[UserSpacePreviewSandboxFlagOption, ...] = (
    {
        "value": "allow-scripts",
        "label": "Scripts",
        "description": "Allows JavaScript execution; without it most apps break, but malicious preview code can actively run in the iframe.",
    },
    {
        "value": "allow-same-origin",
        "label": "Same Origin",
        "description": "Treats the preview as same-origin; this enables normal storage and DOM behavior, but also lets preview code use same-origin browser state more freely.",
    },
    {
        "value": "allow-forms",
        "label": "Forms",
        "description": "Allows form submission; preview pages can post data to remote endpoints instead of being limited to local interaction.",
    },
    {
        "value": "allow-popups",
        "label": "Popups",
        "description": "Allows new windows or tabs; preview code can open external sites or auth flows without extra user mediation.",
    },
    {
        "value": "allow-popups-to-escape-sandbox",
        "label": "Popup Escape",
        "description": "Lets opened popups drop the iframe sandbox; useful for real auth/payment flows, but the new window gets fewer restrictions.",
    },
    {
        "value": "allow-modals",
        "label": "Modals",
        "description": "Allows alert(), confirm(), prompt(), and beforeunload; useful for app UX, but preview code can interrupt users with browser dialogs.",
    },
    {
        "value": "allow-downloads",
        "label": "Downloads",
        "description": "Allows file downloads; preview code can trigger browser downloads, which is convenient but can be abused for noisy or misleading exports.",
    },
    {
        "value": "allow-orientation-lock",
        "label": "Orientation Lock",
        "description": "Allows screen orientation locking; low risk on desktop, but can unexpectedly control device orientation on mobile.",
    },
    {
        "value": "allow-pointer-lock",
        "label": "Pointer Lock",
        "description": "Allows pointer lock; necessary for some immersive UIs, but it can capture the cursor and change expected mouse behavior.",
    },
    {
        "value": "allow-presentation",
        "label": "Presentation",
        "description": "Allows presentation APIs; mostly relevant for slideshow or display apps, but gives preview content more control over presentation surfaces.",
    },
    {
        "value": "allow-storage-access-by-user-activation",
        "label": "Storage Access",
        "description": "Allows storage access requests after user action; useful for embedded auth/session flows, but broadens access to browser-stored state.",
    },
    {
        "value": "allow-top-navigation",
        "label": "Top Navigation",
        "description": "Allows the iframe to navigate the top-level page directly; this is powerful and can replace the Ragtime UI with arbitrary destinations.",
    },
    {
        "value": "allow-top-navigation-by-user-activation",
        "label": "Top Navigation By Activation",
        "description": "Allows top-level navigation only after a user gesture; safer than full top navigation, but clicks in the preview can still leave the app.",
    },
    {
        "value": "allow-top-navigation-to-custom-protocols",
        "label": "Custom Protocol Navigation",
        "description": "Allows links like mailto: or custom app protocols; useful for integrations, but can hand control to external apps on the user machine.",
    },
)

USERSPACE_PREVIEW_SANDBOX_ALL_FLAGS: tuple[str, ...] = tuple(
    option["value"] for option in USERSPACE_PREVIEW_SANDBOX_FLAG_OPTIONS
)

USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS: tuple[str, ...] = (
    "allow-scripts",
    "allow-same-origin",
    "allow-forms",
    "allow-popups",
    "allow-popups-to-escape-sandbox",
    "allow-modals",
    "allow-downloads",
)

_USERSPACE_PREVIEW_SANDBOX_ALLOWED_SET = frozenset(USERSPACE_PREVIEW_SANDBOX_ALL_FLAGS)


def normalize_userspace_preview_sandbox_flags(
    value: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normalize and validate admin-managed iframe sandbox flags."""

    if value is None:
        return list(USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS)

    selected: set[str] = set()
    invalid: set[str] = set()
    for raw in value:
        token = str(raw).strip()
        if not token:
            continue
        if token not in _USERSPACE_PREVIEW_SANDBOX_ALLOWED_SET:
            invalid.add(token)
            continue
        selected.add(token)

    if invalid:
        invalid_values = ", ".join(sorted(invalid))
        allowed_values = ", ".join(USERSPACE_PREVIEW_SANDBOX_ALL_FLAGS)
        raise ValueError(
            "Invalid User Space preview sandbox flags: "
            f"{invalid_values}. Allowed values: {allowed_values}"
        )

    return [flag for flag in USERSPACE_PREVIEW_SANDBOX_ALL_FLAGS if flag in selected]
