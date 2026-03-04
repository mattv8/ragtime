"""Container capability detection helpers."""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ContainerCapabilities:
    """Effective Linux capability flags relevant to containerized features."""

    privileged: bool
    has_sys_admin: bool


def get_container_capabilities() -> ContainerCapabilities:
    """Detect whether the current process has mount-related container capabilities.

    CAP_SYS_ADMIN is capability bit 21 from ``CapEff`` in ``/proc/self/status``.
    """
    privileged = False
    has_sys_admin = False

    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if not line.startswith("CapEff:"):
                continue

            cap_hex = line.split(":", 1)[1].strip()
            cap_bits = int(cap_hex, 16)

            cap_sys_admin_bit = 1 << 21
            has_sys_admin = bool(cap_bits & cap_sys_admin_bit)

            high_privilege_bits = (1 << 27) | (1 << 17) | (1 << 19) | (1 << 21)
            privileged = (cap_bits & high_privilege_bits) == high_privilege_bits
            break
    except Exception:
        pass

    return ContainerCapabilities(privileged=privileged, has_sys_admin=has_sys_admin)
