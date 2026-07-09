from __future__ import annotations

import sys
import types


def install_fake_rag_prompts() -> bool:
    """Install a minimal stub for ``ragtime.rag.prompts`` if it is not already present.

    Returns ``True`` when the caller should later call ``remove_fake_rag_prompts``
    to clean up the stub.
    """
    inserted = "ragtime.rag.prompts" not in sys.modules
    if inserted:
        fake_rag_package = types.ModuleType("ragtime.rag")
        fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
        setattr(
            fake_prompts_module,
            "build_workspace_scm_setup_prompt",
            lambda *args, **kwargs: "",
        )
        setattr(fake_rag_package, "prompts", fake_prompts_module)
        sys.modules.setdefault("ragtime.rag", fake_rag_package)
        sys.modules["ragtime.rag.prompts"] = fake_prompts_module
    return inserted


def remove_fake_rag_prompts(inserted: bool) -> None:
    """Remove the fake ``ragtime.rag.prompts`` stub installed by ``install_fake_rag_prompts``."""
    if inserted:
        sys.modules.pop("ragtime.rag", None)
        sys.modules.pop("ragtime.rag.prompts", None)
