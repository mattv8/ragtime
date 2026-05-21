from typing import Any

# Fallback typing surface for prisma.types generated symbols.
# Keep this permissive because generated symbols vary with schema.

def __getattr__(name: str) -> Any: ...
