from typing import Any

class Prisma:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class Json:
    data: Any
    def __init__(self, data: Any) -> None: ...

class _Fields:
    def Json(self, data: Any) -> Json: ...

fields: _Fields
