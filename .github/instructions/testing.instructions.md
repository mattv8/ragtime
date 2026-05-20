---
applyTo: 'tests/**'
---
# Testing Instructions

## Strategy
- Use Python's stdlib `unittest` framework (`unittest.TestCase` and `unittest.IsolatedAsyncioTestCase`).
- Fast and targeted execution using `pytest`.
- Emphasize integration tests that test the actual flow of the application. Avoid overly mocking databases or core infrastructure where possible unless absolutely necessary for external boundaries (like network calls).

Dirs & patterns
- Python tests: `tests/test_*.py`
- Schema reference: `prisma/schema.prisma`

## BASIC principles
- Readability: descriptive names; flat setup/asserts
- Autonomy: no shared state/order deps
- Speed: prefer integrated tests over heavily mocked ones. Mock only external third-party services (HTTPX) and LLM backends, but use real DB infrastructure when applicable.
- Clarity: one assertion or tight group; comment non-obvious setup

## FastAPI Route Testing
- Avoid firing up a real ASGI server or using `TestClient` for every logic test unless doing an end-to-end route check.
- When unit-testing internal route logic, you can build raw `starlette.requests.Request` objects and manually call route functions with them:
  ```python
  from starlette.requests import Request

  def _build_request(path: str) -> Request:
      return Request({
          "type": "http",
          "method": "GET",
          "path": path,
          "headers": [(b"host", b"ragtime.dev")],
          "scheme": "https",
      })
  ```

## Boundaries and Isolation
- Patch specific external boundaries using `mock.patch.object` and `SimpleNamespace` (e.g. LLM calls or external APIs).
- For database access, prefer using a real test database configuration over replacing Prisma `repository` objects with mock structures.
- To test `asyncio` streams and background tasks, create fake async iterators to mock FastAPI streaming responses if needed.

## Patterns
```python
import unittest
from types import SimpleNamespace
from unittest import mock

class MyFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_does_behavior(self) -> None:
        fake_service = mock.AsyncMock(return_value="mock_value")

        with mock.patch("ragtime.my_module.my_dependency", fake_service):
            result = await do_something()
            self.assertEqual(result, "mock_value")
```

## Commands
```bash
pytest                                      # all tests
pytest tests/test_background_tasks.py       # single file
pytest tests/test_file.py::Class::method    # single method
pytest -k "test_does_behavior"              # filter by name
```

Common issues
- `asyncio` Task Mocking: Need to ensure futures wrap `mock.AsyncMock` when dealing with awaitables heavily nested inside un-patched internals.
- Event Loops: Because of `IsolatedAsyncioTestCase`, if you instantiate `asyncio.Event()` early in `setUp` or `__init__`, ensure you don't bind to an out-of-scope loop.
