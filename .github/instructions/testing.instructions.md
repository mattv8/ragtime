---
applyTo: 'tests/**'
---
# Testing Instructions

## Strategy
- Use Python's stdlib `unittest` framework (`unittest.TestCase` and `unittest.IsolatedAsyncioTestCase`).
- Fast and targeted execution using `pytest`.
- Emphasize integration tests that test the actual flow of the application. Avoid overly mocking databases or core infrastructure where possible unless absolutely necessary for external boundaries (like network calls).
- The default pytest run (CI, `tests/run_all_tests.sh`, the `python-test` Docker target) has no Postgres or other live service. A test that needs one must be opt-in: gate the class with `unittest.skipUnless` on a dedicated environment variable, as `INSTRUCTION_SKILLS_TEST_DATABASE_URL` and `RAGTIME_BYO_INTEGRATION` do, and run it against a throwaway database.

Dirs & patterns
- Python tests: `tests/test_*.py`
- Schema reference: `prisma/schema.prisma`
- Other suites: Vitest files beside frontend sources (`npm test` in `ragtime/frontend/`), the SolidWorks host's xUnit tests in `dotnet/tests/`, and the storage service's Maven tests in `storage/`.

## BASIC principles
- Readability: descriptive names; flat setup/asserts
- Autonomy: no shared state/order deps
- Speed: prefer integrated tests over heavily mocked ones. In default-run tests, mock external third-party services (HTTPX), LLM backends, and the database handle (`get_db`); cover real database behavior in the opt-in tests described above. Each test has a 300-second timeout.
- Autonomy also means no fixed paths or ports, so optional parallel runs (`pytest -n 2 --dist loadfile`) stay safe.
- Clarity: one assertion or tight group; comment non-obvious setup

## Static checks on tests
- mypy and pyright type-check `tests/` in CI. Where a `SimpleNamespace` or mock stands in for a typed model, pass it through `cast(Any, ...)`.
- Ruff lint, format, and import-order checks, `docker/scripts/fix_inline_imports.py tests --check`, and the jscpd duplication check (`.jscpd.json`) also cover `tests/`.

## FastAPI Route Testing
- Avoid firing up a real ASGI server or using `TestClient` for every logic test unless doing an end-to-end route check.
- When unit-testing internal route logic, you can build raw `starlette.requests.Request` objects and manually call route functions with them:
  ```python
  from starlette.requests import Request


  def _build_request(path: str) -> Request:
      return Request(
          {
              "type": "http",
              "method": "GET",
              "path": path,
              "headers": [(b"host", b"ragtime.dev")],
              "scheme": "https",
          }
      )
  ```

## Boundaries and Isolation
- Patch specific external boundaries using `mock.patch.object` and `SimpleNamespace` (e.g. LLM calls or external APIs).
- For database behavior (constraints, queries, migrations), write an opt-in test against a real database rather than replacing Prisma `repository` objects with mock structures. Default-run tests patch `get_db` or the repository function they call.
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
bash tests/run_all_tests.sh                 # full local gate: lint, types, pytest, frontend, .NET, storage
```

Do not run tests through the shared `ragtime-dev` container unless it is serving your checkout; see `.agents/dev-worktree-switching.md`.

Common issues
- `asyncio` Task Mocking: Need to ensure futures wrap `mock.AsyncMock` when dealing with awaitables heavily nested inside un-patched internals.
- Event Loops: Because of `IsolatedAsyncioTestCase`, if you instantiate `asyncio.Event()` early in `setUp` or `__init__`, ensure you don't bind to an out-of-scope loop.
