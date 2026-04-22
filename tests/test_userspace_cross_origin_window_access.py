import importlib.util
import unittest
from pathlib import Path
from unittest import mock


def _load_validator_module():
    """Load ``userspace_window_validator`` without triggering ``ragtime.rag``
    package init (which eagerly imports components.py + langchain/PIL).

    The module is deliberately dependency-free (``re`` only), so loading it
    in isolation via ``importlib.util`` is safe and keeps the test suite
    runnable outside the container.
    """
    module_path = (
        Path(__file__).resolve().parents[1]
        / "ragtime"
        / "rag"
        / "userspace_window_validator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ragtime_userspace_window_validator_under_test", module_path
    )
    assert spec and spec.loader, f"failed to build spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


find_cross_origin_window_access = _load_validator_module().find_cross_origin_window_access


class CrossOriginWindowAccessTests(unittest.TestCase):
    def test_detects_alias_based_parent_handle_leak(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
const parentWindow = runtimeWindow.parent as unknown as RuntimeWindowWithContext;
entries.push(parentWindow.__ragtime_context);
findPostgresComponent(runtimeWindow.parent);
"""

        hits = find_cross_origin_window_access(content)

        self.assertIn("runtimeWindow.parent", hits)

    def test_detects_alias_based_parent_property_access(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
if (runtimeWindow.parent && runtimeWindow.parent !== runtimeWindow) {
  return runtimeWindow.parent.__ragtime_context;
}
"""

        hits = find_cross_origin_window_access(content)

        self.assertIn("runtimeWindow.parent.__ragtime_context", hits)

    def test_allows_postmessage_on_window_alias_parent(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
runtimeWindow.parent.postMessage({ type: 'ping' }, '*');
"""

        hits = find_cross_origin_window_access(content)

        self.assertEqual(hits, [])

    def test_detects_alias_of_alias_parent_access(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
const w = runtimeWindow;
const ctx = w.parent.__ragtime_context;
"""

        hits = find_cross_origin_window_access(content)

        self.assertTrue(
            any("w.parent" in hit for hit in hits),
            f"expected alias-of-alias hit, got {hits!r}",
        )

    def test_detects_globalthis_parent_access(self) -> None:
        content = """
const ctx = globalThis.parent.__ragtime_context;
"""

        hits = find_cross_origin_window_access(content)

        self.assertTrue(
            any("globalThis.parent" in hit for hit in hits),
            f"expected globalThis hit, got {hits!r}",
        )

    def test_detects_inline_cast_parent_access(self) -> None:
        content = """
const ctx = (window as RuntimeWindowWithContext).parent.__ragtime_context;
const opener = (<RuntimeWindow>window).opener;
"""

        hits = find_cross_origin_window_access(content)

        self.assertTrue(
            any(".parent" in hit for hit in hits),
            f"expected cast hit, got {hits!r}",
        )

    def test_detects_destructured_parent_handle(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
const { parent, opener } = runtimeWindow;
console.log(parent, opener);
"""

        hits = find_cross_origin_window_access(content)

        self.assertTrue(
            any("parent" in hit and "=" in hit for hit in hits),
            f"expected destructure hit, got {hits!r}",
        )

    def test_does_not_flag_window_property_reads(self) -> None:
        # Regression: ``const x = window.location`` previously seeded ``x`` as
        # a window alias, which could flag innocent ``x.parent`` reads on
        # unrelated objects. The declaration pattern now requires the RHS to
        # be a bare identifier (no trailing ``.prop``).
        content = """
const loc = window.location;
const origin = window.origin;
const params = new URLSearchParams(loc.search);
"""

        hits = find_cross_origin_window_access(content)

        self.assertEqual(hits, [])

    def test_does_not_flag_self_rebinding(self) -> None:
        # ``self`` is intentionally excluded from the seed alias set because
        # user code frequently rebinds it (e.g. ``const self = this``). This
        # test guards against accidentally re-introducing ``self`` as a seed.
        content = """
class Widget {
  render() {
    const self = this;
    return self.parent.render();
  }
}
"""

        hits = find_cross_origin_window_access(content)

        self.assertEqual(hits, [])

    def test_allows_identity_comparison(self) -> None:
        content = """
const runtimeWindow = window as RuntimeWindowWithContext;
if (runtimeWindow.parent === runtimeWindow) {
  return null;
}
if (runtimeWindow.parent !== window) {
  postMessageToParent();
}
"""

        hits = find_cross_origin_window_access(content)

        self.assertEqual(hits, [])

    def test_timeout_guard_returns_quickly(self) -> None:
        content = "const runtimeWindow = window; runtimeWindow.parent.foo;"

        # Force timeout path deterministically without depending on wall-clock.
        with mock.patch("time.perf_counter", side_effect=[100.0, 100.5]):
            hits = find_cross_origin_window_access(content, timeout_seconds=0.1)

        self.assertEqual(hits, [])

    def test_alias_scan_cap_still_detects_basic_violation(self) -> None:
        aliases = "\n".join(f"const w{i} = window;" for i in range(600))
        content = f"""
{aliases}
const runtimeWindow = window as RuntimeWindowWithContext;
runtimeWindow.parent.__ragtime_context;
"""

        hits = find_cross_origin_window_access(content, max_alias_scan=10)

        self.assertTrue(
            any("runtimeWindow.parent" in hit for hit in hits),
            f"expected base violation hit, got {hits!r}",
        )


if __name__ == "__main__":
    unittest.main()
