import sys
import types
import unittest
from unittest import mock

fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _fake_ensure_copilot_token_fresh(*_args, **_kwargs):
    return None


setattr(fake_copilot_auth, "ensure_copilot_token_fresh", _fake_ensure_copilot_token_fresh)
sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

from ragtime.rag.components import (  # noqa: E402
    extract_userspace_html_asset_references,
    validate_userspace_html_content,
    validate_userspace_python_content,
    validate_userspace_source_content,
)


class UserSpaceCodeValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_validator_reports_real_syntax_error(self) -> None:
        result = await validate_userspace_python_content("def broken(:\n    pass\n", "app.py")

        self.assertFalse(result["ok"])
        self.assertTrue(result["validator_available"])
        self.assertEqual(result["error_count"], 1)
        self.assertIn("app.py:1:", result["errors"][0])
        self.assertIn("invalid syntax", result["errors"][0])

    async def test_html_validator_accepts_template_markup(self) -> None:
        content = """<!doctype html>
<html>
  <body>
    <script src=\"{{ url_for('static', filename='login.js') }}\"></script>
    <form method=\"post\"></form>
  </body>
</html>
"""

        result = await validate_userspace_html_content(content, "templates/login.html")

        self.assertTrue(result["ok"])
        self.assertTrue(result["validator_available"])
        self.assertEqual(result["errors"], [])

    async def test_source_validator_does_not_route_python_or_html_through_typescript(self) -> None:
        with mock.patch(
            "ragtime.rag.components.validate_userspace_typescript_content",
            new_callable=mock.AsyncMock,
            return_value={"ok": True, "validator_available": True, "errors": []},
        ) as ts_validator:
            python_result = await validate_userspace_source_content("print('ok')\n", "app.py")
            html_result = await validate_userspace_source_content("<html></html>", "templates/login.html")

        ts_validator.assert_not_awaited()
        self.assertTrue(python_result["ok"])
        self.assertTrue(html_result["ok"])


class UserSpaceHtmlReferenceExtractionTests(unittest.TestCase):
    def test_extract_html_asset_references_keeps_local_static_paths_only(self) -> None:
        content = """<html>
  <head>
    <script src=\"./dist/main.js?v=1\"></script>
    <script src=\"{{ url_for('static', filename='login.js') }}\"></script>
    <link rel=\"stylesheet\" href=\"/static/app.css#hash\">
    <script src=\"https://cdn.example.com/app.js\"></script>
  </head>
</html>
"""

        references = extract_userspace_html_asset_references(content)

        self.assertEqual(references, ["./dist/main.js", "/static/app.css"])


if __name__ == "__main__":
    unittest.main()
