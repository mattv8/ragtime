import unittest
from unittest import mock

from ragtime.api import auth
from ragtime.oauth_redirects import build_allowed_origins


class OAuthRedirectDefaultsTests(unittest.TestCase):
    def test_default_settings_allow_documented_hosted_callbacks(self) -> None:
        callbacks = (
            "https://claude.ai/oauth/callback",
            "https://claude.ai/api/mcp/auth_callback",
            "https://www.cursor.com/agents/mcp/oauth/callback",
            "https://chatgpt.com/connector_platform_oauth_redirect",
        )

        with (
            mock.patch.object(auth.settings, "oauth_trusted_redirect_uris", ""),
            mock.patch.object(auth.settings, "allowed_origins", ""),
            mock.patch.object(auth.settings, "debug_mode", False),
        ):
            for callback in callbacks:
                with self.subTest(callback=callback):
                    self.assertTrue(auth.validate_redirect_uri(callback).is_valid)

    def test_default_settings_preserve_native_and_loopback_callbacks(self) -> None:
        callbacks = (
            "cursor://oauth/callback",
            "http://localhost:8787/callback",
            "https://127.0.0.1:43110/oauth/callback",
        )

        with (
            mock.patch.object(auth.settings, "oauth_trusted_redirect_uris", ""),
            mock.patch.object(auth.settings, "allowed_origins", ""),
            mock.patch.object(auth.settings, "debug_mode", False),
        ):
            for callback in callbacks:
                with self.subTest(callback=callback):
                    self.assertTrue(auth.validate_redirect_uri(callback).is_valid)

    def test_default_settings_reject_untrusted_hosted_callbacks(self) -> None:
        callbacks = (
            "https://unrelated.example/callback",
            "https://www.cursor.com/unlisted/callback",
            "https://chatgpt.com/unlisted/callback",
            "https://chatgpt.com/connector/oauth/abc123",
            "https://www.cursor.com.evil.example/agents/mcp/oauth/callback",
            "https://oauth.www.cursor.com/agents/mcp/oauth/callback",
            "http://www.cursor.com/agents/mcp/oauth/callback",
            "http://chatgpt.com/connector_platform_oauth_redirect",
            "https://www.cursor.com:444/agents/mcp/oauth/callback",
        )

        with (
            mock.patch.object(auth.settings, "oauth_trusted_redirect_uris", ""),
            mock.patch.object(auth.settings, "allowed_origins", ""),
            mock.patch.object(auth.settings, "debug_mode", False),
        ):
            for callback in callbacks:
                with self.subTest(callback=callback):
                    self.assertFalse(auth.validate_redirect_uri(callback).is_valid)

    def test_operator_override_allows_exact_custom_callback(self) -> None:
        with (
            mock.patch.object(auth.settings, "oauth_trusted_redirect_uris", "https://operator.example/oauth/callback"),
            mock.patch.object(auth.settings, "allowed_origins", ""),
            mock.patch.object(auth.settings, "debug_mode", False),
        ):
            self.assertTrue(auth.validate_redirect_uri("https://operator.example/oauth/callback").is_valid)

    def test_allowed_origins_include_hosted_defaults_and_loopback(self) -> None:
        self.assertEqual(
            build_allowed_origins(""),
            [
                "https://chatgpt.com",
                "https://claude.ai",
                "https://www.cursor.com",
                "http://127.0.0.1",
                "http://localhost",
                "http://[::1]",
            ],
        )

    def test_allowed_origins_merge_custom_values_without_duplicates(self) -> None:
        origins = build_allowed_origins("https://custom.example, https://www.cursor.com, https://custom.example")

        self.assertIn("https://chatgpt.com", origins)
        self.assertIn("https://claude.ai", origins)
        self.assertIn("https://www.cursor.com", origins)
        self.assertIn("https://custom.example", origins)
        self.assertEqual(origins.count("https://www.cursor.com"), 1)
        self.assertEqual(origins.count("https://custom.example"), 1)
        self.assertNotIn("https://unrelated.example", origins)

    def test_allowed_origins_preserve_explicit_wildcard_mode(self) -> None:
        self.assertEqual(build_allowed_origins("*"), ["*"])


if __name__ == "__main__":
    unittest.main()
