import unittest
from urllib.parse import parse_qsl, urlsplit

from ragtime.api.auth import build_oauth_redirect_url


class OAuthAuthorizationRedirectTests(unittest.TestCase):
    def test_preserves_openchamber_query_parameters_as_top_level_values(self) -> None:
        redirect_uri = "http://127.0.0.1:43110/oauth/callback?server=local&directory=%2Ftmp%2Fopenchamber"

        result = build_oauth_redirect_url(redirect_uri, "issued-code", "csrf-state")

        parsed = urlsplit(result)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        self.assertEqual(query["server"], "local")
        self.assertEqual(query["directory"], "/tmp/openchamber")
        self.assertEqual(query["code"], "issued-code")
        self.assertEqual(query["state"], "csrf-state")
        self.assertNotIn("?", query["directory"])

    def test_adds_parameters_to_uri_without_query(self) -> None:
        result = build_oauth_redirect_url("http://127.0.0.1:8000/callback", "code-value", "state-value")

        self.assertEqual(
            urlsplit(result).query,
            "code=code-value&state=state-value",
        )

    def test_preserves_fragment(self) -> None:
        result = build_oauth_redirect_url("myapp://callback/path#complete", "code-value", "state-value")

        parsed = urlsplit(result)
        self.assertEqual(parsed.fragment, "complete")
        self.assertEqual(dict(parse_qsl(parsed.query)), {"code": "code-value", "state": "state-value"})

    def test_replaces_existing_code_and_state_values(self) -> None:
        redirect_uri = "http://localhost/callback?keep=yes&code=old-code&state=old-state"

        result = build_oauth_redirect_url(redirect_uri, "new-code", "new-state")

        self.assertEqual(
            parse_qsl(urlsplit(result).query, keep_blank_values=True),
            [("keep", "yes"), ("code", "new-code"), ("state", "new-state")],
        )

    def test_preserves_blank_query_values_and_omits_empty_state(self) -> None:
        redirect_uri = "http://localhost/callback?empty=&keep=value"

        result = build_oauth_redirect_url(redirect_uri, "code-value")

        self.assertEqual(
            parse_qsl(urlsplit(result).query, keep_blank_values=True),
            [("empty", ""), ("keep", "value"), ("code", "code-value")],
        )
