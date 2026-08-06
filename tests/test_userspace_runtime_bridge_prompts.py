import unittest

from ragtime.core.entrypoint_status import EntrypointState, EntrypointStatus
from ragtime.rag.prompts import build_userspace_entrypoint_nudge


class RuntimeBridgePromptTests(unittest.TestCase):
    def _status(
        self,
        state: EntrypointState = "valid",
        framework: str | None = "node",
        command: str | None = "node server.js",
    ) -> EntrypointStatus:
        return EntrypointStatus(
            state=state,
            framework=framework,
            framework_known=bool(framework),
            command=command or "",
            cwd=".",
            error=None,
        )

    def test_valid_server_entrypoint_includes_bridge_guidance(self) -> None:
        text = build_userspace_entrypoint_nudge(self._status(), is_default_static=False)
        self.assertIn("RAGTIME_BRIDGE_URL", text)
        self.assertIn("RAGTIME_BRIDGE_TOKEN", text)
        self.assertIn("/execute-component", text)
        self.assertIn("/sqlite/query", text)
        self.assertIn("/sqlite/mutate", text)
        self.assertIn(
            "SERVER lane (this bridge, bearer token): follows Workspace Tools access policy.",
            text,
        )
        self.assertIn(
            "Tools default read-only unless the workspace owner has Read+Write access and explicitly enables write",
            text,
        )
        self.assertNotIn("global tool config permits writes", text)
        self.assertIn(
            "ALWAYS read-only, regardless of the workspace write toggle.",
            text,
        )
        self.assertIn(
            "Never attempt INSERT/UPDATE/DELETE or other mutations from browser code.",
            text,
        )
        self.assertIn(
            "The runtime token is the backend service identity; backend mutation routes must enforce their own authz/authn and must never expose that token to browser code.",
            text,
        )
        self.assertIn('"request": {"query": "SELECT ... LIMIT 100"}', text)
        self.assertIn('"request": {"method": "GET", "path": "/customers"}', text)
        self.assertIn('database_name: "app.sqlite3"', text)
        self.assertIn("parameterized SQL", text)
        self.assertIn("positional-list or named-dict `parameters`", text)
        self.assertIn("max_rows` up to 500", text)
        self.assertIn("`columns`, `rows`, `row_count`, and `truncated`", text)
        self.assertIn("1..500 structured `insert`, `upsert`, `update`, or `delete` operations", text)
        self.assertIn("The bridge does not accept raw writable SQL", text)
        self.assertIn("The target owner must grant Shared SQLite access", text)
        self.assertIn("Read grants allow queries; Read / Write grants allow structured mutations", text)
        self.assertIn("source membership plus target viewer membership for query", text)
        self.assertIn("target editor membership for mutation", text)
        self.assertIn("The target workspace owns `.ragtime/db/migrations/*.sql` and all DDL", text)
        self.assertIn("Consumers cannot run DDL, PRAGMA, ATTACH, or extension loading", text)
        self.assertIn("Handle HTTP 409 as busy, 504 as timeout, and 503 as audit unavailable", text)
        self.assertIn("A 503 audit failure occurs before writes and is safe to retry", text)
        self.assertIn("Do not blindly retry other mutation failures", text)
        self.assertIn("server code only", text)
        self.assertNotIn(
            "Runtime bridge component calls are enforced as read-only and platform-limited; do not attempt writes through this bridge.",
            text,
        )
        self.assertNotIn("window.__ragtime_context", text)

    def test_static_framework_keeps_browser_bridge_guidance_only(self) -> None:
        text = build_userspace_entrypoint_nudge(
            self._status(framework="static", command="python3 -m http.server"),
            is_default_static=True,
        )
        self.assertNotIn("RAGTIME_BRIDGE_URL", text)
        self.assertNotIn("/sqlite/query", text)
        self.assertNotIn("/sqlite/mutate", text)
        self.assertIn("context.components", text)

    def test_missing_entrypoint_output_unchanged(self) -> None:
        text = build_userspace_entrypoint_nudge(
            self._status(state="missing", framework=None, command=None),
            is_default_static=False,
        )
        self.assertNotIn("RAGTIME_BRIDGE_URL", text)
        self.assertNotIn("/sqlite/query", text)
        self.assertNotIn("/sqlite/mutate", text)
