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
        self.assertIn("execute-component", text)
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
        self.assertNotIn(
            "Runtime bridge component calls are enforced as read-only and platform-limited; do not attempt writes through this bridge.",
            text,
        )

    def test_static_framework_keeps_browser_bridge_guidance_only(self) -> None:
        text = build_userspace_entrypoint_nudge(
            self._status(framework="static", command="python3 -m http.server"),
            is_default_static=True,
        )
        self.assertNotIn("RAGTIME_BRIDGE_URL", text)
        self.assertIn("context.components", text)

    def test_missing_entrypoint_output_unchanged(self) -> None:
        text = build_userspace_entrypoint_nudge(
            self._status(state="missing", framework=None, command=None),
            is_default_static=False,
        )
        self.assertNotIn("RAGTIME_BRIDGE_URL", text)
