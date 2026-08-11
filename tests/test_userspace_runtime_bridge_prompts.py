import unittest

from ragtime.core.entrypoint_status import EntrypointState, EntrypointStatus
from ragtime.rag.prompts import (
    build_shared_sqlite_prompt_fragment,
    build_userspace_entrypoint_nudge,
    build_userspace_mode_prompt_addition,
)


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
        self.assertIn("`RAGTIME_BRIDGE_URL`", text)
        self.assertIn("`RAGTIME_BRIDGE_TOKEN`", text)
        self.assertIn("`{RAGTIME_BRIDGE_URL}/execute-component`", text)
        self.assertIn(
            "SERVER execution surface (this bridge, bearer token): follows Workspace Tools access policy.",
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
            "BROWSER execution surface (`context.components[...].execute()` in preview/shared pages):",
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
        self.assertNotIn("/sqlite/query", text)
        self.assertNotIn("/sqlite/mutate", text)
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

    def test_shared_sqlite_prompt_fragment_omits_guidance_for_empty_inventory(self) -> None:
        self.assertEqual(build_shared_sqlite_prompt_fragment([]), "")

        prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=False,
            shared_sqlite_databases=[],
        )

        self.assertNotIn("/sqlite/query", prompt)
        self.assertNotIn("/sqlite/mutate", prompt)

    def test_userspace_prompt_combines_data_boundaries_matrix(self) -> None:
        local_only = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=True,
            shared_sqlite_databases=[],
        )
        shared_only = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=False,
            has_live_data_tools=False,
            shared_sqlite_databases=[
                {
                    "workspace_id": "ws_target",
                    "workspace_name": "Target Workspace",
                    "database_name": "app.sqlite3",
                    "access_mode": "read",
                }
            ],
        )
        both = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=True,
            shared_sqlite_databases=[
                {
                    "workspace_id": "ws_target",
                    "workspace_name": "Target Workspace",
                    "database_name": "app.sqlite3",
                    "access_mode": "read_write",
                }
            ],
        )
        neither = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=False,
            has_live_data_tools=False,
            shared_sqlite_databases=[],
        )
        include_without_live_tools = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=True,
            has_live_data_tools=False,
            shared_sqlite_databases=[],
        )

        self.assertIn("#### Data and persistence boundaries", local_only)
        self.assertIn("##### Lane A - Live tool data", local_only)
        self.assertIn("##### Lane B - Primary workspace SQLite", local_only)
        self.assertNotIn("##### Shared SQLite target workspaces", local_only)

        self.assertIn("#### Data and persistence boundaries", shared_only)
        self.assertNotIn("##### Lane A - Live tool data", shared_only)
        self.assertNotIn("##### Lane B - Primary workspace SQLite", shared_only)
        self.assertIn("##### Shared SQLite target workspaces", shared_only)
        self.assertNotIn("Every schema change requires a new numbered migration file", shared_only)

        self.assertIn("#### Data and persistence boundaries", both)
        self.assertLess(both.index("##### Lane A - Live tool data"), both.index("##### Lane B - Primary workspace SQLite"))
        self.assertLess(both.index("##### Lane B - Primary workspace SQLite"), both.index("##### Shared SQLite target workspaces"))
        self.assertIn("These sections define responsibilities, not mandatory work in every turn.", both)
        self.assertIn("Shared SQLite is not a third persistence lane", both)
        self.assertIn("`/sqlite/query` and `/sqlite/mutate` are only for cross-workspace access", both)
        self.assertIn("Browser code must call an app-owned server route", both)
        self.assertIn("must never call bridge endpoints directly or receive `RAGTIME_BRIDGE_TOKEN`", both)

        self.assertNotIn("#### Data and persistence boundaries", neither)

        self.assertIn("#### Data and persistence boundaries", include_without_live_tools)
        self.assertNotIn("##### Lane A - Live tool data", include_without_live_tools)
        self.assertIn("##### Lane B - Primary workspace SQLite", include_without_live_tools)

    def test_shared_sqlite_prompt_fragment_lists_targets_with_sanitized_permissions(self) -> None:
        accessible_databases = [
            {
                "workspace_id": "ws_beta",
                "workspace_name": 'Beta\nWorkspace\t"ops"',
                "database_name": "app.sqlite3",
                "access_mode": "read_write",
            },
            {
                "workspace_id": "ws_alpha",
                "workspace_name": "Alpha\r\nWorkspace" + (" x" * 120),
                "database_name": "app.sqlite3",
                "access_mode": "read",
            },
        ]

        fragment = build_shared_sqlite_prompt_fragment(accessible_databases)
        prompt = build_userspace_mode_prompt_addition(
            include_sqlite_persistence=False,
            shared_sqlite_databases=accessible_databases,
        )

        self.assertIn("##### Shared SQLite target workspaces", fragment)
        self.assertIn("Use Shared SQLite from server code only", fragment)
        self.assertIn("A server-backed entrypoint is required", fragment)
        self.assertNotIn("Contract body examples include", fragment)
        self.assertIn("Shared SQLite is not a third persistence lane", fragment)
        self.assertIn('database_name: "app.sqlite3"', fragment)
        self.assertIn("parameterized SQL", fragment)
        self.assertIn("positional-list or named-dict `parameters`", fragment)
        self.assertIn("max_rows` up to 500", fragment)
        self.assertIn("`columns`, `rows`, `row_count`, and `truncated`", fragment)
        self.assertIn("1..500 structured `insert`, `upsert`, `update`, or `delete` operations", fragment)
        self.assertIn("The bridge does not accept raw writable SQL", fragment)
        self.assertIn("The target owner must grant Shared SQLite access", fragment)
        self.assertIn("Read grants allow queries; Read / Write grants allow structured mutations", fragment)
        self.assertIn("source membership plus target viewer membership for query", fragment)
        self.assertIn("target editor membership for mutation", fragment)
        self.assertIn("The target workspace owns `.ragtime/db/migrations/*.sql` and all DDL", fragment)
        self.assertIn("Consumers cannot run DDL, PRAGMA, ATTACH, or extension loading", fragment)
        self.assertIn("Handle HTTP 409 as busy, 504 as timeout, and 503 as audit unavailable", fragment)
        self.assertIn("A 503 audit failure occurs before writes and is safe to retry", fragment)
        self.assertIn("Do not blindly retry other mutation failures", fragment)
        self.assertIn("`/sqlite/query` and `/sqlite/mutate` are only for cross-workspace access", fragment)
        self.assertIn("Browser code must call an app-owned server route", fragment)
        self.assertIn("must never call bridge endpoints directly or receive `RAGTIME_BRIDGE_TOKEN`", fragment)
        self.assertLess(fragment.index("ws_alpha"), fragment.index("ws_beta"))
        self.assertIn('- Workspace: "Alpha Workspace', fragment)
        self.assertIn('- Workspace: "Beta Workspace \\"ops\\""', fragment)
        self.assertIn("- target_workspace_id: `ws_alpha`", fragment)
        self.assertIn("- target_workspace_id: `ws_beta`", fragment)
        self.assertIn("- database_name: `app.sqlite3`", fragment)
        self.assertIn("- effective_permission: Read", fragment)
        self.assertIn("- effective_permission: Read / Write", fragment)
        self.assertNotIn("Beta\nWorkspace", fragment)
        self.assertNotIn("Alpha\r", fragment)
        self.assertIn(fragment, prompt)
