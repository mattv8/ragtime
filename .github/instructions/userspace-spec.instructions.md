---
applyTo: 'ragtime/userspace/**'
---

# User Space Specification (Current Implemented State)

Last updated: 2026-02-21
Repository: `mattv8/ragtime` (branch: `userspaces`)

## 1) Purpose

This document describes the **current implemented behavior** of User Space in Ragtime.
It is a static specification of what exists now: user experience, API surface, storage model,
authorization behavior, and known boundaries.

---

## 2) Feature Summary

User Space is a workspace-oriented environment for building persistent dashboard/report artifacts with chat-assisted iteration.

Implemented capabilities:

- Dedicated **User Space** navigation/view.
- Left-side chat integrated with the existing conversation/task runtime.
- Viewer role chat is visible but read-only in User Space (send controls disabled in UI; backend still enforces role checks).
- Workspace-scoped file management and artifact preview.
- Workspace-scoped tool selection (admin-configured tools only).
- Snapshot/restore using **git commit history** per workspace.
- User Space-only agent tools for file I/O plus autonomous checkpoint snapshots.
- Workspace sharing with role-based access (`owner`, `editor`, `viewer`).
- Share links with owner-scoped custom slugs.
- Public full-screen shared dashboard rendering via path URL `/{username}/{slug}`.

Persistence root:

- `${INDEX_DATA_PATH}/_userspace` (dev default maps to `.data/_userspace`).

---

## 3) UX and Interaction Model

User Space view is split into two primary columns:

1. **Left column**
   - Workspace selector and creation
   - Workspace tool selector
   - Chat panel (reused existing chat component/runtime)

2. **Right column**
   - File list
   - File editor
   - Artifact preview
   - Snapshot list and restore actions

Implemented workspace actions:

- Create workspace
- Select/update workspace tool set
- Create/edit/delete files
- Create snapshot
- Restore snapshot
- Quick Share (copy slug-based static link)
- Preview (open shared full-screen render)
- Delete workspace (owner only)
- Manage members (owner only)

Workspace creation behavior:

- Create action is single-flight in UI (rapid repeat clicks are ignored while creation is in progress).
- Default naming is assigned server-side using the next available owner-scoped `Workspace N` value.
- Workspace names are enforced server-side as owner-scoped unique values using normalized comparison (trimmed, lowercase, whitespace normalized to underscores).

Editor mode defaults:

- User Space is TypeScript-module-only for interactive report/dashboard generation.
- Artifact format selection is removed from the toolbar; saves always persist as `module_ts`.
- Save action is anchored at the far-right side of the top toolbar.
- Quick Share and Preview actions are available in the User Space top toolbar and are enabled for `owner`/`editor` only.

---

## 4) Backend Architecture

### Primary backend modules

- `ragtime/userspace/models.py`
- `ragtime/userspace/service.py`
- `ragtime/userspace/routes.py`
- `ragtime/indexer/routes.py` (workspace-aware chat/task APIs)
- `ragtime/indexer/background_tasks.py` (workspace tool filtering in background execution)
- `ragtime/rag/components.py` (runtime tool filtering)
- `ragtime/indexer/repository.py` (conversation lookup helpers)
- `ragtime/main.py` (route registration)

### Core runtime behavior

- Userspace routes are exposed under `/indexes/userspace/*`.
- Workspace chat requests reuse the existing RAG/chat pipeline; no duplicate orchestration stack.
- Workspace tool constraints are translated to blocked runtime tool names and applied for sync, stream, and background execution.
- Visualization guidance is layered by context:
  - common UI visualization guidance (applies to all UI requests),
  - chat-specific visualization guidance (applies to chat/non-userspace requests),
  - userspace-specific visualization guidance (applies only when `workspace_context` is present).
- User Space mode applies a dedicated prompt component (`USERSPACE_MODE_PROMPT_ADDITION`) only when `workspace_context` is present, so persistent artifact/file-structure instructions are not mixed into standard chat prompts.
- In User Space mode, the runtime injects internal-only LangChain tools: `assay_userspace_code`, `list_userspace_files`, `read_userspace_file`, `upsert_userspace_file`, `validate_userspace_typescript`, and `create_userspace_snapshot`.
- Runtime tool descriptions are request-scoped and mode-aware:
  - chat mode surfaces chat-focused chart/datatable instructions (explicit per-call payloads),
  - userspace mode surfaces persistent live-wiring instructions (`data_connection`, refreshable request payloads, blocker reporting).

---

## 5) Storage and Data Model

### Userspace filesystem layout

- `workspaces/{workspace_id}/workspace.json`
- `workspaces/{workspace_id}/files/**`
- `workspaces/{workspace_id}/files/.git/`

### Workspace metadata (`workspace.json`)

- `id`, `name`, `description`
- `owner_user_id`
- `members[{user_id, role}]`
- `selected_tool_ids`
- `conversation_ids`
- `created_at`, `updated_at`

Authoritative metadata note:

- Workspace ACL/tool/conversation metadata is persisted in database tables (`userspace_workspaces`, `userspace_members`) as source of truth.
- Filesystem `workspace.json` values are maintained for workspace portability/backfill compatibility.

### File metadata sidecars

- Sidecars are stored as `*.artifact.json` next to files.
- Sidecars are internal implementation files:
  - hidden from file listings,
  - excluded from snapshot `file_count`,
  - blocked from direct read/write/delete by userspace file endpoints.
- Sidecar payload may include:
  - `artifact_type` (currently `module_ts`)
  - `live_data_connections` (for deterministic live wiring metadata)

### Snapshot model

- Snapshot ID = git commit SHA.
- Snapshot creation = stage + commit (`--allow-empty` supported).
- Snapshot listing = git log + per-commit tracked file count.
- Snapshot restore = `git reset --hard <commit>` + `git clean -fd`.

---

## 6) API Contract

Userspace endpoints (`/indexes/userspace`):

- `GET /tools`
- `GET /workspaces`
- `POST /workspaces`
- `GET /workspaces/{workspace_id}`
- `PUT /workspaces/{workspace_id}`
- `DELETE /workspaces/{workspace_id}`
- `PUT /workspaces/{workspace_id}/members`
- `GET /workspaces/{workspace_id}/files`
- `PUT /workspaces/{workspace_id}/files/{file_path:path}`
- `GET /workspaces/{workspace_id}/files/{file_path:path}`
- `DELETE /workspaces/{workspace_id}/files/{file_path:path}`
- `POST /workspaces/{workspace_id}/execute-component`
- `POST /workspaces/{workspace_id}/share-link`
- `GET /workspaces/{workspace_id}/share-link`
- `DELETE /workspaces/{workspace_id}/share-link`
- `PUT /workspaces/{workspace_id}/share-link/slug`
- `GET /workspaces/{workspace_id}/share-link/availability?slug=...`
- `GET /shared/{share_token}`
- `POST /shared/{share_token}/execute-component`
- `GET /shared/{owner_username}/{share_slug}`
- `POST /shared/{owner_username}/{share_slug}/execute-component`
- `GET /workspaces/{workspace_id}/snapshots`
- `POST /workspaces/{workspace_id}/snapshots`
- `POST /workspaces/{workspace_id}/snapshots/{snapshot_id}/restore`

Workspace-aware chat/task endpoints (`/indexes`) accept `workspace_id` and enforce workspace membership/role where applicable.

File endpoint payload behavior:

- `PUT /workspaces/{workspace_id}/files/{file_path:path}` accepts:
  - `content`
  - `artifact_type` (optional)
  - `live_data_requested` (optional, default `false`; set `true` only when user explicitly requested live/refreshable data)
  - `live_data_connections` (optional generally; required by contract only when `live_data_requested=true` for module-source writes under `dashboard/*` or with `artifact_type=module_ts`)
  - `live_data_checks` (optional generally; required by contract when `live_data_requested=true`, and must prove successful connection + transformation for each `live_data_connections.component_id`)
- `GET /workspaces/{workspace_id}/files/{file_path:path}` returns persisted `artifact_type` and `live_data_connections` (when present/valid in sidecar metadata).
- `POST /workspaces/{workspace_id}/execute-component` accepts:
  - `component_id` (must be present in workspace `selected_tool_ids`)
  - `request` (SQL payload; string or object containing `query`/`sql`/`command`)
  - returns structured tabular data: `rows[]`, `columns[]`, `row_count`, optional `error`
- `POST /workspaces/{workspace_id}/share-link` returns:
  - `workspace_id`
  - `share_token`
  - `owner_username`
  - `share_slug`
  - `share_url` (`/{username}/{slug}` using request base URL)
- `GET /shared/{share_token}` returns:
  - `workspace_id`, `workspace_name`
  - `entry_path` (currently `dashboard/main.ts`)
  - `workspace_files` (module map for preview runtime)
  - `live_data_connections` (when present in entry sidecar metadata)
- `POST /shared/{share_token}/execute-component` accepts same execution payload and returns the same structured tabular response as workspace-authenticated execution.

---

## 7) Authorization Model

Roles:

- `owner`
- `editor`
- `viewer`

Effective behavior:

- Read workspace/files/snapshots/conversations: `owner/editor/viewer` with membership.
- Mutate workspace/files/snapshots/conversation state: `owner/editor`.
- Manage members: `owner` only.
- Delete workspace: `owner` only.
- Create share link: `owner/editor` with membership.
- Shared preview/read + shared execute endpoints are anonymous and authorized by a valid signed `share_token`.

Normalization rule:

- Non-owner member payloads with role `owner` are normalized to `editor`.

---

## 8) Tool Access Enforcement

- Workspace stores `selected_tool_ids` (tool config IDs).
- Incoming workspace tool lists are normalized to enabled tool configs and deduplicated.
- Runtime computes blocked tool names from non-selected configs.
- Blocked names are applied to:
  - non-streaming queries,
  - streaming queries,
  - background task execution.

Connection metadata behavior:

- Runtime tool events now include optional connection metadata:
  - `tool_config_id`
  - `tool_config_name`
  - `tool_type`
- Runtime prompts include an "Active Tool Connections" section so the agent sees the currently available connection set in that request context.
- The Active Tool Connections section is layered by mode:
  - common: active tool list and connection identity for the request,
  - userspace-only: reusable dashboard/charts/tables connection-persistence guidance.
- Assistant history reconstruction includes tool connection references (when available), so follow-up turns can reuse the same data sources.

User Space internal tools behavior:

- `create_userspace_snapshot` enables agent-created git checkpoints without adding UI controls/end-user surfacing.
- User Space prompt guidance instructs the agent to create snapshots at each completed user-requested change loop.
- `assay_userspace_code` provides a structured pre-edit assay pass so the agent assesses current file state before editing.
- `upsert_userspace_file` returns AI-facing warnings when hard-coded hex colors are detected in generated module/CSS content, so the agent can replace them with theme tokens.

### Live-data contract enforcement

Enforcement is layered across both agent and service paths:

**Service-level** (`_requires_live_data_contract` in `service.py`):

- Auto-requires the live data contract when the workspace has selected tools (`workspace_has_tools=True`), regardless of the explicit `live_data_requested` flag. Non-module-source or non-dashboard paths are excluded.
- Requires non-empty `live_data_connections` and `live_data_checks` for all qualifying writes.
- Cross-validates that every `live_data_connections.component_id` has a corresponding successful `live_data_checks` entry.
- Verifies server-side execution proofs (`verify_execution_proofs`): each declared `component_id` must have a non-expired `_ExecutionProofRecord` minted by a prior `execute-component` call. Proofs expire after 1 hour.

**Agent-level** (`upsert_userspace_file` wrapper in `components.py`):

- Auto-infers `effective_live_data_requested=true` when workspace has tools and the write targets a dashboard module.
- **AST structural validation** (`validate_live_data_binding`): uses the TypeScript compiler AST to deterministically verify that the module source contains `context.components[componentId].execute()` call patterns. This is non-regex, non-heuristic, and cannot be satisfied by fabricating metadata alone. Entry files (`dashboard/main.ts`) accept local module imports as evidence of deferred binding; non-entry files must contain direct `execute()` calls.
- **Hard tool failure** (`ToolException` with `handle_tool_error=True`): policy violations raise a `ToolException`, which the agent framework surfaces as a tool error rather than a green-checkmark success response. This prevents the agent from ignoring rejected writes.
- **Regex patterns demoted to warning**: `find_hardcoded_data_patterns` is a supplementary warning only; the AST validation above is the authoritative enforcement.
- **No-tools conflict**: when a dashboard module write targets a workspace without selected tools, a warning instructs the agent to report the conflict to the user and request tool configuration. Writes are not blocked but cannot claim live data wiring.
- Each `live_data_connections` item requires a `tool_config` component reference (`component_id`) plus the live `request` payload used for refresh.
- `live_data_connections.component_id` must reference a tool currently selected in the workspace (`selected_tool_ids`); non-selected IDs are rejected.
- For `live_data_requested=true`, `live_data_checks` is required and each check must report:
  - `connection_check_passed=true`
  - `transformation_check_passed=true`
  - `component_id` that matches a selected tool and maps to a corresponding `live_data_connections.component_id`

### Execution proof tracking

- `_ExecutionProofRecord` records are minted server-side when `_execute_component_for_workspace` completes without error.
- Each proof captures `component_id`, `row_count`, `timestamp`, and `query_hash`.
- Proofs expire after `_EXECUTION_PROOF_MAX_AGE_SECONDS` (1 hour).
- The `upsert_workspace_file` service validates that every declared connection has a non-expired proof before persisting.

### Field schemas

- `live_data_connections` is validated against schema fields:
  - `component_kind` = `tool_config`
  - `component_id` (required)
  - `request` (required)
  - `component_name` (optional)
  - `component_type` (optional)
  - `refresh_interval_seconds` (optional, >= 1)
- `live_data_checks` is validated against schema fields:
  - `component_id` (required)
  - `connection_check_passed` (required)
  - `transformation_check_passed` (required)
  - `input_row_count` (optional, >= 0)
  - `output_row_count` (optional, >= 0)
  - `note` (optional)

Execution bridge behavior:

- Preview execution validates workspace membership and selected-tool access per request.
- Preview execution is SQL-only (`postgres`, `mysql`, `mssql`) and always read-only (`allow_write=false`).
- Query limits/timeouts are bounded by tool configuration (`max_results`, `timeout`, `timeout_max_seconds`).
- SQL tool output metadata is parsed into structured response rows/columns for deterministic dashboard wiring.
- Shared-preview execution path resolves workspace context from token, still enforces selected-tool membership and SQL-only/read-only execution constraints.

---

## 9) Artifact Rendering and Security Posture

Supported artifact types:

- `module_ts`

Visualization connection contract:

- `create_chart` and `create_datatable` support optional `data_connection` metadata for persistent/live dashboard wiring.
- Base chart/datatable tool descriptions are mode-neutral; mode-specific usage rules are appended at runtime.
- Chat mode keeps explicit per-call payload guidance for visualization tool calls.
- In User Space mode, `data_connection` is treated as an internal component reference to admin-configured Settings tools.
- Persisted module-source writes subject to the contract use `live_data_connections` (file-level contract metadata) in addition to any per-artifact chart/datatable `data_connection` payloads.
- Recommended `data_connection` fields:
  - `component_kind` (`tool_config`)
  - `component_id` (tool config ID)
  - `component_name` (optional)
  - `component_type` (optional)
  - `request` (query/command payload)
  - `refresh_interval_seconds`

Implemented controls:

- TypeScript source is transpiled client-side and rendered in an isolated iframe runtime.
- iframe sandboxing uses `allow-scripts` only; no same-origin capability is granted.
- Generated preview document applies restrictive CSP bootstrap (`default-src 'none'`, script/style inline constraints).
- Runtime contract expects `export function render(container, context)` from the module.
- Runtime context is isolated and limited to component wiring inputs (`context.components`, `context.componentsList`) and theme-token inputs (`context.themeTokens`).
- Preview runtime hydrates component wiring from persisted `live_data_connections`; `context.components` exposes both keyed (`component_id`) and numeric index access for compatibility.
- Preview runtime now applies robust theme-token fallbacks so inline `var(--token)` style declarations still render when host theme variables are missing.
- Chart.js is preloaded in preview runtime (`window.Chart`), and redundant Chart.js CDN script injection from generated modules is suppressed.
- Preview runtime component `execute()` uses iframe-to-parent `postMessage` RPC.
- Parent runtime forwards execution requests to `/indexes/userspace/workspaces/{workspace_id}/execute-component` and returns structured results back to the iframe.
- In shared-token mode, parent runtime forwards execution requests to `/indexes/userspace/shared/{share_token}/execute-component`.
- Bridge messages are scoped to the preview iframe window and use a dedicated protocol marker (`userspace-exec-v1`) for safer routing.
- Isolated runtime receives current app theme tokens via CSS custom properties, enabling dark/light-consistent module styling.
- Local module import rewriting in preview supports side-effect imports, `from`-clause imports, and dynamic imports for workspace-relative modules.
- Preview bootstrap uses encoded payload transport via iframe DOM attributes and parser-safe runtime initialization to reduce `about:srcdoc` syntax fragility.

---

## 10) Frontend Surface

Primary frontend files:

- `ragtime/frontend/src/App.tsx`
- `ragtime/frontend/src/components/UserSpacePanel.tsx`
- `ragtime/frontend/src/components/UserSpaceArtifactPreview.tsx`
- `ragtime/frontend/src/components/UserSpaceSharedView.tsx`
- `ragtime/frontend/src/components/ChatPanel.tsx`
- `ragtime/frontend/src/api/client.ts`
- `ragtime/frontend/src/types/api.ts`
- `ragtime/frontend/src/styles/components.css`

Frontend behavior includes role-aware disabling of workspace mutation actions for viewers.

Workspace switch behavior:

- Embedded chat panel is remounted by `workspace_id` on workspace change to avoid stale conversation/task polling state crossing workspace boundaries.
- Task polling/stream callbacks are refreshed when `workspace_id` changes.

Shared link behavior:

- App checks `userspace_share_token` query param and renders a shared full-screen dashboard view without the authenticated app shell.
- Shared dashboard view loads preview payload from `/indexes/userspace/shared/{share_token}` and renders with the same iframe runtime.

---

## 11) Current Boundaries / Non-goals

- Workspace file artifacts are filesystem-backed; workspace ACL/tool metadata is DB-backed.
- Renderer isolation is iframe/CSP bootstrap level; not a full multi-origin hardened sandbox.
- Share tokens are persisted per workspace in DB-backed metadata and resolved by token lookup.
- No dedicated automated userspace test suite is defined in this document.
- Member picker currently depends on admin user listing capability in UI flows.

## 12) Verification Criteria (State Conformance)

A deployment conforms to this specification when the following are true:

1. Userspace routes are mounted and reachable.
2. Workspaces persist under `_userspace` with `workspace.json` + `files` + git history.
3. Snapshot create/list/restore uses git commit semantics (SHA IDs).
4. Workspace file operations reject traversal and internal sidecar/git paths.
5. Workspace chat mutations are blocked for viewers and allowed for editors/owners.
6. Non-selected tools are not callable in workspace-scoped runtime execution.
7. TypeScript module previews run in an isolated sandboxed iframe runtime with component-limited context.
8. Workspace switches do not reuse stale embedded chat state across workspace boundaries.
9. Repeated rapid workspace-create clicks in one UI session do not produce duplicate default `Workspace N` names from client-side race conditions.
10. Workspace toolbar offers Quick Share and Preview actions for editable roles.
11. A valid `userspace_share_token` URL renders the shared dashboard without requiring session auth.
12. Shared execution endpoint enforces selected-tool checks and SQL-only read-only execution bounds.

---

## 13) Implementation File Index

Backend:

- `ragtime/main.py`
- `ragtime/userspace/models.py`
- `ragtime/userspace/service.py`
- `ragtime/userspace/routes.py`
- `ragtime/indexer/models.py`
- `ragtime/indexer/repository.py`
- `ragtime/indexer/routes.py`
- `ragtime/indexer/background_tasks.py`
- `ragtime/rag/components.py`

Frontend:

- `ragtime/frontend/src/App.tsx`
- `ragtime/frontend/src/types/api.ts`
- `ragtime/frontend/src/api/client.ts`
- `ragtime/frontend/src/components/ChatPanel.tsx`
- `ragtime/frontend/src/components/UserSpacePanel.tsx`
- `ragtime/frontend/src/components/UserSpaceArtifactPreview.tsx`
- `ragtime/frontend/src/components/index.ts`
- `ragtime/frontend/src/styles/components.css`
