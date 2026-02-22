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
- Workspace-scoped file management and artifact preview.
- Workspace-scoped tool selection (admin-configured tools only).
- Snapshot/restore using **git commit history** per workspace.
- User Space-only agent tools for file I/O plus autonomous checkpoint snapshots.
- Workspace sharing with role-based access (`owner`, `editor`, `viewer`).

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
- Delete workspace (owner only)
- Manage members (owner only)

Workspace creation behavior:

- Create action is single-flight in UI (rapid repeat clicks are ignored while creation is in progress).
- Default naming uses the next available `Workspace N` value from currently loaded workspaces rather than list-length indexing.
- Duplicate names are still possible across independent clients/tabs because name uniqueness is not globally enforced server-side.

Editor mode defaults:

- User Space is TypeScript-module-only for interactive report/dashboard generation.
- Artifact format selection is removed from the toolbar; saves always persist as `module_ts`.
- Save action is anchored at the far-right side of the top toolbar.

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
- In User Space mode, the runtime injects internal-only LangChain tools: `list_userspace_files`, `read_userspace_file`, `upsert_userspace_file`, and `create_userspace_snapshot`.
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
- User Space prompt guidance instructs the agent to create snapshots at meaningful stopping points (milestones/checkpoints).
- `upsert_userspace_file` returns AI-facing warnings when hard-coded hex colors are detected in generated module/CSS content, so the agent can replace them with theme tokens.
- Live-data contract enforcement is deterministic at both agent and userspace service write paths and gated by intent: only writes with `live_data_requested=true` are required to include non-empty `live_data_connections` metadata for eligible module-source paths/types.
- Each `live_data_connections` item requires a `tool_config` component reference (`component_id`) plus the live `request` payload used for refresh.
- `live_data_connections.component_id` must reference a tool currently selected in the workspace (`selected_tool_ids`); non-selected IDs are rejected.
- For `live_data_requested=true`, `live_data_checks` is required and each check must report:
  - `connection_check_passed=true`
  - `transformation_check_passed=true`
  - `component_id` that matches a selected tool and maps to a corresponding `live_data_connections.component_id`
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
- `ragtime/frontend/src/components/ChatPanel.tsx`
- `ragtime/frontend/src/api/client.ts`
- `ragtime/frontend/src/types/api.ts`
- `ragtime/frontend/src/styles/components.css`

Frontend behavior includes role-aware disabling of workspace mutation actions for viewers.

Workspace switch behavior:

- Embedded chat panel is remounted by `workspace_id` on workspace change to avoid stale conversation/task polling state crossing workspace boundaries.
- Task polling/stream callbacks are refreshed when `workspace_id` changes.

---

## 11) Current Boundaries / Non-goals

- Workspace and ACL persistence is filesystem metadata-based, not DB-backed.
- Renderer isolation is iframe/CSP bootstrap level; not a full multi-origin hardened sandbox.
- No dedicated automated userspace test suite is defined in this document.
- Member picker currently depends on admin user listing capability in UI flows.

---

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
