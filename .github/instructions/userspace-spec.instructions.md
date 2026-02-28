---
applyTo: 'ragtime/userspace/**, runtime/**'
---

# User Space Feature Implementation Instructions

Last updated: 2026-02-28 (codebase-scanned; concise agent-focused)

## Scope

- Apply to User Space runtime/collab/share work in `ragtime/userspace/**` and `runtime/**`.
- Keep changes incremental; do not invent new UX or role semantics.
- Endpoint details belong in `/docs` (do not maintain static route dumps here).

## Big Picture (How User Space Actually Works)

- Control plane is in Ragtime app code (`ragtime/userspace/runtime_service.py`, `ragtime/userspace/runtime_routes.py`).
- Data plane runs in runtime manager/worker code (`runtime/manager/**`, `runtime/worker/**`).
- `runtime/main.py` chooses service mode via `RUNTIME_SERVICE_MODE`; manager mode also mounts worker routes.
- Workspace runtime files live under `${INDEX_DATA_PATH}/_userspace/workspaces/{workspace_id}/files`.
- Preview is runtime-only (legacy frontend `srcDoc` fallback is gone).

## Share Routing Split (Critical)

- Public share URLs are top-level routes in `ragtime/main.py`:
  - `/{owner_username}/{share_slug}` (canonical)
  - `/shared/{share_token}` (redirects to canonical slug route)
- Internal editor-preview share routes are in `ragtime/userspace/runtime_routes.py` under `/indexes/userspace/shared/.../preview`.
- Public and internal routes are different paths with different auth contexts; when changing share behavior, update both layers intentionally.
- Public password-protected shares use FastAPI-rendered full-page HTML prompt + scoped cookie (`userspace_share_pw_*`) in `main.py`.

## Runtime + Bootstrap Conventions

- Runtime bootstrap config is workspace-local: `.ragtime/runtime-bootstrap.json`.
- Bootstrap execution stamp is `.ragtime/.runtime-bootstrap.done` (`runtime/shared.py`).
- Default bootstrap template is managed in `ragtime/userspace/service.py` (`_default_runtime_bootstrap_config`, template versioning + auto-update logic).
- Worker startup reads bootstrap config and reruns commands when config digest changes (`runtime/worker/service.py`).
- Runtime launch override is `.ragtime/runtime-entrypoint.json` (command/cwd/framework) consumed by worker service.

## Security + Proxy Rules (Do Not Violate)

- Keep role enforcement strict on runtime/collab mutations: viewer is read-only; editor/owner can mutate.
- Never forward ragtime session credentials to user devservers: preserve blocked header behavior in `_proxy_request_headers`.
- Keep hop-by-hop header filtering in proxy request/response paths.
- No application-layer HTML rewriting in preview proxy; only transport-level root-relative URL rewriting is allowed.
- Keep path normalization/traversal protections (`normalize_file_path` and reserved-path checks) intact.

## Non-Obvious Product Constraints

- Current runtime backend is manager/worker orchestration, not MicroVM leasing yet.
- Devserver port is still effectively dynamic per session (fixed internal `5173` invariant is not fully enforced yet).
- Collaboration is text-sync + version-conflict resync; CRDT/Yjs is not active.

## Known Validation Gaps (Important for Agent Output Quality)

- `validate_userspace_code` can pass while preview is still blank if workspace bootstrap expects `window.render` but bundle format hides it (IIFE export mismatch).
- Component `execute()` timeouts can still blank dashboards if generated code lacks try/catch fallback rendering.
- Prefer prompt/tooling fixes for these issues; do not patch proxy layer to compensate.

## Fast Workflow for User Space Changes

- Check runtime health + logs first (see `.github/instructions/debugging.instructions.md`).
- Validate the exact User Space behavior touched (not full-system sweeps):
  - share route changes: verify both top-level and internal preview flows
  - bootstrap changes: verify config seed + stamp update behavior
  - proxy changes: verify header filtering and websocket pass-through still work
- Use `/docs` for endpoint signatures and payload shapes.

## Sources Used for This Guide

- Conventions search requested by user found `README.md` only (no `AGENTS.md`/`copilot-instructions.md` in repo).
- Primary implementation references: `ragtime/main.py`, `ragtime/userspace/runtime_routes.py`, `ragtime/userspace/runtime_service.py`, `ragtime/userspace/service.py`, `runtime/main.py`, `runtime/shared.py`, `runtime/worker/service.py`.