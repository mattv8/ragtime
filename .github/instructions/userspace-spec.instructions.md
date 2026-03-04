---
applyTo: 'ragtime/userspace/**, runtime/**'
---

# User Space Feature Implementation Instructions

Last updated: 2026-03-03 (codebase-scanned; concise agent-focused)

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
- Default bootstrap template is managed in `ragtime/userspace/service.py` (`_default_runtime_bootstrap_config`, template version 5). Auto-update logic includes `_is_legacy_default_bootstrap` detection for configs missing `managed_by`/`template_version`.
- Worker startup reads bootstrap config and reruns commands when config digest changes (`runtime/worker/service.py`). Digest includes file content of `watch_paths`.
- Bootstrap retry: if devserver exits with "command not found" patterns, the worker invalidates the stamp and retries on next start (`_should_retry_bootstrap_after_exit`).
- Runtime launch is defined by `.ragtime/runtime-entrypoint.json` (command/cwd/framework) consumed by worker service.
- `.ragtime/runtime-entrypoint.json` is authoritative. Runtime no longer falls back to `package.json`, Python entrypoint guesses, or `index.html` heuristics.
- Keep launch commands `$PORT`-aware and bound to `0.0.0.0` for proxy reachability.
- SQLite migration runner and migration files are agent-managed (created via file tools and executed via terminal tool). They are NOT auto-seeded or auto-bootstrapped. The `sqlite_migrate.py` template has been removed; only `dashboard_entrypoint.js` remains in `templates/`.

## Live Data + SQLite Contract (Critical)

- Persistent dashboard data must be live-wired from selected tools via `context.components[componentId].execute()`.
- Contract enforcement is strict for dashboard entry writes (`dashboard/main.ts`) when workspace tools are selected:
  - `live_data_connections` required,
  - `live_data_checks` required,
  - AST-verified `execute()` call patterns required,
  - server-side execution proofs required for referenced component IDs.
- Helper modules under `dashboard/` can receive transformed data from entrypoint modules and do not need their own connection metadata.
- Never substitute mock/static data to bypass live wiring when tools are selected.
- SQLite is for out-of-scope local persistence (cache/preferences/local app state), not as a substitute source of truth for live dashboards.
- Workspace SQLite snapshot mode default is `exclude` (live-data-first default). `include` is opt-in behavior.

## Security + Proxy Rules (Do Not Violate)

- Keep role enforcement strict on runtime/collab mutations: viewer is read-only; editor/owner can mutate.
- Never forward ragtime session credentials to user devservers: preserve blocked header behavior in `_proxy_request_headers`.
- Keep hop-by-hop header filtering in proxy request/response paths.
- No DOM-parsing or content-injection in preview proxy. The only rewriting is regex-based root-relative URL prefixing (`src="/"`, `href="/"`, `action="/"`) via `_rewrite_root_relative_urls` in `runtime_routes.py`.
- Keep path normalization/traversal protections (`normalize_file_path` and reserved-path checks) intact.

## Sandbox + Process Conventions

- Sandbox modes: `pivot_root` (requires `CAP_SYS_ADMIN`) or `chroot` fallback. Detected at startup via `detect_capabilities()` in `runtime/worker/sandbox.py`.
- Per-workspace rootfs layout: `workspaces/{workspace_id}/rootfs/` with system dirs synced from host. Workspace files mirrored into `rootfs/workspace/`.
- `_sync_system_dirs_for_chroot` uses incremental `copytree(dirs_exist_ok=True)` — never `rmtree` on `/workspace` (preserves running process cwd inodes).
- All sandboxed shell commands (devserver, bootstrap) must include explicit `cd /workspace &&` prefix in the `sh -lc` string — `preexec_fn`'s `os.chdir()` can be lost after exec under certain event-loop contexts.
- Worker PTY sessions use `pty_access_token` for terminal access authentication.
- Screenshot capture is available via Playwright (`runtime/worker/screenshot.js`, invoked by `capture_screenshot` in service).

## Non-Obvious Product Constraints

- Current runtime backend is manager/worker orchestration (provider name `microvm_pool_v1` is a label, not actual MicroVM leasing).
- Devserver port is dynamic per session via `_pick_free_port()` in the worker. The control plane uses `5173` as a fallback default when the provider doesn't report a port (`launch_port or _RUNTIME_DEVSERVER_PORT`).
- Collaboration is text-sync + version-conflict resync (`RuntimeVersionConflictError`); CRDT/Yjs is not active.

## Known Validation Gaps (Important for Agent Output Quality)

- `validate_userspace_code` can pass while preview is still blank if workspace bootstrap expects `window.render` but bundle format hides it (IIFE export mismatch).
- Component `execute()` timeouts can still blank dashboards if generated code lacks try/catch fallback rendering.
- Prefer prompt/tooling fixes for these issues; do not patch proxy layer to compensate.
- `validate_userspace_code` + contract metadata alone are not sufficient for live-data persistence; execution-proof checks can still reject writes until component queries have been executed successfully.

## Fast Workflow for User Space Changes

- Check runtime health + logs first (see `.github/instructions/debugging.instructions.md`).
- Validate the exact User Space behavior touched (not full-system sweeps):
  - share route changes: verify both top-level and internal preview flows
  - bootstrap changes: verify config seed + stamp update behavior
  - runtime launch issues: inspect `.ragtime/runtime-entrypoint.json` first (do not add fallback launch logic)
  - live data persistence issues: verify selected tools, execution proofs, and `dashboard/main.ts` contract metadata/call sites
  - proxy changes: verify header filtering and websocket pass-through still work
- Use `/docs` for endpoint signatures and payload shapes.

## Sources Used for This Guide

- Conventions search requested by user found `README.md` only (no `AGENTS.md`/`copilot-instructions.md` in repo).
- Primary implementation references: `ragtime/main.py`, `ragtime/userspace/runtime_routes.py`, `ragtime/userspace/runtime_service.py`, `ragtime/userspace/service.py`, `runtime/main.py`, `runtime/shared.py`, `runtime/worker/service.py`, `runtime/worker/sandbox.py`, `runtime/worker/api.py`.