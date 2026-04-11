---
applyTo: 'ragtime/userspace/**, runtime/**'
---

# User Space Feature Implementation Instructions

Last updated: 2026-04-10 (codebase-scanned; concise agent-focused)

## Scope

- Apply to User Space runtime/collab/share work in `ragtime/userspace/**` and `runtime/**`.
- Keep changes incremental; do not invent new UX or role semantics.
- Endpoint details belong in `/docs` (do not maintain static route dumps here).

## Big Picture (How User Space Actually Works)

- Control plane is in Ragtime app code (`ragtime/userspace/runtime_service.py`, `ragtime/userspace/runtime_routes.py`).
- Data plane runs in runtime manager/worker code (`runtime/manager/**`, `runtime/worker/**`).
- `runtime/main.py` chooses service mode via `RUNTIME_SERVICE_MODE`; manager mode also mounts worker routes.
- Workspace runtime files live under `${INDEX_DATA_PATH}/_userspace/workspaces/{workspace_id}/files`.
- Preview is runtime-only and now uses dedicated per-workspace preview origins dispatched by `PreviewHostDispatchMiddleware` in `ragtime/userspace/preview_host.py`.
- Preview launch is an explicit API step: control-plane routes mint short-lived bootstrap grants and the preview host mints a preview session cookie on first load.
- Browser-auth cookie surfaces are only for `collab` and `runtime_pty`; preview no longer uses capability cookies.

## Share Routing Split (Critical)

- Public share URLs are top-level routes in `ragtime/main.py`:
  - `/{owner_username}/{share_slug}` (canonical)
  - `/shared/{share_token}` (anonymous/token form)
- Public share routes do not proxy app bytes directly anymore; they validate access, then redirect to a dedicated preview origin bootstrap URL.
- Internal editor preview APIs are launch endpoints in `ragtime/userspace/runtime_routes.py` under `/indexes/userspace/runtime/workspaces/.../preview-launch`, and shared launches live under `/indexes/userspace/shared/.../preview-launch`.
- Public and internal routes are still different paths with different auth contexts; when changing share behavior, update both layers intentionally.
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
- SQLite persistence uses `.ragtime/db/migrations/*.sql` as source-of-truth schema history and `.ragtime/scripts/sqlite_migrate.py` as the default runner scaffold.
- The runtime bootstrap template includes a migration apply command for the default runner and watches both the runner path and migrations directory for digest changes.
- Agents may update `.ragtime/scripts/sqlite_migrate.py` and `.ragtime/db/migrations/*` via file tools when implementing persistence changes.
- `dashboard_entrypoint.js`, `runtime_bridge.js`, and `sqlite_migrate.py` templates live in `templates/`.

## Entrypoint Status + Framework Lock-In

- Canonical entrypoint parsing lives in `runtime/shared.py` (`parse_entrypoint_config`, `EntrypointStatus`).
- Both the runtime worker (`runtime/worker/service.py`) and the ragtime prompt builder (`ragtime/rag/components.py`) use this shared parser, so the definition of valid/invalid/missing is always consistent.
- `EntrypointStatus.state` is one of: `missing` (file absent), `invalid` (present but no command or malformed), `valid` (has a usable command).
- `EntrypointStatus.framework_known` indicates whether the framework value is in the recognised set (`KNOWN_FRAMEWORKS` in `runtime/shared.py`).
- `UserSpaceService.is_default_static_entrypoint()` detects the auto-seeded default (`python3 -m http.server`, framework `static`). The prompt layer treats this as equivalent to missing so the agent is nudged to choose a real framework.

### Dynamic System Prompt Behaviour

- When entrypoint is **missing or default-static**: system prompt includes `_USERSPACE_ENTRYPOINT_MISSING_NUDGE` (lightweight suggestion to choose a framework based on the user's request) plus the full `USERSPACE_ENTRYPOINT_SETUP_PROMPT` (runtime contract, examples, module dashboard mode, dependencies).
- When entrypoint is **invalid**: system prompt includes a fix-required notice with the specific error plus the full setup prompt.
- When entrypoint is **valid with a real framework**: system prompt includes only a compact `_USERSPACE_ENTRYPOINT_LOCKED_TEMPLATE` (framework name, command, cwd) and omits all setup guidance.
- This is built by `build_userspace_entrypoint_nudge()` in `ragtime/rag/components.py` and appended per-request in `_build_request_runtime_context`.
- The base userspace prompt (`USERSPACE_MODE_PROMPT_ADDITION`) always includes workspace continuity guidance, persistence rules, file workflow, theme/CSS regardless of entrypoint state. Data wiring rules and resilient data loading are only included when the workspace has live data tools selected (`has_live_data_tools`).
- **Workspace continuity**: `build_workspace_continuity_context()` in `ragtime/rag/prompts.py` produces a conditional per-request block injected into the base prompt via `{workspace_continuity}`. For fresh workspaces (0 user files) it emits a short "starting fresh" note; for existing workspaces it emits concrete state (file count, key files, framework, last snapshot) plus continuity rules (assay first, extend don't replace). No generic/hedging prose is sent — every token reflects actual workspace state.
- The prompt is architecture-agnostic: it supports single-page apps, multi-page apps, API backends, and hybrid structures. The `dashboard/main.ts` module-dashboard pattern is documented as one option, not the only option.

## Live Data + SQLite Contract (Critical)

- Persistent dashboard data must be live-wired from selected tools via `context.components[componentId].execute()`.
- Contract enforcement is strict for dashboard entry writes (`dashboard/main.ts`) when workspace tools are selected:
  - `live_data_connections` required,
  - `live_data_checks` required,
  - AST-verified `execute()` call patterns required,
  - server-side execution proofs required for referenced component IDs.
- Helper modules under `dashboard/` can receive transformed data from entrypoint modules and do not need their own connection metadata.
- Never substitute mock/static data to bypass live wiring when tools are selected.

### Two-lane persistence (when `sqlite_persistence_mode=include`)

- When SQLite local persistence is enabled (`include` mode), the system prompt instructs a **two-lane contract**:
  - **Lane A (live data)**: dashboard datasets fetched at runtime via `context.components[componentId].execute()`. All existing live data enforcement (metadata, AST binding, execution proofs) remains active.
  - **Lane B (SQLite local persistence)**: local app/domain state persisted in `.ragtime/db/app.sqlite3` with numbered SQL migration files in `.ragtime/db/migrations/`.
- The per-turn reminder includes a lane-awareness line so the agent delivers both lanes in a single pass.
- Validation/tool feedback in include mode appends `SQLITE_INCLUDE_MODE_HINT` (from `ragtime/rag/prompts.py`) to error and violation payloads, reinforcing the dual expectation without duplicating prose.
- Service-layer error messages (`ragtime/userspace/service.py`) also append a sqlite suffix when the workspace is in include mode.
- SQLite is **never** a substitute for live dashboard datasets. It supplements live data with local state (preferences, cache, drafts, operational data).
- Default mode is `exclude` (live-data-first, no SQLite prompt block, no lane reminder).

## Live Data Bridge Architecture

- Bridge script is served on the preview origin at `/__ragtime/bridge.js` and rendered by `build_runtime_bridge_content()` in `ragtime/userspace/service.py`. Version tracked via `_RUNTIME_BRIDGE_VERSION`.
- Bridge provides `window.__ragtime_context` with a Proxy-based `components[componentId].execute(params)` that sends postMessage (`USERSPACE_EXEC_BRIDGE` channel) to the parent frame.
- Parent-side handler in `UserSpaceArtifactPreview.tsx` listens for `ragtime-execute` messages, calls `api.executeWorkspaceComponent()`, and sends results back via `ragtime-execute-result` / `ragtime-execute-error`.
- Preview responses auto-inject bridge metadata plus `<script src="/__ragtime/bridge.js?workspace_id=..."></script>` into HTML so workspaces never need to manually include it.
- LLM prompt instructs passing `window.__ragtime_context` as the context argument to `render(container, context)`.

## Security + Proxy Rules (Do Not Violate)

- Keep role enforcement strict on runtime/collab mutations: viewer is read-only; editor/owner can mutate.
- Never forward ragtime session credentials to user devservers: preserve blocked header behavior in `_proxy_request_headers`.
- Keep hop-by-hop header filtering in proxy request/response paths.
- Preview host responses should be same-origin with the running app, so root-relative assets and client-side routing work without preview-path rewriting.
- HTML preview responses still inject bridge config and the bridge script from `/__ragtime/bridge.js`; keep that injection in `runtime_routes.py`.
- Shared and workspace preview auth should remain token-based bootstrap -> preview-session cookie. Do not reintroduce preview capability cookies or `/indexes/userspace/.../preview` proxy routes.
- Proxy response strips `X-Frame-Options`, `Content-Security-Policy`, and `Content-Security-Policy-Report-Only` headers from devserver responses to prevent iframe-blocking policies from reaching the browser (the iframe `sandbox` attribute is the real security boundary).
- Root-relative `Location` headers only need rewriting on routes that still use an explicit proxy base path.
- Keep path normalization/traversal protections (`normalize_file_path` and reserved-path checks) intact.

## Sandbox + Process Conventions

- Sandbox modes: `pivot_root` (requires `CAP_SYS_ADMIN`) or `chroot` fallback. Detected at startup via `detect_capabilities()` in `runtime/worker/sandbox.py`.
- Per-workspace rootfs layout: `workspaces/{workspace_id}/rootfs/` with system dirs synced from host. Workspace files mirrored into `rootfs/workspace/`.
- `_sync_system_dirs_for_chroot` uses incremental `copytree(dirs_exist_ok=True)` — never `rmtree` on `/workspace` (preserves running process cwd inodes).
- All sandboxed shell commands (devserver, bootstrap) must include explicit `cd /workspace &&` prefix in the `sh -lc` string — `preexec_fn`'s `os.chdir()` can be lost after exec under certain event-loop contexts.
- Worker PTY sessions use `pty_access_token` for terminal access authentication.
- Screenshot capture is available via Playwright (`runtime/worker/templates/playwright_broker.js`, invoked by `capture_screenshot` in service).

### Symlink Safety in sandbox.py (Critical — Read Before Any Change)

Modifying `_sync_usr_for_chroot`, `_sync_system_dirs_for_chroot`, `provision_rootfs`, or `_CHROOT_USR_INCLUDE_PATHS` can introduce symlink loops, infinite recursion during copy, or broken sandbox environments. Before making changes:

1. **Audit the source directory for symlinks first.** Run inside the runtime container:
   ```bash
   # Count and classify symlinks in the source tree
   docker exec runtime-dev sh -c 'find <SOURCE_DIR> -type l | wc -l'
   # Identify self-referential symlinks (loops)
   docker exec runtime-dev sh -c 'find <SOURCE_DIR> -type l -exec sh -c \
     "t=\$(readlink -f \"{}\" 2>/dev/null); case \"\$t\" in <SOURCE_DIR>*) \
     echo \"SELF: {} -> \$t\";; esac" \;'
   # Identify external symlinks (will dangle after chroot)
   docker exec runtime-dev sh -c 'find <SOURCE_DIR> -type l -exec sh -c \
     "t=\$(readlink -f \"{}\" 2>/dev/null); case \"\$t\" in <SOURCE_DIR>*) ;; \
     *) echo \"EXT: {} -> \$t\";; esac" \;'
   ```
2. **Never use `shutil.copytree` without `symlinks=True`** — omitting this flag causes copytree to follow symlinks as if they were real directories, which triggers infinite recursion on self-referencing links like `/usr/bin/X11 -> .`.
3. **Always pass `ignore_dangling_symlinks=True`** — Debian packages frequently include relative symlinks to optional peer packages (e.g. `../../javascript/...`) that may not be installed.
4. **Never `rmtree` a directory that may be a running process's cwd** — especially `/workspace` inside rootfs. Use `dirs_exist_ok=True` for incremental updates instead.
5. **Guard top-level include paths against external symlink traversal** — if a path in `_CHROOT_USR_INCLUDE_PATHS` is itself a symlink resolving outside `/usr/`, skip it rather than following it into unexpected host filesystem areas.
6. **Bump `_CHROOT_USR_SYNC_VERSION`** when changing `_CHROOT_USR_INCLUDE_PATHS` or `/usr` sync logic — this triggers a one-time resync for existing workspace rootfs trees via the stamp file (`.ragtime_usr_sync_version`).
7. **Known safe symlink patterns in the current image:**
   - `/usr/bin/X11 -> .` and `/bin/X11 -> .` (Debian self-ref convention) — safe with `symlinks=True`.
   - 20+ intra-`/usr/share/nodejs` symlinks (e.g. `libnpmteam/node_modules -> ../npm/node_modules`) — resolve correctly after chroot because the full subtree is copied.
   - 20+ external relative symlinks in `/usr/share/nodejs` to `/usr/share/javascript/*` and `/usr/share/man` — dangle harmlessly; not required by Node/npm.
8. **Validate after any change** by deleting a test workspace rootfs and running a sandboxed command:
   ```bash
   docker exec runtime-dev sh -c 'rm -rf /data/_userspace/workspaces/<test_id>/rootfs'
   # Then spawn a sandboxed process and verify exit=0
   ```

## Non-Obvious Product Constraints

- Current runtime backend is manager/worker orchestration (provider name `microvm_pool_v1` is a label, not actual MicroVM leasing).
- Devserver port is dynamic per session via `_pick_free_port()` in the worker. The control plane uses `5173` as a fallback default when the provider doesn't report a port (`launch_port or _RUNTIME_DEVSERVER_PORT`).
- Collaboration is text-sync + version-conflict resync (`RuntimeVersionConflictError`); CRDT/Yjs is not active.

## Known Validation Gaps (Important for Agent Output Quality)

- `validate_userspace_code` can pass while preview is still blank if workspace bootstrap expects `window.render` but bundle format hides it (IIFE export mismatch).
- Component `execute()` timeouts can still blank dashboards if generated code lacks try/catch fallback rendering.
- Prefer prompt/tooling fixes for these issues; do not patch proxy layer beyond the existing bridge injection.
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