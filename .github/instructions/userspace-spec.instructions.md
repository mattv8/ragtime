---
applyTo: 'ragtime/userspace/**'
---

# User Space Feature Implementation Instructions

Last updated: 2026-02-25

## Purpose

Use this file as the implementation guardrail for User Space feature work.
Keep changes scoped, incremental, and aligned with runtime hard-cutover goals.

## In Scope (Active)

- Runtime session APIs and lifecycle in `ragtime/userspace/runtime_service.py` and `ragtime/userspace/runtime_routes.py`.
- Runtime preview proxying (HTTP + WS) for workspace and shared routes.
- Collaboration transport and file operations under `/indexes/userspace/collab/*`.
- Terminal websocket behavior and role-based write restrictions.
- Runtime status/session/preview UX wiring in User Space frontend components.

## Out of Scope (Do Not Invent)

- New UX surfaces, pages, or workflows not explicitly requested.
- New role semantics or auth flows outside existing `owner/editor/viewer` policy.
- New runtime provider protocols beyond current `local` + `http` provider seam.

## Current Cutover Status (Authoritative)

Implemented now:

- Runtime session persistence + API routes are live.
- Runtime provider seam is live (inferred from `USERSPACE_RUNTIME_MANAGER_URL` HTTP(S) value).
- HTTP runtime-manager service contract is live (`/sessions/start`, `/sessions/{id}`, `/sessions/{id}/stop`, `/sessions/{id}/restart`).
- Runtime manager code is organized under `runtime/manager/{models,service,api}.py` with `runtime/main.py` as the runtime container entrypoint.
- Runtime manager includes pool/session orchestration scaffolding in-container (warm slots, lease TTL, idle eviction, reconcile loop).
- Preview proxying supports HTTP + websocket upgrades.
- Shared preview proxy supports both slug and token route shapes.
- Collaboration supports text-sync updates, conflict resync, presence events, and file create/rename/delete broadcasts.
- Runtime PTY websocket streams a real subprocess; write access is editor/owner only.

Still pending for full hard-cutover:

- Replace runtime-manager in-memory orchestration scaffold with real MicroVM pool/session orchestration backend.
- Replace text-sync collaboration with CRDT/Yjs semantics.
- Remove remaining legacy `srcDoc` fallback behavior.
- Finish operator runbook + end-to-end runtime validation checklist.

## Required Engineering Rules

- Reuse existing service/router helpers before adding new code paths.
- Enforce workspace role checks in every runtime/collab route:
  - viewer: read-only preview/collab observation
  - editor/owner: mutating collab/fs + terminal input + runtime controls
- Keep runtime and collab writes auditable; do not silently skip failures.
- Keep preview proxy header filtering strict; do not forward hop-by-hop headers.
- Keep path handling traversal-safe using existing normalization helpers.

## API Contract Surface (Feature Work)

- Runtime:
  - `GET /indexes/userspace/runtime/workspaces/{workspace_id}/session`
  - `POST /indexes/userspace/runtime/workspaces/{workspace_id}/session/start`
  - `POST /indexes/userspace/runtime/workspaces/{workspace_id}/session/stop`
  - `GET /indexes/userspace/runtime/workspaces/{workspace_id}/devserver/status`
  - `POST /indexes/userspace/runtime/workspaces/{workspace_id}/devserver/start`
  - `POST /indexes/userspace/runtime/workspaces/{workspace_id}/devserver/restart`
  - `WS /indexes/userspace/runtime/workspaces/{workspace_id}/pty`
- Collaboration:
  - `WS /indexes/userspace/collab/workspaces/{workspace_id}/files/{file_path:path}`
  - `POST /indexes/userspace/collab/workspaces/{workspace_id}/files/create`
  - `POST /indexes/userspace/collab/workspaces/{workspace_id}/files/rename`
  - `POST /indexes/userspace/collab/workspaces/{workspace_id}/files/delete`
- Preview proxy:
  - `ANY /indexes/userspace/workspaces/{workspace_id}/preview[/{path:path}]`
  - `WS  /indexes/userspace/workspaces/{workspace_id}/preview[/{path:path}]`
  - `ANY /indexes/userspace/shared/{owner_username}/{share_slug}[/{path:path}]`
  - `WS  /indexes/userspace/shared/{owner_username}/{share_slug}[/{path:path}]`
  - `ANY /indexes/userspace/shared/{share_token}/preview[/{path:path}]`
  - `WS  /indexes/userspace/shared/{share_token}/preview[/{path:path}]`

## Verification Checklist (Before Claiming Done)

1. `python -m prisma migrate deploy` succeeds in the app container.
2. Authenticated runtime session flow works:
   - session `GET` returns 200
   - `POST .../session/start` returns success
   - `GET .../devserver/status` returns expected state payload
3. Workspace preview route responds (200 when upstream exists, 502 with clear error when unavailable).
4. No new stack traces in `docker logs ragtime-dev` for touched endpoints.
5. Any touched frontend/runtime types pass diagnostics (`get_errors`).

## Runtime Plan Reference

`User Space Runtime Plan v2.md` is the target roadmap.
If a plan item is not implemented, document it explicitly in PR/summary rather than implying completion.