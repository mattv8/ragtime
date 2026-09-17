# User Space runtime domain notes

Scope: runtime execution, workspace file/mount I/O, and preview proxies.

## Product decisions

- One shared runtime container is intentional. Per-workspace containers or
  microVMs are not the performance roadmap.
- Keep mutable devserver responses uncacheable; frequent edits are the primary
  workload. The [fresh-rootfs cost tradeoff](../docs/userspace-runtime-performance.md)
  is accepted in favor of recurring work.

## File-operation sequencing

- `_workspace_startup_lock` spans bootstrap/readiness; `_workspace_file_lock`
  covers file operations and filesystem transitions only. Substituting the
  startup lock for file coordination stalls editing during dependency installs.
- Startup takes its workspace lock and admission semaphore before the file
  lock; mount refresh takes the startup lock then file lock. Materialization's
  semaphore is inside the file lock. File APIs take only the file lock, then
  briefly `_lock` to recapture policy; `_lock` must not span a file-lock wait.
- In `_capture_file_target_locked`, select the deepest **lexical** mount match
  before opening paths. A rejected/read-only matched mount must not become
  "no mount" and fall through to the writable workspace root.
- Recapture target, mount policy, and operation identity after waiting for the
  file lock. Apply completion state only to that captured session/generation.

## Proxy contracts easy to break

- `ragtime/core/http_timeouts.py:get_http_proxy_safe_timeout_seconds()` reads
  `http_proxy_safe_timeout_seconds`. Tool timeouts and pooled-client idle
  defaults do not determine the preview request's elapsed-time budget.
- `X-Ragtime-Internal-HTTP-Budget-Ms` is duplicated across the deployed packages:
  control-plane `_proxy_http_request` replaces incoming values with remaining
  milliseconds; worker `_worker_proxy_budget_seconds` consumes 95%, and
  `_preview_request_headers` strips it before the app. The package split
  prevents sharing a Ragtime import with the worker image.
- This budget covers HTTP/SSE; WebSocket/PTY lifetime follows separate paths.
- An unstarted async generator does not run its `finally` on `aclose()`.
  Upstream connection cleanup therefore also belongs to the response object's
  close callback, including expiry before body iteration begins.
- The worker forwards raw HTML. The control plane owns decoding and removal
  of stale encoding/length headers after decoding, including `service_mode`.

## Copied mount refresh and pending bind optimization

- `_resolve_cloud_mount_source_local_path` downloads into staging, then swaps
  the directory at `cloud_mount_cache/...`. An unchanged path string can name
  a different inode. Retaining the old bind would retain the old tree.
- A pinned FD does not freeze that cache generation. Reject retired directory
  FDs and recheck the source path against the pinned identity around sync and
  fallback; an empty, retired source must not become a successful `--delete`.
- Live-bind no-op detection is still proposed. It must inspect actual source
  identity and mount flags, not merely compare saved source/target strings.
- Explicit filesystem `live_bind` requires mount authority and fails without
  it. Cloud/SSH caches bind read-only when possible and use a copied snapshot
  otherwise; do not silently turn a filesystem live bind into a snapshot.
- Copied mounts use `runtime/worker/mount_sync.py` with distro rsync from the
  runtime image. A normal image rebuild installs the binary; it need not exist
  in each app's rootfs. The launcher runs as an absolute standalone script;
  `python -m runtime.worker...` would import the worker FastAPI app per transfer.
- Rsync eligibility requires the binary and Landlock ABI >=3. If setup is
  unavailable before rsync starts, retain the old full-clear/copytree behavior.
  Exit77 is reserved for that case; transfer errors/partial exits/timeouts do
  not trigger fallback, and no fallback runs unconfined rsync.
- The fixed command preserves symlink text, modes, times and xattrs, checks
  content checksums, and delays source-authoritative deletions. Checksums still
  read files. No ownership/device preservation or writable hardlink shortcut.
- Parent transfers exclude active descendant mount roots with literal escaped
  patterns. Parent-first processing also preserves nested precedence in the
  full-copy fallback. Deletions stay inside copied mount trees, separate from
  canonical `files/` versus `rootfs/workspace` recovery.
- The helper inherits pinned root FDs and a minimal environment. Landlock
  restricts content access, not all chmod/utimensat/xattr operations; hostile
  concurrent metadata races remain possible. A refresh is not a filesystem
  snapshot. The legacy fallback does not gain Landlock's confinement.
- Materialization reuses the existing bootstrap time budget. Cancellation
  signals the child group and drains its thread before releasing the file
  fence; an unreapable kernel I/O operation must not be reported as quiescent.
- Nonzero rsync now fails startup/refresh instead of the former copytree
  warning-and-continue behavior. Partial updates can remain and need retry.
- Keep provider refresh cadence, staging publication, and source-error fallback
  behavior. This local materialization optimization does not eliminate cloud
  downloads or require another cache, manifest service, or configuration knob.

## Session observation

- `_ensure_session_row` must observe matching-provider `starting` rows as well
  as `running` rows. Restarting solely because the DB still says `starting`
  previously produced endless public-preview 503s despite a ready worker.
- `_reconcile_observed_session` checks session-instance identity as well as
  provider/worker IDs; a replacement can reuse the same worker ID.
- Manager heartbeats currently renew leases. The broader state-authority
  clarification is planned, not an implemented DB-owned lease protocol.

## Bridge credential delivery modes

- `Workspace.bridgeCredentialMode` is aspiration; the active mode is fixed when
  a runtime **session** starts (`_runtime_provider_start_session`). Manager and
  worker restart paths accept no mode change, and the session-preserving
  `app/restart` keeps the current mode. `requires_restart` in the mode API
  means a full devserver (session) restart.
- Env refresh must finalize env for the **session's observed** mode
  (`_active_session_bridge_credential_mode`), never the DB mode. Finalizing a
  live env-mode session with a flipped `worker_file` value strips the env token
  and breaks the bridge until a full session restart. The worker independently
  re-applies file-mode invariants in `restart_session` (drop any raw
  `RAGTIME_BRIDGE_TOKEN`, restore `RAGTIME_BRIDGE_TOKEN_FILE`); keep both
  defenses.
- File-mode starts carry the token in `bridge_token_file_initial_token`
  (`exclude=True`). That is safe only because manager→worker is in-process via
  `get_worker_service()`; a serialized remote worker transport would silently
  drop the token.
- The capability key `bridge_credential_file` must appear in worker **session**
  `runtime_capabilities`, not just health metadata: the manager gates refresh
  on session capabilities and the control plane gates mode switches on status
  capabilities. Health-only advertisement makes file mode unreachable.
- The token file lives at rootfs `/run/.ragtime-bridge/token` and is written
  through a directory-FD chain with `O_DIRECTORY|O_NOFOLLOW` per component.
  Leaf-only `O_NOFOLLOW` does not stop parent-symlink traversal into the host;
  do not replace this with Path-based I/O.
- Refresh CAS: an exact `(request_id, token fingerprint)` duplicate returns the
  cached metadata (history bounded to newest 32; evicted replays 409 as
  stale); same id with a different token is 409. The control plane never
  re-POSTs an ambiguous refresh — it re-observes and requires healthy state,
  same session, `worker_file` mode, and an **advanced revision**. Session-id
  match alone is a false-success trap (the file exists from startup).
- Refresh stores the newest token into `bridge_token_file_initial_token`, and
  every startup-pipeline run rewrites the file from that field. That is why an
  app recycle keeps the rotated token, and why a file-mode session whose status
  degrades to `missing` (status `mode` defaults to `env` without credential
  metadata) still converges through the env-style restart recovery.
- Worker `restart_app` 409s while an exec or startup is active. Execs register
  a reservation under `_lock` **before** the spawn releases it, closing the
  restart-vs-spawn race; `_active_execs` and `_app_restart_requests` are pruned
  in the stop path or they leak for the worker process lifetime.
- Exec transport: `runtime_manager_request` retries POSTs on transport failure,
  so exec/restart/refresh must pass `retry_safe=False`; exec adds a 30s grace
  over the requested budget. The worker kills the process group on timeout and
  on `CancelledError` (shielded). The chat diagnostics shell cap
  (`CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS`) is a separate, intentionally
  lower guardrail — do not unify it with workspace exec policy. Workspace
  defaults/ceilings come from admin settings through
  `ragtime.core.userspace_limits`; execution-time service validation is
  authoritative even when the caller holds an older tool schema. Omitted tool
  timeouts must reach the service as `None`, not a captured default.
- The runtime package cannot read app settings. Its independent 3600s exec
  ceiling in `runtime.core.shared` mirrors the control-plane hard-cap constant;
  keep the parity test green when changing either. The agent event stream has
  its own 315s inactivity guard: an active workspace terminal call needs its
  snapshotted extended budget, but the post-tool provider guard stays short.
  Raising only the worker timeout still aborts long chat-initiated commands.
  Omitted-timeout events can arrive after an admin policy change without the
  already-accepted duration; their stream watchdog uses the hard-cap budget
  conservatively, without extending the worker's actual command timeout.
- Durable restart ledger: idempotency and the 60s workspace throttle are
  decided inside `db.tx()` under
  `pg_advisory_xact_lock(hashtextextended('userspace-runtime-restart:{ws}',0))`
  and the intent row commits **before** manager dispatch. The worker
  `request_id` is the ledger row UUID — caller idempotency keys collide across
  users on the same workspace. Ambiguous 5xx marks the row `interrupted`,
  never redispatches; `completed` requires the still-active session to report
  the matching `runtime_operation_id` with phase `ready`.
