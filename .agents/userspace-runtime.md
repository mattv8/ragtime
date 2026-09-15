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
