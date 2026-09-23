# Runtime-owned SQLite history domain notes

Scope: durable workspace SQLite capture, retention, restore, conversion, and
transfer after runtime-history activation. The implementation lives in
`runtime/worker/sqlite_history/`; do not invent a control-plane local writer.

## Authority and ownership

- Start with `coordinator.py`, `service.py`, `operations.py`, `storage.py`,
  `catalog.py`, and `transfer.py`; controller queue/replay/ACK policy is in
  `ragtime/userspace/sqlite_history.py`, `sqlite_backup_queue.py`, and
  `sqlite_backup_queue_store.py`.
- `runtime/manager/api.py:create_app`'s lifespan explicitly calls the embedded
  coordinator because including worker routers does not invoke worker lifespan.
- Run transfer-journal recovery before activation gating, and reconcile non-live
  receipts before ordinary maintenance.
- The runtime owns the repository, service engine, durable operation receipts,
  storage gates, conversion/ledger migration, and transfer/export portability.
- The controller retains owner/admin authorization and its PostgreSQL queue; it
  must not read runtime image paths after activation.
- The runtime repository is `<runtime-root>/_sqlite_history/restic`; its key is
  `<runtime-root>/_sqlite_history/secrets/repository-password`. Cache and scratch
  locations are separate.
- A workspace catalog is
  `<runtime-root>/workspaces/<workspace-id>/sqlite_backups/manifest-v1.json`.
- Runtime activation is `<runtime-root>/_sqlite_history/activation-v1.json`
  (version 2, `active: true`). The controller's separate marker is
  `<INDEX_DATA_PATH>/_userspace/sqlite-history-runtime-activation-v2.json`.
- Before cutover, retain the passive legacy path. After cutover, fail closed on
  an unavailable or ambiguous runtime: never revive a local writer as fallback.

## Capture and operation receipts

- Runtime capture and reconciliation enter through `RuntimeSqliteHistoryService`
  in `service.py`: `capture_workspace_databases()` and
  `reconcile_capture_operation()`.
- A capture has a queue UUID, canonical payload/digest, and per-database UUIDv5
  suboperations. Preserve all three when retrying or observing.
- A GET 404 can permit replay only for the same ID and payload. Transport
  uncertainty does not prove absence; observe rather than blindly retrying.
- A retired ID must never execute again.
- ACK only after a durable SQL terminal projection. Observation rotation changes
  `updated_at`, not `finished_at`.
- An empty controller scan must avoid a runtime call.
- Runtime-mode shutdown detaches rather than cancels; classify from durable
  activation before awaiting the operation.
- A due occurrence is redisplayed until the controller queues it and ACKs it.
- Controller workspace selection is DB-authoritative and rotates batches of 8;
  the runtime API permits up to 100 workspaces per request.
- Catch transient scheduler failures; pending maintenance is best-effort, not a
  fairness or progress guarantee.

## Locks, ordering, and async boundaries

- `storage.py:AsyncRepositoryGate` is the async control-flow gate; history order
  is repository gate, workspace access fence, then catalog access.
- Ordinary access checks the durable maintenance marker inside its workspace
  fence. Destructive restore takes exclusive workspace maintenance.
- The ordinary-capture flock is `_sqlite_history/capture.lock`.
- Two confined slots use `<runtime-root>/sqlite_capture_slots/capture-{0,1}.lock`.
  They align with `<INDEX_DATA_PATH>/_userspace/sqlite_capture_slots` only when
  the shared-root mapping aligns; preserve any custom-root mapping.
- Runtime and legacy wrappers share that physical lock namespace even though
  their retained wrapper implementations are not unified.
- Async waits use `AsyncRepositoryGate`, never default-executor threads.
  Maintenance uses bounded `try_exclusive` and defers while busy; no fairness
  guarantee exists. Synchronous `repository_gate` blocks inside off-loop
  export/import work; do not imply all synchronous gate waits are removed.
- Guarded Git admission uses async context registration, a closing flag, and a
  drain event. Validate a finisher before consuming its future.
- Drain admitted Git and off-loop I/O before releasing a fence or publishing.
- Live operator recovery may bypass fresh gate admission for its owning guard.
- There is no automatic timeout-abandonment guarantee for a guarded operation.

## Integrity, restore, and retention

- Normal capture streams and verifies committed Restic data.
- Pre-restore fully materializes and verifies before publication.
- Default source detection is checksum-based. Full online SQLite copying remains;
  Restic is not a page-level source-incremental capture mechanism.
- Logical quota is unique ready `(sha256, size)` per workspace plus actual
  private temporary bytes. Shared repository compression is not workspace credit.
- `_history_disk_usage` includes scratch, candidates, downloads, and imports;
  it does not count catalog-file bytes.
- Deletion, retention, and quota eviction queue Restic-forget tombstones; remove
  them after successful forget of explicit unreferenced snapshots. Compact retired
  operation-identity tombstones remain retained.
- Capture reconciliation completes an operation only after its tagged snapshot
  has a matching durable catalog and verified bytes; a tag alone is insufficient.
- Export payloads are retained for 24 hours; acknowledged receipts retire after
  30 days (`models.py`).
- Current limitation: receiptless `<runtime-root>/_sqlite_history/transfers/temp`
  uploads can persist and receipt scans are linear. Future cleanup needs
  liveness/reference proof; arbitrary orphan adoption is unsafe.

## Conversion and transfer

- `conversion_ledger.py` is pure identity and validation. In
  `conversion.py`, `LegacyHistoryConverter.migrate()` verifies a fresh artifact
  once as the current pass advances, then reverifies loaded stages on resume.
- Publish a catalog before removing its legacy blob.
- Bootstrap acceptance uses a lightweight inventory; dry-run and full
  verification retain integrity checks.
- The stdlib CLI defaults to no-action inventory. Mutation requires explicit
  `--apply`, `--resume`, or `--cancel`; an unknown 5xx is observed with the same
  run ID.
- Incoming transfers verify ready Restic reference IDs and repository ID before
  mutation. Destination-only references must be incoming-reachable, including a
  same-repository older bundle.
- Replace included catalogs during import; destination catalogs with no Restic
  references may remain legacy. Reject invalid bundles before mutation.
- `transfer.py:export_download_lifetime()` pins shared download liveness against
  exclusive cleanup; acquire it in the ASGI lifecycle.
- Stage only proven immutable Restic packs as same-filesystem hardlinks. Copy
  metadata under the barrier; on EXDEV, check capacity before copying.
- Plain tar work happens outside the gate. Use
  `runtime/core/private_file_response.py` and
  `ragtime/userspace/sqlite_history_transport.py`; never return a runtime path
  for a controller process to read.

## Backup, replacement, and recovery boundaries

- In `ragtime/core/server_backup.py`, activated export handles managed history;
  restore excludes it only when the managed repository exists. Legacy history is
  generic-walker state until activation; do not apply blanket pre-activation
  exclusions.
- Generic replacement/rollback preserves live `_sqlite_history` and per-workspace
  catalog inodes, not all capture-slot lock trees. Runtime transfer is required
  for portable Restic history; a filesystem/object-storage copy alone is not.
- Runtime install/import may replace activation, key, repository, operations, and
  catalog paths under its journal and exclusive gate; do not claim those inodes
  survive runtime installation.
- Include the repository key only in an explicitly encrypted server export.
  Keyless import requires an existing matching key.
- Validate stream SHA and size, while allowing legacy metadata compatibility.
- See `.agents/userspace-object-storage.md` for gateway backup/restore and
  `.agents/dev-worktree-switching.md` for durable-history target selection.

## Change checklist

- Preserve activation fail-closed behavior, receipt idempotency, and gate order.
- Keep repository work runtime-owned and use transport APIs across the boundary.
- Account for temporary private bytes as well as logical ready-image quota.
- Treat scheduler maintenance as deferrable. Observe transport uncertainty; only
  an authoritative same-ID, same-payload 404 permits replay.
