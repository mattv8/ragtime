# User Space object-storage domain notes

- `runtime-s3` is internal-only on the Compose network. The active gateway has
  exactly two mounts: read-write `_userspace/_object_storage` and read-only
  `object-storage-key`; legacy workspace buckets are no longer gateway-mounted.
  Deploy the paired Ragtime and `runtime-s3` images with the updated Compose
  definition. Before starting the upgraded gateway, remove the renamed
  `object-storage` or `object-storage-dev` container if present; two gateway
  processes must not own the same registry. Pull paired images before applying
  the mount-free Compose definition: the pre-migration gateway importer treats
  missing source directories as successful empty imports. New Ragtime also
  requires the paired gateway's `/legacy-import` endpoints; an older gateway's
  404 must never be interpreted as verification authorizing source GC.
  Recreate containers to change mounts.
- Legacy S3 migration is staged asynchronously after Ragtime readiness because
  the gateway starts after Ragtime health. HTTP reads do not import; unfinished
  work is fenced with 503 while the background reconciler advances one workspace
  at a time. Gateway import jobs persist their generation and manifest digest,
  so retries/restarts resume the same staged identity.
- Orchestration lives in `object_storage/legacy_coordinator.py`; filesystem
  staging/GC lives in `legacy_migration.py`. Service methods are thin adapters.
  Python sorts complete manifest paths globally; Java accepts unique safe paths
  in any order so previously published manifests remain retryable, including
  nested and supplementary-Unicode filenames. Preserve manifest-byte digests.
  Adopt pre-upgrade pending/copying/failed states only without an existing bound
  job; never adopt completed or revoked workspaces as fresh imports.
- `LegacyObjectStorageMigrator` copies legacy `s3/buckets` through no-follow,
  descriptor-pinned reads into `_object_storage/_legacy_imports/<workspace>/<generation>`.
  It writes a digest-bound receipt and manifest, and the gateway validates every
  staged file, imports/verifies bytes and relevant S3rver metadata, then writes
  verification evidence. Do not reintroduce direct gateway reads of workspace
  source files or synchronous startup imports.
- Source GC is deliberately conservative: only gateway-verified files whose
  recorded source root, inode identity, digest, and directory entry still match
  are unlinked. GC waits for no active or stopping runtime session and shares
  `workspace_gc_fence()` with runtime-start admission. Acknowledged GC removes
  the staged generation and marks its retained receipt consumed, preventing
  replay. Changed, ambiguous, unfinished-upload, and orphaned artifacts are
  retained and reported; missing sources are reported, never invented as empty.
  Older completed imports without bound provenance must not be replayed or GC'd.
  Consumed receipts use compact v2 identity/count summaries; active receipts
  remain v1. Historical consumed-receipt compaction must hold the restore lock
  before opening/reading the receipt tree, so it cannot overwrite restored work.
- Staging requires 2.25 times the source byte size in free space. The runtime
  fence spans a reconciliation tick, so starts can wait for staging; the shared
  backup lock spans the source copy, so backups/restores may also wait. A failed
  job or active-runtime cleanup deferral can hold up later workspace migrations.
- Backup/restore and staging share `core.file_lock.backup_restore_lock()`;
  `server_backup.locked_operation()` remains a compatibility wrapper. Backups
  omit only unpublished `tmp-` generations; published generations and receipts
  remain resumable backup data. Restores affecting initialized object storage
  require the gateway offline and `OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED=true`.
  Recreate (not restart) `runtime-s3` after Ragtime republishes the managed key.
  These exclusions and offline requirements are object-storage-specific. For
  managed SQLite history, keep `_userspace/_sqlite_history` and workspace
  `sqlite_backups` paths out of generic replacement and rollback. An active
  runtime export replaces those inputs in the generic backup; runtime transfer
  handles their portability. This does not exclude every capture-slot lock path.
  See `.agents/userspace-sqlite-history.md`; do not generalize its runtime-install
  replacement behavior to object storage.
- Keyless FULL/FILES replacement restores preserve existing regular managed/JWT
  key files. Validated archived keys take precedence. A keyless initialized
  storage archive requires an existing authoritative key; reject unsafe key
  paths before mutation rather than inventing replacement keys.
- Focused migration checks: `pytest tests/test_userspace_legacy_migration_lifecycle.py
  tests/test_userspace_legacy_object_storage_migration.py
  tests/test_userspace_object_storage_service.py tests/test_server_backup_object_storage.py`
  and `docker build --target storage-test -f docker/Dockerfile.storage .`.
