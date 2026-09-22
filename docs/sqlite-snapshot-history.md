# SQLite snapshot history

User Space stores protected SQLite history beside each workspace's editable
`files/` directory, under its controlled `sqlite_backups` sibling. PostgreSQL
stores only durable capture-queue metadata; protected blobs and catalogs remain
file-backed. This is recovery protection for managed databases in
`.ragtime/db/`; it is not a general filesystem backup or an atomic transaction
covering workspace code and database writes.

## Capture and catalog

Managed files with the supported SQLite suffixes are captured through SQLite's
online backup API. Manual, snapshot-associated, and hourly captures enter a
durable queue and return an accepted job before SQLite work begins. A queued
capture reads the database when the worker actually executes it, not when it
was requested or when a code snapshot was created; ready-record timestamps
therefore represent execution time. The history feature is independent of the
workspace SQLite persistence prompt setting, so a code snapshot can capture a
managed database even when no code change is otherwise needed.

Pre-restore and guarded code-restore preservation remain inline while
maintenance is held. Their mandatory safety capture is never queued behind
ordinary work.

Ready records contain the capture time, SHA-256, byte size, trigger, and, when
applicable, the User Space snapshot ID and Git commit hash. Failed captures are
diagnostic records with no blob; they never represent database bytes as having
been captured. Deleting a live database does not delete its existing history.

The catalog and protected blobs reject symlinks and non-regular special files.
Source enumeration, publication, sidecar removal, and hashing use
descriptor-relative, no-follow operations. Capture reserves main-file and
SQLite sidecar capacity before work, then checks actual produced bytes before
publication. Preview candidates, temporary downloads, and their disk usage are
also included in the workspace quota.

### Unchanged databases and shared backups

An hourly check skips creating a new record and backup file when it verifies
that the database is unchanged. Manual captures and user/agent code snapshots
still create separate history records, with their own timestamps and snapshot
associations, but can reference the same immutable backup file. A record's
`size_bytes` is the size of the recoverable database; adding record sizes does
not measure physical storage usage. Deleting one record does not remove a file
that another record still references.

The confined child hashes the main database, WAL, and rollback-journal state
through pinned file descriptors and checks that file identities and metadata
remain stable. It ignores SHM synchronization state. A full capture receives a
reusable source token only when its before/after source checks agree. Reuse also
verifies the saved backup's size and checksum. Legacy records without a token,
concurrent writes, replacement, checkpoint changes, or a nonempty rollback
journal cause a conservative full capture rather than an unsafe skip.

These checks still read the database/WAL and the existing backup. They avoid
redundant backup copying, integrity scans, and durability writes; they are not
zero-I/O or incremental backups. They assume coherent local POSIX filesystem
metadata, not coarse or incoherent network-filesystem timestamps.

Before a restore, the service always makes and verifies a fresh safety capture.
Only after that capture succeeds can identical output share an existing file.
Ordinary captures omit logical row fingerprinting and the redundant first
integrity scan; restore preview and drift validation retain their logical
fingerprint checks.

### Restore preparation work

Restore preview and drift preparation create lightweight private copies when
their copy result is used only for later validation. Each path computes one
explicit logical fingerprint of the current private copy. Merge preparation
then reuses that inspected private current copy as its writable candidate
instead of making another copy of the live database. These reductions do not
change the inline, verified pre-restore safety capture.

Logical fingerprints still read source database state and can require SQLite
sorting or temporary disk. Merge and overwrite comparisons remain streamed and
keyed: they do not materialize a whole database in Python memory.

## Confinement and platform requirement

Capture, restore-preview preparation, and live drift fingerprinting run in a
small child process with pinned source and destination directory descriptors.
The child applies a minimal Landlock policy before SQLite opens database,
WAL, SHM, journal, or migration names. Migration SQL is copied through
no-follow descriptors into private scratch before it is fingerprinted or used
to prepare a candidate. This prevents a workspace path or component swap from
redirecting these operations outside the pinned workspace.

This boundary requires Linux Landlock ABI 3. If the ABI, supported architecture,
or confinement setup is unavailable or fails, capture, restore preparation, or
drift checking fails closed with HTTP 503. There is no path-based fallback.

## Access, previews, and restore

Only a workspace owner or global administrator can list, capture, download,
delete, preview, restore, or recover history. Editor membership and
cross-workspace SQL grants do not grant historical-database access.

A restore preview creates a disposable, verified candidate and is bound to its
workspace, backup, database, mode, global and per-table conflict policies,
creator, live-database fingerprint, migration fingerprint, and short expiry.
Apply rechecks candidate integrity and both live fingerprints, preventing a
candidate prepared before a database or migration change from being published.
The browser invalidates pending preview state whenever that context changes;
late preview success, error, or cleanup cannot revive an older result or
overwrite a newer operation. Destructive confirmation and recovery similarly
block duplicate requests.

Merge keeps current-only rows and reports conflicts; overwrite replaces the
live database with the verified candidate. Neither path executes workspace code
or arbitrary migrations with control-plane privileges. Restore stops the
runtime; starting preview again is explicit. Before publication, a live
database receives a mandatory protected safety capture. Publication is
descriptor-relative, verifies the published checksum, and removes stale SQLite
sidecars.

Successful restore receipts are durable, bound to the workspace and initiating
user, and can be replayed by preview ID after preview expiry or cleanup without
draining the runtime. Legacy receipts without explicit ownership are not
attributed to a requester.

## Active, interrupted, and recovery state

SQLite control-plane operations that access workspace files hold a cross-process
shared workspace-operation lock for the whole operation. Maintenance and
recovery hold an exclusive lock. A status probe returns `active` while a live
operation holder exists; it is not an interruption and exposes no completion or
abort action. The durable maintenance marker records crash state, not liveness,
and locks are not stolen by TTL.

`interrupted` identifies recoverable work after exclusive ownership can be
acquired. A restore interrupted before publication can be aborted; once durable
publication has begun, recovery must complete from the verified candidate.
`release_pending` means a terminal receipt exists but the runtime fence still
needs idempotent release. `invalid` means the fence is unsafe or unverifiable
and requires operator recovery. Recovery revalidates marker ownership under its
exclusive lock; online, legacy, or ambiguous markers require a configured
runtime manager rather than guessing that a runtime is offline.

## Retention, quota, and maintenance

The policy retains ordinary ready backups for 30 days, limits them to 100 per
database, and limits all history storage for a workspace to 1 GiB. The newest
ready backup for each database, preview and restore-operation dependencies, and
pre-restore safety points for at least seven days are protected from manual
deletion, retention pruning, and quota eviction. The same policy determines a
record's `can_delete` value.

Quota eviction is planned before disk mutation. If protected usage leaves too
little capacity, the operation fails with HTTP 409 before capture or destructive
publication. When eviction is possible, the reduced catalog is saved before
blobs are removed, so a subsequent capture failure cannot leave a ready record
pointing to an evicted blob. Shared backup files count once toward the quota;
removing an alias frees no file space until its last reference is removed.

Maintenance computes catalog bookkeeping and blob reference counts without
repeated candidate scans. A cleanup that leaves the catalog's canonical
semantic content unchanged does not rewrite the catalog or issue the associated
durability write; saves for claims, intents, completions, and publications are
still required. Catalog updates and cleanup continue to run under their
existing lock scopes.

### Scheduling, queueing, and admission limits

The durable queue runs one ordinary capture globally and one per workspace.
This global cap of one deliberately leaves one of the two confined-child slots
available for inline safety and preview work. Pending jobs are FIFO among rows
whose scheduled availability time has arrived, while a workspace with a running
job is skipped. Repeating an exact request key returns the existing job; a
reused key with different request data is rejected. Only active scheduled work
coalesces. A failed scheduled job receives one new pending retry after five
minutes, while an interrupted job is not replayed automatically.

An owner or global administrator can list a workspace's jobs, observe queued,
running, and terminal status, and cancel a pending job. Running cancellation is
cooperative: the current database capture drains and completed records remain
visible before the job becomes cancelled. Legacy synchronous capture requests
remain compatible by waiting for their accepted job and returning the matching
result; new callers should use the accepted-job API and poll status.

Pending jobs survive a restart. A process that loses durable ownership does not
resume ambiguous work; stale running jobs become `interrupted` after liveness
fencing rather than being silently replayed. Queue rows retain terminal history
for 30 days, separately from the file-backed backup retention policy.

Safety and unchanged-capture behavior are unchanged. Queueing does not relax
confinement, no-follow catalog/blob handling, quota admission, or mandatory
pre-restore capture. It also does not make change detection free: unchanged
checks still read the database/WAL and existing backup as described above.

The scheduler checks for work every 60 seconds. First-seen workspaces receive
a deterministic delay of up to five minutes before their first scheduled
capture, subject to queue load. Successful checks schedule the next attempt
one hour later; failures retry after five minutes. Each pass admits at most
eight workspaces and stops admitting more after 30 seconds. An active capture
is allowed to finish. Rotating discovery order avoids repeatedly preferring
the first workspace. Cached due times reduce catalog reads between attempts.

A per-job liveness lock spans queued capture and completion. A replica can
recover an abandoned claim only after acquiring that lock; a timeout alone does
not displace active work. Maintenance also cleans expired preview candidates,
orphaned backup files, and stale downloads independently of whether a new
capture is due. Active restore candidates remain protected.

Capture requests have a per-process limit of 16 outstanding requests. Confined
capture, probe, preview, and drift subprocesses share two file-lock slots across
app processes using the same `INDEX_DATA_PATH`. Slots live outside editable
workspace files at `_userspace/sqlite_capture_slots`. Waiting for a slot is
limited to five seconds; excess work receives a controlled busy failure rather
than launching unlimited children. A cancelled caller drains its running worker
before releasing admission. Processes with independent storage roots have
independent budgets.

When no job is available, a queue worker backs off its durable claim waits at
one, two, four, and then five seconds. A local enqueue or cancellation wakes
the worker promptly and resets that idle backoff. Another process's enqueue is
observed by the periodic durable claim, with up to approximately five seconds
of idle detection delay. This changes idle polling only: job state remains in
the durable store, and normal job waiting and error recovery retain their
one-second polling cadence.

Logs record admission wait and subprocess duration, capture/reuse/skip outcome,
logical backup size, and new retained blob bytes where applicable. Retained
bytes are not physical device IOPS: deduplicating after a changed-state capture
can still incur copying before the duplicate file is removed.

## Validation and migrations

Capture validates SQLite structural integrity with `PRAGMA integrity_check` but
does not reject an already-existing foreign-key anomaly. Restore candidates are
stricter: they run `foreign_key_check` before becoming applicable. This permits
preservation of an imperfect historical source while preventing publication of
a candidate with foreign-key violations.

Migration files are checksummed and lineage is checked against the migration
ledger before schema conversion. The runner splits complete SQL statements at
semicolon boundaries, including multiple statements on one line, while
preserving semicolons in quotes, comments, and trigger bodies. Each migration's
statements and its ledger entry execute in one transaction. Transaction-control
statements, `PRAGMA`, `ATTACH`, and `DETACH` are rejected in migrations; missing
or changed lineage blocks conversion, but downloading the historical database
remains available.

## Residual limits and verification status

The online capture has a 60-second engine timeout, and restore comparison is
streamed/keyed rather than fully materialized in Python. Large databases can
still consume SQLite temporary disk, I/O time, and process resources; this is
not a universal resource or time guarantee. The feature also cannot protect
against arbitrary workspace shell commands or external filesystem changes made
outside its fenced operation.

The catalog lock remains held during capture and probing, so history listing
or deletion can wait for those operations. Continuously changing databases and
checkpoint-heavy workloads may get fewer skips. Code-snapshot creation awaits
durable queue acceptance; database capture executes later and records its own
outcome. Quotas and
retention bound retained history, not cumulative source hashing, reads, or
write traffic. Fixed queue admission and confined-child capacities remain in
force; no optimization increases them.

Focused automated tests and lint checks are recorded in the SQLite-history
hardening lane reports. A live authenticated smoke result is not recorded here;
do not treat this document as evidence that a live smoke test passed.
