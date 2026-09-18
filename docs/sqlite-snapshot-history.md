# SQLite snapshot history

User Space stores protected SQLite history beside each workspace's editable
`files/` directory, under its controlled `sqlite_backups` sibling. It requires
no database migration, service, mount, or configuration setting. This is
recovery protection for managed databases in `.ragtime/db/`; it is not a
general filesystem backup or an atomic transaction covering workspace code and
database writes.

## Capture and catalog

Managed files with the supported SQLite suffixes are captured through SQLite's
online backup API. Captures can be requested manually, associated with a code
snapshot, scheduled hourly, or required as a pre-restore safety point. The
history feature is independent of the workspace SQLite persistence prompt
setting, so a code snapshot can capture a managed database even when no code
change is otherwise needed.

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
pointing to an evicted blob. Scheduled maintenance claims due work atomically
under the catalog lock, preventing overlapping replicas; a failed attempt can
be retried on a later hourly pass.

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

Focused automated tests and lint checks are recorded in the SQLite-history
hardening lane reports. A live authenticated smoke result is not recorded here;
do not treat this document as evidence that a live smoke test passed.
