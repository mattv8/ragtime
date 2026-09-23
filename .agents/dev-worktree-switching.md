# Agent procedure: switch the local development stack between worktrees

## Scope

Use this procedure only when an agent must implement or test from a linked
checkout. Ragtime has one shared local development stack. Do not switch while
another agent is running active verification against that stack.

Run the switcher from the primary checkout. It preserves the primary checkout's
`.env`, `.data`, Compose project, and development volumes while mounting code
from the selected registered worktree.

## Inspect and select a target

Inspect registered eligible worktrees, then set `TARGET` to the selected branch
name or absolute worktree path before running the dry run:

```sh
bash scripts/switch-dev-worktree.sh --list
bash scripts/switch-dev-worktree.sh --dry-run "$TARGET"
```

Select a target explicitly from the primary checkout:

```sh
bash scripts/switch-dev-worktree.sh "$TARGET"
```

`--dry-run` checks the target, Compose configuration, and migration plan without
building images, stopping services, or changing the database. It may write local
plan or cache artifacts. Do not report a successful switch from file presence or
a dry run alone; the active feature remains under verification until the real
switch and its checks complete.

Do not use a bare `docker compose` command to change checkouts. It bypasses
migration reconciliation and can remount code against the shared database.

## Author migration reversals before switching

`migration.sql` and its optional `down.sql` are authored repository source.
For every branch-only migration that may need to be removed later, author a
matching `down.sql` beside `migration.sql`. A target can require reversal of
later shared migrations to insert an earlier branch migration, so author down
SQL for every migration that could become outgoing.

Write `down.sql` as static transaction-body SQL only: omit `BEGIN`, `COMMIT`,
`ROLLBACK`, and `DO` blocks. The current validator accepts only `ALTER`, `DROP`,
`DELETE`, and `TRUNCATE` statements. The switcher takes no backups; reversing a
branch-only table or column can discard its data.

`scripts/worktree_down_migrations.json` is a required authored fallback input
for the current legacy baseline and branch inverse definitions. Do not ignore,
delete, or treat it as generated state.

## Failure and retry

Routine switching reverses the planned outgoing migrations in one transaction,
then deploys the selected target migrations. The switcher never automatically
reverses a partially failed run. It does not infer reverse SQL, restore backups,
reset the database, run `db push`, repair Prisma metadata, or rewrite checksums.

If a failure occurs after the migration boundary, code writers remain stopped.
The database may be mid-transition; do not restart the original checkout against it.
Correct the authored migration or perform an explicitly scoped normal Prisma
repair, then rerun the complete switch command to create a fresh plan. Never
reuse a stale plan or assume failed-run recovery occurred automatically.

## Local state ownership

`.data/worktree-switch/cache/`, `.data/worktree-switch/runs/`,
`.data/worktree-switch/state.json`, and `.data/worktree-switch/switch.lock` are
local ignored switch artifacts. Do not edit or commit them as migration source.

For maintenance of the active target only, use the override-based Compose
invocation recorded in `.data/worktree-switch/state.json`; it is not a way to
switch worktrees. Use `scripts/switch-dev-worktree.sh` for every checkout change.

## Durable SQLite history

Reversing code and Prisma migrations does not reverse runtime SQLite-history
activation or legacy-to-Restic conversion. When selecting a target against
already-activated data, require compatible history readers and writers; do not
promise automatic rollback. See `.agents/userspace-sqlite-history.md` for the
activation, repository, and recovery contracts and
`.agents/userspace-object-storage.md` for separate gateway-owned backup state.
