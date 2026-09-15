# User Space Runtime Performance Decision

**Accepted:** 2026-09-14

## Decision

For User Space sandboxes, we accept a higher first-rootfs provisioning cost to
preserve independent inode isolation. Product priority is normal starts,
restarts, and public app loading—not minimizing the fresh chroot copy alone.
A fresh-cost result by itself is not a blocker and does not require follow-up.

This applies to the no-mount chroot fallback only. It does not change the
mount-capable chroot or `pivot_root` paths, which expose system directories by
read-only binds rather than the same cold copy.

The companion [runtime domain notes](../.agents/userspace-runtime.md)
record the limited future work on mount refresh and mirrors. Those notes retain
this accepted first-provisioning cost and the decision against asset caching.

## Lifecycle and cost model

* The rootfs is provisioned lazily on the first sandbox launch; it is not
  necessarily created when the workspace is created.
* A persisted rootfs survives runtime stops, so ordinary subsequent starts do
  not repeat the first-rootfs provision.
* System synchronization or migration can repeat after a system-sync version
  update, rootfs recreation/recovery, or missing/invalid sync markers. Rebuilding
  the container image alone does not necessarily resynchronize a persisted rootfs.
* The rootfs-generation marker used by
  [`_sync_system_dirs_for_chroot`](../runtime/worker/sandbox.py) avoids a full
  system hardlink-detachment walk on ordinary starts. Content reconciliation
  still does work and should not be described as free.

## Rationale and evidence limits

Independent copying, instead of prior hardlinking, is the dominant plausible
source of the added write amplification: independent destination inodes prevent
writes through a sandbox copy from modifying the runtime's original file via a
shared inode. A prior local,
five-sample report supported that mechanism, but did not phase-profile the
operation. Exact seconds and wall-time attribution are therefore unprofiled;
those local measurements are neither an SLA nor universal performance data.

Relevant implementation references:

* [`_sync_system_dirs_for_chroot`](../runtime/worker/sandbox.py) performs the
  no-mount chroot system synchronization and generation-marker check.
* [`_copy_system_file_detached`](../runtime/worker/sandbox.py) creates
  independent system-file inodes without writing through legacy hardlinks.
* [`_detach_legacy_system_hardlinks`](../runtime/worker/sandbox.py) performs
  the one-time migration scan when the marker is not valid.
* [`provision_rootfs`](../runtime/worker/sandbox.py) reconciles workspace
  content separately from the system-tree migration decision.

## Guardrails for future changes

Do not restore writable hardlinks, share root-owned writable inodes, weaken
capability or path checks, or move scans into warm requests merely to improve a
cold metric. Do not treat a copy-on-write redesign as required follow-up.

Revisit this decision only when priorities or workloads change, or when a small,
safe, measured fix demonstrates useful benefit. Any optional redesign must be
evidence-based and retain the isolation guarantees above.
