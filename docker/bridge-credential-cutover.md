# Bridge credential and keystore cutover

This release changes two independent deployment contracts:

- Workspace bridge credentials are file-only. Runtime applications receive
  `RAGTIME_BRIDGE_URL` and `RAGTIME_BRIDGE_TOKEN_FILE`; they must read the
  token file for each bridge request. `RUNTIME_AUTH_TOKEN` remains the separate
  Ragtime-to-runtime service authentication secret.
- The instance encryption-key projection is mounted as the Compose volume
  `keystore` at `/run/ragtime-keystore`. Ragtime writes its projection at
  `/run/ragtime-keystore/.encryption_key`; `runtime-s3` mounts it read-only.
  The authoritative managed key remains `/data/.encryption_key`. Bridge tokens
  stay in each workspace rootfs and are never stored in the shared keystore.

Apps that already followed the prior file-read guidance may need no code
change. Check each app before upgrade: environment-only applications are not
compatible with this release.

## Prerequisites

Use a coordinated, tested set of Ragtime, storage, and runtime images built
for this cutover. The reduced Compose files rely on image defaults for the key
export and storage settings. `docker compose up` does not automatically pull a
mutable `:main` tag, and an arbitrary pre-cutover `:main` image is not
sufficient. Record the exact image digests before changing containers. The
runtime image must be recreated with the application and storage images for the
file-only bridge contract. Schema changes must be applied with their matching
application version; bridge rollback also requires a compatible runtime image
and the pre-drop schema.

Do not run this procedure during an active backup/restore or while another
operator is changing the stack. It never deletes a volume; do not use
`docker compose down -v`.

## Preserve and seed the keystore

1. Render the actual project configuration and identify the exact mounted
   volumes from the existing containers. Do not infer a volume name from a
   suffix or assume an unprefixed name. For example, inspect the current
   Ragtime and storage containers and identify mounts by their destinations:

   ```sh
   docker inspect ragtime runtime-s3 --format '{{.Name}}{{range .Mounts}}{{println}}{{.Name}} {{.Source}} -> {{.Destination}} rw={{.RW}}{{end}}'
   docker compose config --volumes
   ```

   Confirm the old projection volume mounted at
   `/run/ragtime-storage-key`, the authoritative `/data` mount, and the new
   project-qualified `keystore` identity before proceeding. Stop if ownership
   or any identity is uncertain.

2. Privately compare only file state and key equality; never print key contents
   or put them in shell history/logs. If both old and new projection files are
   nonempty but differ, stop: do not overwrite either key. If the new keystore
   is populated, do not seed it blindly.

3. If the new keystore is empty and the verified old projection exists, seed
   the new volume with that projection using a controlled, private operation
   that preserves `0600` file permissions and appropriate container ownership.
   Retain the old volume unchanged. If `/data/.encryption_key` is missing while
   the old projection exists, copy the verified projection to the new keystore
   only so Ragtime's existing anti-regeneration guard sees it and fails closed.
   Restore the authoritative `/data/.encryption_key` through the documented
   explicit recovery process; never generate a replacement key.

   The following template is deliberately fail-closed. Substitute only the
   full, exact volume names obtained above; it does not accept guessed project
   prefixes. Run the comparison against the still-running old Ragtime container
   first when its authoritative key exists. None of these commands prints key
   material.

   ```sh
   # Set these from docker inspect output, never from a suffix guess.
   OLD_PROJECTION_VOLUME='exact-old-projection-volume'
   NEW_KEYSTORE_VOLUME='exact-new-keystore-volume'
   test -n "$OLD_PROJECTION_VOLUME" && test -n "$NEW_KEYSTORE_VOLUME"
   test "$OLD_PROJECTION_VOLUME" != "$NEW_KEYSTORE_VOLUME"

   # If the authoritative key is present, it must exactly match the old projection.
   docker exec ragtime sh -ceu '
     test -s /data/.encryption_key
     test -s /run/ragtime-storage-key/.encryption_key
     cmp -s /data/.encryption_key /run/ragtime-storage-key/.encryption_key
   '

   # Refuse a populated target. Copy once without displaying either key.
   docker run --rm \
     --mount "source=$OLD_PROJECTION_VOLUME,target=/old,readonly" \
     --mount "source=$NEW_KEYSTORE_VOLUME,target=/new" \
     alpine:3.20 sh -ceu '
       test -s /old/.encryption_key
       test ! -e /new/.encryption_key
       umask 077
       cp /old/.encryption_key /new/.encryption_key
       chmod 600 /new/.encryption_key
     '

   # Use the selected cutover image's actual ragtime UID:GID, never a guess.
   RAGTIME_IMAGE='registry.example/ragtime@sha256:recorded-cutover-digest'
   RAGTIME_UID_GID="$(docker run --rm --entrypoint sh "$RAGTIME_IMAGE" -ceu 'id -u ragtime; printf :; id -g ragtime')"
   case "$RAGTIME_UID_GID" in [0-9]*:[0-9]*) ;; *) exit 1 ;; esac
   docker run --rm --mount "source=$NEW_KEYSTORE_VOLUME,target=/new" \
     alpine:3.20 sh -ceu "chown '$RAGTIME_UID_GID' /new/.encryption_key"
   ```

   If the authoritative key was absent, skip the first comparison, perform only
   the guarded projection seed, and complete explicit authoritative-key recovery
   before normal startup.

## Upgrade and verify

1. Save the prior image digests and current Compose configuration. Pull or
   build the matched cutover image set explicitly.
2. Apply the matching schema migration, then recreate Ragtime, `runtime-s3`,
   and runtime together so mounts and file-only bridge metadata cannot leave an
   old runtime session falsely healthy. Do not merely restart containers whose
   mount definitions changed.
3. Check application and storage readiness, access to known encrypted data, and
   a workspace bridge call. Confirm the gateway sees the projection through its
   read-only `/run/ragtime-keystore` mount. The runtime worker must not receive
   the shared keystore mount.
4. Keep the old projection volume and the recorded digests through a verified
   rollback window. Roll back only to a mutually compatible application,
   storage, runtime, Compose, and schema set; restoring an old Compose file
   alone is not a safe bridge rollback.

Older instruction prose that describes dual-read bridge credentials is
superseded by this release's file-only contract. Those protected instruction
files are intentionally not edited as part of this deployment change.
