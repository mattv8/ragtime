# User Space object storage

Ragtime starts an internal `object-storage` service by default. It persists local bytes and gateway metadata below `/data/_userspace/_object_storage` and is reachable only on the Compose network at `http://object-storage:9000`. The control plane listens internally on port 9001. Neither port is published to the host.

Workspace applications receive `RAGTIME_OBJECT_STORAGE_*` credentials only in their backend runtime environment. Use standard SigV4 clients with region `us-east-1` and path-style addressing. For Node AWS SDK v3, use `S3Client({ endpoint: process.env.RAGTIME_OBJECT_STORAGE_ENDPOINT, region: "us-east-1", forcePathStyle: true, credentials: { accessKeyId: process.env.RAGTIME_OBJECT_STORAGE_ACCESS_KEY_ID, secretAccessKey: process.env.RAGTIME_OBJECT_STORAGE_SECRET_ACCESS_KEY } })`.

Python backend example:

```python
import os
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["RAGTIME_OBJECT_STORAGE_ENDPOINT"],
    region_name=os.environ.get("RAGTIME_OBJECT_STORAGE_REGION", "us-east-1"),
    aws_access_key_id=os.environ["RAGTIME_OBJECT_STORAGE_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["RAGTIME_OBJECT_STORAGE_SECRET_ACCESS_KEY"],
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)
```

Do not expose workspace or provider credentials to browsers. Browser code uses the app's authenticated object routes. The internal Compose endpoint is not a public download URL. Logical bucket names are workspace metadata; object bytes are accessed through S3. Public/private prefixes are organizational and do not grant anonymous access.

The gateway supports ListBuckets, HeadBucket, ListObjects/V2, Head/Get/Put/DeleteObject, DeleteObjects, CopyObject, and multipart upload/list/complete/abort operations. Bucket creation and deletion remain Ragtime owner/admin API operations. Unsupported bucket policies, lifecycle, CORS, tagging, and similar S3 subresources return `NotImplemented`. During migration, writes are fenced and source bytes are retained after a verified cutover.

## Configure external backing

In Settings → User Space → Object Storage, select external storage and enter the S3 endpoint, region, credentials, and backing bucket. Select automatic bucket creation if the credential can create buckets, or use an existing private bucket. Ragtime creates each workspace's logical buckets and isolated prefixes automatically; administrators do not create a provider bucket per workspace.

Saving a provider changes the default for new logical buckets. Existing buckets retain their current backend. Select **Migrate existing workspaces** to copy and verify existing objects before switching their bindings. Failed jobs retain the source data and can be retried. Conflicting legacy plain-file/S3rver objects require resolving the conflict before the import can complete; Ragtime does not choose a copy silently.

## Operational boundaries

The local gateway is a single service with SQLite metadata and an indexed filesystem object store. It is not a replicated storage cluster. External uploads may use temporary disk-backed staging to satisfy the provider SDK's repeatable-stream requirements without buffering entire bodies in memory.

Workspace deletion revokes access and retains a tombstone; retained namespaces and migration source copies are not automatically purged. Account for retained bytes in capacity planning. Hard per-workspace storage quotas are not implemented; HTTP upload limits do not limit all direct SDK writes. External provider version retention remains provider-managed.

## Backup and restore

Local full/files backups ask the gateway to quiesce and checkpoint before copying its metadata and bytes. If initialized storage cannot provide that consistency lease, the backup fails rather than producing a backup advertised as complete. External backing backups contain Ragtime's configuration, bindings, and manifests; remote provider bytes are not copied. Workspace source snapshots remain code-only.

The Compose gateway receives three mounts: read-write `/data/_userspace/_object_storage`, read-only `/data/_userspace/workspaces` for legacy S3 bucket imports, and the read-only `object-storage-key` volume. Ragtime publishes its persisted managed key into that volume at startup. The key volume is a disposable projection: it is not authoritative and is not a backup source. Encrypted backups made with `--include-secret` retain the existing authoritative managed-key behavior.

The gateway reads its key once at startup. For a full/files restore, stop the gateway before restoring when the archive contains object storage or the destination object-storage directory already exists. A database-only restore that includes a managed key also requires the gateway to be stopped when that destination exists. Set `OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED=true` only after the gateway is offline.

After either restore, restart Ragtime so it republishes the restored authoritative key, wait for Ragtime health, then recreate the gateway with `docker compose up -d --force-recreate object-storage`. Do not use `docker compose start` or `docker compose restart` for this step: recreation renews the gateway's narrowed storage bind after a subtree replacement. A backup of initialized local storage still requires the running gateway for its consistency lease; stopping it is only part of restore ordering.

## Disposable SDK conformance harness

`tests/test_object_storage_s3_contract.py` is disabled unless `OBJECT_STORAGE_S3_CONTRACT_RUN=1` and explicit disposable endpoint plus two tenant credential pairs are provided. It does not target the default service. It covers signed/unsigned/wrong/cross-tenant access, metadata, copy, pagination, multipart, and an opt-in 10,000-key bounded-page check. Set `OBJECT_STORAGE_S3_CONTRACT_NODE_COMMAND` to a disposable Node AWS SDK v3 fixture command to include Node compatibility; no frontend dependency is installed for this check.
