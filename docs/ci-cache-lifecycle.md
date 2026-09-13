# CI Docker cache lifecycle

## Purpose and scope

Self-hosted CI uses a disposable named BuildKit builder for each job. Persistent reusable layers stay in Harbor through the existing repository cache tags. This local lifecycle only manages resources that carry the managed CI name and labels; it does not alter Harbor retention, application containers, application volumes, or legacy anonymous builders. It can remove an orphaned managed BuildKit state volume after its ownership and attachment checks pass.

## Local cache limits

`docker/buildkitd.ci.toml` configures each managed builder with these BuildKit OCI-worker settings:

| Setting | Value |
| --- | --- |
| Reserved cache space | 2 GB |
| Maximum unused cache space | 8 GB |
| Minimum free space | 15 GB |

BuildKit protects active snapshots. The unused-cache budget is per builder, not a host-wide hard cap, and concurrent builders multiply the possible cache use. It also does not bound active image layers or unrelated Docker data.

Before a managed builder is created, the collector checks disk space on Docker's data filesystem. CI requires 15 GiB free after safe managed cleanup. Heavy builds should retain 25 GiB of free space. The current runner has 94 GiB, so capacity planning must include concurrent builders, active images, and non-CI Docker users rather than treating the BuildKit limit as a full-disk guarantee.

## Normal and orphan cleanup

`docker/setup-buildx-action` removes the current builder when the job ends. The managed collector recovers only an orphan from an interrupted job when all ownership checks succeed:

- the builder container and state volume have the exact managed name for this repository, run, attempt, and scope;
- the container is a BuildKit daemon with the expected state-volume mount;
- GitHub reports the exact run attempt as completed and at least one hour old; and
- no Docker container still uses a candidate local image.

Missing, malformed, fresh, queued, in-progress, unauthorized, or network-unavailable run status is a keep decision. The collector does not force-remove images or volumes. A failed local image cleanup logs a warning and leaves recovery to a later managed collection.

The collector does not delete unlabeled historical builders. An operator must handle any measured legacy backlog as a one-time, reviewed operation; do not add global prune automation.

## Operator use

The workflow runs `prepare` before each managed builder. It performs the trusted-event collection and emits the builder and local-image identifiers for that job.

The emitted builder name is `ragtime-ci-<repository-hash8>-<run-id>-<attempt>-<scope-hash8>`. Its local image tag is `ragtime-ci-<repository-hash8>:<run-id>-<attempt>-<scope-hash8>`. These identifiers make builder and loaded-image cleanup repository- and attempt-specific; they are not a general Docker cleanup namespace.

Run collector commands on the Linux Docker host. The collector reads Docker's data filesystem for the headroom check; a macOS Docker client can report a `DockerRootDir` that is not readable from the client filesystem.

For an inspection-only dry run from a checked-out repository, provide the required GitHub context without printing the token:

```bash
GITHUB_REPOSITORY=mattv8/ragtime \
GITHUB_RUN_ID=0 \
GITHUB_RUN_ATTEMPT=0 \
GH_TOKEN="$(gh auth token)" \
python3 docker/scripts/ci_docker_gc.py collect
```

`collect` and `report` require this repository, run, and attempt context even when inspecting. `collect` is dry-run by default. Without `GITHUB_ACTIONS=true`, the command makes no changes; `collect --apply` outside a trusted GitHub Actions event is refused as a no-op. Do not set a fake `GITHUB_ACTIONS` value to bypass that protection. The collector cannot reclaim remote registry bytes, nor can it create disk headroom by deleting resources outside its ownership contract.

## Harbor cache lifecycle

Harbor retains dependency and build cache artifacts independently of runner-local cache. The existing `library` project policy was verified on 2026-09-13:

- Retain the latest 30 pushed artifacts per repository, including untagged artifacts in the selector, with a daily `0 0 3 * * *` retention schedule.
- Run registry garbage collection daily with schedule `0 0 0 * * *`.

These are Harbor's scheduler expressions, not runner-local timezone settings. Retention removes eligible artifacts; registry garbage collection subsequently reclaims unreferenced blobs. The policy also covers `ragtime-base`, so obsolete dependency-image versions are not retained indefinitely. If a needed base tag has expired, CI builds it again from its dependency inputs.

This CI change does not modify the shared Harbor policy. Recheck it when changing registry capacity or retention requirements; runner-local cleanup cannot reclaim remote registry bytes.
