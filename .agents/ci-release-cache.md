# CI, release, and cache domain notes

- `/docs/` is local, ignored reference material. Keep verified codebase
  maintenance guidance for agents in `.agents/`; do not add tracked docs, force
  add ignored docs, or link to local `docs/` files. Route implementations remain
  the authoritative endpoint signatures, rather than duplicated static route dumps.
- `CI Gate` validates pull requests to `beta` and `main`; it requires successful
  dependency-image resolution plus backend, frontend, and storage quality jobs.
  The local equivalent is `bash tests/run_all_tests.sh`, which checks README
  sync and runs Docker targets for Python, frontend format/lint/test/build, and
  storage tests.
- `Build and Push Container` runs on trusted pushes to `beta` and `main`. It
  builds immutable candidate digests, promotes moving branch tags only after
  candidates succeed, and signs promoted digests. Do not use a tag as proof of
  the bytes without its digest.
- Promotion into `main` is repository-`beta` only and requires current `main`
  ancestry (`Main Promotion Guard`). Reconcile direct `main` fixes back to
  `beta`; do not bypass the guard in ordinary work.
- Registry caches are branch-scoped. PRs consume base-branch and `main`
  fallback caches but do not publish registry cache. Managed Buildx cleanup is
  ownership- and completed-run-attempt-scoped; do not add global Docker prune
  automation for CI recovery.
- The merge serializer is metadata-only, never executes PR code, and rebases at
  most one eligible, same-repository, auto-merge-enabled `beta` PR per run. It
  needs `MERGE_SERIALIZER_TOKEN`; fork, conflicted, and `main` reconciliation
  PRs require manual handling.
- `docker/scripts/install_restic.py` is the pinned version, checksum, and download
  timeout source for both runtime and test packaging. Keep its use aligned in
  `docker/Dockerfile` and `docker/Dockerfile.runtime`; do not duplicate an
  unpinned installer. Update `docker/Dockerfile.runtime`'s literal Restic version
  assertion when changing the pinned version.
- `runtime-history-test` is a manual additional target, not invoked by CI Gate,
  `bash tests/run_all_tests.sh`, or the final runtime build. Its `COPY tests/...`
  input list and `pytest` list in `docker/Dockerfile.runtime` must remain
  explicit and synchronized when history coverage changes. Retain real Linux
  Restic, Landlock, and minimal-import coverage rather than replacing it with
  host-only simulations.
  Run it with `docker build --target runtime-history-test -f docker/Dockerfile.runtime .`.
- The test/runtime-manager setting is literally `RUNTIME_MANAGER_URL` (not
  `USERSPACE_RUNTIME_MANAGER_URL`): `userspace_runtime_manager_url` aliases it
  and defaults to `http://runtime:8090`. Tests may patch the settings attribute or
  set the environment empty for an intentionally absent manager. After the
  controller activation marker, a disabled manager returns 503 rather than
  falling back. Keep CI references portable: no personal absolute Docker paths or
  local image tags.
