# CI, release, and cache domain notes

- `/docs/` is local, ignored reference material. Keep verified codebase
  maintenance guidance for agents in `.agents/`; do not add tracked docs or
  links that require local `docs/` files. Route implementations remain the
  authoritative endpoint signatures, rather than duplicated static route dumps.
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
