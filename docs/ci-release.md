# Release and Promotion Workflow

## Branch and Channel Policy

Ragtime uses two release channels, each on its own branch:

- **`beta` branch**: Pre-release channel. New features and fixes are validated and merged here
  first. Images are tagged `beta`, `latest-beta`, and `beta-<shortsha>`.
- **`main` branch**: Release channel. Promotion from beta creates stable releases. Images are
  tagged `main`, `latest`, and `main-<shortsha>`.

Temporary divergence between branches is acceptable. The channel brand (beta vs. main) is baked
at build time via the `VITE_RAGTIME_ENVIRONMENT` variable and cannot be changed after an image
ships; image SHAs are authoritative for content identification.

## Promotion: Beta to Main

Promotion from `beta` to `main` requires two conditions:

1. **Ancestry check**: The current `origin/main` must be an ancestor of (or equal to) the promoted
   commit to maintain a linear history.
2. **CI evidence**: A successful `Build and Push Container` workflow run must exist for the
   promoted commit on the beta branch (proof that the exact bytes were packaged and the app
   built successfully).

Promotion can happen via pull request or via CLI tool.

### Pull Request Route

Open a PR from `beta` to `main`. The required `Main Promotion Guard` check verifies:

- The PR head is on the `beta` branch
- The current `main` is an ancestor of the PR head
- Required checks are passing

The PR must remain up to date with `main` (strict checks enforced by GitHub branch protection)
before merge.

### CLI Route: `promote_beta.py`

For scripted or direct promotions, use the promotion helper:

```bash
python docker/scripts/promote_beta.py [options]
```

#### Options

- `--sha <sha>`: Promote a specific commit (default: `origin/beta` head). The SHA must be an
  ancestor of `origin/beta`.
- `--dry-run`: Plan the promotion without pushing.
- `--skip-evidence`: Force promotion even if the build evidence check fails (emergency use
  only; skipped evidence is logged as a loud warning). In production, this should be used
  rarely and only by designated maintainers.

#### Behavior

The tool:

1. Fetches `origin/beta` and `origin/main` (auto-detects repo root; `--repo-root` override
   for tests).
2. Verifies `origin/main` is an ancestor of the candidate SHA (recursively checks full
   history; auto-unshallows the repo if needed).
3. Queries GitHub API for a successful `Build and Push Container` run on the beta branch for
   that commit (unless `--skip-evidence` is set).
4. Pushes the commit to `main` without force (rejects non-fast-forward, surfacing any race
   between concurrent promotions).

Usage example:

```bash
# Promote current beta head
python docker/scripts/promote_beta.py

# Promote a specific beta commit
python docker/scripts/promote_beta.py --sha abc1234

# Dry-run to check prerequisites
python docker/scripts/promote_beta.py --dry-run
```

## Reconciliation: Main into Beta

When `main` receives a direct maintainer push or emergency fix, reconcile it back into `beta`
to keep the branches aligned:

### Pull Request Route

Open a PR from `main` to `beta` with the fix and merge it normally.

### CLI Route

```bash
git fetch origin
git checkout beta
git merge origin/main  # or git merge --ff-only origin/main for strict fast-forward
git push origin beta
```

## Emergency Bypass: Force-Push

In rare emergencies, a maintainer may force-push to `main` or `beta` to bypass the normal
promotion flow. This skips CI evidence checks and automated validation. Force-push should be
used only when:

- An urgent security fix must ship immediately.
- A release is blocking and normal CI infrastructure is unavailable.

**Important**: Direct force-pushes to `main` trigger the normal `Build and Push Container`
workflow and generate build evidence. Direct force-pushes to `beta` do not trigger the workflow
(beta force-push is blocked by branch rules). To ship images after a force-push to `main`:

1. The workflow runs automatically on push and signs artifacts normally.
2. Document the reason and timing clearly in release notes or commit messages for audit.

For `main` only, branch protection allows maintainers to force-push when needed (e.g., to revert
an unsafe commit); this is your circuit breaker for true emergencies. Beta remains protected
against force-push.

## GitHub Configuration via Rulesets

Branch protection is configured via native GitHub rulesets (Settings → Rules → Rulesets) to
enable layered governance without hardcoded actor lists. The orchestrator applies these via `gh`
during bootstrap; they are not tracked in the repository.

### Default Branch

Set the default branch to `beta` so new clones and PR templates target the correct base.

### Layered Ruleset Architecture

Three rulesets are applied:

1. **Beta PR + Strict CI Gate** (includes PR requirement, up-to-date branch requirement, required
   checks: `CI Gate`)
   - Administrator role bypass: **enabled** (for CLI reconciliation after main pushes)
   - Block non-fast-forward: **not set** (allow merges)
   - Block deletions: **not set**

2. **Main Restricted Updates** (includes PR requirement, up-to-date branch requirement, required
   checks: `CI Gate`, `Main Promotion Guard`, and non-fast-forward protection)
   - Administrator role bypass: **enabled** (for emergency maintainer force-push to main only)
   - Block deletions: **not set**

3. **Deletion + Non-Fast-Forward Protections** (applies to both branches)
   - Block force-push: both `main` and `beta`, **no bypass** (deletion protection is absolute)
   - Block deletions: both `main` and `beta`, **no bypass** (deletion protection is absolute)

This layering allows:

- Administrators to force-push `main` in emergencies (via bypass in ruleset 2).
- Administrators to merge beta directly (via bypass in ruleset 1) for reconciliation.
- Beta force-push is always blocked (no bypass in ruleset 3).
- Both branches protect against deletion (no bypass).

**Note**: On personal or user-owned repositories, rulesets are the native GitHub way to
enforce branch protection without role-based "Restrict who can push" features (which are
org-only). Rulesets use authenticated user identity and configured role bypasses in their
definitions.

## Runner Isolation Follow-Up

Currently, all CI jobs (including PR validation) run on self-hosted runners that also hold
release credentials. This is a known risk: untrusted PR code executes on hosts with access to
signed images and registry secrets.

**Recommended future step**: Move the `CI Gate` PR jobs to GitHub-hosted ephemeral runners
or dedicated isolated self-hosted runners with no registry credentials. Only the
`Build and Push Container` workflow (which publishes images) needs access to secrets, and it
only runs on trusted `main`/`beta` branch pushes, not PRs.

This is a manual operational change (updating runner labels in workflows and provisioning new
infrastructure) and is documented here for future maintainers.

## Test Scope: What CI Gate Covers

The `CI Gate` check runs the full validation suite but **excludes** certain integration and
privileged tests that require external infrastructure or elevated permissions:

**Excluded tests** (gated by environment variables and not run in CI Gate):

- `RAGTIME_BRANCH_INTEGRATION=1`: Opt-in API/Postgres regression checks using disposable
  local conversations (tests/test_conversation_branch_integration.py).
- `LDAP_MIGRATION_TEST_DATABASE_URL`: Disposable-Postgres coverage for LDAP identity
  reconciliation migration (tests/test_ldap_identity_migration.py).
- `OBJECT_STORAGE_*`: Object storage integration and benchmarks (S3-compatible, MinIO).
- MCP client-credentials smoke test: tests/test_mcp_client_credentials.py (requires live
  Ragtime dev stack).
- Root/`CAP_SYS_ADMIN` sandbox integration tests: workspace process sandboxing and privilege
  isolation.
- Linux-only rsync/Landlock mount-sync tests: host filesystem binding and mount isolation
  (macOS/Darwin skipped).

**To run excluded tests locally**:

```bash
# Conversation branch integration (API/Postgres regression)
export RAGTIME_BRANCH_INTEGRATION=1
docker exec ragtime-dev python -m pytest tests/test_conversation_branch_integration.py -xvs

# LDAP identity migration (disposable Postgres)
export LDAP_MIGRATION_TEST_DATABASE_URL=postgres://user:password@host/testdb
python -m pytest tests/test_ldap_identity_migration.py -xvs

# MCP client-credentials smoke (requires running dev stack)
docker exec ragtime-dev python -m pytest tests/test_mcp_client_credentials.py -xvs

# Object storage integration (requires S3-compatible service)
export OBJECT_STORAGE_ACCESS_KEY_ID=... OBJECT_STORAGE_SECRET_ACCESS_KEY=...
python -m pytest tests/test_object_storage_s3_contract.py -xvs
```

Local runs of excluded tests may require:

- External service access (Postgres, S3-compatible storage, running Ragtime dev stack).
- Special privileges (CAP_SYS_ADMIN for sandboxing).
- Linux kernel features (Landlock for mount restrictions).
- Git branches with live-API fixtures configured.

Consult the test file docstrings and environment variable guards for details.
