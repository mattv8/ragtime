# External agent task domain notes

Scope: external build briefs (`/agent/w/{token}`), chat-task execution policy,
provider payment classification, and credit monitoring.

## Execution policy and honest outcomes

- Require-action semantics are gated structurally by the `ExternalBuildRequest`
  ledger row plus a persisted `ChatTask.executionPolicy`
  (`{version:1, task_type, require_action, source}`), never by prompt wording
  or code-path convention. A resumed task prefers its stored policy; a legacy
  null policy keeps old semantics, and replies must pass the stored policy
  through unchanged — defaulting a null-policy reply to `build` retroactively
  changes old conversations.
- `compute_brief_payload_hash` pops the default `task_type="build"` before
  hashing so legacy idempotency keys keep their original hashes; an explicit
  `general` is deliberately a distinct request.
- `activity_summary` counts only non-synthetic `tool_end` rows. The
  "Treat this as a failed tool result" sentinel and `synthetic`/`recovery`
  flags mark internal recovery, and output starting with `Error:` counts as a
  failed execution. Activity is execution evidence only — never present one
  tool call (or all-successful calls) as acceptance-criteria proof.
- `interrupted` is a terminal outcome for `no_actions`/`all_tools_failed`/
  `max_iterations` and sets `completedAt`. Every terminal branch must finalize
  the usage attempt and link assistant snapshots; the interrupted branch
  originally skipped both, leaking open usage attempts on the most common
  external-build ending.
- Unclassified run failures keep `termination_reason` null. Stamping a generic
  `provider_error` blames the provider for platform bugs and misleads
  operators.

## Provider payment flow (cross-boundary)

- `components.py` converts provider failures into **yielded text**, so
  exceptions never reach `background_tasks`. Payment stops survive only as the
  structured terminal event `{type:"error", code:"payment_required", content}`;
  every swallowed-exception path (including the early context-build and
  context-window-fit handlers) must emit it via `_provider_error_stream_event`
  or the task completes as a text-only "success".
- The background loop must consume that event **before** the synthetic
  response / no-text-failure / completion branches. Unknown `error` codes are
  advisory: keep their safe content in `full_response`/events instead of
  dropping the frame.
- `classify_provider_error` has a deliberately narrow stable contract:
  `payment_required` (HTTP 402 or explicit structured
  `error_type`/`type`/`code`) or `None`. No substring matching — generic
  quota/rate-limit wording must not classify as payment. Payment is never
  transient-retryable, and tool-free synthesis must not run after it.

## Credits

- OpenRouter `/key` reports the **per-key spend cap**; `limit_remaining: null`
  means "no cap", not infinite wallet funds. `/credits` needs a management key
  that may belong to a different account — never infer one balance from the
  other.
- `note_openrouter_payment_required()` persists until a **positive wallet
  observation**; a key-cap read alone cannot clear it.
  `get_openrouter_credit_warning()` is the only surface safe for non-admin
  task output (coarse, credential-free, no balances).

## Agent token surface

- The manifest (`GET /agent/w/{token}`) is the machine-read contract external
  agents actually follow — response-shape changes are invisible until it is
  updated. Capability sections must live entirely inside their conditional
  template variable: static template text after the placeholder leaked the
  restart/operations routes to non-restart tokens, and the disabled-token test
  must assert absence of `/runtime/operations`, not just `/runtime/restart`.
- Tokens act as their creator with a per-request re-check of the creator's
  current workspace role; `allow_runtime_restart` is a separate opt-in from
  `allow_task_submission` and the UI re-enable path resets it to false.
  Unknown and disabled tokens are indistinguishable 404s with `no-store` on
  every response, including error paths.
- Builder model routing: `resolve_new_conversation_model` infers `build` for
  any new workspace conversation without an explicit `task_type`, and a
  configured `userspace_build_model` deliberately outranks workspace/user
  personal defaults. Build submission passes the live catalog availability
  snapshot so a configured-but-stale model fails at submit time instead of
  silently downgrading at inference time.
