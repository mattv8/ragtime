# SolidWorks PDM domain notes

- PDM indexing is a pgvector snapshot of checked-in SQL Server vault metadata;
  it is not evidence of unsaved CAD/data-card state. `search_{tool}` is exposed
  to Chat and MCP; exact `lookup_{tool}` is Chat-only.
- Incremental runs (manual, schedule, or webhook) retain deleted documents.
  Only a manual full re-index removes deleted vault documents and rebuilds
  embeddings.
- Schedule and PDM webhook admissions share one per-tool queue. Webhook events
  debounce for 60 seconds, with a five-minute maximum delay; only the latest
  event ID is retained for duplicate suppression. Failed automatic runs need a
  later event or schedule run; they are not automatically retried.
- The webhook is bearer-authenticated at `/webhooks/pdm/{webhook_id}`. Its
  secret is shown only at creation/rotation; do not place it in repository files
  or logs. Route, repository, and service implementations live in
  `ragtime/pdm_automation/`; index behavior is in `ragtime/indexer/pdm_*`.
