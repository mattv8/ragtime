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
- A SolidWorks tool can link one PDM tool (`connection_config.pdm_tool_id`).
  Its `pdm.search` and `pdm.lookup` read this index in Ragtime
  (`ragtime/solidworks/pdm_link.py`) and reach both Chat and MCP through the
  `solidworks_<name>` wrapper, unlike the standalone Chat-only `lookup_{tool}`.
  The SolidWorks tool sends the PDM tool's
  `database` to its host as the vault name; deleting or disabling the PDM tool
  makes the SolidWorks tool answer `pdm_not_linked`. See
  `.agents/solidworks.md`.
- The SQL login reads the vault database for indexing only. Vault reads and
  writes on a SolidWorks machine use that machine's PDM client login, never
  this one.
- The SolidWorks `studying-models` routine finds comparable models with
  `pdm.search`: semantic search by product family, description or part-number
  pattern, then the items' `revision`, `configurations` and `bom_count`. A
  filename match cannot do this. Any other source for this index must supply
  the same data: latest-revision data-card values, configurations, folders,
  BOM rows and the document IDs `pdm.lookup` accepts.
