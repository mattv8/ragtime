# SolidWorks PDM domain notes

- PDM indexing is a pgvector snapshot of checked-in SQL Server vault metadata;
  it is not evidence of unsaved CAD/data-card state. `search_{tool}` is exposed
  to Chat and MCP; exact `lookup_{tool}` is Chat-only. A SolidWorks tool that
  links this PDM tool hides both from callers who can use it, and its
  `pdm.search` and `pdm.lookup` serve the same index (`.agents/solidworks.md`,
  "Linked pair").
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
- `pdm.lookup` (`execute_index_operation` in `ragtime/solidworks/pdm_link.py`;
  the page query is `lookup_pdm_document_page` in `ragtime/indexer/pdm_service.py`)
  needs one of `document_id`, `filename` or `part_number`. Optional arguments:
  `configuration`, `document_type` (`SLDPRT`, `SLDASM` or `SLDDRW`, case-insensitive,
  matched on the file extension), `limit` (1-25, default 10), `offset` (0-1,000)
  and `detail` (`summary`, the default, or `full`). A summary item has the
  identity, `vault_paths`, the first 50 configuration names, the configuration
  and BOM counts. A full item adds the data-card values, configurations and BOM
  rows. `configuration` narrows the configurations an item reports.
- Order: an exact file name (with or without the extension, case-insensitive),
  then names that start with the term, then other CAD files (`.sldasm`, `.sldprt`,
  `.slddrw`), then the rest. CAD files come first inside the first two groups;
  ties go by file name, then `document_id`. Paging is a SQL `OFFSET`; the query
  reads `limit + 1` rows to learn whether a next page exists.
- `next_offset` is null on the last page. Otherwise it is the offset of the first
  row the 32 KB trim dropped, so trimmed rows come back on the next page. Rows
  whose stored state does not parse are skipped and count as read.
- `matched_on` on each item, first match wins: `filename` when the `filename`
  filter sits inside the file name (case-insensitive); `part_number` when the
  `part_number` filter sits inside the part number; `properties` when the term
  matches only the stored state (which can include folder paths); `document_id`
  for a `document_id`-only lookup.
- `pdm.search` is not paged. When the query is one token with a digit, such as a
  part number, up to 5 (`IDENTIFIER_HIT_LIMIT`) filename-prefix hits come first,
  each with `match: "filename"` and honouring `document_type`. Semantic results
  follow, deduplicated by `document_id`, each with `match: "semantic"`. Both
  share `max_results`, boost first.
- The SQL login reads the vault database for indexing only. Vault reads and
  writes on a SolidWorks machine use that machine's PDM client login, never
  this one.
- The SolidWorks `studying-models` routine finds comparable models with
  `pdm.search`: semantic search by product family, description or part-number
  pattern, then the items' `revision`, `configurations` and `bom_count`. A
  filename match cannot do this. Any other source for this index must supply
  the same data: latest-revision data-card values, configurations, folders,
  BOM rows and the document IDs `pdm.lookup` accepts.
