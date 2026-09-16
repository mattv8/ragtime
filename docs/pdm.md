# SolidWorks PDM indexing

## Automatic reindexing

PDM tools can be configured for automatic incremental reindexing on a schedule or via an external webhook. This keeps the indexed snapshot in sync with vault changes without manual intervention.

### How to set up an external webhook

Prerequisite: the PDM tool must be saved at least once (to have connection settings).

1. Open the PDM tool in the Tools panel and go to the **PDM webhook** section.
2. Click **Enable webhook**.
   - A webhook ID and secret are generated. The secret is shown only once.
   - Copy the secret immediately to your secure webhook caller (e.g. PDM change notifications system).
   - The webhook URL is shown for reference; the default is constructed from your configured Ragtime base URL.
3. Paste the URL and secret into your external PDM monitoring tool. See [Webhook request format](#webhook-request-format) below.
4. (Optional) Click **Rotate secret** to invalidate the old secret and generate a new one. Old credentials stop working immediately.
5. (Optional) Click **Pause webhook** to stop accepting events temporarily. A paused webhook is not deleted; click **Resume webhook** to reactivate it.
6. Click **Disable webhook** to disable and remove webhook credentials. The schedule (if configured) remains active.

The webhook accepts events at the published URL. Events arriving while another job is running are coalesced and wait for a follow-up dispatch.

### How to configure an automatic schedule

1. Open the PDM tool and configure **Automatic incremental indexing**.
2. Select a reindex interval from the dropdown. Select **Manual only** to disable the schedule.
3. Optionally set **Start Time** and its **Timezone** for anchored scheduling. The UI supplies the local timezone when you enable a schedule; the saved timezone may be omitted.
4. Click **Save**.

The schedule runs incremental reindexing. A schedule can run independently of the webhook. Both mechanisms share the same job admission queue, so only one reindex runs at a time per tool.

Eligibility is based on the latest terminal attempt (success or failure) and interval. Successful manual full-reindex runs also count toward schedule freshness. With no prior job, an anchored run waits for its configured slot; an unanchored schedule may run immediately after save.

### Webhook request format

**Endpoint:** `POST https://<ragtime_host>/webhooks/pdm/{webhook_id}`

The webhook uses bearer-token authentication only. No session cookies, query strings, or HMAC signatures.

**Authorization header:**
```
Authorization: Bearer <secret>
```

**Body:** JSON object (optional). Permitted fields:

| Field | Type | Max length | Purpose |
| --- | --- | --- | --- |
| `event_id` | string | 200 chars | Optional event identifier. Ragtime suppresses a notification whose ID matches the latest retained event ID. There is no delivery history or exactly-once guarantee. |
| `reason` | string | 500 chars | Optional human-readable reason (e.g. `"document-modified"`, `"weekly-refresh"`). It is accepted but not stored in job logs. |

Empty body or `{}` is valid.

**Example: Bash with environment variables**

Set `PDM_WEBHOOK_URL` and `PDM_WEBHOOK_SECRET` from the URL and one-time secret shown when you enable the webhook.

```bash
curl -X POST "$PDM_WEBHOOK_URL" \
  -H "Authorization: Bearer $PDM_WEBHOOK_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"reason":"document-modified"}'
```

**Example: Windows PowerShell**

Set `$env:PDM_WEBHOOK_URL` and `$env:PDM_WEBHOOK_SECRET` from the URL and one-time secret shown when you enable the webhook.

```powershell
$headers = @{
    "Authorization" = "Bearer $env:PDM_WEBHOOK_SECRET"
    "Content-Type" = "application/json"
}

$body = @{
    "reason" = "document-modified"
} | ConvertTo-Json

Invoke-WebRequest -Uri $env:PDM_WEBHOOK_URL -Method Post -Headers $headers -Body $body
```

**Responses:**

| Status | Body | Meaning |
| --- | --- | --- |
| **202 Accepted** | `{ "status": "accepted" }` | Event is accepted for coalescing and incremental-index dispatch. |
| **202 Accepted** | `{ "status": "ignored" }` | Webhook is paused or the tool is disabled. Request was authenticated but not queued. |
| **404 Not Found** | Error JSON | Webhook ID does not exist or has been deleted. |
| **401 Unauthorized** | Error JSON | Bearer token is missing or invalid for an existing webhook, including a paused webhook. |
| **413 Payload Too Large** | Error JSON | Request body exceeds 65536 bytes. |
| **400 Bad Request** | Error JSON | Request body is not valid JSON or does not parse as a JSON object. |
| **422 Unprocessable Entity** | Error JSON | Request fields are present but do not validate (e.g. `event_id` exceeds 200 chars). |
| **429 Too Many Requests** | Error JSON | Rate limit exceeded (60 requests per minute). Retry after a delay. May be disabled in DEBUG mode. |

### Webhook coalescing and queue semantics

The webhook uses a bounded debounce mechanism:

- **Quiet-period eligibility:** A pending reindex is eligible after 60 seconds without another event.
- **Maximum-delay eligibility:** A pending reindex is eligible 300 seconds (5 minutes) after the first event, even if events continue.
- **Dispatch:** Eligible work starts on a dispatcher pass and waits for any active job. The thresholds do not guarantee an immediate start or exact deadline.
- **Bounded duplicate suppression:** Ragtime retains only the latest event ID for comparison. Use a distinct ID for each new event and reuse that ID when retrying the same event; older IDs are not kept in a delivery history.

You own the responsibility of native PDM event integration. Ragtime does not poll the vault for changes, monitor PDM for modifications, or subscribe to PDM events on your behalf.

### Job visibility and status

Monitor webhook status in the **PDM webhook** settings in the Tools panel:

- **Last event:** Timestamp of the most recently received webhook event.
- **Queue:** `Running`, `Pending`, or `Idle`.
- **Last success:** Timestamp of the most recent successful index.
- **Last error:** A message from the most recent failed job.

The PDM tool card separately shows the document count, schedule summary, and last successful index. Although the webhook response includes `last_attempt_at`, the current component does not display it.

Failed automatic jobs are not automatically retried. A new event (webhook or schedule) can trigger the next job.

### Timezone validation

If supplied with an enabled schedule, the timezone is validated when the PDM tool is saved. An invalid IANA timezone (e.g. `"Unknown/Zone"`) is rejected; the field is nullable.

### Limitations

- Scheduled and webhook runs are incremental. They scan the configured vault metadata and skip unchanged hashes and embeddings.
- Incremental indexing does not remove documents deleted from the vault.
- To remove deleted documents, perform a full reindex via **Full Re-index** (manual action; not available via automation).
- Creating a new PDM tool starts its initial index. Editing an existing tool does not implicitly index it; wait for the next automatic run or use **Index** for an explicit incremental update.
- Configuration changes (e.g. adding variables, changing part number field) take effect on the next reindex, whether automatic or manual.
- A disabled tool suppresses automatic schedule and webhook dispatch. Existing running jobs complete normally.
- A deleted PDM tool resets its automation state, schedule, and webhook configuration. Re-creating a tool with the same name requires webhook re-setup and schedule reconfiguration.

## How to validate current vault data

PDM search and lookup operate on the indexed snapshot, so their results can lag the vault. To audit current vault data directly, create a generic MSSQL tool that connects to the vault database with read-only credentials. The MSSQL tool enforces SELECT-only queries and requires `TOP n` on SELECT statements.

Use a dedicated SQL login with the least privileges needed for the audit. Do not use the indexing snapshot as evidence of unsaved CAD changes or current data-card state.

## Reference

### Chat tools

Each configured SolidWorks PDM tool creates two Chat tools. The configured PDM tool name replaces `{tool}` below.

| Tool | Purpose | Arguments | Results |
| --- | --- | --- | --- |
| `search_{tool}` | Semantic search over the indexed snapshot of checked-in metadata. | `query`, `document_type` | Results are labeled with similarity. |
| `lookup_{tool}` | Deterministic exact lookup over the indexed snapshot. It does not require an embedding provider. | One or more of `document_id`, `filename`, or `part_number`; optional `configuration` | Returns per-configuration values and their origin revisions. |

`document_id` is an exact document lookup. `filename` and `part_number` use case-insensitive substring matching. A `configuration` limits returned values to the exact configuration name, case-insensitively. Configuration names are not globally unique, so matching configurations are returned with their IDs.

Part-number lookup also searches configuration-level part numbers. It searches the stored document state as well, so it can match other stored text in addition to a part number.

Both tools use the PDM tool's `max_results` setting, clamped to 1 through 50. The default is 10.

MCP currently exposes only `search_{tool}` for a PDM tool. `lookup_{tool}` is available in Chat only.

## Explanation

### Indexed checked-in metadata

PDM indexing reads checked-in metadata from the vault SQL Server and stores an indexed snapshot. For each document, a property value is resolved from the latest property version at or before the document's latest checked-in file version. PDM stores property rows sparsely across versions: an unchanged property may have been recorded only at an earlier version. The indexed value therefore records the source revision where it originated.

An explicit cleared value is preserved as empty. Indexing selects the latest eligible value before deciding whether it is blank, so clearing a property does not restore an older nonblank value.

Configuration membership comes from the vault configuration/version table at the document's target revision. This includes the `@` document-level scope, named configurations, and drawing sheets. The `@` scope remains distinct from named configurations; a missing named-configuration value does not fall back to `@`.

Variables are discovered from each vault. Ragtime does not rely on fixed variable IDs. It identifies the part number and description by default through variables named `Part Number` and `Description` (case-insensitive). The `variable_names` connection setting controls which discovered variables are extracted and indexed. Explicit role overrides (`part_number_variable` and `description_variable` in the connection config) take precedence over default names when present.

### Limits and unsupported semantics

- Indexed data represents checked-in state, not unsaved CAD changes or working data-card state.
- Version-free (free-update) variables are detected but do not have a special resolution policy.
- BOM child components are listed without quantities. Vault BOM quantity semantics are not verified.
- Incremental indexing (automatic schedule or webhook) does not remove documents deleted from the vault. Run a full reindex via **Full Re-index** to remove them from the indexed snapshot.
- A full reindex clears and rebuilds embeddings from scratch.
- Automatic indexing (schedule or webhook) never performs full reindex. Full rebuild remains a manual action.
- The latest event ID is retained for bounded duplicate suppression, but no delivery history is maintained.
- Vault change polling is not supported. You must configure an external script or native PDM event integration to call the webhook.
