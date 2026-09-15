# SolidWorks PDM indexing

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
- Incremental indexing does not remove documents deleted from the vault. Run a full re-index to remove them from the indexed snapshot.
- A full re-index clears and rebuilds embeddings from scratch.
