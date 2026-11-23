# SolidWorks tool: overview

This tool reads structured data from SolidWorks desktop on a Windows machine: feature tree,
sketch geometry and constraints, dimensions, mates, components, custom properties and mass
properties. Results are JSON, with no screenshots. Read operations never write SolidWorks documents;
`script.run` can build or edit models when the tool allows scripts (see `scripts`).

## Call shape

Every call goes to the same tool with an `operation` and optional `arguments`:

```json
{"operation": "features.list", "arguments": {"document": "doc-1", "limit": 50}, "timeout": 60}
```

`timeout` (seconds) is optional. Omit `arguments` for operations that take none.

## First steps

1. `guide`: read this file (or another topic, see the index below).
2. `describe`: the exact operation list with argument schemas, from the installed host.
3. `status`: is the host running, which SolidWorks versions are installed, which instances exist.
4. `documents.list` to see open documents, or `document.open` with a local path.
5. Then use the topic-specific operations. Read `model.summary` before listing large models.

## Response envelope

```json
{"ok": true, "operation": "documents.list", "result": {...}, "host": {"solidworks_version": 2024, ...}}
```

- `ok` true: the data is in `result`.
- `ok` false: `error.code` names the problem, `error.message` says it in one sentence,
  `error.details` may add facts. `host` shows the SolidWorks instance involved, if any.
- Errors are returned, not raised. Read `error.code` and act on the table below.

## Units

All lengths are metres, masses kilograms, angles radians, times seconds. Convert for the user
before you answer: millimetres = metres x 1000, inches = metres / 0.0254, degrees = radians x
180 / pi, grams = kilograms x 1000. Quote the unit you used. Results with quantities include
`"units"`.

## Behaviour to keep in mind

- Read operations are read-only: documents are opened read-only and silently, and nothing is saved,
  rebuilt or modified. `script.run` can change and save models when `describe` reports
  `scripts_enabled` true; see `scripts`. Vault changes through a linked PDM vault are separate; see `pdm`.
- A background instance holds a floating SolidWorks licence until it is idle-shutdown. Close
  documents you opened (`document.close`) when you are done, and do not open files speculatively.
- The tool works only while the Windows account has a desktop session. A disconnected remote
  desktop session counts; a logged-out machine does not.
- Only absolute local drive paths open, such as `C:\Parts\bracket.sldprt`, ending in `.sldprt`,
  `.sldasm` or `.slddrw`. UNC and network paths (`\\server\share\...`) and relative paths are
  rejected.
- In attach mode, `document.open` is refused. Only documents the user already has open are
  available. The tool never closes or kills SolidWorks; a script works in that session and can
  change its open documents.

## Error codes and what to do

| Code | Meaning and action |
|---|---|
| `host_not_running` | Host is not installed or did not start. Tell the user to check the tool's host panel; do not retry in a loop. |
| `no_interactive_session` | The Windows account has no desktop session. Ask the user to sign in on that machine. |
| `version_not_installed` | The pinned SolidWorks year is not installed. Ask the user to change the tool's version. |
| `version_mismatch` | Attach mode: the running SolidWorks is not the pinned year. Ask the user to start the right one. |
| `solidworks_not_running` | Attach mode and no SolidWorks instance is open in the session. Ask the user to open SolidWorks. |
| `solidworks_start_failed` | SolidWorks could not start. Report it; the machine may need attention. |
| `instance_limit` | All background instances are busy. Retry later, or close documents you no longer need. |
| `instance_timeout` | The instance did not become ready or is still stuck on an earlier call. Retry later. |
| `operation_timeout` | The call ran out of time. Narrow the request (smaller `limit`, a single feature) and retry. |
| `document_not_found` | The path does not exist. Check the spelling with the user. |
| `document_not_open` | The `document` is not an open document. Call `document.open` first. |
| `not_supported_for_document_type` | Wrong kind of file, for example `components.list` on a part. Choose another operation. |
| `entity_not_found` | A named feature, sketch or configuration does not exist. Call the matching `*.list` first. |
| `stale_snapshot` | The model changed between pages. Restart the list from `offset` 0. |
| `result_too_large` | The result exceeds the size budget. Narrow it, see `paging`. |
| `invalid_arguments` | Arguments are wrong, including a rejected path. Fix them; do not repeat the same call. |

Other codes (`invalid_request`, `protocol_mismatch`, `unknown_operation`, `write_not_allowed`,
`solidworks_error`, `internal_error`) mean the request or the environment is wrong. Report them
with the message rather than guessing.

## Other topics

- `documents`: open, list and close documents; summaries, configurations, properties, equations,
  mass properties.
- `features-and-sketches`: feature tree, feature details, sketch geometry, dimensions.
- `assemblies`: components and mates.
- `paging`: limits, `next_offset`, snapshots, large results.
- `pdm`: PDM vault search, live file state, get latest, check-out and check-in.
- `studying-models`: learn the house conventions from existing models.
- `scripts`: build or change models with C# scripts, when the tool allows scripts.

Read `pdm` when the tool is linked to a PDM vault and you need to find or change vault files.
Read `studying-models` before you propose or change a model, to see how similar models are built.
Read `scripts` for the build-and-save flow.

Call `guide` with `topic` set to one of these ids.
