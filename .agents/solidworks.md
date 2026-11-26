# SolidWorks tool domain notes

Scope: the `solidworks` tool type, its Windows SolidWorks agent, the PDM link,
`script.run`, and model exports and previews. PDM index behavior is in
`.agents/pdm.md`; user-facing setup is in the README's SolidWorks section.

Naming: the Windows program is the SolidWorks agent in the UI, docs and messages. Code, files, folders, JSON fields,
error codes and CI job names keep "host" (`Ragtime.SolidWorks.Host`, `host_version`, `host_updating`, **SolidWorks Host
Tests**); do not rename them. "Chat agent" means the LLM that calls the tool.

## Layout

- Ragtime side: `ragtime/solidworks/`. `tool.py` builds the single
  `solidworks_<name>` wrapper (`operation`, `arguments`, `timeout`) for Chat
  (`ragtime/rag/components.py`) and MCP (`ragtime/mcp/tools.py`). `service.py`
  executes operations; `transport.py` and `protocol.py` carry them;
  `host_status.py`, `host_installer.py` and `host_artifact.py` serve the
  wizard's SolidWorks agent panel; `pdm_link.py` resolves the PDM link; `pdm_rights.py`
  serves the rights check and folder picker; `routes.py` is the admin-only
  `/indexes/solidworks` API; `guidance.py` holds the topic table
  (`GUIDE_TOPICS`) and `guidance/*.md` is the guide the chat agent reads.
  `ragtime/skills/builtin.py` is the built-in skill registry (see "Built-in
  skills"); `linking.py` computes the PDM tools hidden behind a linked
  SolidWorks tool (see "Linked pair"); `pdm_sandbox.py` creates sandbox folders
  (see "Sandbox folders").
  `artifacts.py` stores exported files and previews and `artifact_routes.py`
  serves them (see "Model exports and previews"); `host_artifact.py` is the
  agent package, not an export.
- Agent updates: `host_updater.py` updates outdated agents, and
  `host_activity.py` gates Ragtime's calls while an update runs (see
  "Agent updates" below).
- Config: `SolidworksConnectionConfig` (`ragtime/indexer/models.py`) extends
  the SSH config. `_validate_solidworks_pdm_link` (`ragtime/indexer/routes.py`)
  checks the link on create and update. Quick connect and tool deletion reuse
  Windows Remote pairing (`WINDOWS_PAIRING_TOOL_TYPES`); deleting a tool never
  uninstalls the agent. The heartbeat is the SSH PowerShell check only; it does
  not prove the agent runs.
- Agent: `dotnet/` (build notes in `dotnet/README.md`). Core (netstandard2.0)
  holds the wire protocol, `OperationCatalog`, `RequestProcessor` and
  `PdmOperations`. Host (project `Ragtime.SolidWorks.Host`) is the net48 executable: CLI, pipe server, logon task
  and script worker. Runtime owns the SolidWorks instances, Readers wraps the
  SolidWorks API, Pdm wraps the PDM client, Scripting is `ScriptContext`.
- UI: `ragtime/frontend/src/components/SolidWorks*.tsx`,
  `src/api/solidworks.ts`, `src/utils/solidworksPdm*.ts`,
  `src/hooks/usePdm*.ts`, `src/styles/solidworks.css`. Exports and previews in
  chat: `src/components/ModelArtifact*.tsx`, `ModelViewer.tsx`,
  `modelViewerSupport.ts`, `ArtifactNotice.tsx`, `SolidWorksToolResult.tsx`,
  `src/utils/modelArtifact.ts`, `src/utils/solidworksToolResult.ts`,
  `src/hooks/useArtifactAccess.ts`, and the `model-artifact-*`,
  `chat-message-artifact-*` and `solidworks-result-*` sections of
  `src/styles/chat.css`.

## Request path and contract

- Each call runs `ragtime-solidworks-host.exe call --request-b64 <base64>`
  over SSH. `call` checks that the per-user pipe's server runs as the same
  account (`PipeServerIdentity`), sends one request line and prints one
  response line. The `serve` process answers with `RequestProcessor`, on one
  STA thread per SolidWorks instance or on the single PDM STA thread.
- `describe` comes from the agent's `OperationCatalog`. Ragtime appends
  `guide`, adds `pdm.search` and `pdm.lookup` (answered in Ragtime from the
  index) only when a link resolves, removes `PDM_ADMIN_OPERATIONS`, and caches
  the result per tool until the agent version changes or a requested `pdm.*`
  operation is missing.
- Adding an agent operation: a catalog entry (`SpecShapeSerializationTests`
  asserts the order of the `pdm.*` entries), dispatch, DTOs, `Fakes.cs` and
  tests, an agent version bump, and the guidance topic with
  `tests/test_solidworks_guidance.py`; the guidance tests read operation names
  from `OperationCatalog.cs`, so a documented operation must exist there. Bump `Version` and
  `InformationalVersion` in `dotnet/Directory.Build.props`, the only place the
  agent version is written; Ragtime then updates installed agents when they
  are idle, or the wizard offers **Update agent**. Do not copy the version
  number into docs or guidance. A write operation also goes in
  `PDM_WRITE_OPERATIONS`, an admin-only one in `PDM_ADMIN_OPERATIONS`.
- Error codes in `Wire/ErrorCodes.cs` are wire contract; guidance tables and
  tests name them. Never rename one.
- Request size: Ragtime caps the base64 request at `MAX_ENCODED_REQUEST_LENGTH`,
  or `MAX_ENCODED_SCRIPT_REQUEST_LENGTH` for `script.run`
  (`ragtime/solidworks/protocol.py`). The agent's `call` accepts up to
  `CallRequestArgument.MaxLength` (Core); keep it at least the larger cap and
  within the 32,767-character Windows command line.
- Results use SI units within a 32 KiB budget; lists page with a SHA-256
  snapshot. Ragtime forwards `deadline_seconds` 10 s below its own budget, and
  the agent shares one deadline across all steps of a request. A COM call that
  outlives its deadline leaves its instance or the PDM worker busy until it
  returns.

## Model exports and previews

- `model.export` writes a model file and `model.preview` renders a PNG. Both
  are read-only and take `document`, a `doc-N` id or the full path of an open
  document, optional when exactly one document is open; `path` is rejected.
  Agent code: `Processing/ModelFiles.cs`, `ModelFileCleanup.cs` and
  `Wire/NativeFiles.cs` in Core, `SolidWorksSession.Export.cs` in Readers.
- Formats: parts take every format in `ModelFiles`. Assemblies take all but
  `stl`: SolidWorks writes one STL per component, and forcing one file throws
  `RPC_E_SERVERFAULT`. Drawings export only as native; `model.preview` covers
  pictures of a sheet. A native assembly comes as a zip with every referenced
  file, nested sub-assemblies included.
- The agent saves with `SaveAs3` Silent|Copy and writes only generated names
  (`export-<guid>`, `preview-<guid>`) in the agent work folder, so the source
  document is never saved or renamed. `ModelFileCleanup.DeleteStrays` removes
  extra files SolidWorks writes next to a generated name, and a failed request
  removes the files it wrote.
- SolidWorks behavior the code depends on: the GLB exporter writes the active
  document, so only GLB activates the target and then restores the previous
  document; SolidWorks holds component files open for writing, so native files
  are read with shared access; a sheet needs an explicit zoom to fit. File
  names come from the document path's stem, not the window title.
- Fetch: for these two operations `SolidworksService._fetch_artifacts` runs
  inside the `host_activity` call gate. It accepts only generated files in the
  agent work folder (`_ARTIFACT_PATH`), pulls each over SFTP within
  `MAX_ARTIFACT_BYTES` and the remaining deadline, checks its SHA-256, stores
  it and removes the remote copy. The tool result carries artifact ids and
  URLs, never paths, and stays short enough that chat truncation does not cut
  it. Agent errors pass through, capped by `AGENT_MESSAGE_MAX_CHARS` and
  `AGENT_DETAILS_MAX_CHARS`.
- Store and routes: `artifacts.py` keeps files under `ARTIFACT_DIR` for
  `ARTIFACT_TTL`, tagged with the requesting user; `cleanup_expired_artifacts`
  runs at startup and in `ragtime/indexer/background_tasks.py`.
  `artifact_routes.py` serves `ARTIFACT_URL_PREFIX` + `/{id}` (metadata,
  `expired: true` once expired) and `/{id}/content` (410 once expired;
  `inline=1` for the viewer). `authorize_artifact` allows an admin, the owner,
  or a user whose chat tool access includes the artifact's tool.
- Chat model: `_build_solidworks_reference_content` (`ragtime/rag/components.py`)
  attaches the latest preview image, loaded from the store by validated id,
  unless `_request_supports_reference_images` says the model takes no images
  (always the case for `openai_compatible` endpoints).
- MCP: after the outbound content check, `build_artifact_content`
  (`ragtime/mcp/artifact_content.py`) adds the preview PNG as `ImageContent`
  and a `ResourceLink` built only from `settings.external_base_url`; without
  that setting the link is left out. A failure keeps the plain text result.
- Chat UI: `parseModelArtifact` (`utils/modelArtifact.ts`) accepts only
  artifact objects whose id and URLs match the routes, including from
  truncated output. `ModelArtifactCard` renders under the tool row, and
  `SolidWorksToolResult` renders every SolidWorks result; never show raw JSON
  for these rows.
- `ModelArtifactActions` puts View and Download first in the assistant
  message's action bar, or under the card when the message has none (the
  `ModelArtifactActionsHost` context). View opens `ModelArtifactModal` through
  `modelArtifactModalStore.ts`; `ChatPanel` and `PublicSharedChatView` each
  mount one `ModelArtifactModalHost`. `modelViewerSupport.ts` keeps one live
  3D viewer at a time. `useArtifactAccess` checks access once per artifact id
  and clears on sign-out or a user change. `/?artifact=<id>`
  (`ModelArtifactPage`) stays for MCP links.
- Viewer libraries: `online-3d-viewer`, `occt-import-js` and `draco3d` are
  pinned in `ragtime/frontend/package.json`. The `selfHostModelLibraries`
  plugin in `vite.config.ts` rewrites the viewer's CDN URLs for the other two
  to `/assets/vendor/` and fails the build when an expected URL is missing.
  When you bump any of the three, update the plugin's URL map to the URLs the
  new viewer build references.
- The chat agent's guide topic is `ragtime/solidworks/guidance/previews.md`.

## Agent updates

- `host_updater` keeps its state in memory and runs in the single API worker.
  It reads agent versions from every `host` envelope and from `status`, and it
  updates only when the installed `x.y.z` is below the packaged one. It never
  downgrades.
- `auto_update_host` is a per-tool setting in `connection_config`, on by
  default; a missing key counts as on. If any enabled tool on the same `host_key`
  has it off, automatic updates are off for that agent.
- State is per `host_key` (`host_activity.host_key`: the pairing ID, else SSH
  host, port and user). Tools that share a key share one agent, and agents are
  updated one at a time. Quiet period 300 s after the last call; SSH checks at
  most every 300 s per key; after a failure, backoff of 30 min, 2 h, 8 h, then
  24 h. A new packaged version resets the backoff.
- Before SSH, the updater checks Ragtime's own state. It then calls `status`
  without PDM and skips an agent that is busy, not running, logged off, has
  documents open in a background instance it owns, or reports an in-flight call
  or PDM item.
- `host_activity` gates Ragtime's calls per `host_key`. Calls wait while an
  exclusive update holds the key, then answer `host_updating`. Manual install and
  uninstall take the key exclusively and answer `host_busy` on timeout.
- The agent gates its own stop. `stop --if-idle N` (`host.shutdown` with
  `if_idle_seconds`) refuses with `host_busy` and `reasons` while work is in
  flight, the agent was used within N seconds, an instance is busy, an owned
  background instance has documents open, or PDM is busy. The installer runs it
  before uninstalling an old agent; exit 3 keeps the old agent. The CLI waits up
  to 90 s for the agent to exit after a successful gate, and the gated install
  times out after 240 s (ungated installs use 90 s).
- Activity-neutral operations (`describe`, `status`, `pdm.status`,
  `host.shutdown`) still answer while the agent stops; other work answers
  `host_stopping`. `status` reports `activity` (in-flight count, idle seconds,
  PDM busy).
- Ragtime passes the idle gate only when the installed agent lists
  `stop-if-idle` in its `version` `features`. Older agents get no gate.
- Tunnel client versions: clients send `X-Ragtime-Tunnel-Client-Version`. The
  version is in `TUNNEL_CLIENT_VERSION` in `tunnel_protocol.py` and in both
  client scripts. A hash test pins the scripts, so bump the version and the
  hashes together. The version is display only; Ragtime does not update tunnel
  clients.

## Safety invariants

- Vault writes are gated on both sides; keep both. Ragtime reloads
  `allow_write`, the link and the approved folders from the stored tool on
  every `pdm.*` call (`load_stored_pdm_settings`) and refuses
  `PDM_WRITE_OPERATIONS` early. The agent refuses `write` access without
  `allow_write`, and targets outside the approved folders or with non-ASCII
  paths, before any vault call.
- No delete, rename, move or workflow-state operation. A check-in covers one
  file, never its references. The comment prefix comes from
  `build_pdm_context`.
- The tool finishes only its own check-outs. A claim is bound to the tool ID
  and the exact PDM lock (user ID, computer, lock path), expires after 24
  hours and is revoked before the vault call that ends it. `FilePdmClaimStore`
  removals must be durable.
- `pdm.file.add` reads only from an approved folder's local view or the work
  folder. `pdm.file.get_latest` never overwrites local work
  (`checked_out_here`, `local_untracked`, `local_modified`).
- `document.open` accepts only local drive paths and rejects links and network
  paths (`LocalFileCheck`). Attach mode refuses `document.open` and
  `document.close` and never terminates the user's SolidWorks; only
  `script.run` can change that session.
- `script.run` needs `allow_scripts`, which is separate from `allow_write`.
  Ragtime validates the arguments and the encoded size, reviews the source
  with `authorize_content`, then sends it. The agent checks `allow_scripts`
  again and runs the script in a worker inside a kill-on-close Job Object.
  Scripts are not sandboxed.
- Agent hardening: the pipe ACL allows only the user's SID, the single-instance
  mutex must be owned by that user, `install` runs only from
  `%LOCALAPPDATA%\RagtimeSolidWorks\host`, and the agent log never records
  request bodies.
- Exports and previews never save, rename or change the source document, and
  Ragtime fetches only generated files from the agent work folder. Tool
  results, stored metadata and links carry artifact ids and URLs, never file
  paths; the chat model's preview image comes from the store by validated id,
  never from a path in tool output.
- Artifact content always needs Ragtime sign-in and `authorize_artifact`;
  there are no signed or public URLs, including links handed to MCP clients.
  Anyone who can use the tool can export the full geometry of any file the
  Windows account can open. That was accepted, so there is no separate
  setting for it.

## Built-in skills

- Registry: `BUILTIN_SKILLS` in `ragtime/skills/builtin.py`, keyed by
  `tool_type` (only `solidworks` has entries). Each `BuiltinSkill` gives the
  name, mode, the guide topic whose markdown is the body, an `applies(config)`
  rule and the reason shown when the rule is false. Bodies come from
  `read_topic_markdown` (`ragtime/solidworks/guidance.py`). The topic table is
  the only source: a built-in's topic must be a `GUIDE_TOPICS` id, and
  `read_guide` lists every topic. `overview` and `paging` have no built-in.
- Merge: `SettingsCache.get_enabled_tool_configs` (`ragtime/core/app_settings.py`)
  calls `apply_builtin_skills` for each enabled config, so chat, MCP,
  `read_skill` and `resources/*` see the same list. Built-ins have `id`
  `builtin:<name>` and sort first. DB rows are never touched: export/import,
  duplication, skill CRUD and `/skills/summary` do not see them, and they sit
  outside the 32,000-character always budget (a save-time check on admin rows).
- Override: `admin_skill_names` is computed from every skill row before the
  enabled filter. Any admin skill with the same name, enabled or disabled,
  suppresses the built-in. A disabled same-name row is the way to switch one off.
- Admin view: `GET /indexes/tools/{tool_id}/skills/builtin`
  (`ragtime/skills/routes.py`) returns `list_builtin_skills`: `name`,
  `description`, `mode`, `applies`, `reason`, `overridden`. An unknown tool is
  404. The skills dialog reads it for tool owners only; card badges count admin
  skills only.
- MCP: `tool_skill_lines` (`ragtime/mcp/skills.py`) always inlines `builtin:`
  always-applied skills and counts only admin always-applied skills toward
  `MCP_INLINE_SKILL_TEXT_LIMIT`. `essentials.md` stays under 1,400 characters.
  `_build_tool_description` (`ragtime/mcp/tools.py`) appends `tool_skill_lines`
  to the SolidWorks description, outside its 1,499-character cap.
- Failure: a topic that cannot be read is logged and skipped. The other
  built-ins still load (`builtin_skills_for`).
- Content tests: `tests/test_builtin_skills.py` (every topic resolves to a real
  body) and `tests/test_solidworks_guidance.py`. The guidance tests also check
  the repair snippets: C# 5 only (no `$"`, `?.`, `=>`, `nameof`, `out var`), an
  entry point of `RagtimeScript.Run(ScriptContext ctx)`, the copies-folder guard
  and close order. Snippets marked `<!-- live-check -->` in
  `guidance/repair-assembly.md` are proven only on the Render PC.

## Linked pair

- Rule: `hidden_pdm_config_ids` (`ragtime/solidworks/linking.py`) hides a
  `solidworks_pdm` config when an enabled `solidworks` config in the same
  candidate set links it through `connection_config.pdm_tool_id`. A PDM tool
  that no SolidWorks tool links stays visible.
- MCP: `get_available_tools` (`ragtime/mcp/tools.py`) applies the rule to the
  caller's route-selected, enabled configs. It covers the default route, group
  filters and custom routes. Health does not change it. The SolidWorks config
  gets `pdm_link` (`id`, `name`, `description`, `skills`) and `pdm_link_name`
  from the same set (`pdm_link_name` is kept for the description).
- Skills: `tool_skill_lines` renders the PDM tool's enabled admin skills after
  the SolidWorks skills, under their PDM-owned URIs. `visible_skills`
  (`ragtime/mcp/skills.py`) registers those URIs whether or not the PDM
  definition is hidden. So `read_skill` and `resources/read` resolve them, and
  content protection attributes them to the PDM tool. The `solidworks-pdm`
  built-in is a SolidWorks skill and shows under the SolidWorks tool.
- Chat: `_build_request_runtime_context` (`ragtime/rag/components.py`) applies
  the rule to the request's candidate configs (allowed IDs applied, and the
  SolidWorks runtime tool present). It removes the hidden PDM tool's runtime
  tools and tool-skill entries, so `load_tool_skills` cannot re-add
  `search_<pdm>` or `lookup_<pdm>`. Its configs stay in the instruction loader,
  so the SolidWorks tool's PDM skill URIs resolve. The prompt
  (`ragtime/rag/prompts.py`) lists those skills under the SolidWorks tool.
  Userspace is not affected.
- Description: `create_solidworks_tool` (the shared factory) adds no PDM text.
  Chat clones the SolidWorks runtime tool with `build_solidworks_tool_description`
  and the linked PDM config only when that PDM config is enabled and in the
  candidate set. MCP uses the same builder.
- Line order (`build_solidworks_tool_description`, `ragtime/solidworks/tool.py`):
  the prefix; the SolidWorks admin description; `Vault index: <PDM description>.`
  (SolidWorks text first; each trimmed to fit 1,499 characters); `Scripts: on.`
  or `Scripts: off.`; `Vault changes: on in N approved folder(s).`,
  `Vault changes: off.` or `No PDM vault linked.`; the linked sentence when a
  PDM tool is linked; `Pinned to SolidWorks <version or any installed version>.`;
  then `Load the skills listed below before acting; call describe for argument
  schemas.`

## Assembly diagnostics and copies

- `mates.list` (Readers `SolidWorksSession.Assemblies.cs`, Core
  `dotnet/src/Ragtime.SolidWorks.Core/Processing/ModelListings.cs`): `error` is
  `{code, name, meaning}` or null, from `FeatureErrorCodes`
  (`dotnet/src/Ragtime.SolidWorks.Core/Model/FeatureErrorCodes.cs`). The table is the
  mate codes 38 to 48 in `guidance/assemblies.md`; any other code is `unknown`.
  `FeatureErrorCodesTests` (`dotnet/tests/Ragtime.SolidWorks.Core.Tests/DiagnosticsTableTests.cs`)
  pins the names. `whats_wrong` comes from
  `WhatsWrongTable` (`dotnet/src/Ragtime.SolidWorks.Readers/WhatsWrongTable.cs`), the same source as
  `feature.get`, read once per call. A mate entity that cannot be read is
  `component ""`, `entity_type "unknown"`, `resolved false`, and it does not fail
  the list. Entities add `component_path` (`MateEntityRules.ComponentPath`,
  names joined with `/`) and `resolved`. A thrown reference read is not
  resolved; a legitimate null reference is.
- `only_problems` (both `mates.list` and `components.list`, default false): only
  broken items. Mates count as problems when they have a non-zero error code, a
  warning, or an unresolved entity.
- `components.list`: `file_exists` (`ComponentFiles.LocalFileExists`; UNC and
  other non-local paths count as missing) and `suppression`
  (`resolved`, `lightweight`, `suppressed` or `missing_file`, from
  `GetSuppression2`). `only_problems` keeps the missing-file items.
- `model.summary` on an assembly: `missing_component_count`, counted from
  `GetComponents(false)` at any depth. Parts leave it null (not serialized).
- `document.open`: `load_warnings` names the `OpenDoc6` warning bits
  (`dotnet/src/Ragtime.SolidWorks.Core/Model/FileLoadWarnings.cs`; `references_not_found` is
  `MissingExternalReferences`). `missing_references` lists up to 50 referenced
  paths not on disk, from `MissingReferenceScan` under the request deadline. The
  list can be partial when the deadline runs out, with no warning for that yet.
  An already-open document lists them too.
- `document.copy` (access `read`, not paged): `document`, `include_references`
  (default true). It packs the open document into `<work>\copies\<32 hex digits>`
  with Pack and Go: drawings and simulation results excluded, toolbox components
  and suppressed components included, `FlattenToSingleFolder` false, no prefix or
  suffix. `include_references: false` saves only the top document (SaveAs3
  Copy). The result is `{folder, path, files, count, files_truncated, warnings}`.
  `files` lists at most 50 entries, top document first; `count` is every file;
  `files_truncated` is always present. More than 500 files is `result_too_large`
  with `details: {reason: "file_count", limit: 500, count}`. Sources and every
  destination must be local, plain paths (`CopyDestination`,
  `CopyGuards.RequirePlainChain`). `CopyGuards.RequireBudget` applies the quotas
  in `DocumentCopyLimits`: 2 GiB per copy, 10 GiB for the copies tree, and 5 GiB
  left free. A failed copy deletes its folder in Core. The source is never saved
  or renamed.
- Copy retention: Core `CopyRetention` deletes copy folders older than 7 days,
  but only folders whose own name is 32 hex digits. The host's `CopySweeper`
  runs the sweep at serve start and hourly (`ServeCommand.StartCopySweeps`).
  It skips everything when an ancestor is a link, keeps folders with a document
  open in an instance the agent started, and deletes nothing when an open
  document's path cannot be read. `DiskCopyFileSystem` never follows or deletes
  links. Logs carry counts and exception type names only.
- `pdm.file.add` accepts a source under the work folder, including `copies\...`
  (`ModelFiles.IsInside`).

## Refusals

- Ragtime refusals carry `error.details`: `setting`, `where`
  (`Tool settings > SolidWorks`), `admin_action` and `still_available`
  (`_refusal_details`, `ragtime/solidworks/service.py`). `pdm_folder_not_approved`
  adds `approved_folders`.
- `write_not_allowed` names `PDM vault: Allow changes`. `pdm_folder_not_approved`
  names `PDM vault: Approved folders`. `scripts_not_allowed` names `Allow scripts`.
  The codes are unchanged; the message text is Ragtime's, and the agent uses the
  same wording for scripts.
- `still_available` is computed per call by `_still_available` from the stored
  connection, the vault link and the cached contract. Entries read
  `<capability> (<operation>)`. `document.copy` appears only when the cached
  contract lists it and the tool is not in attach mode, because the copy routine
  needs `document.open`.
- `status` returns `ragtime`, from `_ragtime_status` (the stored tool, not the
  request): `scripts_enabled`, `vault_linked`, `vault_changes_enabled` and
  `approved_folders`.

## Sandbox folders

- Route `POST /indexes/solidworks/tools/{tool_id}/pdm/sandbox`
  (`ragtime/solidworks/routes.py`), body `{pdm_tool_id, parent, name}`, calls
  `create_sandbox` (`ragtime/solidworks/pdm_sandbox.py`). Admin only. The
  vault root is refused (422). A name must match `[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}`
  and must not end in a dot or a space (422 otherwise).
- Order, and what each step may do:
  1. Validate the parent and name (422). Load the saved tool and PDM tool; the
     link must match the stored `pdm_tool_id` (409 `pdm_link_not_saved`). A
     disabled tool is 409 `tool_disabled`.
  2. Compute the approval before any agent call (`_approval`). A target inside an
     approved folder is `covered` and the list stays as it is. A target that
     contains an approved folder, or fails `validate_approved_folders`, is 409
     `sandbox_overlap`.
  3. Rights check on the parent only (`pdm_rights_service.rights`). Missing
     `read`, `add_folders`, `add_files` or `check_out` is 409 `pdm_rights_missing`
     with `details.missing`.
  4. Agent `pdm.folder.create` with `allow_write: true` and
     `approved_folders: [target]` only. The agent's own approval of a folder it
     is told to create is the only approval used. `created: false` (the folder
     already exists) is 409 `sandbox_exists`. The route never approves an
     existing folder.
  5. Persist with one `UPDATE tool_configs` (`jsonb_set` on
     `connection_config.pdm_approved_folders`, `allow_write = true`). The guard
     compares `updated_at` on the tool and on the PDM tool, the link, the vault
     database, the enabled flags and the approved list read in step 2. On zero
     rows it reloads, checks that security state (not the approved list) is
     unchanged, recomputes the approval and retries once without a second create.
     Then 409 `sandbox_conflict`. The statement never rewrites
     `connection_config`, because it holds encrypted fields.
  6. On success: `invalidate_settings_cache()`, `notify_tools_changed()`, and the
     cached contract for the tool is dropped (`solidworks_service._contracts`).
- Agent failures: `host_not_running` and `host_updating` are 503, `timeout` is
  504 (`operation_timeout`), anything else is 502 `pdm_error` with the agent
  message (at most 1,024 characters). Error body: `{"detail": {"code",
  "message", "missing"?}}`.
- Success: `{folder, created, covered, approved_folders, allow_write,
  scripts_enabled}`. `scripts_enabled` is read only; the route never changes
  `allow_scripts`.
- The UI (`SolidWorksPdmSandboxSetup.tsx`) maps the error codes to form text.
  On success the panel adds the folder to the wizard draft and sets
  `allow_write`, so a later Save keeps both.

## PDM link

- `connection_config.pdm_tool_id` links one `solidworks_pdm` tool. Its
  `database` is the vault name sent to the agent, and its index serves
  `pdm.search` and `pdm.lookup`. Without an enabled link, every `pdm.*`
  operation except `pdm.status` answers `pdm_not_linked`.
- The PDM tool's SQL login is for indexing only. The agent's `pdm.*`
  operations log in with `LoginAuto` as the vault-view user of the Windows
  account; that is the only PDM identity for vault files. Ragtime stores no
  PDM credentials.
- `pdm.folder.rights` and `pdm.folder.children` serve the wizard's rights
  check and folder picker through `/indexes/solidworks/tools/{id}/pdm/*`.
  They run with `allow_write: false` and no approved folders, and the
  chat-agent wrapper hides and refuses them. Folder rights do not cover
  workflow-state permissions.

- Lookup arguments, ordering and the identifier boost of `pdm.lookup` and
  `pdm.search` are in `.agents/pdm.md`. Both run in `execute_index_operation`
  (`ragtime/solidworks/pdm_link.py`) without calling the agent.

## Verification

- Agent package: the image ships it at `/opt/ragtime/solidworks-host`
  (`SOLIDWORKS_HOST_ARTIFACT_DIR`). The dev stack mounts `dotnet/artifacts`
  there read-only; build it with the command in `dotnet/README.md` and
  recreate the `ragtime` service, or the wizard shows **Installer missing**.
- Tests: `tests/test_solidworks_*.py` (linked pair in `test_solidworks_linked_pair.py`,
  sandbox in `test_solidworks_pdm_sandbox.py`), `tests/test_builtin_skills.py`,
  `tests/test_mcp_solidworks_artifacts.py`;
  the `SolidWorks*`, `solidworksPdm*`, `usePdm*`, `ModelArtifact*`,
  `ModelViewer*`, `ArtifactNotice*`, `modelArtifact*`, `solidworksToolResult*`
  and `useArtifactAccess*` Vitest files. CI runs pyright over the whole
  project, not only changed files: when `SolidworksTransport` gains a method,
  add it to every transport fake under `tests/`, because pyright checks each
  fake against the Protocol. Agent tests run with
  `docker build --target solidworks-host-test -f docker/Dockerfile .`, also run
  by `tests/run_all_tests.sh` and the required **SolidWorks Host Tests** CI
  job.
- Agent tests run against fakes; no CI job runs Windows, SolidWorks or the PDM
  client. Verify COM, PDM and Windows behavior on a real machine, including
  each export format and preview, and report what was not verified live.
- Restore with `--locked-mode` against `packages.lock.json`. SolidWorks interop
  DLLs ship with SHA-256 hashes pinned in `dotnet/third-party-hashes.txt`. PDM
  interop types are embedded; `build-host-zip.sh` fails if an `EPDM*.dll`
  would ship.
