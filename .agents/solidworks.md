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
  `/indexes/solidworks` API; `guidance/*.md` is the guide the chat agent reads.
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
  `tests/test_solidworks_guidance.py`. Bump `Version` and
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

## Verification

- Agent package: the image ships it at `/opt/ragtime/solidworks-host`
  (`SOLIDWORKS_HOST_ARTIFACT_DIR`). The dev stack mounts `dotnet/artifacts`
  there read-only; build it with the command in `dotnet/README.md` and
  recreate the `ragtime` service, or the wizard shows **Installer missing**.
- Tests: `tests/test_solidworks_*.py` and `tests/test_mcp_solidworks_artifacts.py`;
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
