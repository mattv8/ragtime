# SolidWorks tool domain notes

Scope: the `solidworks` tool type, its Windows host program, the PDM link and
`script.run`. PDM index behavior is in `.agents/pdm.md`; user-facing setup is
in the README's SolidWorks section.

## Layout

- Ragtime side: `ragtime/solidworks/`. `tool.py` builds the single
  `solidworks_<name>` wrapper (`operation`, `arguments`, `timeout`) for Chat
  (`ragtime/rag/components.py`) and MCP (`ragtime/mcp/tools.py`). `service.py`
  executes operations; `transport.py` and `protocol.py` carry them;
  `host_status.py`, `host_installer.py` and `host_artifact.py` serve the
  wizard's host panel; `pdm_link.py` resolves the PDM link; `pdm_rights.py`
  serves the rights check and folder picker; `routes.py` is the admin-only
  `/indexes/solidworks` API; `guidance/*.md` is the agent-facing guide.
- Config: `SolidworksConnectionConfig` (`ragtime/indexer/models.py`) extends
  the SSH config. `_validate_solidworks_pdm_link` (`ragtime/indexer/routes.py`)
  checks the link on create and update. Quick connect and tool deletion reuse
  Windows Remote pairing (`WINDOWS_PAIRING_TOOL_TYPES`); deleting a tool never
  uninstalls the host. The heartbeat is the SSH PowerShell check only; it does
  not prove the host runs.
- Host: `dotnet/` (build notes in `dotnet/README.md`). Core (netstandard2.0)
  holds the wire protocol, `OperationCatalog`, `RequestProcessor` and
  `PdmOperations`. Host is the net48 executable: CLI, pipe server, logon task
  and script worker. Runtime owns the SolidWorks instances, Readers wraps the
  SolidWorks API, Pdm wraps the PDM client, Scripting is `ScriptContext`.
- UI: `ragtime/frontend/src/components/SolidWorks*.tsx`,
  `src/api/solidworks.ts`, `src/utils/solidworksPdm*.ts`,
  `src/hooks/usePdm*.ts`, `src/styles/solidworks.css`.

## Request path and contract

- Each call runs `ragtime-solidworks-host.exe call --request-b64 <base64>`
  over SSH. `call` checks that the per-user pipe's server runs as the same
  account (`PipeServerIdentity`), sends one request line and prints one
  response line. The `serve` process answers with `RequestProcessor`, on one
  STA thread per SolidWorks instance or on the single PDM STA thread.
- `describe` comes from the host's `OperationCatalog`. Ragtime appends
  `guide`, adds `pdm.search` and `pdm.lookup` (answered in Ragtime from the
  index) only when a link resolves, removes `PDM_ADMIN_OPERATIONS`, and caches
  the result per tool until the host version changes or a requested `pdm.*`
  operation is missing.
- Adding a host operation: a catalog entry (`SpecShapeSerializationTests`
  asserts the order of the `pdm.*` entries), dispatch, DTOs, `Fakes.cs` and
  tests, a host version bump in `dotnet/Directory.Build.props` so the wizard
  offers **Update host**, and the guidance topic with
  `tests/test_solidworks_guidance.py`. A write operation also goes in
  `PDM_WRITE_OPERATIONS`, an admin-only one in `PDM_ADMIN_OPERATIONS`.
- Error codes in `Wire/ErrorCodes.cs` are wire contract; guidance tables and
  tests name them. Never rename one.
- Request size: Ragtime caps the base64 request at `MAX_ENCODED_REQUEST_LENGTH`,
  or `MAX_ENCODED_SCRIPT_REQUEST_LENGTH` for `script.run`
  (`ragtime/solidworks/protocol.py`). The host's `call` accepts up to
  `CallRequestArgument.MaxLength` (Core); keep it at least the larger cap and
  within the 32,767-character Windows command line.
- Results use SI units within a 32 KiB budget; lists page with a SHA-256
  snapshot. Ragtime forwards `deadline_seconds` 10 s below its own budget, and
  the host shares one deadline across all steps of a request. A COM call that
  outlives its deadline leaves its instance or the PDM worker busy until it
  returns.

## Safety invariants

- Vault writes are gated on both sides; keep both. Ragtime reloads
  `allow_write`, the link and the approved folders from the stored tool on
  every `pdm.*` call (`load_stored_pdm_settings`) and refuses
  `PDM_WRITE_OPERATIONS` early. The host refuses `write` access without
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
  with `authorize_content`, then sends it. The host checks `allow_scripts`
  again and runs the script in a worker inside a kill-on-close Job Object.
  Scripts are not sandboxed.
- Host hardening: the pipe ACL allows only the user's SID, the single-instance
  mutex must be owned by that user, `install` runs only from
  `%LOCALAPPDATA%\RagtimeSolidWorks\host`, and the host log never records
  request bodies.

## PDM link

- `connection_config.pdm_tool_id` links one `solidworks_pdm` tool. Its
  `database` is the vault name sent to the host, and its index serves
  `pdm.search` and `pdm.lookup`. Without an enabled link, every `pdm.*`
  operation except `pdm.status` answers `pdm_not_linked`.
- The PDM tool's SQL login is for indexing only. Host `pdm.*` operations log
  in with `LoginAuto` as the vault-view user of the Windows account; that is
  the only PDM identity for vault files. Ragtime stores no PDM credentials.
- `pdm.folder.rights` and `pdm.folder.children` serve the wizard's rights
  check and folder picker through `/indexes/solidworks/tools/{id}/pdm/*`.
  They run with `allow_write: false` and no approved folders, and the agent
  wrapper hides and refuses them. Folder rights do not cover workflow-state
  permissions.

## Verification

- Host package: the image ships it at `/opt/ragtime/solidworks-host`
  (`SOLIDWORKS_HOST_ARTIFACT_DIR`). The dev stack mounts `dotnet/artifacts`
  there read-only; build it with the command in `dotnet/README.md` and
  recreate the `ragtime` service, or the wizard shows **Installer missing**.
- Tests: `tests/test_solidworks_*.py`; the `SolidWorks*`, `solidworksPdm*` and
  `usePdm*` Vitest files; host tests with
  `docker build --target solidworks-host-test -f docker/Dockerfile .`, also run
  by `tests/run_all_tests.sh` and the required **SolidWorks Host Tests** CI
  job.
- Host tests run against fakes; no CI job runs Windows, SolidWorks or the PDM
  client. Verify COM, PDM and Windows behavior on a real machine and report
  what was not verified live.
- Restore with `--locked-mode` against `packages.lock.json`. SolidWorks interop
  DLLs ship with SHA-256 hashes pinned in `dotnet/third-party-hashes.txt`. PDM
  interop types are embedded; `build-host-zip.sh` fails if an `EPDM*.dll`
  would ship.
