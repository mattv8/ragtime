# SolidWorks host

`RagtimeSolidWorks.sln` contains Core, Host, Runtime, Readers and Pdm projects in
`src/`, and their tests in `tests/`. The host targets .NET Framework 4.8. Host version: 0.3.0
(`Directory.Build.props`, copied into `manifest.json`).

`Ragtime.SolidWorks.Pdm` links the host to the SOLIDWORKS PDM client API. It uses the PDM interop types embedded
from the compile-time NuGet reference `SolidWorks.EPDM.Interop.epdm` 32.1.0; nothing from that package is
redistributed, so the zip contains no `EPDM*.dll` (`build-host-zip.sh` fails if one appears). The PDM client must be
installed on the target computer for the `pdm.*` operations; without it they answer `client_missing` and
`status` reports `client_installed: false`.

The PDM claims file, `%LOCALAPPDATA%\RagtimeSolidWorks\pdm-claims.txt`, holds one JSON record per claim: the vault,
file id, the tool id, the lock owner, the computer, the local lock path and `ClaimedUtc` (ISO 8601, UTC). A claim is
bound to the tool and to the exact check-out: it is valid only while the file is still checked out by the same user
on the same computer at the same local path, and it expires 24 hours after it was made. Check-in with
`keep_checked_out` renews it. Only valid claims let the tool check in, change or undo a file; anything else is refused
with `not_checked_out_by_tool`. Records that do not parse, including the old `vault|fileId` lines, are skipped, so
those files have no claim. A missing or corrupt file starts empty; the host keeps at most 10,000 claims and drops the
oldest.

Residual limit: a person who undoes and checks out the same file again on the same computer, at the same local path,
within 24 hours and without a tool call in between, looks the same as the tool's own check-out.

The work folder, `%LOCALAPPDATA%\RagtimeSolidWorks\work\`, is created when `serve` starts. `pdm.file.add` may read
from it as well as from the local view of an approved vault folder; any other source path is refused with
`invalid_arguments`. Save new models there or straight into an approved vault folder. If the folder cannot be created,
the host logs it and `pdm.status` reports `work_folder: null`.

Security note: approved folders and the work folder protect against mistakes by the AI. They do not protect against
other programs running as the same Windows user, which can talk to the host pipe directly.

## Scripts

`script.run` executes C# in a separate worker process. Each run uses a user-only folder under
`%LOCALAPPDATA%\RagtimeSolidWorks\scripts\`; completed runs are removed immediately and other runs are retained for
up to 24 hours for diagnosis. On timeout the worker and its children are killed; a script may already have changed
open documents, so re-read them before retrying. Scripts are **not sandboxed**: they run with the Windows user's
file, network and PDM rights.

Build and test in a private copy so concurrent builds do not share `obj/`:

```sh
docker run --rm -v "$PWD/dotnet":/src:ro -v ragtime-sw-nuget:/root/.nuget/packages mcr.microsoft.com/dotnet/sdk:8.0 bash -c 'cp -r /src /build && cd /build && dotnet restore RagtimeSolidWorks.sln --locked-mode && dotnet test RagtimeSolidWorks.sln -c Release --no-restore && bash build-host-zip.sh'
```

To retain the zip and manifest, mount an output directory and pass its path to
`build-host-zip.sh`; its default is `dotnet/artifacts`.

When updating dependencies, regenerate all `packages.lock.json` files with a restore in a private SDK-container copy.
Update `third-party-hashes.txt` from the pinned NuGet DLLs and review the changed hashes.

On Windows, unzip the host into `%LOCALAPPDATA%\RagtimeSolidWorks\host` and run
`ragtime-solidworks-host.exe install` as the SolidWorks user in their desktop session.

## Development

The development stack mounts `dotnet/artifacts/` read-only at `/opt/ragtime/solidworks-host`. Until the package
exists there, the SolidWorks wizard shows "Installer missing". Build it once; the output lands in `dotnet/artifacts/`:

```sh
mkdir -p dotnet/artifacts
docker run --rm -v "$PWD/dotnet":/src:ro -v "$PWD/dotnet/artifacts":/out -v ragtime-sw-nuget:/root/.nuget/packages mcr.microsoft.com/dotnet/sdk:8.0 bash -c 'cp -r /src /build && cd /build && dotnet restore RagtimeSolidWorks.sln --locked-mode && bash build-host-zip.sh /out'
```

Then recreate the dev `ragtime` service (`docker compose -f docker/docker-compose.dev.yml up -d ragtime`) so the mount appears.
