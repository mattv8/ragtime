#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
output="${1:-${SOLIDWORKS_HOST_OUTPUT_DIR:-$root/artifacts}}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

host_project="$root/src/Ragtime.SolidWorks.Host/Ragtime.SolidWorks.Host.csproj"
if [[ ! -f "$root/src/Ragtime.SolidWorks.Host/obj/project.assets.json" ]]; then
  dotnet restore "$host_project" --locked-mode
fi
dotnet publish "$host_project" -c Release --no-restore -o "$work/publish"
# The PDM client interop is embedded in Ragtime.SolidWorks.Pdm; no EPDM assembly may be shipped.
if compgen -G "$work/publish/EPDM*.dll" >/dev/null; then
  echo "EPDM interop assembly must not be shipped: $(basename "$(ls "$work/publish"/EPDM*.dll | head -n 1)")" >&2
  exit 1
fi
mkdir "$work/contents"
for file in ragtime-solidworks-host.exe Ragtime.SolidWorks.Core.dll Ragtime.SolidWorks.Runtime.dll Ragtime.SolidWorks.Readers.dll Ragtime.SolidWorks.Pdm.dll Ragtime.SolidWorks.Scripting.dll; do
  test -f "$work/publish/$file" || { echo "Missing required host file: $file" >&2; exit 1; }
  cp "$work/publish/$file" "$work/contents/"
done
if [[ -f "$work/publish/ragtime-solidworks-host.exe.config" ]]; then
  cp "$work/publish/ragtime-solidworks-host.exe.config" "$work/contents/"
fi
for file in Newtonsoft.Json.dll SolidWorks.Interop.sldworks.dll SolidWorks.Interop.swconst.dll SolidWorks.Interop.swpublished.dll; do
  source="$work/publish/$file"
  if [[ ! -f "$source" ]]; then
    case "$file" in
      Newtonsoft.Json.dll) source="${NUGET_PACKAGES:-$HOME/.nuget/packages}/newtonsoft.json/13.0.3/lib/net45/$file" ;;
      *) source="${NUGET_PACKAGES:-$HOME/.nuget/packages}/xarial.xcad.solidworks.interops/0.4.0/lib/net40/$file" ;;
    esac
  fi
  [[ -f "$source" ]] || { echo "Missing required dependency: $file" >&2; exit 1; }
  cp "$source" "$work/contents/$file"
  expected="$(awk -v file="$file" '$2 == file { print $1 }' "$root/third-party-hashes.txt")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "Missing or invalid expected SHA-256 for $file" >&2; exit 1; }
  actual="$(sha256sum "$work/contents/$file" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]] || { echo "SHA-256 mismatch for $file" >&2; exit 1; }
done
cp "$root/LICENSES.txt" "$work/contents/"
# ZipDirectory preserves file timestamps; normalize them to make repeat builds stable.
touch -t 200001010000 "$work/contents/"*
zip_name=ragtime-solidworks-host-win-x64.zip
dotnet msbuild "$root/zip-host.proj" -t:Zip -p:SourceDirectory="$work/contents" -p:DestinationFile="$output/$zip_name"
# Zip entry names are stored uncompressed, so a byte search finds an EPDM entry without unzip.
if LC_ALL=C grep -aqi 'EPDM[^/]*\.dll' "$output/$zip_name"; then
  rm -f "$output/$zip_name"
  echo "EPDM interop assembly found in $zip_name; refusing to publish it." >&2
  exit 1
fi
version="$(sed -n 's/.*<Version>\([^<]*\)<\/Version>.*/\1/p' "$root/Directory.Build.props")"
[[ "$version" =~ ^[0-9A-Za-z.+-]+$ ]] || { echo "Invalid host version: $version" >&2; exit 1; }
sha="$(sha256sum "$output/$zip_name" | cut -d' ' -f1)"
size="$(wc -c < "$output/$zip_name" | tr -d ' ')"
printf '{"host_version":"%s","protocol":1,"file":"%s","sha256":"%s","size_bytes":%s}\n' "$version" "$zip_name" "$sha" "$size" > "$output/manifest.json"
