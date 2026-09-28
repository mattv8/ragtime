#!/usr/bin/env python3
"""Publish a durable, digest-pinned stable container release.

All mutation boundaries are small classes so the release ordering can be tested
without a registry, GitHub, or Cosign credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}$")
_TAG = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_SHA = re.compile(r"[0-9a-f]{40}$")
_ASSET_HOSTS = {"api.github.com", "release-assets.githubusercontent.com", "github-releases.githubusercontent.com", "objects.githubusercontent.com"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class PublishError(RuntimeError):
    """A failed-closed publication precondition or mutation."""


def _run(command: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(command, input=input_text, capture_output=True, text=True, check=False)
    if result.returncode:
        raise PublishError(f"command failed ({' '.join(command[:3])}): {result.stderr.strip()}")
    return result.stdout


class GitHub:
    def __init__(self, repository: str, token: str) -> None:
        self.repository, self.token = repository, token

    def _api(self, method: str, path: str, body: bytes | None = None, *, uploads: bool = False, optional: bool = False) -> Any:
        host = "https://uploads.github.com" if uploads else "https://api.github.com"
        request = urllib.request.Request(
            f"{host}/repos/{self.repository}/{path.lstrip('/')}",
            body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/octet-stream" if uploads else "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 -- fixed GitHub API host
                payload = response.read()
        except urllib.error.HTTPError as error:
            if optional and method == "GET" and error.code == 404:
                return None
            raise PublishError(f"GitHub {method} {path} failed: HTTP {error.code}") from error
        except OSError as error:
            raise PublishError(f"GitHub {method} {path} failed: {error}") from error
        try:
            return json.loads(payload) if payload else {}
        except json.JSONDecodeError as error:
            raise PublishError(f"GitHub {method} {path} returned invalid JSON") from error

    def remote_main_sha(self) -> str:
        value = self._api("GET", "git/ref/heads/main", optional=True)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("object"), dict)
            or value["object"].get("type") != "commit"
            or not isinstance(value["object"].get("sha"), str)
            or not _SHA.fullmatch(value["object"]["sha"])
        ):
            raise PublishError("remote main ref is missing")
        return value["object"]["sha"]

    def tag_sha(self, tag: str) -> str | None:
        value = self._api("GET", f"git/ref/tags/{urllib.parse.quote(tag, safe='')}", optional=True)
        if value is None:
            return None
        if not isinstance(value, dict) or not isinstance(value.get("object"), dict):
            raise PublishError("GitHub returned malformed tag reference")
        target = value["object"]
        if target.get("type") == "tag":
            annotated = self._api("GET", f"git/tags/{target['sha']}")
            target = annotated.get("object", {}) if isinstance(annotated, dict) else {}
        if target.get("type") != "commit" or not isinstance(target.get("sha"), str) or not _SHA.fullmatch(target["sha"]):
            raise PublishError("GitHub tag does not resolve to a commit")
        return target["sha"]

    def reserve_tag(self, tag: str, sha: str) -> None:
        body = json.dumps({"ref": f"refs/tags/{tag}", "sha": sha}).encode()
        try:
            value = self._api("POST", "git/refs", body)
            if not isinstance(value, dict) or value.get("ref") != f"refs/tags/{tag}":
                raise PublishError("GitHub did not confirm Git tag reservation")
        except PublishError:
            if self.tag_sha(tag) != sha:
                raise

    def _releases(self) -> list[dict[str, object]]:
        releases: list[dict[str, object]] = []
        for page in range(1, 101):
            value = self._api("GET", f"releases?per_page=100&page={page}")
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                raise PublishError("GitHub release listing returned invalid JSON")
            releases.extend(value)
            if len(value) < 100:
                return releases
        raise PublishError("GitHub release listing exceeded pagination limit")

    def _asset_manifest(self, release: dict[str, object]) -> None:
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise PublishError("release assets are malformed; inspect the draft manually")
        matches = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == "release-manifest.json"]
        if not matches:
            return
        if len(matches) != 1 or matches[0].get("state") != "uploaded":
            raise PublishError("release manifest asset is incomplete; inspect the draft manually")
        url = matches[0].get("url")
        if not isinstance(url, str):
            raise PublishError("release manifest asset lacks an API URL")
        release["manifest"] = self._download_manifest(url)

    def _download_manifest(self, url: str) -> object:
        parsed = urllib.parse.urlparse(url)
        expected = f"/repos/{self.repository}/releases/assets/"
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.github.com"
            or parsed.port is not None
            or parsed.username
            or parsed.password
            or not parsed.path.startswith(expected)
            or not parsed.path[len(expected) :].isdigit()
        ):
            raise PublishError("release manifest asset URL is unsafe")
        request = urllib.request.Request(url, headers={"Accept": "application/octet-stream", "Authorization": f"Bearer {self.token}"})
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            response = opener.open(request, timeout=20)
        except urllib.error.HTTPError as error:
            if error.code not in {301, 302, 303, 307, 308}:
                raise PublishError("cannot read durable release manifest") from error
            location = error.headers.get("Location")
            if not isinstance(location, str):
                raise PublishError("release manifest redirect is missing") from error
            redirect = urllib.parse.urlparse(location or "")
            if redirect.scheme != "https" or redirect.hostname not in _ASSET_HOSTS or redirect.port is not None or redirect.username or redirect.password:
                raise PublishError("release manifest redirect is unsafe") from error
            request = urllib.request.Request(location, headers={"Accept": "application/octet-stream"})
            try:
                response = opener.open(request, timeout=20)
            except OSError as redirect_error:
                raise PublishError("cannot read durable release manifest") from redirect_error
        except OSError as error:
            raise PublishError("cannot read durable release manifest") from error
        with response:
            payload = response.read(65537)
        if len(payload) > 65536:
            raise PublishError("release manifest exceeds 64KiB")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise PublishError("release manifest is invalid JSON") from error

    def release(self, tag: str, sha: str) -> dict[str, object] | None:
        value = self._api("GET", f"releases/tags/{urllib.parse.quote(tag, safe='')}", optional=True)
        if isinstance(value, dict):
            target = value.get("target_commitish")
            if isinstance(target, str) and _SHA.fullmatch(target) and target != sha:
                raise PublishError("release target differs from release SHA")
            self._asset_manifest(value)
            return value
        matches = [release for release in self._releases() if release.get("draft") is True and release.get("tag_name") == tag]
        if len(matches) > 1:
            raise PublishError("multiple matching release drafts exist")
        if not matches:
            return None
        value = matches[0]
        target = value.get("target_commitish")
        if isinstance(target, str) and _SHA.fullmatch(target) and target != sha:
            raise PublishError("release draft target differs from release SHA")
        self._asset_manifest(value)
        return value

    def create_draft(self, tag: str, sha: str) -> dict[str, object]:
        value = self._api("POST", "releases", json.dumps({"tag_name": tag, "target_commitish": sha, "draft": True}).encode())
        if not isinstance(value, dict) or not isinstance(value.get("id"), int) or value.get("draft") is not True:
            raise PublishError("GitHub did not return a draft release")
        return value

    def upload_manifest(self, release: dict[str, object], manifest: dict[str, object]) -> None:
        release_id = release.get("id")
        if not isinstance(release_id, int):
            raise PublishError("draft release lacks an id")
        value = self._api("POST", f"releases/{release_id}/assets?name=release-manifest.json", json.dumps(manifest, sort_keys=True).encode(), uploads=True)
        if not isinstance(value, dict) or value.get("name") != "release-manifest.json":
            raise PublishError("GitHub did not confirm manifest asset upload")
        release["manifest"] = manifest

    def _previous_published_tag(self, current: str) -> str | None:
        candidates: list[tuple[tuple[int, int, int], str]] = []
        for release in self._releases():
            tag = release.get("tag_name")
            if not isinstance(tag, str) or not _TAG.fullmatch(tag) or tag == current or release.get("draft") is not False or release.get("prerelease") is True:
                continue
            major, minor, patch = (int(part) for part in tag[1:].split("."))
            candidates.append(((major, minor, patch), tag))
        return max(candidates)[1] if candidates else None

    def publish(self, release: dict[str, object], tag: str, sha: str) -> None:
        release_id = release.get("id")
        if not isinstance(release_id, int):
            raise PublishError("draft release lacks an id")
        payload: dict[str, str] = {"tag_name": tag, "target_commitish": sha}
        previous = self._previous_published_tag(tag)
        if previous:
            payload["previous_tag_name"] = previous
        notes = self._api("POST", "releases/generate-notes", json.dumps(payload).encode())
        if not isinstance(notes, dict) or not isinstance(notes.get("body"), str) or not isinstance(notes.get("name"), str):
            raise PublishError("GitHub did not generate release notes")
        value = self._api("PATCH", f"releases/{release_id}", json.dumps({"draft": False, "name": notes["name"], "body": notes["body"]}).encode())
        if not isinstance(value, dict) or value.get("draft") is not False:
            raise PublishError("GitHub did not confirm release publication")
        release["draft"] = False

    def merged_pr_labels(self, sha: str) -> list[str]:
        # The exact merge commit identity avoids reading labels from a newer PR.
        values = self._api("GET", f"commits/{sha}/pulls?per_page=100")
        if not isinstance(values, list):
            raise PublishError("GitHub merged PR lookup returned an invalid response")
        if len(values) == 100:
            raise PublishError("merged PR lookup may be truncated; refusing ambiguous release metadata")
        matches = [
            item
            for item in values
            if item.get("merge_commit_sha") == sha
            and item.get("merged_at") is not None
            and item.get("base", {}).get("ref") == "main"
            and item.get("head", {}).get("ref") == "beta"
            and isinstance(item.get("head", {}).get("repo"), dict)
            and item["head"]["repo"].get("full_name") == self.repository
        ]
        if len(matches) > 1:
            raise PublishError(f"multiple beta-to-main PRs have merge SHA {sha}")
        if not matches:
            return []
        labels = matches[0].get("labels", [])
        return [str(label["name"]) for label in labels if isinstance(label, dict) and "name" in label]


class Registry:
    def tag_digest(self, image: str, tag: str) -> str | None:
        result = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", "--format", "{{.Manifest.Digest}}", f"{image}:{tag}"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            digest = result.stdout.strip()
            if _DIGEST.fullmatch(digest):
                return digest
            raise PublishError(f"registry returned invalid digest for {image}:{tag}")
        error_lines = {line.strip() for line in result.stderr.splitlines() if line.strip()}
        missing_lines = {f"{image}:{tag}: not found", f"ERROR: {image}:{tag}: not found"}
        if error_lines & missing_lines or "manifest unknown" in result.stderr.lower() or "name unknown" in result.stderr.lower():
            return None
        raise PublishError(f"cannot inspect {image}:{tag}: {result.stderr.strip()}")

    def tag(self, image: str, digest: str, tag: str) -> None:
        _run(["docker", "buildx", "imagetools", "create", "--tag", f"{image}:{tag}", f"{image}@{digest}"])
        if self.tag_digest(image, tag) != digest:
            raise PublishError(f"registry did not preserve {image}:{tag} identity")


class Cosign:
    def sign_and_verify(self, reference: str) -> None:
        _run(["cosign", "sign", "--yes", "--key", "env://COSIGN_PRIVATE_KEY", reference])
        with tempfile.TemporaryDirectory(prefix="ragtime-cosign-") as directory:
            public_key = os.path.join(directory, "public.key")
            _run(["cosign", "public-key", "--key", "env://COSIGN_PRIVATE_KEY", "--outfile", public_key])
            _run(["cosign", "verify", "--key", public_key, reference])


def release_bump(github: Any, sha: str, manual: str) -> str:
    if manual != "auto":
        return manual
    labels = set(github.merged_pr_labels(sha))
    choices = sorted(label.removeprefix("release:") for label in labels if label in {"release:patch", "release:minor", "release:major"})
    if len(choices) > 1:
        raise PublishError("conflicting release bump labels")
    return choices[0] if choices else "auto"


class ReleasePublisher:
    def __init__(self, *, github: Any, registry: Any, signer: Any, images: Mapping[str, str]) -> None:
        self.github, self.registry, self.signer, self.images = github, registry, signer, images

    def _manifest(self, sha: str, tag: str, digests: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "sha": sha,
            "tag": tag,
            "images": {kind: f"{self.images['app' if kind == 'legacy' else kind]}@{digest}" for kind, digest in digests.items()},
        }

    def _validate_manifest(self, manifest: object, sha: str, tag: str) -> dict[str, str]:
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("sha") != sha or manifest.get("tag") != tag:
            raise PublishError("invalid release manifest identity")
        images = manifest.get("images")
        required = {"app", "runtime", "storage", "legacy"}
        if not isinstance(images, dict) or set(images) != required:
            raise PublishError("release manifest has incomplete image roles")
        selected: dict[str, str] = {}
        for role in required:
            image = self.images["app" if role == "legacy" else role]
            reference = images[role]
            if not isinstance(reference, str) or not reference.startswith(f"{image}@"):
                raise PublishError(f"release manifest image repository differs for {role}")
            digest = reference.removeprefix(f"{image}@")
            if not _DIGEST.fullmatch(digest):
                raise PublishError(f"release manifest digest is invalid for {role}")
            selected[role] = digest
        return selected

    def _ensure_tag(self, image: str, tag: str, digest: str) -> None:
        found = self.registry.tag_digest(image, tag)
        if found is None:
            self.registry.tag(image, digest, tag)
        elif found != digest:
            raise PublishError(f"immutable tag conflict for {image}:{tag}")

    def _version_tags_match(self, tag: str, selected: Mapping[str, str]) -> bool:
        for kind, digest in selected.items():
            image = self.images["app" if kind == "legacy" else kind]
            if self.registry.tag_digest(image, f"{tag}-legacy" if kind == "legacy" else tag) != digest:
                return False
        return True

    def publish(self, sha: str, tag: str, candidates: Mapping[str, str]) -> None:
        if self.github.remote_main_sha() != sha:
            raise PublishError("remote main no longer matches release SHA")
        release = self.github.release(tag, sha)
        existing_tag = self.github.tag_sha(tag)
        if existing_tag not in (None, sha):
            raise PublishError(f"Git tag {tag} belongs to another SHA")
        if release is not None and existing_tag != sha:
            raise PublishError("existing release has no matching Git tag")
        if release is None and existing_tag is None:
            self.github.reserve_tag(tag, sha)
        generated = self._manifest(sha, tag, candidates)
        if release is not None and not release.get("draft") and release.get("manifest") is None:
            raise PublishError("published release has no durable manifest")
        manifest = release.get("manifest") if release is not None and release.get("manifest") is not None else generated
        selected = self._validate_manifest(manifest, sha, tag)
        if release and not release.get("draft"):
            if not self._version_tags_match(tag, selected):
                raise PublishError("published release immutable tags differ from manifest")
            return
        if release is None:
            release = self.github.create_draft(tag, sha)
        if release.get("manifest") is None:
            self.github.upload_manifest(release, manifest)
        # Preflight every immutable tag before writing any of them.
        for kind, digest in selected.items():
            image = self.images["app" if kind == "legacy" else kind]
            found = self.registry.tag_digest(image, f"{tag}-legacy" if kind == "legacy" else tag)
            if found is not None and found != digest:
                raise PublishError(f"immutable tag conflict for {image}:{tag}")
        for kind, digest in selected.items():
            image = self.images["app" if kind == "legacy" else kind]
            self.signer.sign_and_verify(f"{image}@{digest}")
        for kind, digest in selected.items():
            image = self.images["app" if kind == "legacy" else kind]
            self._ensure_tag(image, f"{tag}-legacy" if kind == "legacy" else tag, digest)
        if self.github.remote_main_sha() != sha:
            raise PublishError("remote main moved before stable aliases")
        for kind, digest in selected.items():
            image = self.images["app" if kind == "legacy" else kind]
            aliases = ["legacy"] if kind == "legacy" else ["main", "latest", f"main-{sha[:7]}"]
            for alias in aliases:
                self.registry.tag(image, digest, alias)
        if self.github.remote_main_sha() != sha:
            raise PublishError("remote main moved before release publication")
        self.github.publish(release, tag, sha)


def _planner(sha: str, initial_version: str, bump: str) -> dict[str, str]:
    script = os.path.join(os.path.dirname(__file__), "release_version.py")
    output = _run([sys.executable, script, "--ref", sha, "--initial-version", initial_version, "--bump", bump])
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise PublishError("release planner returned invalid JSON") from error
    keys = {"tag", "version", "previous_tag", "sha", "existing_tag"}
    if not isinstance(value, dict) or set(value) != keys or not all(isinstance(value.get(key), str) for key in keys):
        raise PublishError("release planner returned invalid JSON")
    if not _TAG.fullmatch(value["tag"]) or not _SHA.fullmatch(value["sha"]) or value["existing_tag"] not in {"true", "false"}:
        raise PublishError("release planner returned invalid release identity")
    return {key: value[key] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="print the locked-release preplan JSON")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--expected-tag")
    parser.add_argument("--initial-version", default="1.0.0")
    parser.add_argument("--bump", choices=("auto", "patch", "minor", "major"), default="auto")
    parser.add_argument("--app-digest")
    parser.add_argument("--runtime-digest")
    parser.add_argument("--storage-digest")
    parser.add_argument("--legacy-digest")
    parser.add_argument("--app")
    parser.add_argument("--runtime")
    parser.add_argument("--storage")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise PublishError("GITHUB_TOKEN is required for stable publication")
    github = GitHub(args.repository, token)
    planned = _planner(args.sha, args.initial_version, release_bump(github, args.sha, args.bump))
    if args.plan:
        print(json.dumps(planned, sort_keys=True))
        return 0
    required = (
        args.expected_tag,
        args.app_digest,
        args.runtime_digest,
        args.storage_digest,
        args.legacy_digest,
        args.app,
        args.runtime,
        args.storage,
    )
    if not all(required):
        raise PublishError("publication requires image digests, image names, and expected tag")
    if planned["sha"] != args.sha or planned["tag"] != args.expected_tag:
        raise PublishError("release plan changed while waiting for promotion lock; retry the workflow")
    ReleasePublisher(github=github, registry=Registry(), signer=Cosign(), images={"app": args.app, "runtime": args.runtime, "storage": args.storage}).publish(
        args.sha, planned["tag"], {"app": args.app_digest, "runtime": args.runtime_digest, "storage": args.storage_digest, "legacy": args.legacy_digest}
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as error:
        print(f"release publication failed: {error}", file=sys.stderr)
        raise SystemExit(1)
