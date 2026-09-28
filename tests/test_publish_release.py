"""State-machine contracts for durable stable image publication."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_release", ROOT / "docker/scripts/publish_release.py")
assert SPEC and SPEC.loader
publish_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_release)


SHA = "a" * 40
DIGESTS = {
    "app": "sha256:" + "1" * 64,
    "runtime": "sha256:" + "2" * 64,
    "storage": "sha256:" + "3" * 64,
    "legacy": "sha256:" + "4" * 64,
}


class FakeGithub:
    def __init__(self) -> None:
        self.main_sha = SHA
        self.tags: dict[str, str] = {}
        self.releases: dict[str, dict[str, object]] = {}
        self.events: list[str] = []
        self.labels: list[str] = []

    def remote_main_sha(self) -> str:
        return self.main_sha

    def tag_sha(self, tag: str) -> str | None:
        return self.tags.get(tag)

    def reserve_tag(self, tag: str, sha: str) -> None:
        self.events.append("reserve")
        self.tags[tag] = sha

    def release(self, tag: str, sha: str) -> dict[str, object] | None:
        assert sha == SHA
        return self.releases.get(tag)

    def create_draft(self, tag: str, sha: str) -> dict[str, object]:
        self.events.append("draft")
        release = {"tag": tag, "sha": sha, "draft": True, "manifest": None}
        self.releases[tag] = release
        return release

    def upload_manifest(self, release: dict[str, object], manifest: dict[str, object]) -> None:
        self.events.append("manifest")
        release["manifest"] = manifest

    def publish(self, release: dict[str, object], tag: str, sha: str) -> None:
        self.events.append("publish")
        release["draft"] = False

    def merged_pr_labels(self, sha: str) -> list[str]:
        assert sha == SHA
        return self.labels


class FakeRegistry:
    def __init__(self) -> None:
        self.tags: dict[tuple[str, str], str] = {}
        self.events: list[str] = []
        self.fail_tag: str | None = None

    def tag_digest(self, image: str, tag: str) -> str | None:
        return self.tags.get((image, tag))

    def tag(self, image: str, digest: str, tag: str) -> None:
        self.events.append(f"tag:{image}:{tag}")
        if tag == self.fail_tag:
            raise RuntimeError("registry interrupted")
        self.tags[(image, tag)] = digest


class FakeSigner:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[str] = []

    def sign_and_verify(self, reference: str) -> None:
        self.events.append(reference)
        if self.fail:
            raise RuntimeError("signature invalid")


class PublisherTests(unittest.TestCase):
    def make_publisher(self, github: FakeGithub, registry: FakeRegistry, signer: FakeSigner) -> Any:
        return publish_release.ReleasePublisher(
            github=github,
            registry=registry,
            signer=signer,
            images={"app": "registry/app", "runtime": "registry/runtime", "storage": "registry/storage"},
        )

    def test_reserves_tag_then_persists_manifest_before_registry_mutation(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)

        self.assertEqual(github.events[:3], ["reserve", "draft", "manifest"])
        self.assertTrue(registry.events)
        manifest = github.releases["v1.2.3"]["manifest"]
        self.assertEqual(manifest["sha"], SHA)  # type: ignore[index]
        self.assertEqual(manifest["images"]["app"], f"registry/app@{DIGESTS['app']}")  # type: ignore[index]
        self.assertFalse(github.releases["v1.2.3"]["draft"])

    def test_signing_failure_does_not_move_any_tag(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner(fail=True)
        with self.assertRaisesRegex(RuntimeError, "signature"):
            self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)
        self.assertEqual(registry.tags, {})
        self.assertEqual(github.events, ["reserve", "draft", "manifest"])

    def test_partial_retry_reuses_original_manifest_digests(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        registry.fail_tag = "v1.2.3"
        publisher = self.make_publisher(github, registry, signer)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            publisher.publish(SHA, "v1.2.3", DIGESTS)
        original = json.loads(json.dumps(github.releases["v1.2.3"]["manifest"]))
        registry.fail_tag = None
        changed = {**DIGESTS, "app": "sha256:" + "9" * 64}
        publisher.publish(SHA, "v1.2.3", changed)
        self.assertEqual(github.releases["v1.2.3"]["manifest"], original)
        self.assertEqual(registry.tags[("registry/app", "v1.2.3")], DIGESTS["app"])

    def test_conflicting_immutable_tag_fails_before_aliases(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        registry.tags[("registry/app", "v1.2.3")] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(publish_release.PublishError, "immutable"):
            self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)
        self.assertNotIn(("registry/app", "latest"), registry.tags)

    def test_late_immutable_conflict_does_not_write_any_version_tag(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        registry.tags[("registry/storage", "v1.2.3")] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(publish_release.PublishError, "immutable"):
            self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)
        self.assertNotIn(("registry/app", "v1.2.3"), registry.tags)
        self.assertNotIn(("registry/runtime", "v1.2.3"), registry.tags)

    def test_malformed_recorded_manifest_fails_before_signing_or_tagging(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        github.tags["v1.2.3"] = SHA
        github.releases["v1.2.3"] = {"tag": "v1.2.3", "sha": SHA, "draft": True, "manifest": {"schema_version": 1}}
        with self.assertRaisesRegex(publish_release.PublishError, "manifest"):
            self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)
        self.assertEqual(signer.events, [])
        self.assertEqual(registry.tags, {})

    def test_published_retry_verifies_manifest_and_version_tags_without_alias_mutation(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        publisher = self.make_publisher(github, registry, signer)
        publisher.publish(SHA, "v1.2.3", DIGESTS)
        registry.events.clear()
        publisher.publish(SHA, "v1.2.3", DIGESTS)
        self.assertEqual(registry.events, [])


class GitHubAdapterTests(unittest.TestCase):
    def make_publisher(self, github: FakeGithub, registry: FakeRegistry, signer: FakeSigner) -> Any:
        return publish_release.ReleasePublisher(
            github=github,
            registry=registry,
            signer=signer,
            images={"app": "registry/app", "runtime": "registry/runtime", "storage": "registry/storage"},
        )

    def test_mutation_404_is_fatal_and_json_body_has_content_type(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        error = __import__("urllib.error", fromlist=["HTTPError"]).HTTPError("https://api.github.com", 404, "missing", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaisesRegex(publish_release.PublishError, "POST"):
                client.create_draft("v1.2.3", SHA)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_planner_accepts_real_a_cli_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "file").write_text("x", encoding="utf-8")
            subprocess.run(["git", "add", "file"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "feat: bootstrap"], cwd=repo, check=True)
            previous = Path.cwd()
            os.chdir(repo)
            try:
                planned = publish_release._planner("HEAD", "0.1.0", "auto")
            finally:
                os.chdir(previous)
            self.assertEqual(planned["tag"], "v0.1.0")

    def test_registry_only_accepts_exact_buildx_missing_reference(self) -> None:
        image, tag = "registry/app", "v1.2.3"
        missing = subprocess.CompletedProcess([], 1, "", f"ERROR: {image}:{tag}: not found\n")
        with patch("subprocess.run", return_value=missing):
            self.assertIsNone(publish_release.Registry().tag_digest(image, tag))
        unrelated = subprocess.CompletedProcess([], 1, "", "ERROR: credentials not found\n")
        with patch("subprocess.run", return_value=unrelated):
            with self.assertRaisesRegex(publish_release.PublishError, "cannot inspect"):
                publish_release.Registry().tag_digest(image, tag)

    def test_cosign_uses_outfile_and_removes_temporary_public_key(self) -> None:
        commands: list[list[str]] = []

        def run(command: list[str], **_: object) -> str:
            commands.append(command)
            return ""

        with patch.object(publish_release, "_run", side_effect=run):
            publish_release.Cosign().sign_and_verify(f"registry/app@{DIGESTS['app']}")
        public = commands[1]
        self.assertEqual(public[:5], ["cosign", "public-key", "--key", "env://COSIGN_PRIVATE_KEY", "--outfile"])
        self.assertEqual(commands[2][-1], f"registry/app@{DIGESTS['app']}")
        self.assertFalse(Path(public[-1]).exists())

    def test_release_recovers_exactly_one_paginated_draft(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        draft = {"id": 1, "draft": True, "tag_name": "v1.2.3", "target_commitish": SHA, "assets": []}
        with patch.object(client, "_api", side_effect=[None, [draft], []]), patch.object(client, "_asset_manifest") as assets:
            self.assertEqual(client.release("v1.2.3", SHA), draft)
        assets.assert_called_once_with(draft)

    def test_draft_target_mismatch_is_rejected_without_manifest_mutation(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        draft = {"id": 1, "draft": True, "tag_name": "v1.2.3", "target_commitish": "b" * 40, "assets": []}
        with patch.object(client, "_api", side_effect=[None, [draft]]):
            with self.assertRaisesRegex(publish_release.PublishError, "target"):
                client.release("v1.2.3", SHA)

    def test_published_target_mismatch_is_rejected_before_asset_read(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        published = {"id": 1, "draft": False, "tag_name": "v1.2.3", "target_commitish": "b" * 40, "assets": []}
        with patch.object(client, "_api", return_value=published), patch.object(client, "_asset_manifest") as assets:
            with self.assertRaisesRegex(publish_release.PublishError, "target"):
                client.release("v1.2.3", SHA)
        assets.assert_not_called()

    def test_branch_target_metadata_is_accepted_for_draft_and_published_release(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        for draft in (True, False):
            release: dict[str, object] = {"id": 1, "draft": draft, "tag_name": "v1.2.3", "target_commitish": "main", "assets": []}
            with self.subTest(draft=draft), patch.object(client, "_api", return_value=release), patch.object(client, "_asset_manifest") as assets:
                self.assertEqual(client.release("v1.2.3", SHA), release)
                assets.assert_called_once_with(release)

    def test_manifest_download_rejects_spoofed_api_url_before_token_request(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        with patch("urllib.request.build_opener") as opener:
            with self.assertRaisesRegex(publish_release.PublishError, "unsafe"):
                client._download_manifest("https://api.github.com@evil.test/repos/owner/repo/releases/assets/1")
        opener.assert_not_called()

    def test_manifest_redirect_to_current_asset_host_drops_authorization(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        headers = Message()
        headers["Location"] = "https://release-assets.githubusercontent.com/download/manifest"
        redirect = urllib.error.HTTPError(
            "https://api.github.com/repos/owner/repo/releases/assets/1",
            302,
            "redirect",
            headers,
            None,
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"schema_version": 1}'
        opener = MagicMock()
        opener.open.side_effect = [redirect, response]
        with patch("urllib.request.build_opener", return_value=opener):
            self.assertEqual(client._download_manifest("https://api.github.com/repos/owner/repo/releases/assets/1"), {"schema_version": 1})
        redirected_request = opener.open.call_args_list[1].args[0]
        self.assertIsNone(redirected_request.get_header("Authorization"))

    def test_manifest_redirect_to_foreign_host_is_rejected(self) -> None:
        client = publish_release.GitHub("owner/repo", "token")
        headers = Message()
        headers["Location"] = "https://evil.example/manifest"
        redirect = urllib.error.HTTPError("https://api.github.com/repos/owner/repo/releases/assets/1", 302, "redirect", headers, None)
        opener = MagicMock()
        opener.open.side_effect = redirect
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(publish_release.PublishError, "unsafe"):
                client._download_manifest("https://api.github.com/repos/owner/repo/releases/assets/1")
        self.assertEqual(opener.open.call_count, 1)

    def test_stale_main_is_rejected_before_reservation_and_before_aliases(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        github.main_sha = "b" * 40
        with self.assertRaisesRegex(publish_release.PublishError, "main"):
            self.make_publisher(github, registry, signer).publish(SHA, "v1.2.3", DIGESTS)
        self.assertEqual(github.events, [])
        self.assertEqual(registry.tags, {})

    def test_published_release_is_idempotent_only_when_manifest_matches(self) -> None:
        github, registry, signer = FakeGithub(), FakeRegistry(), FakeSigner()
        publisher = self.make_publisher(github, registry, signer)
        publisher.publish(SHA, "v1.2.3", DIGESTS)
        event_count = len(registry.events)
        publisher.publish(SHA, "v1.2.3", DIGESTS)
        self.assertEqual(len(registry.events), event_count)

    def test_conflicting_release_bump_labels_are_rejected(self) -> None:
        github = FakeGithub()
        github.labels = ["release:minor", "release:patch"]
        with self.assertRaisesRegex(publish_release.PublishError, "conflicting"):
            publish_release.release_bump(github, SHA, "auto")
        self.assertEqual(publish_release.release_bump(github, SHA, "major"), "major")


if __name__ == "__main__":
    unittest.main()
