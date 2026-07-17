import hashlib
import hmac
import unittest
from unittest import mock

from ragtime.git_webhooks.models import GitWebhookTarget, GitWebhookTargetType
from ragtime.git_webhooks.verification import (
    WebhookAuthenticationError,
    parse_git_events,
    verify_webhook_request,
)


def _target(
    *,
    provider: str = "generic",
    secret: str = "secret",
    branch: str = "main",
) -> GitWebhookTarget:
    return GitWebhookTarget(
        target_type=GitWebhookTargetType.GIT_INDEX,
        target_id="index-id",
        webhook_id="wh_123",
        secret=secret,
        provider=provider,
        branch=branch,
    )


class GitWebhookVerificationTests(unittest.TestCase):
    def test_github_accepts_sha256_signature(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()

        verify_webhook_request(
            _target(provider="github", secret="secret"),
            {"x-hub-signature-256": signature},
            body,
            None,
        )

    def test_github_rejects_malformed_signature_hex(self) -> None:
        with self.assertRaises(WebhookAuthenticationError):
            verify_webhook_request(
                _target(provider="github", secret="secret"),
                {"x-hub-signature-256": "sha256=not-hex"},
                b"{}",
                None,
            )

    def test_gitlab_rejects_wrong_token(self) -> None:
        with self.assertRaises(WebhookAuthenticationError):
            verify_webhook_request(
                _target(provider="gitlab", secret="secret"),
                {"x-gitlab-token": "wrong"},
                b"{}",
                None,
            )

    def test_gitlab_accepts_matching_token(self) -> None:
        verify_webhook_request(
            _target(provider="gitlab", secret="secret"),
            {"x-gitlab-token": "secret"},
            b"{}",
            None,
        )

    def test_generic_accepts_gitea_raw_hex_signature(self) -> None:
        body = b'{"ref":"refs/heads/main"}'
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-gitea-signature": signature},
            body,
            None,
        )

    def test_generic_accepts_gogs_raw_hex_signature(self) -> None:
        body = b'{"ref":"refs/heads/main"}'
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-gogs-signature": signature},
            body,
            None,
        )

    def test_generic_accepts_github_signature_header(self) -> None:
        body = b'{"ref":"refs/heads/main"}'
        signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()

        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-hub-signature-256": signature},
            body,
            None,
        )

    def test_generic_accepts_gitlab_token_header(self) -> None:
        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-gitlab-token": "secret"},
            b"{}",
            None,
        )

    def test_generic_accepts_uppercase_raw_hex_signature(self) -> None:
        body = b'{"ref":"refs/heads/main"}'
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest().upper()

        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-gitea-signature": signature},
            body,
            None,
        )

    def test_generic_accepts_ragtime_token_header(self) -> None:
        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"x-ragtime-webhook-token": "secret"},
            b"{}",
            None,
        )

    def test_generic_accepts_bearer_auth(self) -> None:
        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {"authorization": "Bearer secret"},
            b"{}",
            None,
        )

    def test_generic_accepts_query_token_fallback(self) -> None:
        verify_webhook_request(
            _target(provider="generic", secret="secret"),
            {},
            b"{}",
            "secret",
        )

    def test_generic_rejects_missing_credentials(self) -> None:
        with self.assertRaises(WebhookAuthenticationError):
            verify_webhook_request(
                _target(provider="generic", secret="secret"),
                {},
                b"{}",
                None,
            )

    def test_generic_rejects_malformed_signature_hex(self) -> None:
        with self.assertRaises(WebhookAuthenticationError):
            verify_webhook_request(
                _target(provider="generic", secret="secret"),
                {"x-gitea-signature": "not-hex"},
                b"{}",
                None,
            )

    def test_compare_digest_is_used_for_every_credential_form(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        sha_signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        raw_signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        uppercase_raw_signature = raw_signature.upper()
        calls: list[tuple[str, str]] = []

        def _compare_digest(left: str, right: str) -> bool:
            calls.append((left, right))
            return left == right

        with mock.patch("ragtime.git_webhooks.verification.hmac.compare_digest", side_effect=_compare_digest):
            verify_webhook_request(_target(provider="github"), {"x-hub-signature-256": sha_signature}, body, None)
            verify_webhook_request(_target(provider="gitlab"), {"x-gitlab-token": "secret"}, body, None)
            verify_webhook_request(_target(provider="generic"), {"x-gitea-signature": raw_signature}, body, None)
            verify_webhook_request(_target(provider="generic"), {"x-gogs-signature": uppercase_raw_signature}, body, None)
            verify_webhook_request(_target(provider="generic"), {"x-hub-signature-256": sha_signature}, body, None)
            verify_webhook_request(_target(provider="generic"), {"x-gitlab-token": "secret"}, body, None)
            verify_webhook_request(_target(provider="generic"), {"x-ragtime-webhook-token": "secret"}, body, None)
            verify_webhook_request(_target(provider="generic"), {"authorization": "Bearer secret"}, body, None)
            verify_webhook_request(_target(provider="generic"), {}, body, "secret")

        self.assertEqual(
            calls,
            [
                (sha_signature, sha_signature),
                ("secret", "secret"),
                (raw_signature, raw_signature),
                (raw_signature, raw_signature),
                (sha_signature, sha_signature),
                ("secret", "secret"),
                ("secret", "secret"),
                ("secret", "secret"),
                ("secret", "secret"),
            ],
        )


class GitWebhookParsingTests(unittest.TestCase):
    def test_parses_ref_push(self) -> None:
        events = parse_git_events(
            "github",
            {"x-github-event": "push", "x-github-delivery": "delivery-1"},
            {"ref": "refs/heads/main", "after": "abc123"},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "push")
        self.assertIsNone(event.message)
        self.assertEqual(event.branch, "main")
        self.assertEqual(event.head_commit, "abc123")
        self.assertEqual(event.provider_delivery_id, "delivery-1")
        self.assertEqual(event.event_name, "push")

    def test_parses_gitlab_push_headers(self) -> None:
        events = parse_git_events(
            "gitlab",
            {"x-gitlab-event": "Push Hook", "x-gitlab-event-uuid": "delivery-2"},
            {"ref": "refs/heads/release", "after": "def456"},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "push")
        self.assertEqual(event.branch, "release")
        self.assertEqual(event.head_commit, "def456")
        self.assertEqual(event.provider_delivery_id, "delivery-2")
        self.assertEqual(event.event_name, "Push Hook")

    def test_parses_bitbucket_style_push(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-key": "repo:push", "x-request-uuid": "delivery-3"},
            {"push": {"changes": [{"new": {"type": "branch", "name": "develop", "target": {"hash": "ghi789"}}}]}},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "push")
        self.assertEqual(event.branch, "develop")
        self.assertEqual(event.head_commit, "ghi789")
        self.assertEqual(event.provider_delivery_id, "delivery-3")
        self.assertEqual(event.event_name, "repo:push")

    def test_parses_multiple_bitbucket_branch_changes(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-key": "repo:push", "x-request-uuid": "delivery-4"},
            {
                "push": {
                    "changes": [
                        {"new": {"type": "branch", "name": "develop", "target": {"hash": "111"}}},
                        {"new": {"type": "branch", "name": "main", "target": {"hash": "222"}}},
                    ]
                }
            },
        )

        self.assertEqual([event.branch for event in events], ["develop", "main"])
        self.assertTrue(all(event.kind == "push" for event in events))

    def test_parses_multiple_azure_branch_updates(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-type": "git.push", "x-request-uuid": "delivery-5"},
            {
                "resource": {
                    "refUpdates": [
                        {"name": "refs/heads/main", "newObjectId": "aaa"},
                        {"name": "refs/heads/develop", "newObjectId": "bbb"},
                    ]
                }
            },
        )

        self.assertEqual([event.branch for event in events], ["main", "develop"])
        self.assertEqual([event.head_commit for event in events], ["aaa", "bbb"])
        self.assertTrue(all(event.kind == "push" for event in events))

    def test_tag_refs_are_ignored(self) -> None:
        events = parse_git_events(
            "github",
            {"x-github-event": "push", "x-github-delivery": "delivery-6"},
            {"ref": "refs/tags/v1.0.0", "after": "abc123"},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertIsNone(event.branch)
        self.assertEqual(event.message, "Push branch could not be determined.")

    def test_deleted_branch_is_ignored(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-key": "repo:push", "x-request-uuid": "delivery-7"},
            {"push": {"changes": [{"closed": True, "new": None, "old": {"type": "branch", "name": "main", "target": {"hash": "deadbeef"}}}]}},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertEqual(event.branch, "main")
        self.assertEqual(event.message, "Deleted branches are ignored.")

    def test_non_push_event_is_ignored(self) -> None:
        events = parse_git_events(
            "github",
            {"x-github-event": "ping", "x-github-delivery": "delivery-8"},
            {"zen": "keep it logically awesome"},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertEqual(event.message, "Non-push webhook events are ignored.")
        self.assertEqual(event.event_name, "ping")

    def test_unparseable_payload_is_ignored(self) -> None:
        events = parse_git_events("generic", {}, {"repository": {"name": "repo"}})

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertEqual(event.message, "Push branch could not be determined.")
        self.assertIsNone(event.branch)

    def test_tag_only_bitbucket_payload_is_ignored(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-key": "repo:push", "x-request-uuid": "delivery-9"},
            {"push": {"changes": [{"new": {"type": "tag", "name": "v1.0.0", "target": {"hash": "abc123"}}}]}},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertEqual(event.message, "Push branch could not be determined.")

    def test_tag_only_azure_payload_is_ignored(self) -> None:
        events = parse_git_events(
            "generic",
            {"x-event-type": "git.push", "x-request-uuid": "delivery-10"},
            {"resource": {"refUpdates": [{"name": "refs/tags/v1.0.0", "newObjectId": "abc123"}]}},
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, "ignored")
        self.assertEqual(event.message, "Push branch could not be determined.")


if __name__ == "__main__":
    unittest.main()
