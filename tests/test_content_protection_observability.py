import unittest
from unittest import mock

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, ProtectionContext


class ContentProtectionObservabilityTests(unittest.IsolatedAsyncioTestCase):
    def _policy(self):
        return service._ResolvedPolicy(True, "all", {"u"}, {"u": set()}, {"u": None}, [[{"id": "standard", "scope": "ordinary"}]])

    async def test_finalized_log_is_payload_free_for_allowed_and_denied_content(self) -> None:
        config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")
        canary = "CANARY-SECRET-DO-NOT-LOG"
        for verdict, expected in (("allow", "permitted"), ("deny", "denied")):
            service._decision_cache.clear()
            with (
                mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
                mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
                mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
                mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": verdict, "reason_code": "restricted_content"})),
                mock.patch.object(service, "_audit", mock.AsyncMock()),
                mock.patch.object(service.logger, "info") as log,
            ):
                if verdict == "deny":
                    with self.assertRaises(ContentProtectionError):
                        await service.authorize_content(canary, direction="inbound", context=ProtectionContext(user_id="u"))
                else:
                    await service.authorize_content(canary, direction="inbound", context=ProtectionContext(user_id="u"))
            log_call = log.call_args
            assert log_call is not None
            payload = log_call.kwargs["extra"]["content_protection"]
            self.assertEqual(payload["outcome"], expected)
            self.assertNotIn(canary, str(payload))
            self.assertEqual(
                set(payload), {"direction", "outcome", "initial_ms", "queue_ms", "provider_ms", "release_recheck_ms", "audit_ms", "total_ms", "boundary_count"}
            )

    async def test_release_recheck_failure_is_audited(self) -> None:
        config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")
        audit = mock.AsyncMock()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(side_effect=[config, RuntimeError("db unavailable")])),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "allow", "reason_code": "permitted"})),
            mock.patch.object(service, "_audit", audit),
        ):
            with self.assertRaisesRegex(ContentProtectionError, "classifier_unavailable"):
                await service.authorize_content("safe", direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(audit.await_count, 1)
        audit_call = audit.await_args
        assert audit_call is not None
        self.assertEqual(audit_call.args[1]["stage"], "release_recheck")
