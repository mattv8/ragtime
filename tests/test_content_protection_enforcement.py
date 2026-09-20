import unittest
from typing import Literal
from unittest import mock

from ragtime.content_protection import service
from ragtime.content_protection.models import (
    ContentProtectionConfig,
    ContentProtectionError,
    GroupProfile,
    Profile,
    ProtectionContext,
    Requirement,
    UserOverride,
)


class ContentProtectionEnforcementTests(unittest.IsolatedAsyncioTestCase):
    def _config(
        self,
        *,
        revision: int = 0,
        coverage_mode: Literal["all_supported_traffic", "selected_scopes"] = "all_supported_traffic",
        profiles: list[Profile] | None = None,
        group_profiles: list[GroupProfile] | None = None,
        requirements: list[Requirement] | None = None,
        user_overrides: list[UserOverride] | None = None,
    ) -> ContentProtectionConfig:
        return ContentProtectionConfig(
            enabled=True,
            classifier_model="openai::classifier",
            revision=revision,
            coverage_mode=coverage_mode,
            profiles=profiles if profiles is not None else [Profile(id="standard", name="Standard", level=0, scope="ordinary")],
            group_profiles=group_profiles if group_profiles is not None else [],
            requirements=requirements if requirements is not None else [],
            user_overrides=user_overrides if user_overrides is not None else [],
        )

    async def test_group_requirement_and_standard_baseline_are_sent_as_audience_constraints(self) -> None:
        config = self._config(
            coverage_mode="selected_scopes",
            profiles=[Profile(id="standard", name="Standard", level=0, scope="operational"), Profile(id="finance", name="Finance", level=1, scope="finance")],
            group_profiles=[GroupProfile(group_id="finance-group", profile_id="finance")],
            requirements=[Requirement(scope_kind="group", scope_key="finance-group", mode="require")],
        )
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(side_effect=[config, config])),
            mock.patch.object(service, "resolve_identities", mock.AsyncMock(return_value=({"u"}, {"u": {"finance-group"}}, {"u": None}))),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "allow", "reason_code": "permitted"})) as classify,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
        ):
            await service.authorize_content("report", direction="outbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(classify.await_count, 1)
        await_args = classify.await_args
        assert await_args is not None
        scopes = await_args.args[1]["audience_constraints"]
        self.assertEqual({scope["id"] for scope in scopes[0]}, {"standard", "finance"})

    async def test_exact_cache_binds_supporting_context_and_rechecks_membership(self) -> None:
        config = self._config()
        identities = mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(side_effect=[config, config, config, config])),
            mock.patch.object(service, "resolve_identities", identities),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "allow", "reason_code": "permitted"})) as classify,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
        ):
            await service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u"), supporting_context={"a": 1})
            await service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u"), supporting_context={"a": 2})
        self.assertEqual(classify.await_count, 2)
        self.assertGreaterEqual(identities.await_count, 4)

    async def test_denial_terminates_nested_turn_before_subsequent_provider_call(self) -> None:
        config = self._config()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "resolve_identities", mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))),
            mock.patch.object(
                service,
                "classify",
                mock.AsyncMock(return_value={"verdict": "deny", "reason_code": "restricted_content", "reason": "This request is outside the allowed policy."}),
            ) as classify,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            service.protection_context(ProtectionContext(user_id="u")),
        ):
            with self.assertRaisesRegex(ContentProtectionError, "content_denied") as raised:
                await service.authorize_content("first", direction="inbound")
            with self.assertRaisesRegex(ContentProtectionError, "content_denied"):
                await service.authorize_content("second", direction="outbound")
        self.assertEqual(classify.await_count, 1)
        self.assertEqual(raised.exception.public_detail()["reason"], "This request is outside the allowed policy.")
        classify_call = classify.await_args
        assert classify_call is not None
        self.assertTrue(classify_call.kwargs["include_reason"])

    async def test_uncertain_denial_uses_honest_fallback_without_auditing_reason(self) -> None:
        config = self._config()
        audit = mock.AsyncMock()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "resolve_identities", mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "deny", "reason_code": "uncertain"})) as classify,
            mock.patch.object(service, "_audit", audit),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content("candidate", direction="inbound", context=ProtectionContext(user_id="u"))

        detail = raised.exception.public_detail()
        self.assertEqual(raised.exception.code, "content_denied")
        self.assertEqual(detail["reason_code"], "uncertain")
        self.assertEqual(detail["reason"], "This request could not be safely classified under the access policy.")
        classify_call = classify.await_args
        audit_call = audit.await_args
        assert classify_call is not None and audit_call is not None
        self.assertTrue(classify_call.kwargs["include_reason"])
        self.assertNotIn("reason", audit_call.args[1])

    async def test_allow_discards_optional_reason_before_caching_and_audit(self) -> None:
        service._decision_cache.clear()
        config = self._config()
        candidate = {"text": "ordinary"}
        audit = mock.AsyncMock()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "resolve_identities", mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "allow", "reason_code": "permitted", "reason": "Not retained."})),
            mock.patch.object(service, "_audit", audit),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
        ):
            await service.authorize_content(candidate, direction="inbound", context=ProtectionContext(user_id="u"))

        self.assertEqual(candidate, {"text": "ordinary"})
        self.assertTrue(service._decision_cache)
        self.assertTrue(all("reason" not in cached[1] for cached in service._decision_cache.values()))
        audit_call = audit.await_args
        assert audit_call is not None
        self.assertNotIn("reason", audit_call.args[1])

    async def test_revision_change_rechecks_once_then_fails_closed_when_it_changes_again(self) -> None:
        first = self._config(revision=1)
        second = self._config(revision=2)
        third = self._config(revision=3)
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(side_effect=[first, second, second, third])),
            mock.patch.object(service, "resolve_identities", mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))),
            mock.patch.object(service, "classify", mock.AsyncMock(return_value={"verdict": "allow", "reason_code": "permitted"})),
            mock.patch.object(service, "_audit", mock.AsyncMock()),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
        ):
            with self.assertRaisesRegex(ContentProtectionError, "policy_changed"):
                await service.authorize_content("candidate", direction="outbound", context=ProtectionContext(user_id="u"))
