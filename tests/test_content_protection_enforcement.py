import unittest
from unittest import mock

from ragtime.content_protection import service
from ragtime.content_protection.models import AccessLevel, ContentProtectionConfig, ContentProtectionError, GroupAccessLevel, ProtectionContext


def _detection(config: ContentProtectionConfig, **probabilities: float) -> dict[str, object]:
    return {
        "probabilities": {category.id: probabilities.get(category.id, 0.0) for category in config.categories},
        "model": "test",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "transport": "test",
        "cache_hit": False,
    }


class ContentProtectionEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def test_mapped_level_does_not_inherit_default_grants(self) -> None:
        config = ContentProtectionConfig(
            access_levels=[
                AccessLevel(id="standard", name="Standard", granted_category_ids=["operational"]),
                AccessLevel(id="public_only", name="Public only", granted_category_ids=[]),
            ],
            group_access_levels=[GroupAccessLevel(group_id="public", access_level_id="public_only")],
        )
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=({"u"}, {"u": {"public"}}, {"u": None}))):
            resolved = await service._resolve(config, ProtectionContext(user_id="u"))
        self.assertEqual(resolved.granted_category_ids, set())

    async def test_service_baseline_without_audience_uses_default_level(self) -> None:
        config = ContentProtectionConfig()
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=(set(), {}, {}))):
            resolved = await service._resolve(config, ProtectionContext(baseline="service"))
        self.assertEqual(resolved.granted_category_ids, {"operational"})

    async def test_explicit_none_and_unknown_audience_contribute_empty_grants(self) -> None:
        config = ContentProtectionConfig()
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))):
            none_recipient = await service._resolve(config, ProtectionContext(user_id="u", audience_user_ids=(None,)))
            unknown_recipient = await service._resolve(config, ProtectionContext(user_id="u", audience_user_ids=("disabled",)))
        self.assertEqual(none_recipient.granted_category_ids, set())
        self.assertEqual(unknown_recipient.granted_category_ids, set())

    async def test_mapped_levels_union_and_audience_intersection(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            access_levels=[
                AccessLevel(id="standard", name="Standard", granted_category_ids=[]),
                AccessLevel(id="finance", name="Finance", granted_category_ids=["company_finance"]),
            ],
            group_access_levels=[GroupAccessLevel(group_id="finance", access_level_id="finance")],
        )
        with mock.patch.object(
            service, "resolve_identities", new=mock.AsyncMock(return_value=({"a", "b"}, {"a": {"finance"}, "b": set()}, {"a": None, "b": None}))
        ):
            resolved = await service._resolve(config, ProtectionContext(user_id="a", audience_user_ids=("b",)))
        self.assertEqual(resolved.granted_category_ids, set())

    async def test_missing_audience_identity_contributes_no_grants(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=({"a"}, {"a": set()}, {"a": None}))):
            resolved = await service._resolve(config, ProtectionContext(user_id="a", audience_user_ids=("missing",)))
        self.assertEqual(resolved.granted_category_ids, set())

    async def test_ungranted_category_probability_denies_with_admin_message(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        with (
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=({"u"}, {"u": set()}, {"u": None}))),
            mock.patch.object(service, "_provider_settings_identity", new=mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "detect", new=mock.AsyncMock(return_value=_detection(config, company_finance=0.25))),
            mock.patch.object(service, "_audit", new=mock.AsyncMock()),
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content("financial report", direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(raised.exception.reason_code, "restricted_content")
        self.assertEqual(raised.exception.reason, "Company financial information is not available for this audience.")

    def test_thresholds_are_inclusive_and_rule_override_is_ungrantable(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        self.assertEqual(
            service._authorize_probabilities(config, {category.id: 0.0 for category in config.categories} | {"rule_override": 0.25}, {"operational"})[
                "verdict"
            ],
            "deny",
        )
        with self.assertRaises(ValueError):
            ContentProtectionConfig(access_levels=[AccessLevel(id="standard", name="Standard", granted_category_ids=["rule_override"])])
