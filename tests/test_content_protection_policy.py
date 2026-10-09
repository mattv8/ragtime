import unittest

from pydantic import ValidationError

from ragtime.content_protection.models import AccessLevel, ContentCategory, ContentProtectionConfig, ProtectionContext, Requirement, UserOverride
from ragtime.content_protection.policy import resolve_required
from ragtime.content_protection.store import _is_recognizable_legacy_config


class ContentProtectionPolicyTests(unittest.TestCase):
    def test_v2_defaults_have_protected_ungrantable_rule_override(self) -> None:
        config = ContentProtectionConfig()
        self.assertEqual(config.schema_version, 2)
        self.assertTrue(next(category for category in config.categories if category.id == "rule_override").system)
        with self.assertRaises(ValidationError):
            ContentProtectionConfig(access_levels=[AccessLevel(id="standard", name="Standard", granted_category_ids=["rule_override"])])

    def test_system_category_cannot_be_edited_or_replaced(self) -> None:
        categories = ContentProtectionConfig().categories
        categories[-1] = ContentCategory(id="rule_override", name="Changed", description="Changed", denial_message="Changed", system=True)
        with self.assertRaises(ValidationError):
            ContentProtectionConfig(categories=categories)

    def test_only_strictly_validated_v1_documents_can_reset(self) -> None:
        legacy = {"enabled": True, "profiles": [{"id": "standard", "name": "Standard", "level": 0, "scope": "ordinary"}], "group_profiles": []}
        self.assertTrue(_is_recognizable_legacy_config(legacy))
        self.assertFalse(_is_recognizable_legacy_config({**legacy, "profiles": [{"id": "standard"}]}))
        self.assertFalse(_is_recognizable_legacy_config({**legacy, "schema_version": 3}))

    def test_master_off_never_requires_classification(self) -> None:
        required, provenance = resolve_required(ContentProtectionConfig(enabled=False), ProtectionContext(user_id="u"))
        self.assertFalse(required)
        self.assertEqual(provenance, "master_off")

    def test_never_override_does_not_exempt_another_audience(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            coverage_mode="selected_scopes",
            requirements=[Requirement(scope_kind="surface", scope_key="chat", mode="require")],
            user_overrides=[UserOverride(user_id="caller", mode="never_classify")],
        )
        self.assertTrue(resolve_required(config, ProtectionContext(user_id="caller", audience_user_ids=("recipient",)))[0])

    def test_always_override_requires_even_without_scope_match(self) -> None:
        config = ContentProtectionConfig(enabled=True, coverage_mode="selected_scopes", user_overrides=[UserOverride(user_id="u", mode="always_classify")])
        self.assertEqual(resolve_required(config, ProtectionContext(user_id="u")), (True, "user_override:always_classify"))
