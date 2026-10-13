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

    def test_skip_group_exempts_inherited_identity_when_no_other_slot_requires(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            coverage_mode="all_supported_traffic",
            requirements=[Requirement(scope_kind="group", scope_key="skip", mode="skip")],
        )

        self.assertEqual(
            resolve_required(config, ProtectionContext(user_id="u"), {"u": {"skip"}}, {"u"}),
            (False, "group_override:skip"),
        )

    def test_require_group_wins_over_skip_group(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            coverage_mode="selected_scopes",
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
            ],
        )

        self.assertEqual(
            resolve_required(config, ProtectionContext(user_id="u"), {"u": {"skip", "require"}}, {"u"}),
            (True, "group_override:require"),
        )

    def test_public_slot_prevents_personal_never_exemption(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            user_overrides=[UserOverride(user_id="u", mode="never_classify")],
        )

        self.assertTrue(resolve_required(config, ProtectionContext(user_id="u", public=True), {}, {"u"})[0])

    def test_personal_never_does_not_exempt_another_required_audience(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            coverage_mode="selected_scopes",
            requirements=[Requirement(scope_kind="group", scope_key="require", mode="require")],
            user_overrides=[UserOverride(user_id="u", mode="never_classify")],
        )

        self.assertTrue(resolve_required(config, ProtectionContext(user_id="u", audience_user_ids=("v",)), {"u": {"require"}, "v": {"require"}}, {"u", "v"})[0])

    def test_none_audience_slot_requires_coverage_without_treating_absent_primary_as_slot(self) -> None:
        config = ContentProtectionConfig(enabled=True, requirements=[Requirement(scope_kind="group", scope_key="skip", mode="skip")])

        self.assertTrue(resolve_required(config, ProtectionContext(audience_user_ids=(None,)), {}, set())[0])
        self.assertFalse(resolve_required(config, ProtectionContext(audience_user_ids=("u",)), {"u": {"skip"}}, {"u"})[0])

    def test_skip_provenance_is_not_used_when_uncovered_inherited_slot_remains(self) -> None:
        config = ContentProtectionConfig(
            enabled=True,
            coverage_mode="selected_scopes",
            requirements=[Requirement(scope_kind="group", scope_key="skip", mode="skip")],
        )

        self.assertEqual(resolve_required(config, ProtectionContext(user_id="u", audience_user_ids=("v",)), {"u": {"skip"}, "v": set()}, {"u", "v"}), (False, "no_requirement"))

    def test_skip_mode_is_rejected_for_non_group_scope(self) -> None:
        with self.assertRaises(ValidationError):
            Requirement(scope_kind="surface", scope_key="chat", mode="skip")
