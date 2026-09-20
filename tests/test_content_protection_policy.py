import unittest

from ragtime.content_protection.models import ContentProtectionConfig, ProtectionContext, Requirement, UserOverride
from ragtime.content_protection.policy import resolve_required


class ContentProtectionPolicyTests(unittest.TestCase):
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
