"""Explicit default-off policy fixture for isolated legacy unit tests.

These tests already replace their repository/database boundaries. Supply the
new configuration boundary too, without bypassing the guard itself or changing
the enabled-policy integration tests.
"""

from unittest import TestCase, mock

from ragtime.content_protection.models import ContentProtectionConfig


def use_disabled_content_protection(test_case: TestCase) -> None:
    policy = mock.patch(
        "ragtime.content_protection.service.load_config",
        new=mock.AsyncMock(return_value=ContentProtectionConfig(enabled=False)),
    )
    policy.start()
    test_case.addCleanup(policy.stop)
