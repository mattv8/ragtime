import unittest
from unittest import mock

from ragtime.content_protection.models import ContentProtectionError


class ContentProtectionExternalRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_outbound_denial_advises_one_safe_alternative_without_counter(self) -> None:
        from ragtime.content_protection import external

        denied = ContentProtectionError(
            "content_denied",
            "request-1",
            reason="Restricted result.",
            reason_code="restricted_content",
        )
        core = mock.Mock()
        core.ProtectionContext.side_effect = lambda **kwargs: kwargs
        core.authorize_content = mock.AsyncMock(side_effect=denied)

        with mock.patch.object(external, "_core", return_value=core):
            with self.assertRaises(ContentProtectionError) as raised:
                await external.authorize_external_content(
                    {"content": "restricted bytes"},
                    direction="outbound",
                    surface="mcp",
                    execution_completed=True,
                )

        detail = external.public_error_detail(raised.exception)
        self.assertEqual(raised.exception.code, "content_denied")
        self.assertEqual(raised.exception.request_id, "request-1")
        self.assertEqual(detail["reason"], "Restricted result.")
        self.assertEqual(detail["reason_code"], "restricted_content")
        self.assertEqual(detail["recovery_action"], "produce_allowed_alternative")
        self.assertEqual(detail["execution_status"], "completed_response_withheld")
        self.assertNotIn("attempts_remaining", detail)
        self.assertNotIn("restricted bytes", str(detail))

    async def test_inbound_and_operational_errors_never_advertise_recovery(self) -> None:
        from ragtime.content_protection import external

        denied = ContentProtectionError("content_denied", "request-2", reason="Restricted input.")
        core = mock.Mock()
        core.ProtectionContext.side_effect = lambda **kwargs: kwargs
        core.authorize_content = mock.AsyncMock(side_effect=denied)

        with mock.patch.object(external, "_core", return_value=core):
            with self.assertRaises(ContentProtectionError) as raised:
                await external.authorize_external_content(
                    {"content": "restricted input"},
                    direction="inbound",
                    surface="mcp",
                    execution_completed=True,
                )

        self.assertNotIn("recovery_action", external.public_error_detail(raised.exception))
        self.assertNotIn(
            "recovery_action",
            external.public_error_detail(ContentProtectionError("classifier_unavailable", "request-3")),
        )

    async def test_external_detail_strips_hosted_attempt_counter(self) -> None:
        from ragtime.content_protection.external import public_error_detail

        class _HostedError(Exception):
            def public_detail(self) -> dict[str, str]:
                return {
                    "code": "content_denied",
                    "message": "Restricted.",
                    "reason": "Restricted.",
                    "next_step": "Use permitted information.",
                    "request_id": "request-4",
                    "recovery_action": "produce_allowed_alternative",
                    "attempts_remaining": "1",
                }

        detail = public_error_detail(_HostedError())

        self.assertEqual(detail["recovery_action"], "produce_allowed_alternative")
        self.assertNotIn("attempts_remaining", detail)


if __name__ == "__main__":
    unittest.main()
