import unittest

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionError, ProtectionContext


class ContentProtectionRecoveryTests(unittest.TestCase):
    def test_eligible_error_has_compatible_public_recovery_metadata(self) -> None:
        error = ContentProtectionError(
            "content_denied",
            "r1",
            recovery_eligible=True,
            execution_status="completed_response_withheld",
            attempts_remaining=1,
        )
        self.assertEqual(error.public_detail()["recovery_action"], "produce_allowed_alternative")
        self.assertEqual(error.public_detail()["attempts_remaining"], "1")

    def test_recovery_successor_keeps_record_but_old_attempt_stays_terminal(self) -> None:
        with service.protection_context(ProtectionContext(user_id="u")):
            original = service._turn.get()
            assert original is not None
            error = ContentProtectionError("content_denied", "r1", recovery_eligible=True)
            original.terminal = error
            successor = service.begin_recovery(error)
            self.assertIsNot(successor, original)
            self.assertIs(successor.record, original.record)
            self.assertIs(original.terminal, error)
            self.assertTrue(successor.record.recovery_consumed)
            with self.assertRaises(ContentProtectionError):
                service.begin_recovery(error)

    def test_recovery_context_restores_terminal_attempt_on_failure(self) -> None:
        with service.protection_context(ProtectionContext(user_id="u")):
            original = service._turn.get()
            assert original is not None
            error = ContentProtectionError("content_denied", "r1", recovery_eligible=True)
            original.terminal = error
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with service.recovery_attempt(error) as successor:
                    self.assertIsNone(successor.terminal)
                    raise RuntimeError("failed")
            self.assertIs(service.terminal_error(), error)
            self.assertTrue(original.record.recovery_consumed)

    def test_consumed_recovery_does_not_advertise_another_attempt(self) -> None:
        error = ContentProtectionError("content_denied", "r2", recovery_eligible=False)
        self.assertNotIn("recovery_action", error.public_detail())
        self.assertNotIn("attempts_remaining", error.public_detail())
