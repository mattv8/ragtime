import unittest
from typing import cast
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service


class _Denied(Exception):
    def public_detail(self):
        return {
            "code": "content_denied",
            "message": "This request conflicts with the access policy. Try rephrasing your question.",
            "reason": "This request conflicts with the access policy.",
            "next_step": "Try rephrasing your question.",
            "request_id": "request-1",
            "reason_code": "profile_mismatch",
        }


class ContentProtectionExternalTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_error_detail_preserves_refusal_reason_and_next_step(self) -> None:
        from ragtime.content_protection.external import public_error_detail

        detail = public_error_detail(_Denied("restricted body"))

        self.assertEqual(detail["reason"], "This request conflicts with the access policy.")
        self.assertEqual(detail["next_step"], "Try rephrasing your question.")
        self.assertNotIn("restricted body", str(detail))

    async def test_development_input_denial_prevents_operation_execution(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        with (
            mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock(side_effect=_Denied())),
            mock.patch.object(development_service, "_execute_unprotected", new=mock.AsyncMock()) as execute,
        ):
            with self.assertRaises(Exception) as raised:
                await development_service.execute(principal, "workspace-1", "file_read", {"path": "secret.txt"})
        self.assertEqual(cast(dict[str, str], cast(HTTPException, raised.exception).detail)["code"], "content_denied")
        execute.assert_not_awaited()

    async def test_development_output_denial_is_not_released(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        authorize = mock.AsyncMock(side_effect=[None, _Denied()])
        with (
            mock.patch("ragtime.userspace.development_service.authorize_external_content", new=authorize),
            mock.patch.object(development_service, "_execute_unprotected", new=mock.AsyncMock(return_value={"content": "secret"})),
        ):
            with self.assertRaises(Exception) as raised:
                await development_service.execute(principal, "workspace-1", "file_read", {"path": "secret.txt"})
        self.assertEqual(cast(dict[str, str], cast(HTTPException, raised.exception).detail)["code"], "content_denied")

    async def test_user_identity_is_preserved_for_override_resolution(self) -> None:
        from ragtime.content_protection import external

        fake_core = mock.Mock()
        fake_core.ProtectionContext.side_effect = lambda **kwargs: kwargs
        with mock.patch.object(external, "_core", return_value=fake_core):
            context = external.context_for_principal(DevelopmentPrincipal(user_id="user-1", is_admin=False), surface="development")
        self.assertEqual(context["user_id"], "user-1")
        self.assertEqual(context["baseline"], "user")

    async def test_service_baseline_has_no_user_override_identity(self) -> None:
        from ragtime.content_protection import external

        fake_core = mock.Mock()
        fake_core.ProtectionContext.side_effect = lambda **kwargs: kwargs
        with mock.patch.object(external, "_core", return_value=fake_core):
            context = external.context_for_principal(None, surface="component", baseline="service")
        self.assertIsNone(context["user_id"])
        self.assertEqual(context["baseline"], "service")


if __name__ == "__main__":
    unittest.main()
