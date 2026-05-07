import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    fake_prompts_module.build_workspace_scm_setup_prompt = lambda *args, **kwargs: ""
    fake_rag_package.prompts = fake_prompts_module
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.models import CloudOAuthCallbackRequest, UserCloudOAuthAccount
from ragtime.userspace.service import UserSpaceService


class _CloudOAuthService(UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.callback_request: CloudOAuthCallbackRequest | None = None

    async def complete_user_cloud_oauth(
        self,
        user_id: str,
        request: CloudOAuthCallbackRequest,
    ) -> UserCloudOAuthAccount:
        self.callback_request = request
        now = datetime(2026, 5, 7, tzinfo=timezone.utc)
        return UserCloudOAuthAccount(
            id="account-1",
            provider=request.provider,
            account_email="user@example.com",
            created_at=now,
            updated_at=now,
        )


class UserSpaceCloudOAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_callback_uses_redirect_uri_from_state(self) -> None:
        service = _CloudOAuthService()
        service._cloud_oauth_states["state-1"] = {
            "user_id": "user-1",
            "provider": "google_drive",
            "redirect_uri": "http://localhost:8001/indexes/userspace/cloud-oauth/callback",
            "workspace_id": None,
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        }

        await service.complete_user_cloud_oauth_browser_callback(
            "user-1",
            code="code-1",
            state="state-1",
        )

        self.assertIsNotNone(service.callback_request)
        self.assertIsNone(service.callback_request.redirect_uri)


if __name__ == "__main__":
    unittest.main()
