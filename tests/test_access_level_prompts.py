"""Effective audience guidance must survive both hosted and external projection."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

from ragtime.content_protection.models import ProtectionContext
from ragtime.core import openrouter
from ragtime.rag import prompts
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service
from ragtime.userspace.instruction_bundle import build_instruction_bundle


def guidance_snapshot():
    return {
        "share_with_assistant": True,
        "policy_revision": 7,
        "guidance_revision": "effective-standard",
        "granted_category_ids": ["operational"],
        "categories": [
            {"id": "operational", "name": "Operations", "description": "Nonpublic operational records"},
            {"id": "company_finance", "name": "Company finance", "description": "Nonpublic financial records"},
        ],
        "access_levels": [[{"name": "Standard"}], [{"name": "Finance"}]],
        "guidance": ["Explain operational procedures only."],
    }


def test_guidance_projects_effective_grants_without_audience_identifiers():
    snapshot = guidance_snapshot()
    snapshot["user_id"] = "private-user-id"
    snapshot["groups"] = ["private-group-id"]
    fragment = prompts.build_access_level_prompt_fragment(snapshot)
    assert "Operations" in fragment
    assert "Company finance" in fragment
    assert "Explain operational procedures only." in fragment
    assert "private-user-id" not in fragment
    assert "private-group-id" not in fragment
    assert "takes precedence over" in fragment


def test_advisory_switch_does_not_depend_on_enforcement():
    snapshot = guidance_snapshot()
    snapshot["enabled"] = False
    assert prompts.build_access_level_prompt_fragment(snapshot)
    snapshot["share_with_assistant"] = False
    assert prompts.build_access_level_prompt_fragment(snapshot) == ""
    assert prompts.build_access_level_prompt_fragment(None) == ""


def test_bundle_revision_changes_on_effective_grant_change_without_policy_revision():
    context = {"access_guidance": guidance_snapshot()}
    before = build_instruction_bundle(context)
    changed = deepcopy(context)
    changed["access_guidance"]["granted_category_ids"] = []
    changed["access_guidance"]["guidance_revision"] = "revoked-membership"
    after = build_instruction_bundle(changed)
    assert "Explain operational procedures only." in before["system_instructions"]["access_level"]
    assert before["context_revision"] != after["context_revision"]


def test_jev_decision_models_cannot_appear_as_chat_models():
    for model in ("typesafe/jev-1.13", "~typesafe/jev-latest", "typesafe/jev-1.13-20260917"):
        assert not openrouter.supports_chat({"id": model, "architecture": {"input_modalities": ["text"]}})
    assert openrouter.supports_chat({"id": "openai/gpt-4.1"})
    # Jev Router is a chat routing product, not the Jev classifier endpoint.
    assert openrouter.supports_chat({"id": "typesafe/jev-router"})


class ExternalDevelopmentGuidanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_guidance_uses_authenticated_development_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="authenticated-user", is_admin=False)
        fake_db = SimpleNamespace(
            user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(username="local:user", displayName="User", role="user")))
        )
        guidance = mock.AsyncMock(side_effect=RuntimeError("stop after guidance boundary"))

        with (
            mock.patch(
                "ragtime.userspace.development_service.planning_service.get_workspace_context",
                new=mock.AsyncMock(return_value={}),
            ),
            mock.patch("ragtime.userspace.development_service.get_db", new=mock.AsyncMock(return_value=fake_db)),
            mock.patch(
                "ragtime.userspace.development_service.content_protection_service.access_guidance",
                new=guidance,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after guidance boundary"):
                await development_service._context(principal, "workspace-1")

        context = guidance.await_args.args[0]
        self.assertIsInstance(context, ProtectionContext)
        self.assertEqual(context.user_id, "authenticated-user")
        self.assertEqual(context.audience_user_ids, ("authenticated-user",))
        self.assertEqual(context.surface, "workspace_development")
        self.assertEqual(context.resource_id, "workspace-1")
        self.assertEqual(context.baseline, "user")
