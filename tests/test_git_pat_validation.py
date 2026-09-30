import asyncio
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import unquote, urlsplit

import httpx
from fastapi import HTTPException

from ragtime.core import git
from ragtime.indexer import routes
from ragtime.indexer.file_utils import build_authenticated_git_url
from ragtime.indexer.models import FetchBranchesRequest
from ragtime.indexer.repository import IndexerRepository


class GitPatValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_branches_uses_git_read_probe_and_parses_slash_refs(self) -> None:
        process = SimpleNamespace(returncode=0, communicate=mock.AsyncMock(return_value=(b"a\trefs/heads/main\nb\trefs/heads/release/2026\n", b"")))
        with mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)):
            branches, error = await git.fetch_branches("https://github.com/acme/repo.git", "github_pat_safe")
        self.assertEqual(branches, ["main", "release/2026"])
        self.assertIsNone(error)

    async def test_fetch_branches_classifies_github_write_403_as_auth_without_leaking_token(self) -> None:
        token = "github_pat_do_not_leak"
        process = SimpleNamespace(
            returncode=128,
            communicate=mock.AsyncMock(return_value=(b"", b"remote: Write access to repository not granted.\nfatal: The requested URL returned error: 403")),
        )
        with mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)):
            _, error = await git.fetch_branches("https://github.com/acme/repo.git", token)
        self.assertIn("Authentication failed", error or "")
        self.assertNotIn(token, error or "")

    async def test_fetch_branches_rejects_pat_on_non_https_and_empty_repo_is_accessible(self) -> None:
        branches, error = await git.fetch_branches("git@github.com:acme/repo.git", "github_pat_safe")
        self.assertEqual(branches, [])
        self.assertIn("SSH key", error or "")

        process = SimpleNamespace(returncode=0, communicate=mock.AsyncMock(return_value=(b"", b"")))
        with mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)):
            branches, error = await git.fetch_branches("https://github.com/acme/empty.git", "github_pat_safe")
        self.assertEqual(branches, [])
        self.assertIsNone(error)

    async def test_visibility_does_not_let_public_metadata_mask_invalid_stored_token(self) -> None:
        with (
            mock.patch("ragtime.core.git._check_repo_access", new=mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.core.git._probe_git_read_access", new=mock.AsyncMock(return_value=([], git.GIT_AUTH_FAILURE_MESSAGE))),
        ):
            result = await git.check_repo_visibility("https://github.com/acme/repo.git", "github_pat_bad")
        self.assertTrue(result.has_stored_token)
        self.assertTrue(result.needs_token)
        self.assertEqual(result.visibility, git.RepoVisibility.PUBLIC)

    async def test_config_rejects_invalid_candidate_without_persisting_and_accepts_valid_trimmed_candidate(self) -> None:
        metadata = SimpleNamespace(
            sourceType="git",
            source="https://github.com/acme/repo.git",
            configSnapshot={},
            gitBranch="main",
            webhookId=None,
            webhookSecret=None,
            documentCount=1,
            chunkCount=1,
        )
        repo = mock.AsyncMock()
        repo.get_index_metadata = mock.AsyncMock(return_value=metadata)
        repo.update_index_config = mock.AsyncMock(return_value=True)
        with (
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], git.GIT_AUTH_FAILURE_MESSAGE))),
        ):
            with self.assertRaises(HTTPException) as error:
                await routes.update_index_config("idx", routes.UpdateIndexConfigRequest(git_token=" bad "), _user=mock.sentinel.user)
        self.assertEqual(error.exception.status_code, 400)
        repo.update_index_config.assert_not_awaited()

        with mock.patch.object(routes, "repository", repo), mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], None))):
            await routes.update_index_config("idx", routes.UpdateIndexConfigRequest(git_token=" good "), _user=mock.sentinel.user)
        self.assertEqual(repo.update_index_config.await_args.kwargs["git_token"], "good")

    async def test_config_rejects_replacement_without_source_before_probe_or_persistence(self) -> None:
        repo = mock.AsyncMock()
        repo.get_index_metadata.return_value = SimpleNamespace(sourceType="git", source=None, configSnapshot={})
        with (
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], None))) as probe,
        ):
            with self.assertRaises(HTTPException) as failure:
                await routes.update_index_config("idx", routes.UpdateIndexConfigRequest(git_token="candidate"), _user=mock.sentinel.user)
        self.assertEqual(failure.exception.status_code, 400)
        probe.assert_not_awaited()
        repo.update_index_config.assert_not_awaited()

    async def test_config_without_replacement_does_not_require_source(self) -> None:
        repo = mock.AsyncMock()
        repo.get_index_metadata.return_value = SimpleNamespace(sourceType="git", source=None, configSnapshot={})
        with (
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock()) as probe,
        ):
            await routes.update_index_config("idx", routes.UpdateIndexConfigRequest(chunk_size=512), _user=mock.sentinel.user)
        probe.assert_not_awaited()
        repo.update_index_config.assert_awaited_once()

    async def test_reindex_validates_explicit_candidate_before_creating_job(self) -> None:
        metadata = SimpleNamespace(
            sourceType="git", source="https://github.com/acme/repo.git", description="", configSnapshot={}, gitBranch="main", gitToken=None
        )
        with (
            mock.patch.object(routes.repository, "get_index_metadata", new=mock.AsyncMock(return_value=metadata)),
            mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], git.GIT_AUTH_FAILURE_MESSAGE))),
            mock.patch.object(routes.indexer, "create_index_from_git", new=mock.AsyncMock()) as create_job,
        ):
            with self.assertRaises(HTTPException) as error:
                await routes.reindex_from_git("idx", routes.ReindexGitRequest(git_token="candidate"), _user=mock.sentinel.user, _=None)
        self.assertEqual(error.exception.status_code, 400)
        create_job.assert_not_awaited()

    def test_git_auth_pair_matches_authenticated_url_for_hosts_prefixes_and_encoded_values(self) -> None:
        cases = [
            ("https://github.com/acme/repo.git", "glpat-wrong-provider"),
            ("https://gitlab.example.com/acme/repo.git", "github_pat_wrong_provider"),
            ("https://bitbucket.org/acme/repo.git", "glpat-wrong-provider"),
            ("https://code.example.com/acme/repo.git", "generic token:@/"),
        ]
        for url, token in cases:
            with self.subTest(url=url, token=token):
                authenticated = urlsplit(build_authenticated_git_url(url, token))
                expected = git.git_auth_pair(url, token)
                self.assertEqual((unquote(authenticated.username or ""), unquote(authenticated.password or "")), expected)

    async def test_probe_keeps_token_out_of_argv_and_scopes_header_without_discarding_global_config(self) -> None:
        token = "generic secret:@/"
        process = SimpleNamespace(returncode=0, communicate=mock.AsyncMock(return_value=(b"", b"")))
        with (
            mock.patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_GLOBAL": "/example/gitconfig",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.extraheader",
                    "GIT_CONFIG_VALUE_0": "Authorization: old-header",
                },
            ),
            mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)) as spawn,
        ):
            await git.fetch_branches("https://code.example.com/acme/repo.git", token)
        spawn_call = spawn.await_args
        assert spawn_call is not None
        args = spawn_call.args
        env = spawn_call.kwargs["env"]
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], "/example/gitconfig")
        self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertNotIn("Authorization: old-header", env.values())
        self.assertNotIn(token, " ".join(args))
        self.assertEqual(args[-2:], ("--", "https://code.example.com/acme/repo.git"))
        entries = [(env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"]) for i in range(int(env["GIT_CONFIG_COUNT"]))]
        self.assertIn(("http.followRedirects", "false"), entries)
        self.assertIn(("http.https://code.example.com/.extraheader", ""), entries)
        authorization = next(value for key, value in entries if key.endswith(".extraheader") and value.startswith("Authorization:"))
        self.assertNotIn(token, authorization)

    async def test_probe_timeout_kills_and_awaits_process(self) -> None:
        async def blocked_communication():
            await asyncio.Event().wait()

        process = SimpleNamespace(
            pid=123, returncode=None, communicate=mock.AsyncMock(side_effect=blocked_communication), kill=mock.Mock(), wait=mock.AsyncMock(return_value=0)
        )
        with (
            mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)),
            mock.patch("ragtime.core.git.os.name", "nt"),
        ):
            _, error = await git.fetch_branches("https://github.com/acme/repo.git", timeout=0.01)
        self.assertIn("timed out", error or "")
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_probe_cancellation_kills_and_awaits_process(self) -> None:
        process = SimpleNamespace(
            pid=123, returncode=None, communicate=mock.AsyncMock(side_effect=asyncio.CancelledError), kill=mock.Mock(), wait=mock.AsyncMock(return_value=0)
        )
        with (
            mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)),
            mock.patch("ragtime.core.git.os.name", "nt"),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await git.fetch_branches("https://github.com/acme/repo.git")
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    @unittest.skipUnless(os.name == "posix", "POSIX process group cleanup")
    async def test_probe_cancellation_reaps_the_https_helper_process_group(self) -> None:
        process = SimpleNamespace(pid=123, returncode=None, communicate=mock.AsyncMock(side_effect=asyncio.CancelledError), wait=mock.AsyncMock(return_value=0))
        with (
            mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)) as spawn,
            mock.patch("ragtime.core.git.os.killpg") as kill_group,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await git.fetch_branches("https://github.com/acme/repo.git")
        spawn_call = spawn.await_args
        assert spawn_call is not None
        self.assertTrue(spawn_call.kwargs["start_new_session"])
        kill_group.assert_called_once_with(process.pid, git.signal.SIGKILL)
        process.wait.assert_awaited_once()

    async def test_old_git_check_is_inconclusive_and_does_not_spawn_anonymous_probe(self) -> None:
        with (
            mock.patch("ragtime.core.git._supports_git_config_environment", return_value=False),
            mock.patch("ragtime.core.git.asyncio.create_subprocess_exec", new=mock.AsyncMock()) as spawn,
        ):
            _, error = await git.fetch_branches("https://github.com/acme/repo.git", "candidate")
        self.assertIsNotNone(error)
        spawn.assert_not_awaited()
        with mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], error))):
            result = await routes.fetch_branches(
                FetchBranchesRequest(git_url="https://github.com/acme/repo.git", git_token="candidate"), _user=mock.sentinel.user
            )
        self.assertFalse(result.needs_token)

    def test_rate_limits_are_not_classified_as_authentication_failures(self) -> None:
        for detail in (
            "fatal: unable to access: The requested URL returned error: 429",
            "remote: API rate limit exceeded; requested URL returned error: 403",
            "remote: secondary rate limit. HTTP 403",
        ):
            with self.subTest(detail=detail):
                self.assertFalse(git.is_git_auth_error(detail))

    async def test_standard_provider_network_probe_is_inconclusive_not_private(self) -> None:
        with (
            mock.patch("ragtime.core.git._check_repo_access", new=mock.AsyncMock(return_value=False)),
            mock.patch("ragtime.core.git._probe_git_read_access", new=mock.AsyncMock(return_value=([], "Repository access check timed out. Please retry."))),
        ):
            result = await git.check_repo_visibility("https://github.com/acme/repo.git", "github_pat_token")
        self.assertEqual(result.visibility, git.RepoVisibility.ERROR)
        self.assertFalse(result.needs_token)

    async def test_repository_not_found_requests_repair_for_standard_and_generic_hosts(self) -> None:
        for url in ("https://github.com/acme/repo.git", "https://code.example.com/acme/repo.git"):
            with (
                self.subTest(url=url),
                mock.patch("ragtime.core.git._probe_git_read_access", new=mock.AsyncMock(return_value=([], git.GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE))),
            ):
                if "github.com" in url:
                    with mock.patch("ragtime.core.git._check_repo_access", new=mock.AsyncMock(return_value=False)):
                        result = await git.check_repo_visibility(url, "token")
                else:
                    result = await git.check_repo_visibility(url, "token")
                self.assertTrue(result.needs_token)

    async def test_api_transport_failure_is_not_collapsed_into_repository_denial(self) -> None:
        client = mock.create_autospec(httpx.AsyncClient, instance=True, spec_set=True)
        client.get = mock.AsyncMock(side_effect=httpx.ConnectError("offline"))
        parsed = git.parse_git_url("https://github.com/acme/repo.git")
        assert parsed is not None
        with self.assertRaises(httpx.ConnectError):
            await git._check_repo_access(client, parsed)

    async def test_fetch_branches_stored_token_is_same_source_only_and_explicit_blank_does_not_fallback(self) -> None:
        metadata = SimpleNamespace(source="https://github.com/acme/repo.git", gitToken="encrypted")
        repo = mock.AsyncMock()
        repo.get_index_metadata = mock.AsyncMock(return_value=metadata)
        probe = mock.AsyncMock(return_value=(["main"], None))
        with (
            mock.patch.object(routes, "repository", repo),
            mock.patch.object(routes, "decrypt_secret", return_value="stored"),
            mock.patch.object(routes, "git_fetch_branches", new=probe),
        ):
            await routes.fetch_branches(FetchBranchesRequest(git_url=metadata.source, index_name="idx"), _user=mock.sentinel.user)
            probe_call = probe.await_args
            assert probe_call is not None
            self.assertEqual(probe_call.kwargs["token"], "stored")
            await routes.fetch_branches(FetchBranchesRequest(git_url="https://github.com/other/repo.git", index_name="idx"), _user=mock.sentinel.user)
            probe_call = probe.await_args
            assert probe_call is not None
            self.assertIsNone(probe_call.kwargs["token"])
            await routes.fetch_branches(FetchBranchesRequest(git_url=metadata.source, index_name="idx", git_token="  "), _user=mock.sentinel.user)
            probe_call = probe.await_args
            assert probe_call is not None
            self.assertIsNone(probe_call.kwargs["token"])
            await routes.fetch_branches(FetchBranchesRequest(git_url=metadata.source, index_name="idx", git_token=" candidate "), _user=mock.sentinel.user)
            probe_call = probe.await_args
            assert probe_call is not None
            self.assertEqual(probe_call.kwargs["token"], "candidate")

    def test_core_git_import_does_not_initialize_indexer_settings(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import ragtime.core.git; assert 'ragtime.indexer.models' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "ENCRYPTION_KEY": "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    async def test_metadata_overwrite_clears_credentials_unless_preservation_is_explicit(self) -> None:
        repository = IndexerRepository()
        db = SimpleNamespace(indexmetadata=SimpleNamespace(upsert=mock.AsyncMock()))
        with mock.patch.object(repository, "_get_db", new=mock.AsyncMock(return_value=db)):
            for preserve in (False, True):
                await repository.upsert_index_metadata(
                    name="idx",
                    path="/indexes/idx",
                    document_count=1,
                    chunk_count=1,
                    size_bytes=1,
                    source_type="git",
                    source="https://new.example/repo.git",
                    config_snapshot=None,
                    preserve_git_token=preserve,
                )
                update = db.indexmetadata.upsert.await_args.kwargs["data"]["update"]
                if preserve:
                    self.assertNotIn("gitToken", update)
                else:
                    self.assertIn("gitToken", update)
                    self.assertIsNone(update["gitToken"])

    async def test_retry_rejects_candidate_before_starting_another_job(self) -> None:
        job = SimpleNamespace(status=routes.IndexStatus.FAILED, source_type="git", git_url="https://github.com/acme/repo.git", git_token="old")
        with (
            mock.patch.object(routes.repository, "get_job", new=mock.AsyncMock(return_value=job)),
            mock.patch.object(routes, "git_fetch_branches", new=mock.AsyncMock(return_value=([], git.GIT_AUTH_FAILURE_MESSAGE))),
            mock.patch.object(routes.indexer, "create_index_from_git", new=mock.AsyncMock()) as create,
        ):
            with self.assertRaises(HTTPException) as failure:
                await routes.retry_failed_job("job", routes.RetryJobRequest(git_token="bad"), _user=mock.sentinel.user, _=None)
            self.assertEqual(failure.exception.status_code, 400)
            create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
