import asyncio
import unittest
from unittest import mock

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, ProtectionContext


class ContentProtectionPerformanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        service._decision_cache.clear()

    @staticmethod
    def _config(revision: int = 1) -> ContentProtectionConfig:
        return ContentProtectionConfig(enabled=True, classifier_model="openai::classifier", revision=revision)

    @staticmethod
    def _policy(user: str = "u") -> service._ResolvedPolicy:
        return service._ResolvedPolicy(True, "all_supported_traffic", {user}, {user: set()}, {user: None}, [[{"id": "standard", "scope": "ordinary"}]])

    async def test_identical_waiters_share_provider_but_audit_and_release_independently(self) -> None:
        entered, second_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        settings_calls = 0

        async def settings():
            nonlocal settings_calls
            settings_calls += 1
            if settings_calls == 2:
                second_ready.set()
            return "settings"

        async def classify(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"verdict": "allow", "reason_code": "permitted"}

        audit = mock.AsyncMock()
        load_config = mock.AsyncMock(return_value=self._config())
        with (
            mock.patch.object(service, "load_config", load_config),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(side_effect=settings)),
            mock.patch.object(service, "classify", side_effect=classify) as provider,
            mock.patch.object(service, "_audit", audit),
        ):
            first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await entered.wait()
            second = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await second_ready.wait()
            release.set()
            await asyncio.gather(first, second)

        self.assertEqual(provider.await_count, 1)
        self.assertEqual(audit.await_count, 2)
        # Each caller loads config before classification and again at release.
        self.assertEqual(load_config.await_count, 4)

    async def test_complete_fingerprint_dimensions_do_not_coalesce(self) -> None:
        """Audience, revision, and provider settings each isolate live work."""
        for dimension in ("audience", "revision", "settings"):
            entered_twice, release = asyncio.Event(), asyncio.Event()
            calls: list[object] = []

            async def classify(*_args, **_kwargs):
                calls.append(object())
                if len(calls) == 2:
                    entered_twice.set()
                await release.wait()
                return {"verdict": "allow", "reason_code": "permitted"}

            def task_value(first: object, second: object) -> object:
                task = asyncio.current_task()
                assert task is not None
                return first if task.get_name() == "first" else second

            async def resolve(_config, context, **_kwargs):
                user = context.user_id or "anonymous"
                # Only the audience case needs a distinct resolved policy.  The
                # others deliberately use identical policy/candidate values.
                return self._policy(user)

            config = mock.AsyncMock(side_effect=lambda: task_value(self._config(1), self._config(2)) if dimension == "revision" else self._config())
            settings = mock.AsyncMock(side_effect=lambda: task_value("one", "two") if dimension == "settings" else "same")
            first_context = ProtectionContext(user_id="first" if dimension == "audience" else "u")
            second_context = ProtectionContext(user_id="second" if dimension == "audience" else "u")
            with (
                mock.patch.object(service, "load_config", config),
                mock.patch.object(service, "_resolve", mock.AsyncMock(side_effect=resolve)),
                mock.patch.object(service, "_provider_settings_identity", settings),
                mock.patch.object(service, "classify", side_effect=classify) as provider,
                mock.patch.object(service, "_audit", mock.AsyncMock()),
            ):
                first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=first_context), name="first")
                second = asyncio.create_task(service.authorize_content("same", direction="inbound", context=second_context), name="second")
                await entered_twice.wait()
                release.set()
                await asyncio.gather(first, second)
            self.assertEqual(provider.await_count, 2, dimension)

    async def test_shared_failure_has_per_caller_request_ids_and_audits(self) -> None:
        entered, second_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        settings_calls = 0

        async def settings():
            nonlocal settings_calls
            settings_calls += 1
            if settings_calls == 2:
                second_ready.set()
            return "settings"

        async def classify(*_args, **_kwargs):
            entered.set()
            await release.wait()
            raise RuntimeError("external provider unavailable")

        audit = mock.AsyncMock()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(side_effect=settings)),
            mock.patch.object(service, "classify", side_effect=classify) as provider,
            mock.patch.object(service, "_audit", audit),
        ):
            first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await entered.wait()
            second = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await second_ready.wait()
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)

        self.assertEqual(provider.await_count, 1)
        self.assertTrue(all(isinstance(result, ContentProtectionError) for result in results))
        first_error, second_error = results
        assert isinstance(first_error, ContentProtectionError)
        assert isinstance(second_error, ContentProtectionError)
        self.assertNotEqual(first_error.request_id, second_error.request_id)
        self.assertEqual({call.args[0] for call in audit.await_args_list}, {first_error.request_id, second_error.request_id})

    async def test_failed_and_cancelled_waits_charge_the_turn_budget(self) -> None:
        for outcome in ("failure", "cancellation"):
            state = service._TurnState()
            token = service._turn.set(state)
            entered = asyncio.Event()

            async def classify(*_args, **_kwargs):
                entered.set()
                if outcome == "failure":
                    raise RuntimeError("fixture failure")
                await asyncio.Future()

            try:
                with (
                    mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
                    mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
                    mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
                    mock.patch.object(service, "classify", side_effect=classify),
                    mock.patch.object(service, "_audit", mock.AsyncMock()),
                ):
                    waiter = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
                    await entered.wait()
                    if outcome == "failure":
                        with self.assertRaises(ContentProtectionError):
                            await waiter
                    else:
                        waiter.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await waiter
                self.assertGreater(state.record.spent, 0.0, outcome)
            finally:
                service._turn.reset(token)

    async def test_each_waiter_rechecks_current_policy_before_its_own_release(self) -> None:
        entered, second_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls_by_task: dict[str, int] = {}

        async def load_config():
            task = asyncio.current_task()
            assert task is not None
            name = task.get_name()
            calls_by_task[name] = calls_by_task.get(name, 0) + 1
            if name == "second" and calls_by_task[name] == 1:
                second_ready.set()
            # The second caller observes a revision change only at release;
            # its retry sees the same revised policy and gets a fresh verdict.
            return self._config(2 if name == "second" and calls_by_task[name] >= 2 else 1)

        verdicts = [{"verdict": "allow", "reason_code": "permitted"}, {"verdict": "deny", "reason_code": "restricted_content"}]

        async def classify(*_args, **_kwargs):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return verdicts.pop(0)

        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(side_effect=load_config)),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "classify", side_effect=classify) as provider,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
        ):
            first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")), name="first")
            await entered.wait()
            second = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")), name="second")
            await second_ready.wait()
            release.set()
            await first
            with self.assertRaisesRegex(ContentProtectionError, "content_denied"):
                await second

        self.assertEqual(provider.await_count, 2)

    async def test_one_cancelled_waiter_leaves_other_waiter_running(self) -> None:
        entered, second_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        settings_calls = 0

        async def settings():
            nonlocal settings_calls
            settings_calls += 1
            if settings_calls == 2:
                second_ready.set()
            return "settings"

        async def classify(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"verdict": "allow", "reason_code": "permitted"}

        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(side_effect=settings)),
            mock.patch.object(service, "classify", side_effect=classify) as provider,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
        ):
            first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await entered.wait()
            second = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await second_ready.wait()
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            self.assertFalse(first.done())
            release.set()
            await first
        self.assertEqual(provider.await_count, 1)

    async def test_last_cancel_settles_provider_and_removes_registry_entry(self) -> None:
        entered, provider_cancelled = asyncio.Event(), asyncio.Event()

        async def classify(*_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

        loop = asyncio.get_running_loop()
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "classify", side_effect=classify),
            mock.patch.object(service, "_audit", mock.AsyncMock()),
        ):
            waiter = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
            await entered.wait()
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertTrue(provider_cancelled.is_set())
            self.assertEqual(service._inflight_by_loop[loop].entries, {})

    async def test_same_turn_near_budget_rejects_second_waiter_without_extra_provider_call(self) -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def classify(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"verdict": "allow", "reason_code": "permitted"}

        state = service._TurnState()
        state.record.spent = service._TURN_BUDGET - 5.0
        state_token = service._turn.set(state)
        try:
            with (
                mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
                mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
                mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
                mock.patch.object(service, "classify", side_effect=classify) as provider,
                mock.patch.object(service, "_audit", mock.AsyncMock()),
            ):
                first = asyncio.create_task(service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u")))
                await entered.wait()
                with self.assertRaisesRegex(ContentProtectionError, "classifier_unavailable"):
                    await service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u"))
                release.set()
                # A budget exhaustion is sticky for the shared turn, so the
                # pre-existing sibling must not later release its allow.
                with self.assertRaisesRegex(ContentProtectionError, "classifier_unavailable"):
                    await first
            self.assertEqual(provider.await_count, 1)
            self.assertLessEqual(state.record.spent, service._TURN_BUDGET)
        finally:
            service._turn.reset(state_token)

    async def test_stale_allowing_sibling_cannot_release_after_other_sibling_denies(self) -> None:
        deny_entered, allow_entered = asyncio.Event(), asyncio.Event()
        deny_release, allow_release = asyncio.Event(), asyncio.Event()

        async def classify(_config, envelope, **_kwargs):
            if envelope["candidate"] == "deny":
                deny_entered.set()
                await deny_release.wait()
                return {"verdict": "deny", "reason_code": "restricted_content"}
            allow_entered.set()
            await allow_release.wait()
            return {"verdict": "allow", "reason_code": "permitted"}

        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "classify", side_effect=classify),
            mock.patch.object(service, "_audit", mock.AsyncMock()) as audit,
            service.protection_context(ProtectionContext(user_id="u")),
        ):
            denied = asyncio.create_task(service.authorize_content("deny", direction="inbound"))
            allowed = asyncio.create_task(service.authorize_content("allow", direction="inbound"))
            await asyncio.gather(deny_entered.wait(), allow_entered.wait())
            deny_release.set()
            with self.assertRaises(ContentProtectionError) as denied_error:
                await denied
            allow_release.set()
            with self.assertRaises(ContentProtectionError) as stale_error:
                await allowed

        self.assertIs(stale_error.exception, denied_error.exception)
        self.assertFalse(any(call.args[1].get("code") == "permitted" for call in audit.await_args_list))

    async def test_denied_reason_is_not_cached(self) -> None:
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=self._config())),
            mock.patch.object(service, "_resolve", mock.AsyncMock(return_value=self._policy())),
            mock.patch.object(service, "_provider_settings_identity", mock.AsyncMock(return_value="settings")),
            mock.patch.object(
                service,
                "classify",
                mock.AsyncMock(
                    side_effect=[
                        {"verdict": "deny", "reason_code": "restricted_content", "reason": "no"},
                        {"verdict": "allow", "reason_code": "permitted"},
                    ]
                ),
            ) as provider,
            mock.patch.object(service, "_audit", mock.AsyncMock()),
        ):
            with self.assertRaises(ContentProtectionError):
                await service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u"))
            await service.authorize_content("same", direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(provider.await_count, 2)
