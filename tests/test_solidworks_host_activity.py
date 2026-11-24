import asyncio
import threading
import unittest
from contextlib import asynccontextmanager

from ragtime.solidworks.host_activity import HostActivity, HostActivitySnapshot, HostBusyError, HostUpdatingError, host_activity, host_key, run_blocking

KEY = "pairing:pair-1"
POLL = 0.01


@asynccontextmanager
async def held_gate(activity, key, wait_seconds, *, exclusive):
    started, release = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        await release.wait()

    task = asyncio.create_task(activity.run_exclusive(key, wait_seconds, work) if exclusive else activity.run_call(key, wait_seconds, lambda ticket: work()))
    ready = asyncio.create_task(started.wait())
    try:
        await asyncio.wait((task, ready), return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
        yield
    finally:
        release.set()
        ready.cancel()
        await asyncio.gather(task, ready, return_exceptions=True)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class HostKeyTests(unittest.TestCase):
    def test_pairing_id_wins_and_is_lower_cased(self):
        config = {"connection_config": {"pairing_id": "Pair-ABC", "host": "Host", "user": "Alice"}}
        self.assertEqual(host_key(config), "pairing:pair-abc")

    def test_ssh_key_defaults_port_and_lower_cases(self):
        config = {"connection_config": {"host": "Render-PC", "user": "Alice"}}
        self.assertEqual(host_key(config), "ssh:render-pc:22:alice")

    def test_ssh_key_uses_explicit_port(self):
        config = {"connection_config": {"host": "render", "port": 2222, "user": "alice"}}
        self.assertEqual(host_key(config), "ssh:render:2222:alice")

    def test_empty_pairing_id_falls_back_to_ssh(self):
        config = {"connection_config": {"pairing_id": "", "host": "render", "user": "alice"}}
        self.assertEqual(host_key(config), "ssh:render:22:alice")

    def test_non_string_pairing_id_is_ignored(self):
        config = {"connection_config": {"pairing_id": 7, "host": "render", "user": "alice"}}
        self.assertEqual(host_key(config), "ssh:render:22:alice")

    def test_missing_host_or_user_has_no_key(self):
        self.assertIsNone(host_key({"connection_config": {"host": "render"}}))
        self.assertIsNone(host_key({"connection_config": {"user": "alice"}}))
        self.assertIsNone(host_key({"connection_config": {"host": "", "user": "alice"}}))
        self.assertIsNone(host_key({"connection_config": {"host": "render", "user": 3}}))
        self.assertIsNone(host_key({}))


class EventLoopTests(unittest.TestCase):
    def test_same_instance_works_across_event_loops(self):
        activity = HostActivity(poll_seconds=POLL)

        async def use_once():
            async with held_gate(activity, KEY, 0, exclusive=False):
                pass
            async with held_gate(activity, KEY, 0, exclusive=True):
                pass

        asyncio.run(use_once())
        asyncio.run(use_once())
        self.assertEqual(activity.snapshot(KEY).in_flight, 0)

    def test_loop_shutdown_keeps_orphaned_gates_across_loops(self):
        for exclusive in (False, True):
            with self.subTest(exclusive=exclusive):
                activity = HostActivity(poll_seconds=0.001)
                started, finish = threading.Event(), threading.Event()

                def blocking():
                    started.set()
                    finish.wait(5)

                async def work():
                    await run_blocking(blocking)

                async def first_loop():
                    runner = activity.run_exclusive(KEY, 1, work) if exclusive else activity.run_call(KEY, 1, lambda ticket: work())
                    caller = asyncio.create_task(runner)
                    while not started.is_set():
                        await asyncio.sleep(0.001)
                    caller.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await caller
                    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    await asyncio.sleep(0)

                async def second_loop():
                    self.assertFalse(await activity.drain(0.01))
                    if exclusive:
                        with self.assertRaises(HostUpdatingError):
                            await activity.run_call(KEY, 0.01, lambda ticket: asyncio.sleep(0))
                    else:
                        with self.assertRaises(HostBusyError):
                            await activity.run_exclusive(KEY, 0.01, lambda: asyncio.sleep(0))
                    finish.set()
                    self.assertTrue(await activity.drain(1))
                    await activity.run_call(KEY, 0, lambda ticket: asyncio.sleep(0))
                    await activity.run_exclusive(KEY, 0, lambda: asyncio.sleep(0))
                    self.assertEqual(activity.snapshot(KEY).in_flight, 0)
                    self.assertFalse(activity.snapshot(KEY).exclusive)

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(first_loop())
                    loop.close()
                    asyncio.run(second_loop())
                finally:
                    finish.set()
                    if not loop.is_closed():
                        loop.close()


class ShieldedGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.activity = HostActivity(poll_seconds=0.001)

    async def wait_until(self, condition):
        async def wait():
            while not condition():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), 1)

    async def test_ticket_reports_actual_contention(self):
        async def work(ticket):
            return ticket.waited

        clock_value = 0

        def clock():
            nonlocal clock_value
            clock_value += 100
            return clock_value

        slow = HostActivity(clock=clock)
        self.assertFalse(await slow.run_call(KEY, 1000, work))
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            task = asyncio.create_task(self.activity.run_call(KEY, 1, work))
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
        self.assertTrue(await task)

    async def test_cancelled_call_keeps_gate_until_thread_finishes(self):
        started = threading.Event()
        finish = threading.Event()

        def blocking():
            started.set()
            finish.wait(2)

        async def work(ticket):
            await asyncio.to_thread(blocking)

        task = asyncio.create_task(self.activity.run_call(KEY, 1, work))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.activity.snapshot(KEY).in_flight, 1)
            with self.assertRaises(HostBusyError):
                await self.activity.run_exclusive(KEY, 0, lambda: asyncio.sleep(0))
        finally:
            finish.set()
        await self.wait_until(lambda: not self.activity.snapshot(KEY).in_flight)
        await self.activity.run_exclusive(KEY, 0, lambda: asyncio.sleep(0))

    async def test_cancelled_exclusive_keeps_gate_until_thread_finishes(self):
        started = threading.Event()
        finish = threading.Event()

        def blocking():
            started.set()
            finish.wait(2)

        async def work():
            await asyncio.to_thread(blocking)

        task = asyncio.create_task(self.activity.run_exclusive(KEY, 1, work))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(self.activity.snapshot(KEY).exclusive)
            with self.assertRaises(HostUpdatingError):
                await self.activity.run_call(KEY, 0, lambda ticket: asyncio.sleep(0))
        finally:
            finish.set()
        await self.wait_until(lambda: not self.activity.snapshot(KEY).exclusive)
        await self.activity.run_call(KEY, 0, lambda ticket: asyncio.sleep(0))

    async def test_exception_propagates_and_releases(self):
        async def fail():
            raise ValueError("failure")

        with self.assertRaises(ValueError):
            await self.activity.run_call(KEY, 0, lambda ticket: fail())
        self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)
        with self.assertRaises(ValueError):
            await self.activity.run_exclusive(KEY, 0, fail)
        self.assertFalse(self.activity.snapshot(KEY).exclusive)

    async def test_none_key_passthrough(self):
        async def call(ticket):
            return ticket.waited

        async def exclusive():
            return 42

        self.assertFalse(await self.activity.run_call(None, 0, call))
        self.assertEqual(await self.activity.run_exclusive(None, 0, exclusive), 42)

    async def test_work_cancellation_releases_when_no_worker_remains(self):
        async def cancelled():
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await self.activity.run_call(KEY, 0, lambda ticket: cancelled())
        self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)
        with self.assertRaises(asyncio.CancelledError):
            await self.activity.run_exclusive(KEY, 0, cancelled)
        self.assertFalse(self.activity.snapshot(KEY).exclusive)

    async def test_offload_without_hold(self):
        self.assertEqual(await run_blocking(lambda a, b: a + b, 2, 3), 5)

    async def test_finished_work_waits_for_detached_worker_once(self):
        started, finish = threading.Event(), threading.Event()

        def blocking():
            started.set()
            finish.wait(2)

        async def work(ticket):
            asyncio.create_task(run_blocking(blocking))
            while not started.is_set():
                await asyncio.sleep(0.001)
            return "done"

        try:
            self.assertEqual(await self.activity.run_call(KEY, 1, work), "done")
            self.assertEqual(self.activity.snapshot(KEY).in_flight, 1)
            self.assertFalse(await self.activity.drain(0.01))
        finally:
            finish.set()
        self.assertTrue(await self.activity.drain(1))
        self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)

    async def test_drain_waits_for_detached_work_and_times_out(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def work():
            started.set()
            await finish.wait()

        task = asyncio.create_task(self.activity.run_exclusive(KEY, 1, work))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(await self.activity.drain(0.01))
        finish.set()
        self.assertTrue(await self.activity.drain(1))
        self.assertFalse(self.activity.snapshot(KEY).exclusive)

    async def test_detached_exception_is_retrieved(self):
        started = asyncio.Event()
        finish = asyncio.Event()
        captured = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: captured.append(context))

        async def fail(ticket):
            started.set()
            await finish.wait()
            raise ValueError("failure")

        try:
            task = asyncio.create_task(self.activity.run_call(KEY, 0, fail))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            finish.set()
            await self.wait_until(lambda: not self.activity.snapshot(KEY).in_flight)
            await asyncio.sleep(0)
            self.assertEqual(captured, [])
        finally:
            loop.set_exception_handler(previous)


class CallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.activity = HostActivity(poll_seconds=POLL)

    async def test_call_counts_in_flight_and_records_idle(self):
        clock = FakeClock()
        activity = HostActivity(clock=clock, poll_seconds=POLL)
        self.assertEqual(activity.snapshot(KEY), HostActivitySnapshot(in_flight=0, idle_seconds=None, exclusive=False))
        async with held_gate(activity, KEY, 0, exclusive=False):
            self.assertEqual(activity.snapshot(KEY).in_flight, 1)
            self.assertEqual(activity.snapshot(KEY).idle_seconds, 0.0)
        clock.now += 42
        self.assertEqual(activity.snapshot(KEY), HostActivitySnapshot(in_flight=0, idle_seconds=42.0, exclusive=False))

    async def test_call_records_end_time_when_body_raises(self):
        with self.assertRaises(RuntimeError):
            async with held_gate(self.activity, KEY, 0, exclusive=False):
                raise RuntimeError("boom")
        self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)
        self.assertIsNotNone(self.activity.snapshot(KEY).idle_seconds)

    async def test_call_waits_during_exclusive_then_proceeds(self):
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            task = asyncio.create_task(self._call_once(KEY, 2.0))
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
        self.assertTrue(await asyncio.wait_for(task, 2.0))

    async def test_call_times_out_during_exclusive_and_is_not_counted(self):
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            with self.assertRaises(HostUpdatingError):
                async with held_gate(self.activity, KEY, 0.05, exclusive=False):
                    self.fail("call must not start while exclusive is held")
            self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)

    async def test_exclusive_waits_for_in_flight_calls_to_drain(self):
        release = asyncio.Event()

        async def long_call():
            async with held_gate(self.activity, KEY, 0, exclusive=False):
                await release.wait()

        call_task = asyncio.create_task(long_call())
        await asyncio.sleep(0.02)
        exclusive_task = asyncio.create_task(self._hold_exclusive(KEY, 2.0))
        await asyncio.sleep(0.05)
        self.assertFalse(exclusive_task.done())
        self.assertTrue(self.activity.snapshot(KEY).exclusive)
        release.set()
        await asyncio.wait_for(call_task, 2.0)
        self.assertTrue(await asyncio.wait_for(exclusive_task, 2.0))
        self.assertFalse(self.activity.snapshot(KEY).exclusive)

    async def test_exclusive_timeout_releases_mark_and_raises_busy(self):
        release = asyncio.Event()

        async def long_call():
            async with held_gate(self.activity, KEY, 0, exclusive=False):
                await release.wait()

        call_task = asyncio.create_task(long_call())
        await asyncio.sleep(0.02)
        with self.assertRaises(HostBusyError):
            async with held_gate(self.activity, KEY, 0.05, exclusive=True):
                self.fail("exclusive must not be granted while a call is in flight")
        self.assertFalse(self.activity.snapshot(KEY).exclusive)
        release.set()
        await call_task
        self.assertTrue(await self._call_once(KEY, 0))

    async def test_second_exclusive_raises_busy_when_not_granted_in_time(self):
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            with self.assertRaises(HostBusyError):
                async with held_gate(self.activity, KEY, 0.05, exclusive=True):
                    self.fail("second exclusive must not be granted")
            self.assertTrue(self.activity.snapshot(KEY).exclusive)

    async def test_second_exclusive_waits_then_is_granted(self):
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            second = asyncio.create_task(self._hold_exclusive(KEY, 2.0))
            await asyncio.sleep(0.05)
            self.assertFalse(second.done())
        self.assertTrue(await asyncio.wait_for(second, 2.0))

    async def test_no_call_starts_while_exclusive_is_held(self):
        started = []
        async with held_gate(self.activity, KEY, 0, exclusive=True):
            for _ in range(5):
                await asyncio.sleep(0)
                with self.assertRaises(HostUpdatingError):
                    async with held_gate(self.activity, KEY, 0, exclusive=False):
                        started.append(self.activity.snapshot(KEY).in_flight)
                self.assertEqual(self.activity.snapshot(KEY).in_flight, 0)
        self.assertEqual(started, [])

    async def test_snapshot_reports_exclusive_and_in_flight(self):
        clock = FakeClock()
        activity = HostActivity(clock=clock, poll_seconds=POLL)
        async with held_gate(activity, KEY, 0, exclusive=False):
            pass
        async with held_gate(activity, KEY, 0, exclusive=True):
            self.assertEqual(activity.snapshot(KEY), HostActivitySnapshot(in_flight=0, idle_seconds=0.0, exclusive=True))
        clock.now += 5
        self.assertEqual(activity.snapshot(KEY), HostActivitySnapshot(in_flight=0, idle_seconds=5.0, exclusive=False))

    async def test_none_key_is_a_no_op(self):
        async with held_gate(self.activity, None, 0, exclusive=False):
            pass
        async with held_gate(self.activity, None, 0, exclusive=True):
            async with held_gate(self.activity, None, 0, exclusive=False):
                pass
        self.assertEqual(self.activity.snapshot(None), HostActivitySnapshot(in_flight=0, idle_seconds=None, exclusive=False))

    async def test_keys_are_independent(self):
        async with held_gate(self.activity, "ssh:a:22:alice", 0, exclusive=True):
            self.assertTrue(await self._call_once("ssh:b:22:alice", 0))

    def test_module_singleton_exists(self):
        self.assertIsInstance(host_activity, HostActivity)

    async def _call_once(self, key, wait_seconds):
        async with held_gate(self.activity, key, wait_seconds, exclusive=False):
            return True

    async def _hold_exclusive(self, key, wait_seconds):
        async with held_gate(self.activity, key, wait_seconds, exclusive=True):
            return True


if __name__ == "__main__":
    unittest.main()
