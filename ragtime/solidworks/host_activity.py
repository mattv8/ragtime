"""In-process gate that keeps SolidWorks host updates away from running calls."""

import asyncio
import contextvars
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeVar

DEFAULT_POLL_SECONDS = 0.1
SSH_WORKERS = 16
_executor = ThreadPoolExecutor(max_workers=SSH_WORKERS, thread_name_prefix="solidworks-ssh")
_hold_lock = threading.RLock()
T = TypeVar("T")


@dataclass(frozen=True)
class GateTicket:
    waited: bool


class HostUpdatingError(Exception):
    """The host is held for an update, so the call cannot start."""


class HostBusyError(Exception):
    """The host stayed busy for the whole wait, so the exclusive hold was not granted."""


@dataclass(frozen=True)
class HostActivitySnapshot:
    in_flight: int
    idle_seconds: float | None
    exclusive: bool


@dataclass
class _KeyState:
    in_flight: int = 0
    exclusive: bool = False
    last_ended: float | None = None


@dataclass(eq=False)
class _Hold:
    owner: "HostActivity"
    release: Callable[[], None]
    finished: bool = False
    pending: set[Future[Any]] = field(default_factory=set)

    def finish(self) -> None:
        with _hold_lock:
            self.finished = True
            self._release_if_done()

    def _release_if_done(self) -> None:
        if self.finished and not self.pending and self in self.owner._holds:
            self.release()
            self.owner._holds.remove(self)


_current_hold: contextvars.ContextVar[_Hold | None] = contextvars.ContextVar("solidworks_gate_hold", default=None)


async def run_blocking(fn: Callable[..., T], /, *args: Any) -> T:
    """Offload SSH work and keep its gate held until the worker actually finishes."""
    hold = _current_hold.get()
    with _hold_lock:
        future: Future[T] = _executor.submit(partial(contextvars.copy_context().run, fn, *args))
        if hold is not None:
            hold.pending.add(future)

    if hold is not None:

        def complete(done: Future[Any]) -> None:
            with _hold_lock:
                hold.pending.discard(done)
                hold._release_if_done()

        future.add_done_callback(complete)
    return await asyncio.wrap_future(future)


def host_key(config: dict[str, Any]) -> str | None:
    """Return the activity key for a SolidWorks tool config, or None when it cannot be identified."""
    connection = config.get("connection_config") or {}
    pairing_id = connection.get("pairing_id")
    if isinstance(pairing_id, str) and pairing_id:
        return f"pairing:{pairing_id}".lower()
    host = connection.get("host")
    user = connection.get("user")
    if isinstance(host, str) and host and isinstance(user, str) and user:
        port = connection.get("port") or 22
        return f"ssh:{host}:{port}:{user}".lower()
    return None


class HostActivity:
    """Counts calls per host key and lets one exclusive holder block new calls.

    State lives behind one threading.Lock and waits poll with asyncio.sleep, so the same
    instance works from any event loop.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, poll_seconds: float = DEFAULT_POLL_SECONDS):
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._states: dict[str, _KeyState] = {}
        self._holds: set[_Hold] = set()

    async def run_call(self, key: str | None, wait_seconds: float, work: Callable[[GateTicket], Awaitable[T]]) -> T:
        """Run work under a shared gate, retaining it until cancelled work finishes."""
        if key is None:
            return await work(GateTicket(False))
        waited = await self._acquire_call(key, wait_seconds)
        return await self._run_work(lambda: work(GateTicket(waited)), lambda: self._end_call(key))

    async def run_exclusive(self, key: str | None, wait_seconds: float, work: Callable[[], Awaitable[T]]) -> T:
        """Run work under an exclusive gate, retaining it until cancelled work finishes."""
        if key is None:
            return await work()
        await self._acquire_exclusive(key, wait_seconds)
        return await self._run_work(work, lambda: self._clear_exclusive(key))

    async def _acquire_call(self, key: str, wait_seconds: float) -> bool:
        return await self._wait(lambda: self._try_start_call(key), self._deadline(wait_seconds), HostUpdatingError)

    async def _acquire_exclusive(self, key: str, wait_seconds: float) -> None:
        deadline = self._deadline(wait_seconds)
        await self._wait(lambda: self._try_mark_exclusive(key), deadline, HostBusyError)
        try:
            await self._wait(lambda: self._is_drained(key), deadline, HostBusyError)
        except BaseException:
            self._clear_exclusive(key)
            raise

    async def _run_work(self, work: Callable[[], Awaitable[T]], release: Callable[[], None]) -> T:
        hold = _Hold(self, release)
        with _hold_lock:
            self._holds.add(hold)

        async def wrapped() -> T:
            token = _current_hold.set(hold)
            try:
                return await work()
            finally:
                _current_hold.reset(token)
                hold.finish()

        coroutine = wrapped()
        try:
            task = asyncio.ensure_future(coroutine)
        except BaseException:
            coroutine.close()
            hold.finish()
            raise

        def finish(completed: asyncio.Future[T]) -> None:
            hold.finish()
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finish)
        return await asyncio.shield(task)

    async def drain(self, timeout_seconds: float) -> bool:
        """Wait for gated work and its worker threads to finish."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            with _hold_lock:
                if not self._holds:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(self._poll_seconds, remaining))

    def snapshot(self, key: str | None) -> HostActivitySnapshot:
        """Return the current activity of a key; idle_seconds is 0 while calls run, None before any call ended."""
        if key is None:
            return HostActivitySnapshot(in_flight=0, idle_seconds=None, exclusive=False)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return HostActivitySnapshot(in_flight=0, idle_seconds=None, exclusive=False)
            if state.in_flight > 0:
                idle_seconds: float | None = 0.0
            elif state.last_ended is None:
                idle_seconds = None
            else:
                idle_seconds = max(0.0, self._clock() - state.last_ended)
            return HostActivitySnapshot(in_flight=state.in_flight, idle_seconds=idle_seconds, exclusive=state.exclusive)

    def _deadline(self, wait_seconds: float) -> float:
        return self._clock() + max(0.0, wait_seconds)

    async def _wait(self, attempt: Callable[[], bool], deadline: float, error: type[Exception]) -> bool:
        waited = False
        while not attempt():
            waited = True
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise error()
            await asyncio.sleep(min(self._poll_seconds, remaining))
        return waited

    def _try_start_call(self, key: str) -> bool:
        with self._lock:
            state = self._states.setdefault(key, _KeyState())
            if state.exclusive:
                return False
            state.in_flight += 1
            return True

    def _end_call(self, key: str) -> None:
        with self._lock:
            state = self._states[key]
            state.in_flight -= 1
            state.last_ended = self._clock()

    def _try_mark_exclusive(self, key: str) -> bool:
        with self._lock:
            state = self._states.setdefault(key, _KeyState())
            if state.exclusive:
                return False
            state.exclusive = True
            return True

    def _is_drained(self, key: str) -> bool:
        with self._lock:
            return self._states[key].in_flight == 0

    def _clear_exclusive(self, key: str) -> None:
        with self._lock:
            self._states[key].exclusive = False


host_activity = HostActivity()
