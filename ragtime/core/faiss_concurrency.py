from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Any, Callable, TypeVar

try:
    from ragtime.core.logging import get_logger
except Exception:  # pragma: no cover - test fallback for minimal environments
    get_logger = logging.getLogger

logger = get_logger(__name__)

T = TypeVar("T")
_DEFAULT_MODE = "per_index"
_DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 30.0


async def get_app_settings() -> dict[str, Any]:
    from ragtime.core.app_settings import get_app_settings as _get_app_settings  # inline-import: keep

    return await _get_app_settings()


class FaissSearchConcurrencyMode(str, Enum):
    PER_INDEX = "per_index"
    GLOBAL = "global"


class FaissSearchBusyError(RuntimeError):
    pass


class FaissSearchCoordinator:
    def __init__(
        self,
        *,
        acquire_timeout_seconds: float = _DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
        faiss_module: Any | None = None,
    ) -> None:
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._faiss_module = faiss_module
        self._global_semaphore = asyncio.Semaphore(1)
        self._per_index_semaphores: dict[str, asyncio.Semaphore] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=min(4, max(1, os.cpu_count() or 1)),
            thread_name_prefix="ragtime-faiss-",
        )
        self._shutdown = False

    async def run(
        self,
        index_name: str,
        operation: Callable[..., T],
        *args: Any,
        mode: FaissSearchConcurrencyMode | str | None = None,
        **kwargs: Any,
    ) -> T:
        if self._shutdown:
            raise RuntimeError("FAISS search coordinator is shut down")

        resolved_mode = await self._resolve_mode(mode)
        semaphore = self._get_semaphore(index_name, resolved_mode)
        wait_start = time.perf_counter()
        permit_acquired = False

        try:
            permit_acquired = await self._acquire_semaphore(semaphore)
        except asyncio.TimeoutError as exc:
            queue_wait = time.perf_counter() - wait_start
            logger.warning(
                "FAISS search busy: mode=%s index=%s queue_wait=%.3fs",
                resolved_mode.value,
                index_name,
                queue_wait,
            )
            raise FaissSearchBusyError(f"Timed out waiting for FAISS search slot for index '{index_name}'") from exc

        queue_wait = time.perf_counter() - wait_start
        operation_start = time.perf_counter()

        try:
            result = await self._run_in_executor(operation, *args, **kwargs)
        finally:
            operation_duration = time.perf_counter() - operation_start
            if permit_acquired:
                semaphore.release()
            logger.info(
                "FAISS search completed: mode=%s index=%s queue_wait=%.3fs duration=%.3fs",
                resolved_mode.value,
                index_name,
                queue_wait,
                operation_duration,
            )

        return result

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("FAISS search executor shut down")

    async def _resolve_mode(
        self,
        mode: FaissSearchConcurrencyMode | str | None,
    ) -> FaissSearchConcurrencyMode:
        candidate = mode
        if candidate is None:
            try:
                settings = await get_app_settings()
            except Exception as exc:
                logger.debug("Could not read faiss_search_concurrency_mode: %s", exc)
                settings = {}
            candidate = settings.get("faiss_search_concurrency_mode")

        if isinstance(candidate, FaissSearchConcurrencyMode):
            return candidate

        if hasattr(candidate, "value"):
            candidate = getattr(candidate, "value")

        normalized = str(candidate or _DEFAULT_MODE).strip().lower()
        try:
            return FaissSearchConcurrencyMode(normalized)
        except ValueError:
            return FaissSearchConcurrencyMode.PER_INDEX

    async def _acquire_semaphore(self, semaphore: asyncio.Semaphore) -> bool:
        acquire_task = asyncio.create_task(semaphore.acquire())
        try:
            await asyncio.wait_for(
                asyncio.shield(acquire_task),
                timeout=self._acquire_timeout_seconds,
            )
            return True
        except asyncio.TimeoutError:
            if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                semaphore.release()
            else:
                await self._cancel_acquire_task(acquire_task)
            raise
        except asyncio.CancelledError:
            if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                semaphore.release()
            else:
                await self._cancel_acquire_task(acquire_task)
            raise

    async def _cancel_acquire_task(self, acquire_task: asyncio.Task[bool]) -> None:
        if acquire_task.done():
            return
        acquire_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await acquire_task

    def _get_semaphore(
        self,
        index_name: str,
        mode: FaissSearchConcurrencyMode,
    ) -> asyncio.Semaphore:
        if mode is FaissSearchConcurrencyMode.GLOBAL:
            return self._global_semaphore
        return self._per_index_semaphores.setdefault(index_name, asyncio.Semaphore(1))

    async def _run_in_executor(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._execute_operation, operation, *args, **kwargs),
        )

    def _execute_operation(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        faiss_module = self._faiss_module
        if faiss_module is None:
            try:
                import faiss as faiss_module  # type: ignore[import-not-found]
            except Exception:
                faiss_module = None

        omp_set_num_threads = getattr(faiss_module, "omp_set_num_threads", None)
        if callable(omp_set_num_threads):
            omp_set_num_threads(1)

        return operation(*args, **kwargs)


faiss_search_coordinator = FaissSearchCoordinator()
