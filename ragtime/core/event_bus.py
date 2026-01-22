import asyncio
from typing import Dict, Set


class TaskEventBus:
    _instance = None
    _subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskEventBus, cls).__new__(cls)
            cls._instance._subscribers = {}
        return cls._instance

    async def publish(self, task_id: str, data: dict):
        if task_id in self._subscribers:
            # Copy set to avoid size change during iteration if something unsubscribes concurrently (unlikely but safe)
            for queue in list(self._subscribers[task_id]):
                await queue.put(data)

    async def subscribe(self, task_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        if task_id not in self._subscribers:
            self._subscribers[task_id] = set()
        self._subscribers[task_id].add(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        if task_id in self._subscribers:
            self._subscribers[task_id].discard(queue)
            if not self._subscribers[task_id]:
                del self._subscribers[task_id]


task_event_bus = TaskEventBus()
