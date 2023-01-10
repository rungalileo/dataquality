import threading
from typing import Any, Iterable, Set


class ThreadSafeSet:
    def __init__(self) -> None:
        self._set: Set = set()
        self._lock = threading.Lock()

    def add(self, value: Any) -> None:
        with self._lock:
            self._set.add(value)

    def remove(self, value: Any) -> None:
        with self._lock:
            self._set.remove(value)

    def update(self, values: Iterable) -> None:
        with self._lock:
            self._set.update(values)

    def __iter__(self) -> Iterable:
        with self._lock:
            return iter(self._set)
