from typing import Any, Iterable, Iterator, Set

from dataquality.utils.thread_pool import lock


class ThreadSafeSet:
    def __init__(self) -> None:
        self._set: Set = set()
        self._lock = lock

    def add(self, value: Any) -> None:
        with self._lock:
            self._set.add(value)

    def remove(self, value: Any) -> None:
        with self._lock:
            self._set.remove(value)

    def update(self, values: Iterable[Any]) -> None:
        with self._lock:
            self._set.update(values)

    def __iter__(self) -> Iterator:
        with self._lock:
            return iter(self._set)

    def difference(self, values: Iterable[Any]) -> Set:
        with self._lock:
            return self._set.difference(values)

    def __len__(self) -> int:
        with self._lock:
            return len(self._set)
            return len(self._set)
