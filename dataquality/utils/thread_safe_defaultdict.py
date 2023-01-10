import threading
from collections import defaultdict
from typing import Any


class ThreadSafeDefaultDict(defaultdict):
    def __init__(self, default_factory: Any = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(default_factory, *args, **kwargs)
        self._lock = threading.Lock()

    def __missing__(self, key: Any) -> None:
        with self._lock:
            if key not in self:
                self[key] = set()
            return self[key]
