import weakref
from typing import Any, Callable


class RefManager:
    """Manage weak references to objects and call a cleanup function when
    the object is garbage collected."""

    def __init__(self, cleanup_func: Callable):
        """Initialize the cleanup manager.
        :param cleanup_func: The function to call when an object is garbage
        collected."""
        self.cleanup_func: Callable = cleanup_func
        self._instances: set = set()

    def register(self, instance: Any) -> None:
        """Register an object with the cleanup manager.
        :param instance: The object to register."""
        weak_instance = weakref.ref(instance, self._cleanup_callback)
        self._instances.add(weak_instance)

    def _cleanup_callback(self, weak_instance: Any) -> None:
        """Call the cleanup function when the object is garbage collected.
        :param weak_instance: The weak reference to the object that was
        garbage collected."""
        self.cleanup_func()
        self._instances.discard(weak_instance)


class Cleanup:
    """Base class for objects that need to be cleaned up when they are
    garbage collected."""

    def __init__(self, cleanup_manager: RefManager) -> None:
        """Register this object with the cleanup manager.
        :param cleanup_manager: The cleanup manager to register with."""
        self.cleanup_manager = cleanup_manager
        self.cleanup_manager.register(self)

    def __del__(self) -> None:
        """Call the cleanup function when the object is garbage collected."""
        # This will be called when the object is garbage collected.
        # However, it's not guaranteed to be called in all situations.
        self.cleanup_manager._cleanup_callback(weakref.ref(self))
