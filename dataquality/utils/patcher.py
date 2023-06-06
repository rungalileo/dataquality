import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


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


class Borg:
    """Borg class making class attributes global"""

    _shared_state: Dict = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state


class PatchManager(Borg):
    """Class to manage patches"""

    def __init__(self) -> None:
        """Class to manage patches"""
        super().__init__()
        if not hasattr(self, "patches"):
            self.patches: List[Patch] = []

    def add_patch(self, patch: "Patch") -> None:
        """Add patch to list of patches
        :param patch: patch to add
        """
        self.patches.append(patch)

    def unpatch(self) -> None:
        """Unpatch all patches"""
        for patch in self.patches[:]:
            patch._unpatch()
            self.patches.remove(patch)


class Patch(ABC):
    is_patch = True
    manager = PatchManager()
    name = "patch"

    def patch(self) -> None:
        """Patch function. Call _patch function and adds patch to manager."""
        patch = self._patch()
        self.manager.add_patch(patch)

    @abstractmethod
    def _patch(self) -> "Patch":
        """Abstract patch function. Returns patch object."""

    @abstractmethod
    def _unpatch(self) -> None:
        """Abstract unpatch function."""

    def unpatch(self) -> None:
        """Unpatch function. Call _unpatch function and removes patch from manager."""
        self.manager.unpatch()
