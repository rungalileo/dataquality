import weakref


class RefManager:
    def __init__(self, cleanup_func):
        self.cleanup_func = cleanup_func
        self._instances = set()

    def register(self, instance):
        weak_instance = weakref.ref(instance, self._cleanup_callback)
        self._instances.add(weak_instance)

    def _cleanup_callback(self, weak_instance):
        self.cleanup_func()
        self._instances.discard(weak_instance)


class Cleanup:
    def __init__(self, cleanup_manager):
        self.cleanup_manager = cleanup_manager
        self.cleanup_manager.register(self)

    def __del__(self):
        # This will be called when the object is garbage collected.
        # However, it's not guaranteed to be called in all situations.
        self.cleanup_manager._cleanup_callback(weakref.ref(self))
