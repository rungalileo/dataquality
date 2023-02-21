import os
import threading
from threading import Thread
from time import sleep
from typing import Any, Callable, Iterable, List, Optional

from dataquality.exceptions import GalileoException

lock = threading.Lock()


class ThreadPoolManager:
    """
    A class for managing the threaded logging calls throughout dataquality
    """

    THREADS: List[Thread] = []
    MAX_THREADS = os.cpu_count() or 10

    @staticmethod
    def add_thread(target: Callable, args: Optional[Iterable[Any]] = None) -> None:
        """
        Start a new function in a thread and store that in the global list of threads

        :param target: The callable
        :param args: The arguments to the function
        """
        ThreadPoolManager.wait_for_thread()
        thread = Thread(target=target, args=args or [])
        try:
            thread.start()
            ThreadPoolManager.THREADS.append(thread)
        # Push up to the user
        except Exception as e:
            raise GalileoException(e)

    @staticmethod
    def wait_for_threads() -> None:
        """
        Joins all currently active threads and waits for all to be done
        """
        ThreadPoolManager._cleanup()
        # Waits for each thread to finish
        for i in ThreadPoolManager.THREADS:
            i.join()
        ThreadPoolManager._cleanup()

    @staticmethod
    def wait_for_thread() -> None:
        """
        Waits for an open slot in the ThreadPool for another thread.
        """
        ThreadPoolManager._cleanup()
        while len(ThreadPoolManager.THREADS) >= ThreadPoolManager.MAX_THREADS:
            # this sleep is necessary to prevent the thread from hogging compute
            sleep(0.05)
            ThreadPoolManager._cleanup()

    @staticmethod
    def _cleanup() -> None:
        """
        Cleans up the ThreadPoolManager, removing any dead ones
        """
        ThreadPoolManager.THREADS = [
            i for i in ThreadPoolManager.THREADS if i.is_alive()
        ]
