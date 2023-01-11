import os
from threading import Thread
from time import sleep
from typing import Any, Callable, Iterable, List

from dataquality.exceptions import GalileoException


class ThreadPoolManager:
    """
    A class for managing the threaded logging calls throughout dataquality
    """

    THREADS: List[Thread] = []
    MAX_THREADS = os.cpu_count() or 10

    @staticmethod
    def add_thread(target: Callable, args: Iterable[Any] = None) -> None:
        """
        Start a new function in a thread and store that in the global list of threads

        :param target: The callable
        :param args: The arguments to the function
        :return: None
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

        :return: None
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
            sleep(0.05)
            ThreadPoolManager._cleanup()

    @staticmethod
    def _cleanup() -> None:
        """
        Cleans up the ThreadPoolManager, removing any dead ones

        :return: None
        """
        ThreadPoolManager.THREADS = [
            i for i in ThreadPoolManager.THREADS if i.is_alive()
        ]
