from collections.abc import Callable
from typing import List, Any
from threading import Thread
from time import sleep


class ThreadPoolManager:
    """
    A class for managing the threaded logging calls throughout dataquality
    """
    THREADS: List[Thread] = []
    MAX_THREADS = 100


    @staticmethod
    def add_thread(target: Callable, args: List[Any] = None) -> None:
        """
        Start a new function in a thread and store that in the global list of threads

        :param target: The callable
        :param args: The arguments to the function
        :return: None
        """
        ThreadPoolManager.wait_for_thread()
        thread = Thread(target=target, args=args)
        thread.start()
        ThreadPoolManager.THREADS.append(thread)

    @staticmethod
    def wait_for_threads() -> None:
        """
        Joins all currently active threads and waits for all to be done

        :return: None
        """
        ThreadPoolManager._cleanup()
        # Waits for each thread to finish
        [i.join() for i in ThreadPoolManager.THREADS]
        ThreadPoolManager._cleanup()

    @staticmethod
    def wait_for_thread() -> None:
        """
        Waits for an open slot in the ThreadPool for another thread.
        """
        ThreadPoolManager._cleanup()
        while len(ThreadPoolManager.THREADS) >= ThreadPoolManager.MAX_THREADS:
            sleep(0.25)
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
