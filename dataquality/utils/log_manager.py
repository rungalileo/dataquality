import os
import threading
# from concurrent.futures.process import ProcessPoolExecutor
import multiprocessing as mp
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Thread
from time import sleep
from typing import Any, Callable, Iterable, List

from dataquality.exceptions import GalileoException

lock = mp.Lock()


class LogManager:
    """
    A class for managing the async logging calls throughout dataquality
    """

    MAX_LOGGERS = 3
    EXECUTOR = ProcessPoolExecutor(max_workers=MAX_LOGGERS)

    @staticmethod
    def add_logger(target: Callable, args: Iterable[Any] = None) -> None:
        """
        Start a new function in a thread and store that in the global list of threads

        :param target: The callable
        :param args: The arguments to the function
        :return: None
        """
        LogManager.EXECUTOR.submit(target, *(args or []))


    @staticmethod
    def wait_for_loggers() -> None:
        """
        Joins all currently active processes and waits for all to be done

        :return: None
        """
        LogManager.EXECUTOR.shutdown()
