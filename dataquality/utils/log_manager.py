# from concurrent.futures.process import ProcessPoolExecutor
import multiprocessing as mp
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable

from dataquality.schemas.task_type import TaskType

lock = mp.Lock()


class LogManager:
    """
    A class for managing the async logging calls throughout dataquality

    Depending on the task, we use either a ThreadPoolExecutor or a ProcessPoolExecutor

    For TC, MLTC, and IC, we use a ThreadPoolExecutor, because the majority of the work
    is in I/O, and these tasks require write access to global variables (logger_config
    vars like `observed_num_labels` and `observed_ids`

    For NER, we use a ProcessPoolExecutor because the majority of the work is CPU bound,
    in `process_sample` (see TextNERModelLogger), not I/O. It does NOT need any global
    variable write access, so it's safe to be using a ProcessPoolExecutor
    """

    MAX_LOGGERS = 3
    PEXECUTOR = ProcessPoolExecutor(max_workers=MAX_LOGGERS)
    TEXECUTOR = ThreadPoolExecutor(max_workers=MAX_LOGGERS)

    @staticmethod
    def add_logger(target: Callable, task_type: TaskType) -> None:
        """
        Start a new function in a thread and store that in the global list of threads

        :param target: The callable
        :param args: The arguments to the function
        :return: None
        """
        executor = (
            LogManager.PEXECUTOR
            if task_type == TaskType.text_ner
            else LogManager.TEXECUTOR
        )
        executor.submit(target)

    @staticmethod
    def wait_for_loggers() -> None:
        """
        Joins all currently active processes and waits for all to be done

        :return: None
        """
        LogManager.TEXECUTOR.shutdown()
        LogManager.PEXECUTOR.shutdown()
        LogManager.PEXECUTOR = ProcessPoolExecutor(max_workers=LogManager.MAX_LOGGERS)
        LogManager.TEXECUTOR = ThreadPoolExecutor(max_workers=LogManager.MAX_LOGGERS)
