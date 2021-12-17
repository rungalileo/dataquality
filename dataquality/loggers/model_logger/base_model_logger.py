from abc import abstractmethod
from typing import Any, Dict, Optional

from vaex.dataframe import DataFrame

from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager


class BaseGalileoModelLogger(BaseGalileoLogger):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.epoch: Optional[int] = None

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def _log(self) -> None:
        """The target log function implemented by the child class"""

    def _add_threaded_log(self) -> None:
        try:
            self._log()
        except Exception as e:
            print(f"An error occurred while logging: {str(e)}")
            import traceback

            traceback.print_exc()

    def log(self) -> None:
        """The top level log function that try/excepts it's child"""
        ThreadPoolManager.add_thread(target=self._add_threaded_log)

    @abstractmethod
    def write_model_output(self, model_output: DataFrame) -> None:
        ...

    @classmethod
    def upload(cls) -> None:
        """The upload function is implemented in the sister DataConfig class"""
        BaseGalileoDataLogger.get_logger(TaskType[cls.__logger_name__]).upload()

    @staticmethod
    def get_model_logger_attr(cls: object) -> str:
        """
        Returns the attribute that corresponds to the GalileoModelConfig class.
        This assumes only 1 GalileoModelConfig object exists in the class

        :param cls: The class
        :return: The attribute name
        """
        for attr in dir(cls):
            member_class = getattr(cls, attr)
            if isinstance(member_class, BaseGalileoModelLogger):
                return attr
        raise AttributeError("No GalileoModelConfig attribute found!")
