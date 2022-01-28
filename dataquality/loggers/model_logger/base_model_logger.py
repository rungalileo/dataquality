from abc import abstractmethod
from typing import Any, Dict, Optional
from uuid import uuid4

import vaex
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _save_hdf5_file, _try_concat_df


class BaseGalileoModelLogger(BaseGalileoLogger):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.epoch: Optional[int] = None

    def _log(self) -> None:
        """Threaded logger target implemented by child"""
        try:
            self.validate()
        except AssertionError as e:
            raise GalileoException(f"The provided logged data is invalid. {e}")
        data = self._get_data_dict()
        self.write_model_output(model_output=vaex.from_dict(data))

    @abstractmethod
    def _get_data_dict(self) -> Dict[str, Any]:
        """Returns the formatted data for hdf5 storage"""

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

    def write_model_output(self, model_output: DataFrame) -> None:
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )

        epoch, split = model_output[["epoch", "split"]][0]
        path = f"{location}/{split}/{epoch}"
        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        _save_hdf5_file(path, object_name, model_output)
        _try_concat_df(path)

    @abstractmethod
    def validate(self) -> None:
        super().validate()
        assert self.epoch is not None, "You didn't log an epoch!"
        assert isinstance(
            self.epoch, int
        ), f"epoch must be int but was {type(self.epoch)}"

    @classmethod
    def upload(cls) -> None:
        """The upload function is implemented in the sister DataConfig class"""
        BaseGalileoDataLogger.get_logger(TaskType[cls.__logger_name__]).upload()

    @staticmethod
    def get_model_logger_attr(cls: object) -> str:
        """
        Returns the attribute that corresponds to the GalileoModelLogger class.
        This assumes only 1 GalileoModelLogger object exists in the class

        :param cls: The class
        :return: The attribute name
        """
        for attr in dir(cls):
            member_class = getattr(cls, attr)
            if isinstance(member_class, BaseGalileoModelLogger):
                return attr
        raise AttributeError("No model logger attribute found!")
