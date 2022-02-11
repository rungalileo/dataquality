from abc import abstractmethod
from typing import Any, Dict, Optional
from uuid import uuid4

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _save_hdf5_file


class BaseGalileoModelLogger(BaseGalileoLogger):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.epoch: Optional[int] = None

    @abstractmethod
    def validate(self) -> None:
        ...

    def _log(self) -> None:
        """Threaded logger target"""
        try:
            self.validate()
        except AssertionError as e:
            raise GalileoException(f"The provided logged data is invalid. {e}")
        data = self._get_data_dict()
        self.write_model_output(data)

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

    def write_model_output(self, data: Dict) -> None:
        """Creates an hdf5 file from the data dict"""
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        epoch, split = data["epoch"][0], data["split"][0]
        path = f"{location}/{split}/{epoch}"
        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        _save_hdf5_file(path, object_name, data)

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

    @abstractmethod
    def _get_data_dict(self) -> Dict:
        """Constructs a dictionary of arrays from logged model output data"""
