from abc import abstractmethod
from typing import List, Any

from vaex.dataframe import DataFrame

from dataquality.core.integrations.config import GalileoModelConfig
from dataquality.loggers.config.base_config import BaseGalileoConfig
from dataquality.utils.thread_pool import ThreadPoolManager


class BaseGalileoModelConfig(BaseGalileoConfig):
    def __init__(self):
        super().__init__()
        self.is_data_config = False
        self.is_model_config = True

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def _log(self) -> None:
        """The target log function implemented by the child class"""
        pass

    @staticmethod
    def _add_threaded_log(self) -> None:
        try:
            self._log()
        except Exception as e:
            print(f"An error occurred while logging: {str(e)}")
            import traceback

            traceback.print_exc()

    def log(self):
        """The top level log function that try/excepts it's child"""
        ThreadPoolManager.add_thread(target=self._add_threaded_log)

    @staticmethod
    def upload():
        pass

    def get_configs(self, task_type: str) -> List:
        super().get_config(task_type)
        configs = {i.__name__: i for i in BaseGalileoModelConfig.__subclasses__()}
        return configs[task_type]

    @abstractmethod
    def write_model_output(self, model_output: DataFrame) -> None:
        pass

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in GalileoModelConfig.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of GalileoModelConfig. "
                f"Only {GalileoModelConfig.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
