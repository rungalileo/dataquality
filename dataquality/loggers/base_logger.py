import os
import shutil
from abc import abstractmethod
from enum import Enum, unique
from glob import glob
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from dataquality.core._config import _Config, config
from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType

try:
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@unique
class BaseLoggerAttributes(str, Enum):
    """A collection of all default attributes across all loggers"""

    text = "text"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging
    prob = "prob"
    gold = "gold"
    emb = "emb"
    probs = "probs"
    epoch = "epoch"
    data_error_potential = "data_error_potential"
    aum = "aum"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, BaseLoggerAttributes))


class BaseGalileoLogger:
    """
    An abstract base class that all model logger and data loggers inherit from
    """

    __logger_name__ = ""
    LOG_FILE_DIR = f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs"

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.split: Optional[str] = None

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return BaseLoggerAttributes.get_valid()

    @abstractmethod
    def validate(self) -> None:
        pass

    def is_valid(self) -> bool:
        try:
            self.validate()
        except AssertionError:
            return False
        return True

    @abstractmethod
    def log(self) -> None:
        pass

    @staticmethod
    def _convert_tensor_ndarray(
        arr: Union[List, np.ndarray], attr: Optional[str] = None
    ) -> np.ndarray:
        """Handles numpy arrays and tensors conversions"""
        if TORCH_AVAILABLE:
            if isinstance(arr, Tensor):
                arr = arr.detach().cpu().numpy()
        if TF_AVAILABLE:
            if isinstance(arr, tf.Tensor):
                arr = arr.cpu().numpy()
        if isinstance(arr, np.ndarray):
            if attr in ("Embedding", "Prob"):
                shp = arr.shape
                assert (
                    len(shp) == 2
                ), f"{attr} tensor must be 2D shape, but got shape {shp}"
        return np.array(arr)

    @staticmethod
    def validate_task(task_type: str) -> None:
        if task_type not in TaskType.get_valid_tasks():
            raise GalileoException(
                f"Task type {task_type} not valid. Choose one of "
                f"{TaskType.get_valid_tasks()}"
            )

    @classmethod
    def _cleanup(cls) -> None:
        """
        Cleans up the current run data locally
        """
        assert config.current_project_id
        assert config.current_run_id
        location = (
            f"{cls.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        print("🧹 Cleaning up")
        for path in glob(f"{location}/*"):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

    @classmethod
    def upload(cls) -> None:
        pass

    @classmethod
    def get_all_subclasses(cls) -> List[Type["BaseGalileoLogger"]]:
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_all_subclasses())

        return all_subclasses

    @classmethod
    def get_logger(cls, task_type: str) -> Type["BaseGalileoLogger"]:
        cls.validate_task(task_type)
        loggers = {i.__logger_name__: i for i in cls.get_all_subclasses()}
        return loggers[task_type]
