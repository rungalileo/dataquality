import os
import shutil
from abc import abstractmethod
from enum import Enum, unique
from glob import glob
from typing import List, Optional, Type, TypeVar, Union

import numpy as np

from dataquality.core._config import _Config, config
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.base_logger_config import (
    BaseLoggerConfig,
    base_logger_config,
)
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


T = TypeVar("T", bound="BaseGalileoLogger")


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
    logger_config: BaseLoggerConfig = base_logger_config

    def __init__(self) -> None:
        self.split: Optional[str] = None

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return BaseLoggerAttributes.get_valid()

    @abstractmethod
    def validate(self) -> None:
        ...

    def is_valid(self) -> bool:
        try:
            self.validate()
        except AssertionError:
            return False
        return True

    @abstractmethod
    def log(self) -> None:
        ...

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
            if attr == "Embedding":
                assert (
                    len(arr.shape) == 2
                ), f"{attr} tensor must be 2D shape, but got shape {arr.shape}"
            elif attr == "Prob":
                if config.task_type != TaskType.text_multi_label:
                    assert (
                        len(arr.shape) == 2
                    ), f"{attr} tensor must be 2D shape, but got shape {arr.shape}"
                else:
                    # Because probs in multi label may not have a clear shape (each
                    # task may have a different number of probabilities
                    arr = np.array(arr, dtype=object)
            elif attr == "Labels" and config.task_type == TaskType.text_multi_label:
                arr = np.array(arr, dtype=object)
        return np.array(arr)

    @staticmethod
    def validate_task(task_type: Union[str, TaskType]) -> None:
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
        ...

    @classmethod
    def get_all_subclasses(cls: Type[T]) -> List[Type[T]]:
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_all_subclasses())

        return all_subclasses

    @classmethod
    def get_logger(cls: Type[T], task_type: TaskType) -> Type[T]:
        cls.validate_task(task_type)
        loggers = {i.__logger_name__: i for i in cls.get_all_subclasses()}
        return loggers[task_type]

    @classmethod
    def doc(cls) -> None:
        print(cls.__doc__)
