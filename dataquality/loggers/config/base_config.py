from abc import abstractmethod
from enum import unique, Enum
from typing import List, Optional, TypeVar, Type

import numpy as np

from dataquality.core._config import _Config
from dataquality.exceptions import GalileoException

try:
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


T = TypeVar('T', bound='BaseGalileoDataConfig')


@unique
class TaskTypes(str, Enum):
    """Valid task types supported for logging by Galileo"""
    text_classification = "text_classification"

    @staticmethod
    def get_valid_tasks() -> List[str]:
        return list(map(lambda x: x.value, TaskTypes))


@unique
class BaseConfigAttributes(str, Enum):
    """A collection of all default attributes across all configs"""
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
        return list(map(lambda x: x.value, BaseConfigAttributes))


class BaseGalileoConfig:
    """
    An abstract base class that all model configs and data config inherit from
    """
    LOG_FILE_DIR = f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs"

    def __init__(self):
        self.is_model_config = False
        self.is_data_config = False

    @abstractmethod
    def validate(self):
        pass

    def is_valid(self) -> bool:
        try:
            self.validate()
        except AssertionError:
            return False
        return True

    @abstractmethod
    def log(self):
        pass

    @staticmethod
    def upload():
        pass

    @staticmethod
    def _convert_tensor_ndarray(arr: List, attr: Optional[str] = None) -> np.ndarray:
        """Handles numpy arrays and tensors conversions"""
        if TORCH_AVAILABLE:
            if isinstance(arr, Tensor):
                arr = arr.detach().cpu().numpy()
        if isinstance(arr, np.ndarray):
            if attr in ("Embedding", "Prob"):
                shp = arr.shape
                assert len(
                    shp) == 2, f"{attr} tensor must be 2D shape, but got shape {shp}"
        return np.array(arr)

    @abstractmethod
    def get_config(self, task_type: str):
        if task_type not in TaskTypes.get_valid_tasks():
            raise GalileoException(
                f"Task type {task_type} not valid. Choose one of "
                f"{TaskTypes.get_valid_tasks()}"
            )
        pass

