import os
import shutil
from abc import abstractmethod
from enum import Enum, unique
from glob import glob
from typing import Any, List, Optional, Type, TypeVar, Union

import numpy as np

from dataquality.core._config import ConfigData, config
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.base_logger_config import (
    BaseLoggerConfig,
    base_logger_config,
)
from dataquality.schemas.split import Split, conform_split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.cloud import is_galileo_cloud
from dataquality.utils.dq_logger import upload_dq_log_file
from dataquality.utils.tf import TF_AVAILABLE, is_tf_2

try:
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TF_AVAILABLE:
    import tensorflow as tf

try:
    import datasets

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


T = TypeVar("T", bound="BaseGalileoLogger")


@unique
class BaseLoggerAttributes(str, Enum):
    """A collection of all default attributes across all loggers"""

    texts = "texts"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging
    prob = "prob"
    gold = "gold"
    embs = "embs"
    probs = "probs"
    logits = "logits"
    epoch = "epoch"
    data_error_potential = "data_error_potential"
    aum = "aum"
    text_tokenized = "text_tokenized"
    gold_spans = "gold_spans"
    pred_emb = "pred_emb"
    gold_emb = "gold_emb"
    pred_spans = "pred_spans"
    dep_scores = "dep_scores"
    text_token_indices = "text_token_indices"
    gold_dep = "gold_dep"
    pred_dep = "pred_dep"
    text_token_indices_flat = "text_token_indices_flat"
    log_helper_data = "log_helper_data"
    inference_name = "inference_name"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, BaseLoggerAttributes))


class BaseGalileoLogger:
    """
    An abstract base class that all model logger and data loggers inherit from
    """

    __logger_name__ = ""
    LOG_FILE_DIR = f"{ConfigData.DEFAULT_GALILEO_CONFIG_DIR}/logs"
    logger_config: BaseLoggerConfig = base_logger_config

    def __init__(self) -> None:
        self.split: Optional[str] = None
        self.inference_name: Optional[str] = None

    def write_output_dir(self) -> str:
        return (
            f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
            f"{config.current_run_id}"
        )

    def split_name(self) -> str:
        split = self.split
        if split == Split.inference:
            split = f"{split}_{self.inference_name}"
        return str(split)

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return BaseLoggerAttributes.get_valid()

    @abstractmethod
    def validate(self) -> None:
        """Validates params passed in during logging. Implemented by child"""

    def set_split_epoch(self) -> None:
        if not self.split:
            if self.logger_config.cur_split:
                self.split = self.logger_config.cur_split
            else:
                raise GalileoException(
                    "You didn't log a split and did not set a split. Use "
                    "'dataquality.set_split' to set the split"
                )
        self.split = self.validate_split(self.split)
        # Set this config variable in validation, right before logging split data
        setattr(self.logger_config, f"{self.split}_logged", True)

    def is_valid(self) -> bool:
        try:
            self.validate()
        except AssertionError:
            return False
        return True

    @classmethod
    def non_inference_logged(cls) -> bool:
        """Return true if training, test, or validation data is logged

        If just inference data is logged then append data rather than overwriting.
        This flag is also used by the api to know which processing jobs to run.
        """
        return any(
            [
                cls.logger_config.training_logged,
                cls.logger_config.test_logged,
                cls.logger_config.validation_logged,
            ]
        )

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
                if is_tf_2():
                    arr = arr.cpu().numpy()
                else:  # Just for TF1.x
                    arr = arr.numpy()
        if isinstance(arr, np.ndarray):
            if attr == "Embedding":
                assert (
                    len(arr.shape) == 2
                ), f"{attr} tensor must be 2D shape, but got shape {arr.shape}"
            elif attr == "Prob":
                if config.task_type != TaskType.text_multi_label:
                    assert (
                        arr.ndim == 2
                    ), f"Probs/logits must have ndim=2, but got shape {arr.shape}"
                else:
                    # Because probs in multi label may not have a clear shape (each
                    # task may have a different number of probabilities
                    arr = np.array(arr, dtype=object)
                    assert arr.ndim > 1, (
                        f"Probs/logits must have at least 2 dimensions, "
                        f"they have {arr.ndim}"
                    )
            elif attr == "Labels" and config.task_type == TaskType.text_multi_label:
                arr = np.array(arr, dtype=object)
        return np.array(arr)

    @staticmethod
    def _convert_tensor_to_py(v: Any) -> Union[str, int]:
        """Converts pytorch and tensorflow tensors to strings or ints"""
        if isinstance(v, (int, str)):
            return v
        if TF_AVAILABLE:
            if isinstance(v, tf.Tensor):
                v = v.numpy()
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                else:
                    v = int(v)
        if TORCH_AVAILABLE:
            if isinstance(v, Tensor):
                v = int(v.numpy())  # Torch tensors cannot hold strings
        if isinstance(v, np.ndarray):
            if np.issubdtype(v, np.integer):
                v = int(v)
            else:
                v = str(int(v))
        if not isinstance(v, (int, str)):
            raise GalileoException(
                f"Logged data should be of type int, string, pytorch tensor, "
                f"or tf tensor, but got {type(v)}"
            )
        return v

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
        Cleans up the current run data and metadata locally
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
        cls.logger_config.reset()

    def upload(self) -> None:
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

    @classmethod
    def validate_split(cls, split: Union[str, Split]) -> str:
        split = conform_split(split).value
        if is_galileo_cloud() and split == Split.inference:
            raise GalileoException(
                "You cannot log inference data from a Galileo Cloud account, only "
                "enterprise accounts can access this feature. Please email us at "
                "team@rungalileo.io for more information."
            )
        return split

    @classmethod
    def check_for_logging_failures(cls) -> None:
        """When a threaded logging call fails, it sets the logger_config.exception

        If that field is set, raise an exception here and stop the main process
        """
        # If a currently active thread crashed, check and raise a top level exception
        if cls.logger_config.exception:
            upload_dq_log_file()
            raise GalileoException(cls.logger_config.exception)

    @classmethod
    def is_hf_dataset(cls, df: Any) -> bool:
        if HF_AVAILABLE:
            return isinstance(df, datasets.Dataset)
        return False
