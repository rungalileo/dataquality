import os
import shutil
from abc import abstractmethod
from enum import Enum, unique
from functools import lru_cache
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from dataquality.core._config import config, config_data
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.base_logger_config import (
    BaseLoggerConfig,
    base_logger_config,
)
from dataquality.schemas.split import Split, conform_split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dq_logger import upload_dq_log_file
from dataquality.utils.file import _shutil_rmtree_retry
from dataquality.utils.imports import hf_available, tf_available, torch_available
from dataquality.utils.patcher import PatchManager
from dataquality.utils.tf import is_tf_2

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
    gold_conf_prob = "gold_conf_prob"
    gold_loss_prob = "gold_loss_prob"
    gold_loss_prob_label = "gold_loss_prob_label"
    pred_conf_prob = "pred_conf_prob"
    pred_loss_prob = "pred_loss_prob"
    pred_loss_prob_label = "pred_loss_prob_label"
    gold = "gold"
    embs = "embs"
    probs = "probs"
    logits = "logits"
    epoch = "epoch"
    aum = "aum"
    text_tokenized = "text_tokenized"
    gold_spans = "gold_spans"
    pred_emb = "pred_emb"
    gold_emb = "gold_emb"
    pred_spans = "pred_spans"
    text_token_indices = "text_token_indices"
    text_token_indices_flat = "text_token_indices_flat"
    log_helper_data = "log_helper_data"
    inference_name = "inference_name"
    image = "image"
    token_label_positions = "token_label_positions"
    token_label_offsets = "token_label_offsets"
    label = "label"
    token_deps = "token_deps"
    text = "text"
    id = "id"
    token_gold_probs = "token_gold_probs"
    tokenized_label = "tokenized_label"
    # Seq2Seq attributes
    input = "input"
    target = "target"
    generated_output = "generated_output"
    input_cutoff = "input_cutoff"
    target_cutoff = "target_cutoff"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, BaseLoggerAttributes))


class BaseGalileoLogger:
    """
    An abstract base class that all model logger and data loggers inherit from
    """

    __logger_name__ = ""
    LOG_FILE_DIR = f"{config_data.DEFAULT_GALILEO_CONFIG_DIR}/logs"  # type: ignore
    logger_config: BaseLoggerConfig = base_logger_config

    def __init__(self) -> None:
        self.split: Optional[str] = None
        self.inference_name: Optional[str] = None

    @property
    def proj_run(self) -> str:
        """Returns the project and run id

        Example:
            proj-id/run-id
        """
        return f"{config.current_project_id}/{config.current_run_id}"

    @property
    def write_output_dir(self) -> str:
        """Returns the path to the output directory for the current run

        Example:
            /Users/username/.galileo/logs/proj-id/run-id
        """
        return f"{BaseGalileoLogger.LOG_FILE_DIR}/{self.proj_run}"

    @property
    def split_name(self) -> str:
        """Returns the name of the current split

        If the split is inference, it will return the name of the inference
        concatenated to the end of the split name

        Example:
            training
            inference_inf-name1
        """
        split = self.split
        if split == Split.inference:
            split = f"{split}_{self.inference_name}"
        return str(split)

    @property
    def split_name_path(self) -> str:
        """Returns the path part of the current split

        If the split is inference, it will return the name of the inference
        run after the split name

        Example:
            training
            inference/inf-name1
        """
        split = self.split
        if split == Split.inference:
            split = f"{split}/{self.inference_name}"
        return str(split)

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return BaseLoggerAttributes.get_valid()

    @abstractmethod
    def validate_and_format(self) -> None:
        """Validates params passed in during logging. Implemented by child"""

    def set_split_epoch(self) -> None:
        """Sets the split for the current logger

        If the split is not set, it will use the split set in the logger config
        """
        if not self.split:
            if self.logger_config.cur_split:
                self.split = self.logger_config.cur_split
            else:
                raise GalileoException(
                    "You didn't log a split and did not set a split. Use "
                    "'dataquality.set_split' to set the split"
                )

        # Inference split must have inference name
        if self.split == Split.inference and self.inference_name is None:
            if self.logger_config.cur_inference_name is not None:
                self.inference_name = self.logger_config.cur_inference_name
            else:
                raise GalileoException(
                    "For inference split you must either log an inference name "
                    "or set it before logging. Use `dataquality.set_split` to set "
                    "inference_name"
                )

        self.split = self.validate_split(self.split)
        # Set this config variable in validation, right before logging split data
        setattr(self.logger_config, f"{self.split}_logged", True)

    def is_valid(self) -> bool:
        try:
            self.validate_and_format()
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
        if torch_available():
            from torch import Tensor

            if isinstance(arr, Tensor):
                arr = arr.detach().cpu().numpy()
        if tf_available():
            import tensorflow as tf

            if isinstance(arr, tf.Tensor):
                if is_tf_2():
                    with tf.device("cpu"):
                        arr = tf.identity(arr).numpy()
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
        return np.array(arr) if not isinstance(arr, np.ndarray) else arr

    @staticmethod
    def _convert_tensor_to_py(v: Any) -> Union[str, int]:
        """Converts pytorch and tensorflow tensors to strings or ints"""
        if isinstance(v, (int, str)):
            return v
        if tf_available():
            import tensorflow as tf

            if isinstance(v, tf.Tensor):
                v = v.numpy()
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                else:
                    v = int(v)
        if torch_available():
            from torch import Tensor

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
    def validate_task(task_type: Union[str, TaskType]) -> TaskType:
        """Raises error if task type is not a valid TaskType"""
        try:
            return TaskType[task_type]
        except KeyError:
            raise GalileoException(
                f"Task type {task_type} not valid. Choose one of "
                f"{TaskType.get_valid_tasks()}"
            )

    @classmethod
    def _cleanup(cls) -> None:
        """Cleans up the current run data and metadata locally

        Does so by deleting the run directory and resetting the logger config

        Example:
            # Deletes all files in the run directory
            /Users/username/.galileo/logs/proj-id/run-id
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
                # Sometimes the directory is not deleted immediately
                # This can happen if the client is using an nfs
                # so we try again after a short delay
                try:
                    shutil.rmtree(path)
                except OSError:
                    _shutil_rmtree_retry(path)

        cls.logger_config.reset()
        pm = PatchManager()
        pm.unpatch()

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
        """Raises error if split is not a valid Split"""
        return conform_split(split).value

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
        if hf_available():
            import datasets

            return isinstance(df, datasets.Dataset)
        return False

    @staticmethod
    @lru_cache(1)
    def _label_idx_map(labels: Tuple[str]) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(labels)}

    @property
    def label_idx_map(self) -> Dict[str, int]:
        """Convert a list of labels to a dictionary of label to index

        Example:
        --------
        >>> labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        >>> label_idx_map(labels)
        {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}
        """
        return self._label_idx_map(tuple(self.logger_config.labels or []))

    def labels_to_idx(self, gold_sequence: List[str]) -> np.ndarray:
        """Convert a list of labels to a np array of indices

        Example:
        --------
        # labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        >>> gold_sequence = ["O", "B-LOC", "B-PER", "I-PER", "O"]
        >>> labels_to_idx(gold_sequence)
        [0, 3, 1, 2, 0]
        """
        return np.array([self.label_idx_map[s] for s in gold_sequence])
