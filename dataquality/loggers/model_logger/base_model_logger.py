import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from scipy.special import softmax

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.stdout_logger import get_stdout_logger
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _save_hdf5_file


class BaseGalileoModelLogger(BaseGalileoLogger):
    def __init__(
        self,
        embs: Union[List, np.ndarray] = None,
        probs: Union[List, np.ndarray] = None,
        logits: Union[List, np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.embs: Union[List, np.ndarray] = embs if embs is not None else []
        self.logits: Union[List, np.ndarray] = logits if logits is not None else []
        self.probs: Union[List, np.ndarray] = probs if probs is not None else []
        self.ids: Union[List, np.ndarray] = ids if ids is not None else []
        self.epoch = epoch
        self.split: str = split
        self.inference_name = inference_name

    def _log(self) -> None:
        """Threaded logger target"""
        try:
            self.validate()
        except AssertionError as e:
            get_stdout_logger().error("Validation of data failed")
            raise GalileoException(
                f"The provided logged data is invalid: {e}"
            ) from None
        data = self._get_data_dict()
        self.write_model_output(data)

    def _add_threaded_log(self) -> None:
        try:
            self._log()
        except Exception as e:
            get_stdout_logger().exception("Logging of model outputs failed")
            warnings.warn(f"An issue occurred while logging: {str(e)}")

    def log(self) -> None:
        """The top level log function that try/excepts it's child"""
        get_stdout_logger().info("Starting logging process from thread")
        ThreadPoolManager.add_thread(target=self._add_threaded_log)

    def write_model_output(self, data: Dict) -> None:
        """Creates an hdf5 file from the data dict"""
        get_stdout_logger().info("Writing model output")
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        epoch, split = data["epoch"][0], data["split"][0]

        if split == Split.inference:
            inference_name = data["inference_name"][0]
            path = f"{location}/{split}/{inference_name}"
        else:
            path = f"{location}/{split}/{epoch}"

        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        get_stdout_logger().info("Saving hdf5 file")
        _save_hdf5_file(path, object_name, data)

    @abstractmethod
    def validate(self) -> None:
        super().validate()
        if self.split == Split.inference and self.inference_name is None:
            if self.logger_config.cur_inference_name is not None:
                self.inference_name = self.logger_config.cur_inference_name
            else:
                raise GalileoException(
                    "For inference split you must either log an inference name "
                    "or set it before logging. Use `dataquality.set_split` to set"
                    "inference_name"
                )
        # Epoch can be ignored for inference split
        if self.split != Split.inference and self.epoch is None:
            if self.logger_config.cur_epoch is not None:
                self.epoch = self.logger_config.cur_epoch
            else:
                raise GalileoException(
                    "You must either log an epoch or set it before logging. Use "
                    "`dataquality.set_epoch` to set the epoch"
                )

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

    @abstractmethod
    def _get_data_dict(self) -> Dict:
        """Constructs a dictionary of arrays from logged model output data"""

    def convert_logits_to_probs(
        self, sample_logits: Union[List[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Converts logits to probs via softmax"""
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        return softmax(np.array(sample_logits), axis=-1)
