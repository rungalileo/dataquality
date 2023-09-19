import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from scipy.special import expit, softmax

from dataquality import config
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException, GalileoWarning, LogBatchError
from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dq_logger import get_dq_logger
from dataquality.utils.hdf5_store import _save_hdf5_file
from dataquality.utils.thread_pool import ThreadPoolManager

analytics = Analytics(ApiClient, config)  # type: ignore


class BaseGalileoModelLogger(BaseGalileoLogger):
    log_file_ext = "hdf5"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
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
        """Threaded logger target

        If validation fails with an assertion error, we stop the model training process
        (something is wrong)

        If validation fails with a LogBatchError, we simply warn and skip logging this
        batch, but do not halt model training
        (this batch is bad, but we can continue logging)
        """
        try:
            self.validate_and_format()
        except AssertionError as e:
            get_dq_logger().error(
                "Validation of data failed", split=self.split, epoch=self.epoch
            )
            raise GalileoException(
                f"The provided logged data is invalid: {e}"
            ) from None
        except LogBatchError as e:
            warnings.warn(
                f"An error occurred logging this batch, it will be skipped. Error: {e}",
                GalileoWarning,
            )
            return
        data = self._get_data_dict()
        data = self._downcast_data_dict(data)
        self.write_model_output(data)

    def _downcast_data_dict(self, data: Dict) -> Dict:
        """Downcasts any float/int 64 types to 32 before saving batch"""
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    data[key] = val.astype(np.int32)
                elif val.dtype == np.float64:
                    data[key] = val.astype(np.float32)
        return data

    def _add_threaded_log(self) -> None:
        try:
            self._log()
        except Exception as e:
            get_dq_logger().exception(
                "Logging of model outputs failed", split=self.split, epoch=self.epoch
            )
            err_msg = (
                f"An issue occurred while logging model outputs. Address any issues in "
                f"your logging and make sure to call dq.init before restarting:\n"
                f"{repr(e)}"
            )
            warnings.warn(err_msg)
            try:
                analytics.set_config(config)
                analytics.capture_exception(e)
            except Exception:
                pass
            self.logger_config.exception = err_msg

    def log(self) -> None:
        """The top level log function that try/excepts its child"""
        self.check_for_logging_failures()
        # We validate split and epoch before entering the thread because we reference
        # global variables (cur_split and cur_epoch) that are subject to change
        # between subsequent threads
        self.set_split_epoch()
        ThreadPoolManager.add_thread(target=self._add_threaded_log)

    def write_model_output(self, data: Dict) -> None:
        """Creates an hdf5 file from the data dict"""
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        split = data["split"][0]

        if split == Split.inference:
            inference_name = data["inference_name"][0]
            path = f"{location}/{split}/{inference_name}"
        else:
            epoch = data["epoch"][0]
            path = f"{location}/{split}/{epoch}"

        object_name = f"{str(uuid4()).replace('-', '')[:12]}.{self.log_file_ext}"
        self._write_dict_to_disk(path, object_name, data)

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        _save_hdf5_file(path, object_name, data)

    def set_split_epoch(self) -> None:
        super().set_split_epoch()

        # Non-inference split must have an epoch
        if self.split != Split.inference and self.epoch is None:
            if self.logger_config.cur_epoch is not None:
                self.epoch = self.logger_config.cur_epoch
            else:
                raise GalileoException(
                    "You must either log an epoch or set it before logging. Use "
                    "`dataquality.set_epoch` to set the epoch"
                )

    def upload(self) -> None:
        """The upload function is implemented in the sister DataConfig class"""
        BaseGalileoDataLogger.get_logger(TaskType[self.__logger_name__])().upload()

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

    def convert_logits_to_prob_binary(self, sample_logits: np.ndarray) -> np.ndarray:
        """Converts logits to probs in the binary case

        Takes the sigmoid of the single class logits and adds the negative
        lass prediction (1-class pred)
        """
        sample_probs = expit(sample_logits)
        probs_1 = np.expand_dims(sample_probs, axis=-1)
        probs_0 = 1 - probs_1
        probs = np.concatenate([probs_0, probs_1], axis=-1)
        return probs

    def convert_logits_to_probs(
        self, sample_logits: Union[List[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Converts logits to probs via softmax"""
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        if not isinstance(sample_logits, np.ndarray):
            sample_logits = self._convert_tensor_ndarray(sample_logits)

        # If shape is (num_samples, 1) or (num_samples,) then we have a binary case
        if len(sample_logits.shape) == 1 or sample_logits.shape[-1] == 1:
            if len(sample_logits.shape) > 1:
                # Remove final empty dimension if it's there
                sample_logits = sample_logits.reshape(-1)
            return self.convert_logits_to_prob_binary(sample_logits)

        if not isinstance(sample_logits, np.ndarray):
            sample_logits = np.ndarray(sample_logits)
        return softmax(sample_logits, axis=-1)
