import warnings
from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


@unique
class GalileoModelLoggerAttributes(str, Enum):
    emb = "emb"
    probs = "probs"
    logits = "logits"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"
    inference_name = "inference_name"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelLoggerAttributes))


class TextClassificationModelLogger(BaseGalileoModelLogger):
    """
    Class for logging model output data of Text Classification models to Galileo.

    * emb: Union[List, np.ndarray, torch.Tensor, tf.Tensor]. The Embeddings per
    text sample input. Only one embedding vector is allowed per input sample.
    the `emb` parameter can be formatted either as:
        * np.ndarray
        * torch.tensor / tf.tensor
        * A list of List[float]
        * A list of numpy arrays
        * A list of tensorflow tensors
        * A list of pytorch tensors
    * logits: Union[List, np.ndarray, torch.Tensor, tf.Tensor] outputs from
     forward pass. If provided, probs will be converted automatically and DO NOT need
     to be provided. Can be formatted either as:
        * np.ndarray
        * torch.tensor / tf.tensor
        * A list of List[float]
        * A list of numpy arrays
        * A list of tensorflow tensors
        * A list of pytorch tensors
    * probs: Deprecated - the probabilities for each output sample (use logits instead)
    * ids: Indexes of each input field: List[int]. These IDs must align with the input
    IDs for each sample input. This will be used to join them together for analysis
    by Galileo.
    * split: The model training/test/validation split for the samples being logged
    """

    __logger_name__ = "text_classification"
    logger_config = text_classification_logger_config

    def __init__(
        self,
        emb: Union[List, np.ndarray] = None,
        probs: Union[List, np.ndarray] = None,
        logits: Union[List, np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb: Union[List, np.ndarray] = emb if emb is not None else []
        self.logits: Union[List, np.ndarray] = logits if logits is not None else []
        self.probs: Union[List, np.ndarray] = probs if probs is not None else []
        self.ids: Union[List, np.ndarray] = ids if ids is not None else []
        self.split: str = split
        self.epoch = epoch

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that this logger accepts
        :return: List[str]
        """
        return GalileoModelLoggerAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        super().validate()

        if len(self.logits):
            self.logits = self._convert_tensor_ndarray(self.logits, "Prob")
            self.probs = self.convert_logits_to_probs(self.logits)
            del self.logits
        elif len(self.probs):
            warnings.warn("Usage of probs is deprecated, use logits instead")
            self.probs = self._convert_tensor_ndarray(self.probs, "Prob")

        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        self.emb = self._convert_tensor_ndarray(self.emb, "Embedding")
        self.ids = self._convert_tensor_ndarray(self.ids)

        assert self.emb.ndim == 2, "Only one embedding vector is allowed per input."

        assert emb_len and prob_len and id_len, (
            f"All of emb, probs, and ids for your logger must be set, but "
            f"got emb:{bool(emb_len)}, probs:{bool(prob_len)}, ids:{bool(id_len)}"
        )

        assert emb_len == prob_len == id_len, (
            f"All of emb, probs, and ids for your logger must be the same "
            f"length, but got (emb, probs, ids) -> ({emb_len},{prob_len}, {id_len})"
        )

        # User may manually pass in 'train' instead of 'training' / 'test' vs 'testing'
        # but we want it to conform
        try:
            self.split = Split[self.split].value
        except KeyError:
            raise AssertionError(
                f"Split should be one of {Split.get_valid_attributes()} "
                f"but got {self.split}"
            )

        if self.epoch:
            assert isinstance(self.epoch, int), (
                f"If set, epoch must be int but was " f"{type(self.epoch)}"
            )
            if self.epoch > self.logger_config.last_epoch:
                self.logger_config.last_epoch = self.epoch

    def write_model_output(self, model_output: Dict) -> None:
        self._set_num_labels(model_output)
        super().write_model_output(model_output)

    def _get_data_dict(self) -> Dict[str, Any]:
        data = defaultdict(list)
        for record_id, prob, emb in zip(self.ids, self.probs, self.emb):
            # Handle binary classification by making it 2-class classification
            p = [prob[0], 1 - prob[0]] if len(prob) == 1 else prob
            record = {
                "id": record_id,
                "epoch": self.epoch,
                "split": Split[self.split].value,
                "emb": emb,
                "prob": p,
                "pred": int(np.argmax(prob)),
                "data_schema_version": __data_schema_version__,
            }
            if self.split == Split.inference:
                record.update(inference_name=self.inference_name)
            for k in record.keys():
                data[k].append(record[k])
        return data

    def _set_num_labels(self, data: Dict) -> None:
        self.logger_config.observed_num_labels = len(data["prob"][0])

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
