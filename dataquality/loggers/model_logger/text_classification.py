from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

import numpy as np
import vaex
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__


@unique
class GalileoModelLoggerAttributes(str, Enum):
    emb = "emb"
    probs = "probs"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelLoggerAttributes))


class TextClassificationModelLogger(BaseGalileoModelLogger):
    """
    Class for logging model output data of Text Classification models to Galileo.

    * emb: List[Union[List, np.ndarray, torch.Tensor, tf.Tensor]]. The Embeddings per
    text sample input. Only one embedding vector is allowed per input sample.
    the `emb` parameter can be formatted either as:
        * A list of List[float]
        * A list of numpy arrays
        * A list of tensorflow tensors
        * A list of pytorch tensors
    * prob: List[Union[List, np.ndarray, torch.Tensor, tf.Tensor]] probabilities from
    forward passes during model training/evaluation.
    the `prob` parameter can be formatted either as:
        * A list of List[float]
        * A list of numpy arrays
        * A list of tensorflow tensors
        * A list of pytorch tensors
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
        ids: Union[List, np.ndarray] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
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
        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        self.emb = self._convert_tensor_ndarray(self.emb, "Embedding")
        self.probs = self._convert_tensor_ndarray(self.probs, "Prob")
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

    def write_model_output(self, model_output: DataFrame) -> None:
        self._set_num_labels(model_output)
        super().write_model_output(model_output)

    def _log(self) -> None:
        """Threaded logger target implemented by child"""
        try:
            self.validate()
        except AssertionError as e:
            raise GalileoException(f"The provided logged data is invalid. {e}")
        data = self._get_data_dict()
        self.write_model_output(model_output=vaex.from_dict(data))

    def _get_data_dict(self) -> Dict[str, Any]:
        data = defaultdict(list)
        for record_id, prob, emb in zip(self.ids, self.probs, self.emb):
            # Handle binary classification by making it 2-class classification
            p = [prob[0], 1 - prob[0]] if len(prob) == 1 else prob
            record = {
                "id": record_id,
                "epoch": self.epoch,
                "split": self.split,
                "emb": emb,
                "prob": p,
                "pred": int(np.argmax(prob)),
                "data_schema_version": __data_schema_version__,
            }
            for k in record.keys():
                data[k].append(record[k])
        return data

    def _set_num_labels(self, df: DataFrame) -> None:
        self.logger_config.observed_num_labels = len(df[:1]["prob"].values[0])

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
