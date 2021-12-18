from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import vaex
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.vaex import _save_hdf5_file, _try_concat_df


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

    * Embeddings: List[List[Union[int,float]]]. The Embeddings per text sample input.
    Only one embedding vector is allowed per input
    * Probabilities from forward passes during model training/evaluation.
    List[List[float]]
    * ids: Indexes of each input field: List[int]. These IDs must align with the input
    IDs for each sample input. This will be used to join them together for analysis
    by Galileo.
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
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoModelLoggerAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        # We add validation here instead of requiring the params at init because
        # for lightning callbacks, we add these automatically for the user, so they
        # can create the config in their training loop and we will manage this metadata
        assert self.split, "Your GalileoModelConfig has no split!"
        assert self.epoch is not None, "Your GalileoModelConfig has no epoch!"

        self.emb = self._convert_tensor_ndarray(self.emb, "Embedding")
        self.probs = self._convert_tensor_ndarray(self.probs, "Prob")
        self.ids = self._convert_tensor_ndarray(self.ids)

        assert self.emb.ndim == 2, "Only one embedding vector is allowed per input."

        assert emb_len and prob_len and id_len, (
            f"All of emb, probs, and ids for your GalileoModelConfig must be set, but "
            f"got emb:{bool(emb_len)}, probs:{bool(prob_len)}, ids:{bool(id_len)}"
        )

        assert emb_len == prob_len == id_len, (
            f"All of emb, probs, and ids for your GalileoModelConfig must be the same "
            f"length, but got (emb, probs, ids) -> ({emb_len},{prob_len}, {id_len})"
        )
        if self.split:
            # User may manually pass in 'train' instead of 'training'
            # but we want it to conform
            self.split = Split.training.value if self.split == "train" else self.split
            assert (
                isinstance(self.split, str)
                and self.split in Split.get_valid_attributes()
            ), (
                f"Split should be one of {Split.get_valid_attributes()} "
                f"but got {self.split}"
            )

        if self.epoch:
            assert isinstance(self.epoch, int), (
                f"If set, epoch must be int but was " f"{type(self.epoch)}"
            )

    def write_model_output(self, model_output: DataFrame) -> None:
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )

        self._set_num_labels(model_output)
        epoch, split = model_output[["epoch", "split"]][0]
        path = f"{location}/{split}/{epoch}"
        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        _save_hdf5_file(path, object_name, model_output)
        _try_concat_df(path)

    def _log(self) -> None:
        """Threaded logger target implemented by child"""
        try:
            self.validate()
        except AssertionError as e:
            raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
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
