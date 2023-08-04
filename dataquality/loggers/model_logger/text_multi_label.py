from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.special import expit

from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig,
    text_multi_label_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.dq_logger import get_dq_logger


class TextMultiLabelModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

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
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )

    def _has_len(self, arr: Any) -> bool:
        """Checks if an array has length

        Array can be list, numpy array, or tensorflow tensor. Tensorflow tensors don't
        let you call len(), they throw a TypeError so we catch that here and check
        shape https://github.com/tensorflow/tensorflow/blob/master/tensorflow/...
        python/framework/ops.py#L929
        """
        try:
            has_len = len(arr) != 0
        except TypeError:
            has_len = bool(arr.shape[0])
        return has_len

    def validate_and_format(self) -> None:
        """
        Validates that the current config is correct.
        * embs, probs, and ids must exist and be the same length
        :return:
        """
        super().validate_and_format()
        for ind, prob_per_label in enumerate(self.probs):
            assert len(prob_per_label) == self.logger_config.observed_num_labels, (
                f"Expected {self.logger_config.observed_num_labels} probability vectors "
                f"per input (based on input data logging) but found "
                f"{len(prob_per_label)} for input {ind}."
            )
        # check that the number of logits equals the number of labels
        for ind, logits_per_label in enumerate(self.logits):
            assert len(logits_per_label) == self.logger_config.observed_num_labels, (
                f"Expected {self.logger_config.observed_num_labels} logits vectors "
                f"per input (based on input data logging) but found "
                f"{len(logits_per_label)} for input {ind}."
            )
        has_logits = self._has_len(self.logits)
        if has_logits:
            self.logits = self._convert_tensor_ndarray(self.logits, "Prob")
            self.probs = self.convert_logits_to_probs(self.logits)
            del self.logits

        self.embs = self._convert_tensor_ndarray(self.embs, "Embedding")
        self.ids = self._convert_tensor_ndarray(self.ids)

        embs_len = len(self.embs)
        probs_len = len(self.probs)
        ids_len = len(self.ids)

        assert self.embs.ndim == 2, "Only one embedding vector is allowed per input."

        assert embs_len and probs_len and ids_len, (
            f"All of emb, probs, and ids for your logger must be set, but "
            f"got emb:{bool(embs_len)}, probs:{bool(probs_len)}, ids:{bool(ids_len)}"
        )

        assert embs_len == probs_len == ids_len, (
            f"All of emb, probs, and ids for your logger must be the same "
            f"length, but got (emb, probs, ids) -> ({embs_len},{probs_len}, {ids_len})"
        )

        # User may manually pass in 'train' instead of 'training' / 'test' vs 'testing'
        # but we want it to conform
        try:
            self.split = Split[self.split].value
        except KeyError:
            get_dq_logger().error("Provided a bad split", split=self.split)
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

    def _get_data_dict(self) -> Dict[str, Any]:
        data = defaultdict(list)
        for record_id, prob, emb in zip(self.ids, self.probs, self.embs):
            record = {
                "id": record_id,
                "epoch": self.epoch,
                "split": self.split,
                "emb": emb,
                "prob": prob,
                "data_schema_version": __data_schema_version__,
            }
            for k in record.keys():
                data[k].append(record[k])
        return data

    def convert_logits_to_probs(
        self, sample_logits: Union[List, np.ndarray]
    ) -> np.ndarray:
        if not isinstance(sample_logits, np.ndarray):
            sample_logits = self._convert_tensor_ndarray(sample_logits)

        return expit(sample_logits)
