from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.special import softmax

from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig,
    text_multi_label_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


class TextMultiLabelModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

    def __init__(
        self,
        embs: Optional[Union[List[List], List[np.ndarray]]] = None,
        probs: Optional[Union[List[List], List[np.ndarray]]] = None,
        logits: Optional[Union[List[List], List[np.ndarray]]] = None,
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
        # For each record, the task probability vector should be the same length
        for task_ind in range(self.logger_config.observed_num_labels):
            num_prob_per_task = set([len(p[task_ind]) for p in self.probs])
            assert len(num_prob_per_task) == 1, (
                f"Task {task_ind} has an inconsistent number of probabilities in this "
                f"batch. Found {num_prob_per_task} unique number of labels. "
                f"Every input for a given task should have the same number of "
                f"values in its probability vector."
            )

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
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        probs = []
        for sample_logits in sample_logits:
            task_probs = []
            for task_logits in sample_logits:
                task_logits = self._convert_tensor_ndarray(task_logits)
                task_probs.append(softmax(task_logits.astype(np.float_), axis=-1))
            probs.append(task_probs)
        return np.array(probs, dtype=object)
