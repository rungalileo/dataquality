from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig,
    text_multi_label_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
from dataquality.schemas import __data_schema_version__


class TextMultiLabelModelLogger(TextClassificationModelLogger):
    """
    Class for logging model outputs of Multi Label Text classification models to Galileo

    * emb: (Embeddings) List[List[Union[int,float]]]. Embeddings per text sample input.
    Only one embedding vector is allowed per input (len(emb) == len(text)
    and emb.shape==2)
    * logits: Output from forward pass during model training/evalutation.
    List[List[List[float]]] or List[np.ndarray].
    For each text input, a list of lists of floats is expected (one list/array per task)
    The number of inner lists must be the number of tasks (matching the labels logged).
    The order of the inner lists is assumed to match the order of the inner list of
    labels when logging input data (matching the tasks provided by the call to
    dataquality.set_tasks_for_run()).
    * probs: (Probabilities) deprecated, use logits instead.
    * ids: Indexes of each input field: List[int]. These IDs must align with the input
    IDs for each sample input. This will be used to join them together for analysis
    by Galileo.
    """

    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

    def __init__(
        self,
        emb: Union[List, np.ndarray] = None,
        probs: Union[List[List[List]], List[np.ndarray]] = None,
        logits: Union[List[List[List]], List[np.ndarray]] = None,
        ids: Union[List, np.ndarray] = None,
        split: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        self.probs = probs if probs is not None else []
        self.logits = logits if logits is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.epoch = epoch

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        super().validate()
        for ind, probs_per_task in enumerate(self.probs):
            assert len(probs_per_task) == self.logger_config.observed_num_tasks, (
                f"Expected {self.logger_config.observed_num_tasks} probability vectors "
                f"per input (based on input data logging) but found "
                f"{len(probs_per_task)} for input {ind}."
            )
            for task_ind, task_probs in enumerate(probs_per_task):
                assert (
                    len(task_probs) > 0
                ), f"Cannot log empty probability list for task {task_ind}."
        # For each record, the task probability vector should be the same length
        for task_ind in range(self.logger_config.observed_num_tasks):
            num_prob_per_task = set([len(p[task_ind]) for p in self.probs])
            assert len(num_prob_per_task) == 1, (
                f"Task {task_ind} has an inconsistent number of probabilities in this "
                f"batch. Found {num_prob_per_task} unique number of labels. "
                f"Every input for a given task should have the same number of "
                f"values in its probability vector."
            )

    def _get_data_dict(self) -> Dict[str, Any]:
        data = defaultdict(list)
        for record_id, prob_per_task, emb in zip(self.ids, self.probs, self.emb):
            # Handle binary classification by making it 2-class classification
            record = {
                "id": record_id,
                "epoch": self.epoch,
                "split": self.split,
                "emb": emb,
                "data_schema_version": __data_schema_version__,
            }
            # Break out the probabilities and predictions into a col per task
            for task_num in range(self.logger_config.observed_num_tasks):
                task_probs: List[float] = prob_per_task[task_num]
                if len(task_probs) == 1:  # Handle binary classification case
                    task_probs = [task_probs[0], 1 - task_probs[0]]
                record[f"prob_{task_num}"] = task_probs
                record[f"pred_{task_num}"] = int(np.argmax(task_probs))

            for k in record.keys():
                data[k].append(record[k])

        return data

    def _set_num_labels(self, data: Dict) -> None:
        num_labels_per_task = []
        for task_num in range(self.logger_config.observed_num_tasks):
            len(data[f"prob_{task_num}"][0])
            num_labels = len(data[f"prob_{task_num}"][0])
            num_labels_per_task.append(num_labels)
        self.logger_config.observed_num_labels = num_labels_per_task

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes() and not key.startswith("prob_"):
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)

    def convert_logits_to_probs(
        self, sample_logits: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Converts logits to probs via softmax per sample"""
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        probs = []
        for sample_logits in sample_logits:
            task_probs = []
            for task_logits in sample_logits:
                task_probs.append(
                    super().convert_logits_to_probs(task_logits.astype(np.float_))
                )
            probs.append(task_probs)
        return np.array(probs, dtype=object)
