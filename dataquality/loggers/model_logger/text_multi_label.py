from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.special import softmax

from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig,
    text_multi_label_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


class TextMultiLabelModelLogger(TextClassificationModelLogger):
    """
    Class for logging model outputs of Multi Label Text classification models to Galileo

    * embs: (Embeddings) List[List[Union[int,float]]]. Embeddings per text sample input.
    Only one embedding vector is allowed per input (len(embs) == len(text)
    and embs.shape==2)
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

    ex:
    .. code-block:: python

        dq.set_epoch(0)
        dq.set_split("train")

        # 3 samples, embedding dim 768. Only 1 embedding vector can be logged for all
        # tasks. Each task CANNOT have it's own embedding vector
        embs: np.ndarray = np.random.rand(3, 768)
        # Logits per task. In this example, tasks "task_0" and "task_2" have 3 classes
        # but task "task_1" has 2
        logits: List[np.ndarray] = [
            np.random.rand(3, 3),  # 3 samples, 3 classes
            np.random.rand(3, 3),  # 3 samples, 2 classes
            np.random.rand(3, 3)  # 3 samples, 3 classes
        ]
        ids: List[int] = [0, 1, 2]

        dq.log_model_outputs(embs=embs, logits=logits, ids=ids)
    """

    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

    def __init__(
        self,
        embs: Union[List, np.ndarray] = None,
        probs: Union[List[List[List]], List[np.ndarray]] = None,
        logits: Union[List[List[List]], List[np.ndarray]] = None,
        ids: Union[List, np.ndarray] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
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

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * embs, probs, and ids must exist and be the same length
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

    def _get_data_dict_binary(self) -> Dict[str, Any]:
        """In the binary ML case, we can optimize this operation

        Because all probs are of the same dimension, we can create a numpy array and
        not loop through every row input
        """
        num_samples = len(self.ids)
        data = {
            "id": self.ids,
            "emb": self.embs,
            "split": [Split[self.split].value] * num_samples,
            "epoch": [self.epoch] * num_samples,
            "data_schema_version": [__data_schema_version__] * num_samples,
        }
        task_probs = np.array(self.probs)
        for task_num in range(self.logger_config.observed_num_tasks):
            probs = task_probs[:, task_num]
            # Only the single probability was logged. Add the negative case
            if probs.shape[-1] == 1:
                probs = np.column_stack((1 - probs, probs))
            data[f"prob_{task_num}"] = probs
            data[f"pred_{task_num}"] = np.argmax(probs, axis=1)
        return data

    def _get_data_dict(self) -> Dict[str, Any]:
        if self.logger_config.binary:
            return self._get_data_dict_binary()
        data = defaultdict(list)
        for record_id, prob_per_task, emb in zip(self.ids, self.probs, self.embs):
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
                    task_probs = [1 - task_probs[0], task_probs[0]]
                record[f"prob_{task_num}"] = np.array(task_probs, dtype=np.float32)
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

    def convert_logits_to_prob_binary(self, sample_logits: np.ndarray) -> np.ndarray:
        """Converts logits to probs in the binary case

        Takes the sigmoid of the single class logits and adds the negative
        lass prediction (1-class pred)
        """
        assert sample_logits.ndim == 2, (
            f"In binary multi-label, your logits should have 2 dimensions, but "
            f"they currently have {sample_logits.ndim}. Do you mean to use to "
            f"binary multi-label? If not, call dq.set_tasks_for_run(tasks) without "
            f"the binary=True flag. Or call dq.init() to reset."
        )
        return super().convert_logits_to_prob_binary(sample_logits)

    def convert_logits_to_probs(
        self, sample_logits: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Converts logits to probs via softmax per sample

        In the case of binary multi-label, we don't run softmax, we use sigmoid
        """
        if self.logger_config.binary:
            return self.convert_logits_to_prob_binary(
                np.array(sample_logits, dtype=np.float32)
            )
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
