import warnings
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import MetaType
from dataquality.loggers.data_logger.text_classification import (
    MetasType,
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig,
    text_multi_label_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split

DATA_FOLDERS = ["emb", "prob", "data"]


class TextMultiLabelDataLogger(TextClassificationDataLogger):
    """
    Class for logging input data/metadata of Text Multi Label models to Galileo.

    * texts: The raw text inputs for model training. List[str]
    * task_labels: the list of ground truth labels aligned to each text field. Each text
    field input must have the same number of labels (which must be the number of tasks)
    List[List[str]]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[int]]
    * split: The split for training/test/validation

    .. code-block:: python

        task_labels = [["A", "B", "C"], ["Foo", "Bar"], ["Apple", "Orange", "Grape"]]
        tasks = ["Task_0", "Task_1", "Task_2"]
        dq.init("text_multi_label")
        dq.set_tasks_for_run(tasks)
        dq.set_labels_for_run(labels = task_labels)

        texts: List[str] = [
            "Text sample 1",
            "Text sample 2",
            "Text sample 3",
        ]

        task_labels: List[str] = [
            ["A", "Foo", "Grape"],
            ["C", "Foo", "Apple"],
            ["B", "Bar", "Orange"]
        ]

        ids: List[int] = [0, 1, 2]
        meta = {"sample_quality": [5.3, 9.1, 2.7]}
        split = "training"

        dq.log_data_samples(
            texts=texts, task_labels=task_labels, ids=ids, meta=meta, split=split
        )
    """

    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

    def __init__(
        self,
        texts: List[str] = None,
        labels: List[List[str]] = None,
        ids: List[int] = None,
        split: str = None,
        meta: MetasType = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param labels: the ground truth labels aligned to each text field.
        List[List[str]]
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(texts=texts, ids=ids, split=split, meta=meta)
        if labels is not None:
            self.labels: List[List[str]] = [
                [str(i) for i in tl] for tl in labels  # type: ignore
            ]
        else:
            self.labels = []

    def log_data_sample(
        self,
        *,
        text: str,
        id: int,
        label: Optional[str] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetaType] = None,
        task_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a single input sample for text multi-label
        :param text: str the text sample
        :param id: The sample ID
        :param split: train/test/validation/inference. Can be set here or via
            dq.set_split
        :param meta: Dict[str, Union[str, int, float]]. Metadata for the text sample
            Format is the {"metadata_field_name": metadata_field_value}
        :param task_labels: List[str] The label of each task for this sample
            Required if split is not inference
        """
        self.validate_kwargs(kwargs)
        if label:
            raise GalileoException("In multi-label, use task_labels instead of label")

        task_labels = [str(i) for i in task_labels] if task_labels else []

        self.texts = [text]
        self.ids = [id]
        self.split = split
        self.labels = [task_labels] if task_labels else []
        self.inference_name = inference_name
        self.meta = {i: [meta[i]] for i in meta} if meta else {}
        self.log()

    def log_data_samples(
        self,
        *,
        texts: List[str],
        ids: List[int],
        labels: Optional[List[str]] = None,
        split: Optional[Split] = None,
        inference_name: str = None,
        meta: MetasType = None,
        task_labels: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Log input samples for text multi-label

        :param texts: List[str] text samples
        :param ids: List[int,str] IDs for each text sample
        :param split: train/test/validation/inference. Can be set here or via
            dq.set_split
        :param meta: Dict[str, List[str, int, float]]. Metadata for each text sample
            Format is the {"metadata_field_name": [metdata value per sample]}
        :param task_labels: List[List[str]] list of labels for each task for each
            text sample. Required if not in inference
        """
        self.validate_kwargs(kwargs)
        if labels is not None:
            raise GalileoException("In multi-label, use task_labels instead of labels")
        self.texts = texts
        self.ids = ids
        self.split = split
        self.meta = meta or {}
        if task_labels is not None:
            self.labels = [[str(i) for i in tl] for tl in task_labels]
        else:
            self.labels = []
        self.log()

    def _process_label(self, batches: DefaultDict, label: Any) -> DefaultDict:
        """In multi-label, label will be a list of strings instead of a string"""
        # In binary multi-label, it will be a single string
        if self.logger_config.binary:
            return super()._process_label(batches, label)
        batches["label"].append(self._convert_tensor_ndarray(label).tolist())
        return batches

    def _log_dict(
        self, d: Dict, meta: Dict, split: Split = None, inference_name: str = None
    ) -> None:
        self.log_data_samples(
            texts=d["text"],
            task_labels=d["label"],
            ids=d["id"],
            split=split,
            inference_name=inference_name,
            meta=meta,
        )

    def _process_binary_labels(self) -> None:
        """In binary multi labels, users will simply log the samples that are
        active in this case. If tasks are ["A","B","C"] and sample 1 ground truth is
        ["A"], sample 2 is ["A","C"] the user will log
        [
            ("Sample 1", ["A"]),
            ("Sample 1", ["A","C"])
        ]

        We want to convert that to
        [
            ("Sample 1", ["A", "NOT_B", "NOT_C"]),
            ("Sample 1", ["A","NOT_B", "C"])
        ]
        """
        # Assert for mypy
        assert self.logger_config.tasks, (
            "You must call dq.set_tasks_for_run(..., binary=True) before logging"
            " in binary multi-label"
        )
        clean_task_labels = []
        for sample_task_labels in self.labels:
            clean_sample_labels = []
            sample_label_set = set(sample_task_labels)
            for task in self.logger_config.tasks:
                if task in sample_label_set:
                    clean_sample_labels.append(task)
                else:
                    clean_sample_labels.append(f"NOT_{task}")
            clean_task_labels.append(clean_sample_labels)
        self.labels = clean_task_labels

    def validate(self) -> None:
        """
        Parent validation (text_classification) with additional validation on labels

        in multi_label modeling, each element in self.labels should itself be a list
        """
        if self.logger_config.binary:
            self._process_binary_labels()
        self.logger_config.observed_num_tasks = len(self.labels[0])
        super().validate()

    def validate_logged_labels(self) -> None:
        for ind, input_labels in enumerate(self.labels):
            assert isinstance(
                input_labels, (list, np.ndarray, pd.Series)
            ), "labels must be a list of lists in multi-label tasks"
            assert len(input_labels) == self.logger_config.observed_num_tasks, (
                f"Each {self.split} input must have the same number of labels. "
                f"Expected {self.logger_config.observed_num_tasks} based on record 0 "
                f"but saw {len(input_labels)} for input record {ind}. If this is a "
                f"binary multi label and you are logging the active tasks, call"
                f"dq.set_tasks_for_run(tasks, binary=True) and log again"
            )

            for task_ind, task_label in enumerate(input_labels):
                # In binary case its just the 1 label so we make it a list to conform
                self._validate_task_labels(ind, task_ind, task_label)

    def _validate_task_labels(
        self, input_ind: int, task_ind: int, task_label: str
    ) -> None:
        # Capture the observed labels per task
        self.logger_config.observed_labels[task_ind].update({task_label})
        observed_task_labels = self.logger_config.observed_labels[task_ind]
        # If the user has already set labels validate logged ones are valid
        if self.logger_config.labels:
            set_task_labels = self.logger_config.labels[task_ind]
            assert observed_task_labels.issubset(set_task_labels), (
                f"The input labels you log must be exactly the same as the "
                f"labels you set in set_labels_for_run. Input record {input_ind}: "
                f"logged labels: {observed_task_labels}, set labels for that "
                f"task (task #{task_ind}): {set_task_labels}"
            )

    def _get_input_df(self) -> DataFrame:
        inp = dict(
            id=self.ids,
            text=self.texts,
            split=self.split,
            data_schema_version=__data_schema_version__,
            **self.meta,
        )
        if self.split != Split.inference.value:
            gold_array = np.array(self.labels)
            for task_num in range(self.logger_config.observed_num_tasks):
                inp[f"gold_{task_num}"] = gold_array[:, task_num]
        return vaex.from_pandas(pd.DataFrame(inp))

    @classmethod
    def _get_prob_cols(cls) -> List[str]:
        prob_cols = [f"prob_{i}" for i in range(cls.logger_config.observed_num_tasks)]
        gold_cols = [f"gold_{i}" for i in range(cls.logger_config.observed_num_tasks)]
        return ["id"] + prob_cols + gold_cols

    @classmethod
    def validate_labels(cls) -> None:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )
        if not cls.logger_config.tasks:
            cls.logger_config.tasks = [
                f"task_{i}" for i in range(cls.logger_config.observed_num_tasks)
            ]
            warnings.warn(
                f"No tasks were set for this run. Setting tasks to "
                f"{cls.logger_config.tasks}"
            )

        assert len(cls.logger_config.tasks) == cls.logger_config.observed_num_tasks, (
            f"You set your task names as {cls.logger_config.tasks} "
            f"({len(cls.logger_config.tasks)} tasks but based on training, your model "
            f"has {cls.logger_config.observed_num_tasks} "
            f"tasks. Use dataquality.set_tasks_for_run to update your config tasks."
        )

        assert len(cls.logger_config.labels) == cls.logger_config.observed_num_tasks, (
            f"You set your labels to be {cls.logger_config.labels} "
            f"({len(cls.logger_config.labels)} tasks) but based on training, your "
            f"model has {cls.logger_config.observed_num_tasks} tasks. "
            f"Use dataquality.set_labels_for_run to update your config labels."
        )
        assert isinstance(cls.logger_config.observed_num_labels, list), (
            f"Is your task_type correct? The observed number of labels is "
            f"{cls.logger_config.observed_num_labels}, but this should be a list of "
            f"ints, one per task. Should task {config.task_type} be set?"
        )
        assert (
            len(cls.logger_config.observed_num_labels)
            == cls.logger_config.observed_num_tasks
        ), (
            "Something went wrong with model output logging. Based on training, the "
            f"observed number of labels per task is "
            f"{cls.logger_config.observed_num_labels} indicating "
            f"{len(cls.logger_config.observed_num_labels)} tasks, but the observed "
            f"number of tasks is only {cls.logger_config.observed_num_tasks}. Ensure "
            f"you are using the logger properly and that your task_type is "
            f"correct ({config.task_type})."
        )
        for task_num, (task, task_labels, num_task_labels) in enumerate(
            zip(
                cls.logger_config.tasks,
                cls.logger_config.labels,
                cls.logger_config.observed_num_labels,
            )
        ):
            assert isinstance(task_labels, list), (
                "In the multi-label case, your config labels should be a list of lists "
                "of strings. See dataquality.set_labels_for_run to update your config "
                "labels."
            )
            assert len(task_labels) == num_task_labels, (
                f"Task {task} is set to have {len(task_labels)} labels ({task_labels}) "
                f"but based on training, your model has {num_task_labels} labels "
                f"for that task. See dataquality.set_labels_for_run to update your "
                f"config labels"
            )
            logged_labels = cls.logger_config.observed_labels[task_num]
            assert logged_labels.issubset(task_labels), (
                f"The labels set for task #{task_num} ({task}) do not align with the "
                "observed labels during logging. Labels logged for this task during "
                f"input logging: {logged_labels} -- labels set for this task: "
                f"{task_labels}. Update the labels for this task using "
                "`dq.set_labels_for_run`"
            )

    def _get_num_labels(self, df: DataFrame) -> List[int]:
        return [len(i) for i in df[:1]["prob"].values[0]]
