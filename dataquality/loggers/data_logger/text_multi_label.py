import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.loggers.data_logger.text_classification import (
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

    * text: The raw text inputs for model training. List[str]
    * labels: the list of ground truth labels aligned to each text field. Each text
    field input must have the same number of labels (which must be the number of tasks)
    List[List[str]]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[int]]
    * split: The split for training/test/validation
    """

    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = (
        text_multi_label_logger_config  # type: ignore
    )

    def __init__(
        self,
        text: List[str] = None,
        labels: List[List[str]] = None,
        ids: List[int] = None,
        split: str = None,
        meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param labels: the ground truth labels aligned to each text field.
        List[List[str]]
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(text=text, ids=ids, split=split, meta=meta)
        if labels is not None:
            self.labels = [[str(i) for i in tl] for tl in labels]  # type: ignore
        else:
            self.labels = []

    def validate(self) -> None:
        """
        Parent validation (text_classification) with additional validation on labels

        in multi_label modeling, each element in self.labels should itself be a list
        """
        super().validate()
        self.logger_config.observed_num_tasks = len(self.labels[0])
        for ind, input_labels in enumerate(self.labels):
            assert isinstance(
                input_labels, (list, np.ndarray, pd.Series)
            ), "labels must be a list of lists in multi-label tasks"
            assert len(input_labels) == self.logger_config.observed_num_tasks, (
                f"Each {self.split} input must have the same number of labels. "
                f"Expected {self.logger_config.observed_num_tasks} based on record 0 "
                f"but saw {len(input_labels)} for input record {ind}."
            )

    def _get_input_df(self) -> DataFrame:
        inp = dict(
            id=self.ids,
            text=self.text,
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
        for task, task_labels, num_task_labels in zip(
            cls.logger_config.tasks,
            cls.logger_config.labels,
            cls.logger_config.observed_num_labels,
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

    def _get_num_labels(self, df: DataFrame) -> List[int]:
        return [len(i) for i in df[:1]["prob"].values[0]]
