from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
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
    """

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
        """
        super().__init__(text=text, ids=ids, split=split, meta=meta)
        if labels is not None:
            self.labels = [[str(i) for i in tl] for tl in labels]  # type: ignore
        else:
            self.labels = []

    __logger_name__ = "text_multi_label"

    def validate(self) -> None:
        """
        Parent validation (text_classification) with additional validation on labels

        in multi_label modeling, each element in self.labels should itself be a list
        """
        super().validate()
        config.observed_num_tasks = len(self.labels[0])
        for ind, input_labels in enumerate(self.labels):
            assert isinstance(
                input_labels, (list, np.ndarray, pd.Series)
            ), "labels must be a list of lists in multi-label tasks"
            assert len(input_labels) == config.observed_num_tasks, (
                f"Each {self.split} input must have the same number of labels. "
                f"Expected {config.observed_num_tasks} based on record 0 but saw "
                f"{len(input_labels)} for input record {ind}."
            )
        config.update_file_config()

    def _get_input_dict(self) -> Dict[str, Any]:
        inp = dict(
            id=self.ids,
            text=self.text,
            split=self.split,
            data_schema_version=__data_schema_version__,
            **self.meta,
        )
        if self.split != Split.inference.value:
            gold_array = np.array(self.labels)
            for task_num in range(config.observed_num_tasks):
                inp[f"gold_{task_num}"] = gold_array[:, task_num]
        return inp

    @classmethod
    def split_dataframe(cls, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Overrides parent split because the multi-label case has different columns"""
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files
        prob_cols = [f"prob_{i}" for i in range(config.observed_num_tasks)]
        gold_cols = [f"gold_{i}" for i in range(config.observed_num_tasks)]
        prob = df_copy[["id"] + prob_cols + gold_cols]
        emb = df_copy[["id", "emb"]]
        ignore_cols = ["emb", "split_id"] + prob_cols + gold_cols
        other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
        data_df = df_copy[other_cols]
        return prob, emb, data_df

    @classmethod
    def validate_labels(cls) -> None:
        if not config.tasks:
            config.tasks = [f"task_{i}" for i in range(config.observed_num_tasks)]

        assert config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )

        assert len(config.tasks) == config.observed_num_tasks, (
            f"You set your task names as {config.tasks} ({len(config.tasks)} tasks "
            f"but based on training, your model has {config.observed_num_tasks} "
            f"tasks. Use dataquality.set_tasks_for_run to update your config tasks."
        )

        assert len(config.labels) == config.observed_num_tasks, (
            f"You set your labels to be {config.labels} ({len(config.labels)} tasks) "
            f"but based on training, your model has {config.observed_num_tasks} tasks. "
            f"Use dataquality.set_labels_for_run to update your config labels."
        )
        assert isinstance(config.observed_num_labels, list), (
            f"Is your task_type correct? The observed number of labels is "
            f"{config.observed_num_labels}, but this should be a list of ints, one per"
            f"task. Should task {config.task_type} be set?"
        )
        assert len(config.observed_num_labels) == config.observed_num_tasks, (
            "Something went wrong with model output logging. Based on training, the "
            f"observed number of labels per task is {config.observed_num_labels} "
            f"indicating {len(config.observed_num_labels)} tasks, but the observed "
            f"number of tasks is only {config.observed_num_tasks}. Ensure you are "
            f"using the logger properly and that your task_type is correct "
            f"({config.task_type})."
        )
        for task, task_labels, num_task_labels in zip(
            config.tasks, config.labels, config.observed_num_labels
        ):
            assert isinstance(task_labels, list), (
                "In the multi-label case, your config labels should be a list of lists "
                "of strings. See dataquality.set_labels_for_run to update your config "
                "labels."
            )
            assert len(task_labels) == num_task_labels, (
                f"Task {task} is set to have {len(task_labels)} labels ({task_labels}) "
                f"but based on training, your model has only {num_task_labels} labels "
                f"for that task. See dataquality.set_labels_for_run to update your "
                f"config labels"
            )
