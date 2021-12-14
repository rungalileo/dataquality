import os
from glob import glob
from typing import Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
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
                input_labels, list
            ), "labels must be a list of lists in multi-label tasks"
            assert len(input_labels) == config.observed_num_tasks, (
                f"Each {self.split} input must have the same number of labels. "
                f"Expected {config.observed_num_tasks} based on record 0 but saw "
                f"{len(input_labels)} for input record {ind}."
            )
        config.update_file_config()

    def log(self) -> None:
        self.validate()
        write_input_dir = (
            f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
            f"{config.current_run_id}"
        )
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)
        inp = dict(
            id=self.ids,
            text=self.text,
            split=self.split,
            data_schema_version=__data_schema_version__,
            gold=None,
            **self.meta,
        )
        if self.split != Split.inference.value:
            gold_array = np.array(self.labels)
            for task_num in range(config.observed_num_tasks):
                inp[f"gold_{task_num}"] = gold_array[:, task_num]

        df = vaex.from_pandas(pd.DataFrame(inp))
        file_path = f"{write_input_dir}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
        if os.path.isfile(file_path):
            new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
            os.rename(file_path, new_name)
            vaex.concat([df, vaex.open(new_name)]).export(file_path)
            os.remove(new_name)
        else:
            df.export(file_path)
        df.close()

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
        other_cols = [
            i for i in df_copy.get_column_names() if i not in ignore_cols
        ]
        data_df = df_copy[other_cols]
        return prob, emb, data_df
