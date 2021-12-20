from typing import List, Optional

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextMultiLabelLoggerConfig(BaseLoggerConfig):
    labels: Optional[List[List[str]]] = None
    observed_num_labels: Optional[List[int]] = None
    tasks: Optional[List[str]] = None
    observed_num_tasks: int = 0

    class Config:
        validate_assignment = True

    @validator("labels", always=True, pre=True)
    def clean_labels(cls, labels: List[List[str]]) -> List[List[str]]:
        cleaned_labels = []
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if labels is not None:
            assert isinstance(labels, List), "Labels must be a list of lists"
            for task_labels in labels:
                assert isinstance(task_labels, List), "Labels must be a list of lists"
                if len(task_labels) == 1:
                    task_labels = [task_labels[0], f"NOT_{task_labels[0]}"]
                cleaned_labels.append([str(i) for i in task_labels])
        return cleaned_labels


text_multi_label_logger_config = TextMultiLabelLoggerConfig()
