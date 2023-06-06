from collections import defaultdict
from typing import DefaultDict, List, Optional, Set

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextMultiLabelLoggerConfig(BaseLoggerConfig):
    labels: Optional[List[List[str]]] = None
    observed_num_labels: Optional[List[int]] = None
    observed_labels: DefaultDict[int, Set] = defaultdict(set)
    tasks: Optional[List[str]] = None
    observed_num_tasks: int = 0
    binary: bool = True  # For binary multi label

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
                assert isinstance(task_labels, List), (
                    "Labels must be a list of lists. If you are running a binary "
                    "multi-label case, use `dq.set_tasks_for_run(labels, binary=True)"
                )
                if len(task_labels) == 1:
                    task_labels = [f"NOT_{task_labels[0]}", task_labels[0]]
                cleaned_labels.append([str(i) for i in task_labels])
        return cleaned_labels


text_multi_label_logger_config = TextMultiLabelLoggerConfig()
