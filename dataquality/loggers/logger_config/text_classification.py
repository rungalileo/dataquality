from typing import List, Optional, Set

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextClassificationLoggerConfig(BaseLoggerConfig):
    labels: Optional[List[str]] = None
    observed_num_labels: int = 0
    observed_labels: Set[str] = set()

    class Config:
        validate_assignment = True

    @validator("labels", always=True, pre=True, allow_reuse=True)
    def clean_labels(cls, labels: List[str]) -> List[str]:
        if labels is None:
            return labels
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if labels is not None:
            assert isinstance(labels, List), "Labels must be a list"
        if labels and len(labels) == 1:
            labels = [f"NOT_{labels[0]}", labels[0]]
        return [str(i) for i in labels]


text_classification_logger_config = TextClassificationLoggerConfig()
