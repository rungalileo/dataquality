from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextNERLoggerConfig(BaseLoggerConfig):
    gold_spans: Dict[str, List[Tuple[int, int, str]]] = {}
    sample_length: Dict[str, int] = {}
    user_data: Dict[str, Any] = {}

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    def get_sample_key(self, split: str, sample_id: int) -> str:
        return f"{split}_{sample_id}"

    @validator("labels", always=True, pre=True, allow_reuse=True)
    def clean_labels(cls, labels: List[str]) -> List[str]:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if labels is not None:
            assert isinstance(labels, List), "Labels must be a list"
        if labels and len(labels) == 1:
            labels = [labels[0], f"NOT_{labels[0]}"]
        return labels


text_ner_logger_config = TextNERLoggerConfig()
