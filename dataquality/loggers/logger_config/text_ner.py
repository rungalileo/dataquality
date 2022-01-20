from typing import List, Optional

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig

MAX_SPANS = 5


class TextNERLoggerConfig(BaseLoggerConfig):
    labels: Optional[List[str]] = None
    observed_num_labels: int = 0
    max_spans: int = MAX_SPANS
    num_emb: int = 0
    # max_gold_spans: Dict[str, int] = {
    #     Split.training.value: 0,
    #     Split.test.value: 0,
    #     Split.validation.value: 0,
    #     Split.inference.value: 0
    # }
    # max_pred_spans: Dict[str, int] = {
    #     Split.training.value: 0,
    #     Split.test.value: 0,
    #     Split.validation.value: 0,
    #     Split.inference.value: 0
    # }

    class Config:
        validate_assignment = True

    @validator("labels", always=True, pre=True, allow_reuse=True)
    def clean_labels(cls, labels: List[str]) -> List[str]:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if labels is not None:
            assert isinstance(labels, List), "Labels must be a list"
        if labels and len(labels) == 1:
            labels = [labels[0], f"NOT_{labels[0]}"]
        return labels

    @validator("max_spans", always=True, pre=True, allow_reuse=True)
    def max_spans_limit(cls, new_span_max: int) -> int:
        return min(MAX_SPANS, new_span_max)


text_ner_logger_config = TextNERLoggerConfig()
