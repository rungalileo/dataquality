from enum import Enum, unique
from typing import List, Optional

import numpy as np
from pydantic import validator
from vaex.dataframe import DataFrame

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig

MAX_SPANS = 5


@unique
class TaggingSchema(str, Enum):
    BIO = "BIO"
    IOB2 = "IOB2"
    IOB = "IOB"
    BILOU = "BILOU"
    BILOES = "BILOES"


class TextNERLoggerConfig(BaseLoggerConfig):
    max_spans: int = MAX_SPANS
    num_emb: int = 0
    input_data: Optional[DataFrame]

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
        arbitrary_types_allowed = True

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

    @validator("tagging_schema", allow_reuse=True)
    def validate_tagging_schema(cls, tagging_schema: str) -> str:
        tagging_schema = tagging_schema.upper()
        if tagging_schema not in list(TaggingSchema):
            raise ValueError(  # Ignore here because mypy can't find member_names_ attr
                f"Tagging schema {tagging_schema} invalid, must be one of "
                f"{TaggingSchema.member_names_}"  # type: ignore
            )
        return tagging_schema


text_ner_logger_config = TextNERLoggerConfig()
