from enum import Enum, unique
from typing import Dict, List, Tuple

import numpy as np
from pydantic import validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


@unique
class TaggingSchema(str, Enum):
    BIO = "BIO"
    BILOU = "BILOU"
    BIOES = "BIOES"
    # IOB2 = "IOB2"
    # IOB = "IOB"
    # BILOES = "BILOES"


class TextNERLoggerConfig(BaseLoggerConfig):
    num_emb: int = 0
    gold_spans: Dict[str, List[Tuple[int, int, str]]] = {}
    sample_length: Dict[str, int] = {}

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
