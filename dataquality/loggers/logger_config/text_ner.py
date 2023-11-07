from typing import Dict, List, Tuple

import numpy as np
from pydantic import ConfigDict, validator

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextNERLoggerConfig(BaseLoggerConfig):
    gold_spans: Dict[str, List[Tuple[int, int, str]]] = {}
    sample_length: Dict[str, int] = {}
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def get_sample_key(self, split: str, sample_id: int) -> str:
        return f"{split}_{sample_id}"

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("labels", always=True, pre=True, allow_reuse=True)
    def clean_labels(cls, labels: List[str]) -> List[str]:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if labels is not None:
            assert isinstance(labels, List), "Labels must be a list"
        if labels and len(labels) == 1:
            labels = [labels[0], f"NOT_{labels[0]}"]
        return labels

    def reset(self, factory: bool = False) -> None:
        """Don't clear the "ner" or "ner_config" variables if they are set"""
        if factory:
            return super().reset()
        nlp = None
        ner_config = None
        if self.helper_data.get("nlp") is not None:
            nlp = self.helper_data["nlp"]
        if self.helper_data.get("ner_config") is not None:
            ner_config = self.helper_data["ner_config"]
        super().reset()
        if nlp is not None:
            self.helper_data["nlp"] = nlp
        if ner_config is not None:
            self.helper_data["ner_config"] = ner_config


text_ner_logger_config = TextNERLoggerConfig()
