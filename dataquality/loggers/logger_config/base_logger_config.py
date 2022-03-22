from typing import Any, Dict, Optional

from pydantic import BaseModel, validator

from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    tagging_schema: Optional[TaggingSchema]
    last_epoch: int = 0
    cur_epoch: Optional[int]
    cur_split: Optional[Split]
    cur_inference_name: Optional[str]
    training_logged: bool = False
    validation_logged: bool = False
    test_logged: bool = False
    inference_logged: bool = False

    class Config:
        validate_assignment = True

    def reset(self) -> None:
        """Reset all class vars"""
        self.__init__()  # type: ignore

    @validator("cur_split")
    def inference_sets_inference_name(
        cls, field_value: Split, values: Dict[str, Any]
    ) -> Split:
        if field_value == Split.inference:
            assert values.get(
                "cur_inference_name"
            ), "Please specify inference_name when setting split to inference"
        return field_value


base_logger_config = BaseLoggerConfig()
