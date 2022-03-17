from typing import Any, Optional

from pydantic import BaseModel

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

    def reset(self) -> None:
        """Reset all class vars"""
        self.__init__()  # type: ignore


base_logger_config = BaseLoggerConfig()
