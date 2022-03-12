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

    def reset(self) -> None:
        """Reset all class vars"""
        self.__init__()  # type: ignore


base_logger_config = BaseLoggerConfig()
