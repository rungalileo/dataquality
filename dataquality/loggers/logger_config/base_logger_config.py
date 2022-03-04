from typing import Any, Optional

from pydantic import BaseModel

from dataquality.schemas.ner import TaggingSchema


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    tagging_schema: Optional[TaggingSchema]
    last_epoch: int = 0


base_logger_config = BaseLoggerConfig()
