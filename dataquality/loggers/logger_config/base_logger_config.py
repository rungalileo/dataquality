from typing import Any, Optional

from pydantic import BaseModel


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    tagging_schema: Optional[str]
    last_epoch: int = 0


base_logger_config = BaseLoggerConfig()
