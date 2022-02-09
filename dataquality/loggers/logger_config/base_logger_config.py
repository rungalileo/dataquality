from typing import Any

from pydantic import BaseModel


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    last_epoch: int = 0


base_logger_config = BaseLoggerConfig()
