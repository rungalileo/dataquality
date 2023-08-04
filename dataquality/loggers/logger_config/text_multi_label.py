from typing import List, Optional, Set

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class TextMultiLabelLoggerConfig(BaseLoggerConfig):
    labels: Optional[List[str]] = None
    observed_labels: Set = set()
    observed_num_labels: Optional[int] = None

    class Config:
        validate_assignment = True


text_multi_label_logger_config = TextMultiLabelLoggerConfig()
