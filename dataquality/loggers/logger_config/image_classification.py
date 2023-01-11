from typing import Dict

from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)
from dataquality.utils.thread_safe_set import ThreadSafeSet


class ImageClassificationLoggerConfig(TextClassificationLoggerConfig):
    # Keep track of the ids that have been observed in the current epoch
    # the key is the split and epoch like observed_ids["train_0"] = {0, 1, 2, 3}
    observed_ids: Dict[str, ThreadSafeSet] = dict()

    class Config:
        arbitrary_types_allowed = True


image_classification_logger_config = ImageClassificationLoggerConfig()
