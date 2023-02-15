from typing import Dict

from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class StructuredClassificationLoggerConfig(TextClassificationLoggerConfig):
    # NOTE: By inheriting from TCLoggerCongif we get cleaned labels
    feature_importances: Dict[str, float] = {}


structured_classification_logger_config = StructuredClassificationLoggerConfig()
