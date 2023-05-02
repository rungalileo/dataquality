from typing import Dict

from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class TabularClassificationLoggerConfig(TextClassificationLoggerConfig):
    # NOTE: By inheriting from TCLoggerCongif we get cleaned labels
    feature_importances: Dict[str, float] = {}


tabular_classification_logger_config = TabularClassificationLoggerConfig()
