from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class StructuredClassificationLoggerConfig(TextClassificationLoggerConfig):
    # NOTE: By inheriting from TCLoggerCongif we get cleaned labels
    pass


structured_classification_logger_config = StructuredClassificationLoggerConfig()
