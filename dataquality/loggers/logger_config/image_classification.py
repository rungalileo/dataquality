from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class ImageClassificationLoggerConfig(TextClassificationLoggerConfig):
    ...


image_classification_logger_config = ImageClassificationLoggerConfig()
