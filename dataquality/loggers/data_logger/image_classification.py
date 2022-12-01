from enum import Enum, unique
from typing import List

from dataquality.loggers.data_logger.text_classification import (
    GalileoDataLoggerAttributes,
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)


class ImageClassificationDataLogger(TextClassificationDataLogger):
    __logger_name__ = "image_classification"
    logger_config: ImageClassificationLoggerConfig = (
        image_classification_logger_config  # type: ignore
    )

    def __init__(
        self,
    ) -> None:
        super().__init__()
