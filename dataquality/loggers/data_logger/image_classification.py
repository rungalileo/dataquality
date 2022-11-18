from enum import Enum, unique
from typing import List

from dataquality.loggers.data_logger.base_data_logger import MetasType
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)


@unique
class GalileoDataLoggerAttributes(str, Enum):
    img_tn = "img_tn"
    img_path = "img_path"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging
    inference_name = "inference_name"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataLoggerAttributes))


class ImageClassificationDataLogger(TextClassificationDataLogger):
    __logger_name__ = "image_classification"
    logger_config: ImageClassificationLoggerConfig = (
        image_classification_logger_config  # type: ignore
    )

    def __init__(
        self,
        images: List[str] = None,
    ) -> None:
        super().__init__()
        self.images = images if images else []
