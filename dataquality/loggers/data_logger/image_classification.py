from typing import List

from dataquality.loggers.data_logger.base_data_logger import MetasType
from dataquality.loggers.data_logger.text_classification import (
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
        texts: List[str] = None,
        labels: List[str] = None,
        ids: List[int] = None,
        split: str = None,
        meta: MetasType = None,
        inference_name: str = None,
    ) -> None:
        super().__init__(
            texts=texts,
            labels=labels,
            ids=ids,
            split=split,
            meta=meta,
            inference_name=inference_name,
        )

    def update_logged_dataset_text(self) -> None:
        # TODO
        pass
