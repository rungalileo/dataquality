from typing import List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.image_classification import (
    image_classification_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)


class ImageClassificationModelLogger(TextClassificationModelLogger):
    __logger_name__ = "image_classification"
    logger_config = image_classification_logger_config

    def __init__(
        self,
        embs: Union[List, np.ndarray] = None,
        probs: Union[List, np.ndarray] = None,
        logits: Union[List, np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )
