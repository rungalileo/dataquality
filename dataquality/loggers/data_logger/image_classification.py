import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from dataquality.loggers.data_logger.base_data_logger import ITER_CHUNK_SIZE, MetasType
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)
from dataquality.schemas.split import Split
from dataquality.utils.cv import _img_to_b64_str


class ImageClassificationDataLogger(TextClassificationDataLogger):
    __logger_name__ = "image_classification"
    logger_config: ImageClassificationLoggerConfig = image_classification_logger_config

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

    def log_image_dataset(
        self,
        dataset: pd.DataFrame,
        imgs_dir: str,
        *,
        imgs_location_colname: Optional[str] = "relpath",
        batch_size: int = ITER_CHUNK_SIZE,
        id: Union[str, int] = "id",
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
    ) -> None:
        dataset["text"] = dataset[imgs_location_colname].apply(
            lambda x: _img_to_b64_str(img=Image.open(os.path.join(imgs_dir, x)))
        )
        self.log_dataset(
            dataset=dataset,
            batch_size=batch_size,
            text="text",
            id=id,
            label=label,
            split=split,
            meta=meta,
        )

    def _get_data_dict(self) -> Dict[str, Any]:
        # Handle the binary case by converting it to 2-class classification
        probs = np.array(self.probs)
        if probs.shape[-1] == 1:
            self.probs = np.column_stack((1 - probs, probs))
        num_samples_in_batch = len(self.ids)
        data = {
            "id": self.ids,
            "emb": self.embs,
            "prob": self.probs,
            "pred": np.argmax(self.probs, axis=1),
            "split": [Split[self.split].value] * num_samples_in_batch,
            "data_schema_version": [__data_schema_version__] * num_samples_in_batch,
            "epoch": [self.epoch] * num_samples_in_batch,
        }
        if self.split == Split.inference:
            data["inference_name"] = [self.inference_name] * num_samples_in_batch
        return data
