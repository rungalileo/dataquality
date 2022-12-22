from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.image_classification import (
    image_classification_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
from dataquality.schemas.split import Split


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

    def _get_data_dict(self) -> Dict[str, Any]:
        # Handle the binary case by converting it to 2-class classification
        probs = np.array(self.probs)
        if probs.shape[-1] == 1:
            self.probs = np.column_stack((1 - probs, probs))
        num_samples_in_batch = len(self.ids)
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(
                f"Duplicate ids found in batch: {self.ids}. "
                "Please make sure that each sample has a unique id."
            )
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
