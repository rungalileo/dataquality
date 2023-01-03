from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.image_classification import (
    image_classification_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.dq_logger import get_dq_logger
from dataquality.utils.thread_pool import lock


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

    def _filter_duplicate_ids(self) -> None:
        """To avoid duplicate ids, when augmentation is used.
        Filter out duplicate ids in the batch. This is done by keeping track of
        the ids that have been observed in the current epoch in the config"""

        observed_ids = self.logger_config.observed_ids[f"{self.split}_{self.epoch}"]
        unique_ids = set(self.ids).difference(observed_ids)
        observed_ids.update(unique_ids)
        # If there are duplicate ids, filter out the duplicates
        if len(self.ids) > len(unique_ids):
            get_dq_logger().warning(
                f"Duplicate ids found in epoch. "
                f"Batch size: {len(self.ids)}, "
                f"Unique ids: {len(unique_ids)}"
            )
            unique_indices = [list(self.ids).index(id) for id in unique_ids]
            if len(self.embs) > 0:
                self.embs = np.array(self.embs)[unique_indices]
            if len(self.probs) > 0:
                self.probs = np.array(self.probs)[unique_indices]
            if len(self.ids) > 0:
                self.ids = np.array(self.ids)[unique_indices]

    def write_model_output(self, model_output: Dict) -> None:
        """Only write model output if there is data to write.

        In image classification, it is possible that after filtering
        duplicate IDs, there are none to write. In that case,
        we'll get an error trying to write them, so we skip
        """
        if len(model_output["id"]):
            return super().write_model_output(model_output)

    def _get_data_dict(self) -> Dict[str, Any]:
        # Handle the binary case by converting it to 2-class classification
        with lock:
            self._filter_duplicate_ids()
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
