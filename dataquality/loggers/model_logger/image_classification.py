from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.image_classification import (
    image_classification_logger_config,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
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
        id_to_index = {id: index for index, id in enumerate(self.ids)}

        # If there are duplicate ids, filter out the duplicates
        if len(self.ids) > len(unique_ids):
            get_dq_logger().warning(
                f"Duplicate ids found in epoch. "
                f"Batch size: {len(self.ids)}, "
                f"Unique ids: {len(unique_ids)}"
            )
            unique_indices = [id_to_index[id] for id in unique_ids]
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

        return super()._get_data_dict()
