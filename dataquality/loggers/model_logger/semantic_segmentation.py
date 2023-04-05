from typing import Dict, List, Optional, Union

import numpy as np
import torch

from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split
from dataquality.utils.semantic_segmentation.contours import find_and_upload_contours
from dataquality.utils.semantic_segmentation.errors import (
    calculate_false_positives,
    calculate_missing_segments,
)
from dataquality.utils.semantic_segmentation.metrics import (
    calculate_and_upload_dep,
    calculate_mean_iou,
)


class SemanticSegmentationModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "semantic_segmentation"
    logger_config: SemanticSegmentationLoggerConfig = (
        semantic_segmentation_logger_config
    )

    def __init__(
        self,
        image_ids: List[int],
        gt_masks: torch.Tensor,
        pred_mask: torch.Tensor,
        gold_boundary_masks: torch.Tensor,
        pred_boundary_masks: torch.Tensor,
        output_probs: torch.Tensor,
        # Below fields must be present, linting from parent class
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        """Takes in SemSeg inputs as a list of batches

        Args:
            image_ids: List of image ids
            gt_masks: List of ground truth masks
                np.ndarray of shape (batch_size, height, width)
            pred_mask: List of prediction masks
                np.ndarray of shape (batch_size, height, width)
            gold_boundary_masks: List of gold boundary masks
                np.ndarray of shape (batch_size, height, width)
            pred_boundary_masks: List of predicted boundary masks
                np.ndarray of shape (batch_size, height, width)
            output_probs: Model probability predictions
                np.ndarray of shape (batch_size, height, width, num_classes)
        """
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )
        self.image_ids = image_ids
        self.gt_masks = gt_masks
        self.pred_mask = pred_mask
        self.gold_boundary_masks = gold_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        self.output_probs = output_probs
        # assert ids is not None

    def validate_and_format(self) -> None:
        pass

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary of data to be logged as a DataFrame"""
        # assert self.logger_config.image_cloud_path is not None, (
        #     "Must have image cloud path, set using `dq.set_image_cloud_path`. "
        #     "Must be set before training model."
        # )
        image_dep = calculate_and_upload_dep(
            self.output_probs,
            self.gt_masks,
            self.image_ids,
            f"{self.proj_run}/{self.split_name_path}/dep",
        )
        find_and_upload_contours(
            self.image_ids,
            self.pred_mask,
            f"{self.proj_run}/{self.split_name_path}/contours",
        )

        mean_ious = calculate_mean_iou(self.pred_mask, self.gt_masks)
        boundary_ious = calculate_mean_iou(
            self.pred_boundary_masks, self.gold_boundary_masks
        )

        false_positives = calculate_false_positives(self.pred_mask, self.gt_masks)
        missing_segments = calculate_missing_segments(self.pred_mask, self.gt_masks)

        data = {
            # "id": self.ids,
            "image_id": self.image_ids,
            "height": [img.shape[-1] for img in self.gt_masks],
            "width": [img.shape[-2] for img in self.gt_masks],
            "data_error_potential": image_dep,
            "mean_iou": mean_ious,
            "boundary_iou": boundary_ious,
            "error_false_positive": false_positives,
            "error_missing_segment": missing_segments,
            "split": [self.split] * len(self.image_ids),
            "epoch": [self.epoch] * len(self.image_ids),
        }
        if self.split == Split.inference:
            data["inference_name"] = [self.inference_name] * len(self.image_ids)
        return data
