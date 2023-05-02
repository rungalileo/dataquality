from typing import Dict, List, Optional, Union

import numpy as np
import torch

from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger


class SemanticSegmentationModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "semantic_segmentation"
    logger_config: SemanticSegmentationLoggerConfig = (
        semantic_segmentation_logger_config
    )

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        image_ids: Optional[List[int]] = None,
        gt_masks: Optional[torch.Tensor] = None,
        pred_masks: Optional[torch.Tensor] = None,
        gt_boundary_masks: Optional[torch.Tensor] = None,
        pred_boundary_masks: Optional[torch.Tensor] = None,
        output_probs: Optional[torch.Tensor] = None,
        mislabeled_pixels: Optional[torch.Tensor] = None,
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
            pred_masks: List of prediction masks
                np.ndarray of shape (batch_size, height, width)
            gold_boundary_masks: List of gold boundary masks
                np.ndarray of shape (batch_size, height, width)
            pred_boundary_masks: List of predicted boundary masks
                np.ndarray of shape (batch_size, height, width)
            output_probs: Model probability predictions
                np.ndarray of shape (batch_size, height, width, num_classes)
            mislabeled_pixels: Model confidence predictions in the GT label
                torch.Tensor of shape (batch_size, height, width)
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
        self.bucket_name = bucket_name
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.gt_masks = gt_masks
        self.pred_masks = pred_masks
        self.gt_boundary_masks = gt_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        self.output_probs = output_probs
        self.mislabled_pixels = mislabeled_pixels

    def validate_and_format(self) -> None:
        pass

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary of data to be logged as a DataFrame"""
        return {}
