from typing import Dict, List, Optional, Union

import numpy as np
import torch

from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.semantic_segmentation import ErrorType
from dataquality.schemas.split import Split
from dataquality.utils.semantic_segmentation.errors import (
    calculate_misclassified_object,
    calculate_undetected_object,
)
from dataquality.utils.semantic_segmentation.lm import upload_mislabeled_pixels
from dataquality.utils.semantic_segmentation.metrics import (
    calculate_and_upload_dep,
    calculate_mean_iou,
)
from dataquality.utils.semantic_segmentation.polygons import (
    find_polygon_maps,
    upload_polygon_map,
)


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
        gold_boundary_masks: Optional[torch.Tensor] = None,
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
        self.gold_boundary_masks = gold_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        self.output_probs = output_probs
        self.mislabled_pixels = mislabeled_pixels

    def validate_and_format(self) -> None:
        pass

    @property
    def lm_path(self) -> str:
        return f"{self.proj_run}/{self.split_name_path}/LM"

    @property
    def dep_path(self) -> str:
        return f"{self.proj_run}/{self.split_name_path}/dep"

    @property
    def pred_mask_path(self) -> str:
        return f"{self.proj_run}/{self.split_name_path}/masks/pred"

    @property
    def gt_mask_path(self) -> str:
        return f"{self.proj_run}/{self.split_name_path}/masks/ground_truth"

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary of data to be logged as a DataFrame"""
        # assert self.logger_config.image_cloud_path is not None, (
        #     "Must have image cloud path, set using `dq.set_image_cloud_path`. "
        #     "Must be set before training model."
        # )

        # DEP
        image_dep = calculate_and_upload_dep(
            self.output_probs,
            self.gt_masks,
            self.image_ids,
            obj_prefix=self.dep_path,
        )

        # Likely Mislabeled
        mean_mislabeled = torch.mean(self.mislabled_pixels, dim=(1, 2)).numpy()
        upload_mislabeled_pixels(
            self.mislabled_pixels, self.image_ids, prefix=self.lm_path
        )

        # Image Metrics (IoU)
        mean_ious, category_iou = calculate_mean_iou(self.pred_masks, self.gt_masks)
        boundary_mean_ious, boundary_category_iou = calculate_mean_iou(
            self.pred_boundary_masks, self.gold_boundary_masks
        )

        # Image masks
        pred_polygon_maps = find_polygon_maps(self.pred_masks)
        gt_polygon_maps = find_polygon_maps(self.gt_masks)
        # Errors
        misclassified_objects = calculate_misclassified_object(
            self.gt_masks, pred_polygon_maps
        )
        undetected_objects = calculate_undetected_object(
            self.pred_masks, gt_polygon_maps
        )
        # Add errors to polygons and upload to Minio
        for i, image_id in enumerate(self.image_ids):
            upload_polygon_map(
                pred_polygon_maps[i],
                image_id,
                self.pred_mask_path,
                misclassified_objects[i],
                ErrorType.classification,
            )
            upload_polygon_map(
                gt_polygon_maps[i],
                image_id,
                self.gt_mask_path,
                undetected_objects[i],
                ErrorType.undetected,
            )

        data = {
            "image": [
                f"{self.bucket_name}/{pth.lstrip('./')}" for pth in self.image_paths
            ],  # "gs://.../image_id.png
            "image_id": self.image_ids,
            "height": [img.shape[-1] for img in self.gt_masks],
            "width": [img.shape[-2] for img in self.gt_masks],
            "data_error_potential": image_dep,
            "mean_iou": mean_ious,
            "mean_lm_score": [i for i in mean_mislabeled],
            "category_iou": category_iou,
            "boundary_iou": boundary_mean_ious,
            "boundary_category_iou": boundary_category_iou,
            "classification_errors": misclassified_objects,  # str of polygon ids
            "undetected_errors": undetected_objects,  # str of polygon ids
            "split": [self.split] * len(self.image_ids),
            "epoch": [self.epoch] * len(self.image_ids),
        }
        if self.split == Split.inference:
            data["inference_name"] = [self.inference_name] * len(self.image_ids)
        return data
