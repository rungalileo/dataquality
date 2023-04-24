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
from dataquality.utils.semantic_segmentation.contours import (
    find_polygon_maps,
    upload_polygon_map,
)
from dataquality.utils.semantic_segmentation.errors import (
    calculate_misclassified_object,
    calculate_undetected_object,
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
        image_ids: Optional[List[int]] = None,
        gt_masks: Optional[torch.Tensor] = None,
        pred_masks: Optional[torch.Tensor] = None,
        gold_boundary_masks: Optional[torch.Tensor] = None,
        pred_boundary_masks: Optional[torch.Tensor] = None,
        output_probs: Optional[torch.Tensor] = None,
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
        self.pred_masks = pred_masks
        self.gold_boundary_masks = gold_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        self.output_probs = output_probs
        # assert ids is not None

    def validate_and_format(self) -> None:
        pass

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

        # path to dep map and contours is
        # {self.proj_run}/{self.split_name_path}/dep/image_id.json
        image_dep = calculate_and_upload_dep(
            self.output_probs,
            self.gt_masks,
            self.image_ids,
            self.dep_path,
        )

        mean_ious = calculate_mean_iou(self.pred_masks, self.gt_masks)
        boundary_ious = calculate_mean_iou(
            self.pred_boundary_masks, self.gold_boundary_masks
        )

        pred_polygons = find_polygon_maps(self.image_ids, self.pred_masks)
        gt_polygons = find_polygon_maps(self.image_ids, self.gt_masks)
        misclassified_objects = calculate_misclassified_object(
            self.gt_masks, pred_polygons
        )
        undetected_objects = calculate_undetected_object(self.pred_masks, gt_polygons)
        for image_id in self.image_ids:
            upload_polygon_map(
                pred_polygons[image_id],
                image_id,
                self.pred_mask_path,
                misclassified_objects[image_id],
                ErrorType.classification,
            )
            upload_polygon_map(
                gt_polygons[image_id],
                image_id,
                self.gt_mask_path,
                undetected_objects[image_id],
                ErrorType.undetected,
            )

        data = {
            "image_id": self.image_ids,
            "height": [img.shape[-1] for img in self.gt_masks],
            "width": [img.shape[-2] for img in self.gt_masks],
            "data_error_potential": image_dep,
            "mean_iou": mean_ious,
            "boundary_iou": boundary_ious,
            "classification_errors": misclassified_objects,  # str of polygon ids
            "undetected_errors": undetected_objects,  # str of polygon ids
            "split": [self.split] * len(self.image_ids),
            "epoch": [self.epoch] * len(self.image_ids),
            # "pred_contour_path": [
            #     f"{pred_contour_prefix}/{image_id}.json" for image_id in self.image_ids
            # ],
            # "gt_contour_path": [
            #     f"{gt_contour_prefix}/{image_id}.json" for image_id in self.image_ids
            # ],
            # "dep_path": [f"{dep_prefix}/{image_id}.png" for image_id in self.image_ids],
        }
        if self.split == Split.inference:
            data["inference_name"] = [self.inference_name] * len(self.image_ids)
        return data
