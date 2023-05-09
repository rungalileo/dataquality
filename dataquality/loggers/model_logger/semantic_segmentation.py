from typing import Dict, List, Optional, Union

import numpy as np
import torch

import dataquality as dq
from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split
from dataquality.utils.semantic_segmentation.errors import (
    calculate_misclassified_polygons_batch,
    calculate_undetected_polygons_batch,
)
from dataquality.utils.semantic_segmentation.lm import upload_mislabeled_pixels
from dataquality.utils.semantic_segmentation.metrics import (
    calculate_and_upload_dep,
    calculate_mean_iou,
)
from dataquality.utils.semantic_segmentation.polygons import find_polygons_batch


class SemanticSegmentationModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "semantic_segmentation"
    logger_config: SemanticSegmentationLoggerConfig = (
        semantic_segmentation_logger_config
    )

    def __init__(
        self,
        bucket_name: str = "",
        image_paths: List[str] = [],
        image_ids: List[int] = [],
        gt_masks: torch.Tensor = torch.empty(0),
        pred_masks: torch.Tensor = torch.empty(0),
        gt_boundary_masks: torch.Tensor = torch.empty(0),
        pred_boundary_masks: torch.Tensor = torch.empty(0),
        output_probs: torch.Tensor = torch.empty(0),
        mislabeled_pixels: torch.Tensor = torch.empty(0),
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

    @property
    def lm_path(self) -> str:
        """Minio path for Likely Mislabeled heatmaps"""
        return f"{self.proj_run}/{self.split_name_path}/LM"

    @property
    def dep_path(self) -> str:
        """Minio path for Data Error Potential heatmaps"""
        return f"{self.proj_run}/{self.split_name_path}/dep"

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary of data to be logged as a DataFrame"""
        # DEP & likely mislabeled
        mean_mislabeled = torch.mean(self.mislabled_pixels, dim=(1, 2)).numpy()
        upload_mislabeled_pixels(
            self.mislabled_pixels, self.image_ids, prefix=self.lm_path
        )

        image_dep = calculate_and_upload_dep(
            self.output_probs,
            self.gt_masks,
            self.image_ids,
            obj_prefix=self.dep_path,
        )

        # Image Metrics (IoU)
        iou, iou_per_class = calculate_mean_iou(self.pred_masks, self.gt_masks)
        boundary_iou, boundary_iou_per_class = calculate_mean_iou(
            self.pred_boundary_masks, self.gt_boundary_masks
        )

        # Image masks
        pred_polygons_batch, gt_polygons_batch = find_polygons_batch(
            self.pred_masks, self.gt_masks
        )
        # Errors
        calculate_misclassified_polygons_batch(self.pred_masks, gt_polygons_batch)
        calculate_undetected_polygons_batch(self.pred_masks, gt_polygons_batch)

        image_data = {
            "image": [
                f"{self.bucket_name}/{pth}" for pth in self.image_paths
            ],  # E.g. https://storage.googleapis.com/bucket_name/.../image_id.png
            "id": self.image_ids,
            "height": [img.shape[-1] for img in self.gt_masks],
            "width": [img.shape[-2] for img in self.gt_masks],
            "image_data_error_potential": image_dep,
            "mean_lm_score": [i for i in mean_mislabeled],
            "mean_iou": iou,
            "mean_iou_per_class": iou_per_class,
            "boundary_iou": boundary_iou,
            "boundary_iou_per_class": boundary_iou_per_class,
            # "epoch": [self.epoch] * len(self.image_ids),
        }
        not_meta = ["id", "image"]
        meta_keys = [k for k in image_data.keys() if k not in not_meta]
        dq.log_dataset(
            image_data,
            split=Split[self.split],
            inference_name=self.inference_name,
            meta=meta_keys,
        )

        image_ids = []
        preds = []
        golds = []
        contours = []
        data_error_potentials = []
        errors = []
        for i, image_id in enumerate(self.image_ids):
            pred_polygons = pred_polygons_batch[i]
            for polygon in pred_polygons:
                image_ids.append(image_id)
                preds.append(polygon.label_idx)
                golds.append(-1)
                contours.append(polygon.contours)
                data_error_potentials.append(0.0)
                errors.append(polygon.error_type)
            gt_polygons = gt_polygons_batch[i]
            for polygon in gt_polygons:
                image_ids.append(image_id)
                preds.append(-1)
                golds.append(polygon.label_idx)
                contours.append(polygon.contours)
                data_error_potentials.append(0.0)
                errors.append(polygon.error_type.value)

        polygon_data = {
            "image_id": image_ids,
            "pred": preds,
            "gold": golds,
            "contours": np.array(contours),
            "data_error_potential": data_error_potentials,
            "galileo_error_type": errors,
        }

        return polygon_data
