from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.utils.semantic_segmentation import (
    calculate_false_positives,
    calculate_mean_iou,
    calculate_missing_segments,
    probs_to_preds,
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
            gold_boundary_masks: List of gold boundary masks
                np.ndarray of shape (batch_size, height, width)
            pred_boundary_masks: List of predicted boundary masks
                np.ndarray of shape (batch_size, height, width)
        """
        # super().__init__(
        #     embs=embs,
        #     probs=probs,
        #     logits=logits,
        #     ids=ids,
        #     split=split,
        #     epoch=epoch,
        #     inference_name=inference_name,
        # )
        self.image_ids = image_ids
        self.gt_masks = gt_masks
        self.pred_mask = pred_mask
        self.gold_boundary_masks = gold_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        self.output_probs = output_probs
        # assert ids is not None

    def validate_and_format(self) -> None:
        return

    def create_contours(self, pred_mask: torch.Tensor) -> Dict[int, List]:
        """Returns a list of GT contours from the pred mask

        A contour is a list of points that make up the boundary of a shape.
        Each image can be represented as a dictionary mapping a GT class to
          its corresponding contours.

        Example:
        {
          "7": [  # Class '7' has 2 contours
              ((13, 17), (19, 25), (22, 21), (13, 17)),  # contour 1
              ((0, 3), (2, 5), (4, 6), (2, 2), (0,3)),  # contour 2
          ],
          "15": [  # Class '15' has 1 contour
              ((11, 17), (19, 25), (22, 21), (11, 17)),  # contour 1
          ],
        }
        """
        contours_map = {}
        for label in np.unique(pred_mask).astype(int).tolist():
            if label == 0:
                continue

            mask = pred_mask == label
            mask = mask.astype(np.uint8)  # maybe don't need this
            # contours is a tuple of numpy arrays
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_map[label] = [self.format_contour(c) for c in contours]

        return contours_map

    def format_contour(self, contour: np.ndarray) -> List[Tuple[int]]:
        """Converts a contour from a numpy array to a list of pixel coordinates

        Example input:
        contour = np.array([[[13, 17]], [[19, 25]], [[22, 21]], [[13, 17]]])
        print(contour.shape)
          => (4, 1, 2)

        Example output:
        [
            (13, 17),
            (19, 25),
            (22, 21),
            (13, 17),
        ]
        """
        return list(map(tuple, contour.squeeze(1).tolist()))

    def _upload_contour(self, image_id: int, contour: Dict[int, List]) -> None:
        """Uploads a contour to the cloud for a given image"""
        # assert self.logger_config.image_cloud_path is not None, (
        #     "Must have image cloud path, set using `dq.set_image_cloud_path`. "
        #     "Must be set before training model."
        # )
        # raise NotImplementedError

    def upload_contours(self, contours: List[Dict[int, List]]) -> None:
        """Uploads contours to the cloud"""
        for image_id, contour in zip(self.image_ids, contours):
            self._upload_contour(image_id, contour)

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary of data to be logged as a DataFrame"""
        # assert self.logger_config.image_cloud_path is not None, (
        #     "Must have image cloud path, set using `dq.set_image_cloud_path`. "
        #     "Must be set before training model."
        # )
        # dep_heatmaps = calculate_dep_heatmap(self.output_probs, self.gt_masks)
        # image_dep = calculate_image_dep(dep_heatmaps)

        # probs = self._convert_tensor_ndarray(self.output_probs)
        # pred_masks = probs_to_preds(self.output_probs)
        # contours = [self.create_contours(pred_mask) for pred_mask in pred_masks]
        # self.upload_contours(contours)


        mean_ious = calculate_mean_iou(self.pred_mask, self.gt_masks)
        boundary_ious = calculate_mean_iou(
            self.pred_boundary_masks, self.gold_boundary_masks
        )
        false_positives = calculate_false_positives(self.pred_mask, self.gt_masks)
        missing_segments = calculate_missing_segments(self.pred_mask, self.gt_masks)

        obj = {
            # "id": self.ids,
            "image_id": self.image_ids,
            "height": [img.shape[-1] for img in self.gt_masks],
            "width": [img.shape[-2] for img in self.gt_masks],
            # "data_error_potential": image_dep,
            "mean_iou": mean_ious,
            "boundary_iou": boundary_ious,
            "error_false_positive": false_positives,
            "error_missing_segment": missing_segments,
            # "split": [self.split] * len(self.image_ids),
        }

        return obj
