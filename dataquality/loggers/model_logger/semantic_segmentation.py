from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import evaluate
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
        image_ids: List[int],
        gt_masks: np.ndarray,
        gold_boundary_masks: np.ndarray,
        pred_boundary_masks: np.ndarray,
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
        self.gold_boundary_masks = gold_boundary_masks
        self.pred_boundary_masks = pred_boundary_masks
        # assert ids is not None
        self._get_data_dict()

    def validate_and_format(self) -> None:
        return

    def probs_to_preds(self, probs: np.ndarray) -> np.ndarray:
        """Takes pixel-wise arg-max to return preds"""
        return np.argmax(probs, axis=-1)

    def create_contours(self, pred_mask: np.ndarray) -> Dict[int, List]:
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

    def calculate_dep_heatmap(
        self, probs: np.ndarray, gt_masks: np.ndarray, logits: bool = False
    ) -> float:
        """Calculates the Data Error Potential (DEP) for each image in the batch"""
        # probs: float, (bs, n_rows, n_classes)
        # y: int, (bs, n_rows,)
        # if `logits` is True, we convert the probabilities to logits, and compute the margin on the logits.
        bs = probs.shape[0]
        if logits:
            # if we have probabilities, we can convert to logits by taking a logarithm.
            # we clip the probabilities first to avoid taking the log of 0.
            values = torch.log(torch.clamp(probs, min=1e-8))
        else:
            values = probs

        # CHANGE TO PASS IN AS TORCH
        gt_masks = torch.tensor(gt_masks, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.int64)
        gt_indices = gt_masks.reshape((bs, -1, 1)).expand(-1, -1, values.shape[2])
        value_at_ground_truth = torch.gather(values, 2, gt_indices)[:, :, 0]

        next_highest = values.clone()
        next_highest.scatter_(2, gt_indices, 0)
        next_highest = next_highest.max(dim=2).values

        return 1 - (value_at_ground_truth - next_highest)

    def calculate_image_dep(self, dep_heatmap: np.ndarray) -> List[float]:
        """Calculates the Data Error Potential (DEP) for each image in the batch"""
        return dep_heatmap.sum(axis=1)

    def calculate_mean_iou(
        self, probs: List[np.ndarray], gt_masks: List[np.ndarray], nc: int = 21
    ) -> List[float]:
        """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch"""
        metric = evaluate.load("mean_iou")
        ious = []
        for i in range(len(probs)):
            iou = metric._compute(
                probs[i : i + 1].cpu(),
                gt_masks[i : i + 1].cpu(),
                num_labels=nc,
                ignore_index=255,
            )
            ious.append(iou["mean_iou"].item().cpu().numpy())
        return ious

    def calculate_boundary_iou(
        self, probs: List[np.ndarray], gt_masks: List[np.ndarray]
    ) -> List[float]:
        """Calculates the Boundary Intersection Over Union (bIoU) for each image in the batch"""
        raise NotImplementedError

    def calculate_false_positives(
        self, preds: List[np.ndarray], gt_masks: List[np.ndarray]
    ) -> List[Set[int]]:
        """Calculates a set of False Positive classes for each image in the batch

        For each image, returns a set of classes that were predicted but not
            present in the ground truth.
        """
        false_positives: List[Set[int]] = []
        for image in range(len(preds)):
            pred_mask = preds[image]
            gt_mask = gt_masks[image]

            # Calculate classes present in predictions and ground truth
            pred_classes = set(np.unique(pred_mask).astype(int))
            gt_classes = set(np.unique(gt_mask).astype(int))

            # Calculate classes present in predictions but not ground truth
            fp_classes = pred_classes.difference(gt_classes)
            false_positives.append(fp_classes)

        return false_positives

    def calculate_missing_segments(
        self, preds: List[np.ndarray], gt_masks: List[np.ndarray]
    ) -> List[Set[int]]:
        """Calculates a set of Missing Segment classes for each image in the batch

        For each image, returns a set of classes that were in the ground truth but not
            present in the predictions.
        """
        missing_segments: List[Set[int]] = []
        for image in range(len(preds)):
            pred_mask = preds[image]
            gt_mask = gt_masks[image]

            # Calculate classes present in predictions and ground truth
            pred_classes = set(np.unique(pred_mask).astype(int))
            gt_classes = set(np.unique(gt_mask).astype(int))

            # Calculate classes present in predictions but not ground truth
            ms_classes = gt_classes.difference(pred_classes)
            missing_segments.append(ms_classes)

        return missing_segments

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
        self.probs = self.convert_logits_to_probs(self.logits)
        del self.logits

        pred_masks = self.probs_to_preds(self.probs)
        contours = [self.create_contours(pred_mask) for pred_mask in pred_masks]
        self.upload_contours(contours)
        import pdb

        # pdb.set_trace()

        dep_heatmaps = self.calculate_dep_heatmap(self.probs, self.gt_masks)
        image_dep = self.calculate_image_dep(dep_heatmaps)
        import pdb

        # pdb.set_trace()
        mean_ious = self.calculate_mean_iou(pred_masks, self.gt_masks)
        import pdb

        # pdb.set_trace()
        boundary_ious = self.calculate_boundary_iou([pred_masks], self.gt_masks)
        import pdb

        # pdb.set_trace()
        false_positives = self.calculate_false_positives(pred_masks, self.gt_masks)
        import pdb

        # pdb.set_trace()
        missing_segments = self.calculate_missing_segments(pred_masks, self.gt_masks)
        import pdb

        # pdb.set_trace()

        obj = {
            "id": self.ids,
            "image_id": self.image_ids,
            "height": [img.shape[0] for img in self.probs],
            "width": [img.shape[1] for img in self.probs],
            "data_error_potential": dep_heatmaps,
            "image_dep": image_dep,
            "mean_iou": mean_ious,
            "boundary_iou": boundary_ious,
            "error_false_positive": false_positives,
            "error_missing_segment": missing_segments,
            "split": [self.split] * len(self.image_ids),
        }

        return obj
