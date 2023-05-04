from typing import List

import numpy as np
import torch

from dataquality.schemas.semantic_segmentation import ErrorType, Polygon
from dataquality.utils.semantic_segmentation.polygons import draw_polygon

ERROR_THRES = 0.5


def polygon_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the accuracy of one ground truth polygon
    accuracy = (number of correct pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape =  height, width)

    returns: pixel accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    pointwise_accuracy = (preds == gt_mask)[relevant_region]
    return pointwise_accuracy.sum() / relevant_region.sum()


def calculate_misclassified_polygons(
    gt_mask: np.ndarray,
    pred_polygons: List[Polygon],
) -> None:
    """Checks for polygon misclassifications and sets the Polygon error_type field

    A misclassified polygon is a polygon from the predicted mask
    that has a pixel accuracy less than ERROR_THRES (0.5)

    In other words, if less than 50% of the pixels in the predicted polygon
    were correct, then the polygon is misclassified.

    Args:
        gt_mask (np.ndarray): ground truth mask
        pred_polygons (List[Polygon]):
            predicted polygon map for one image
    """
    for polygon in pred_polygons:
        out_polygon = draw_polygon(polygon, gt_mask.shape[-2:])
        if polygon_accuracy(gt_mask, out_polygon) < ERROR_THRES:
            polygon.error_type = ErrorType.classification


def calculate_misclassified_polygons_batch(
    gt_masks: torch.Tensor,
    pred_polygons_batch: List[List[Polygon]],
) -> None:
    """Calculates a set of misclassified polygon ids from the
    predicted mask for each image in a batch

    Also sets the error type field on the bad polygons to "misclassified"

    Args:
        gt_mask(torch.tensor): ground truth mask
        pred_polygon_maps(List[List[Polygon]]):
            list of predicted polygons for each image in a batch
    """
    for idx in range(len(gt_masks)):
        gt_mask = gt_masks[idx].numpy()
        pred_polygons = pred_polygons_batch[idx]
        calculate_misclassified_polygons(gt_mask, pred_polygons)


def undetected_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the amount of background predicted on a polygon
    calculated as (number of background pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape = (height, width)

    returns: accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    return (preds == 0)[relevant_region].sum() / relevant_region.sum()


def calculate_undetected_polygons(
    pred_mask: np.ndarray, gt_polygons: List[Polygon]
) -> None:
    """Checks for polygon misclassifications and sets the Polygon error_type field

    Args:
        pred_mask(np.ndarray): argmax of the prediction probabilities
        deserialized_polygon_map(Dict[int, List[List[np.ndarray]]]):
            ground truth polygons that we examine for undetected objects
    """
    for polygon in gt_polygons:
        polygon_img = draw_polygon(polygon, pred_mask.shape[-2:])
        if undetected_accuracy(pred_mask, polygon_img) > ERROR_THRES:
            polygon.error_type = ErrorType.undetected


def calculate_undetected_polygons_batch(
    pred_masks: torch.Tensor,
    gt_polygons_batch: List[List[Polygon]],
) -> None:
    """Calculates a set of undetected polygon ids from the ground truth
    that are not detected by the model for images in a batch

    Args:
        preds(torch.tensor): argmax of the prediction probabilities
        gt_polygons_batch(List[List[Polygon]]):
            ground truth polygons that we examine for undetected objects
    """
    for idx in range(len(pred_masks)):
        pred_mask = pred_masks[idx].numpy()
        gt_polygons = gt_polygons_batch[idx]
        calculate_undetected_polygons(pred_mask, gt_polygons)
