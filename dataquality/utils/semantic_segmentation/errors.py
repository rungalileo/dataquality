from typing import Dict, List

import numpy as np
import torch

from dataquality.schema.semantic_segmentation import Polygon
from dataquality.utils.semantic_segmentation.polygons import (
    deserialize_polygon_map,
    draw_polygon,
)

ERROR_THRES = 0.5


def polygon_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the accuracy of one ground truth polygon
    accuracy = (number of correct pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape =  height, width)
    returns: accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    pointwise_accuracy = (preds == gt_mask)[relevant_region]
    return pointwise_accuracy.sum() / relevant_region.sum()


def polygon_ids_to_string(polygon_ids: List[str]) -> str:
    """Converts a list of polygon ids to a string

    Example:
        polygon_ids_to_string([1, 2, 3]) -> "1,2,3"

    Args:
        polygon_ids(List[int]): list of polygon ids

    Returns:
        str: comma separated string of polygon ids
    """
    return ",".join([str(polygon_id) for polygon_id in polygon_ids])


def calculate_misclassified_polygons(
    gt_mask: np.ndarray,
    pred_pmap: Dict[int, List[Polygon]],
) -> str:
    """Calculates a set of misclassified polygon ids for one image

    Args:
        gt_mask(np.ndarray): ground truth mask
        pred_polygon_map(Dict[int, List[Polygon]]):
            predicted polygon map for one image

    Returns:
        List[str]: list of misclassified objects by their polygon id
    """
    missing_polygons = []
    counter = 0
    deserialized_pmap = deserialize_polygon_map(pred_pmap)

    for label_idx in sorted(deserialized_pmap.keys()):
        for polygon in deserialized_pmap[label_idx]:
            out_polygon = draw_polygon(polygon, gt_mask, label_idx)
            if polygon_accuracy(gt_mask, out_polygon) < ERROR_THRES:
                missing_polygons.append(str(counter))
            counter += 1

    return polygon_ids_to_string(missing_polygons)


def calculate_misclassified_polygons_batch(
    gt_mask: torch.Tensor,
    pred_polygon_maps: List[Dict],
) -> List[str]:
    """Calculates a set of misclassified polygon ids from the
    predicted mask for each image in a batch

    Args:
        gt_mask(torch.tensor): ground truth mask
        pred_polygon_maps(List[Dict[int, List[List[np.ndarray]]]]):
            list of predicted polygon maps for each image in a batch
    Returns
        List[List[str]]: list of lists of undetected objects by their
            polygon id for each image in a batch

    """
    misclassified_polygons_batch = []
    for image in range(len(gt_mask)):
        pred_mask = gt_mask[image].numpy()
        deserialized_pmap = pred_polygon_maps[image]
        # Calculate classes present in predictions and ground truth
        missing_polygons = calculate_misclassified_polygons(
            pred_mask, deserialized_pmap
        )
        misclassified_polygons_batch.append(missing_polygons)
    return misclassified_polygons_batch


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
    pred_mask: np.ndarray, gt_pmap: Dict[int, List[Polygon]]
) -> str:
    """Calculates a set of undetected object polygon ids for one image

    Args:
        pred_mask(np.ndarray): argmax of the prediction probabilities
        deserialized_polygon_map(Dict[int, List[List[np.ndarray]]]):
            ground truth polygons that we examine for undetected objects

    Returns:
        List[str]: list of undetected objects by their polygon id for an image


    Example:
        >>> pred_mask = np.array([[0, 0, 0, 0, 0],
        ...                       [0, 1, 1, 1, 0],
        ...                       [0, 1, 1, 1, 0],
        ...                       [0, 1, 1, 1, 0],
        ...                       [0, 0, 0, 0, 0]])
        >>> gt_pmap = {1: [Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])]}
        >>> calculate_undetected_polygons(pred_mask, gt_pmap)
        '0'
    """
    undetected_polygons = []
    counter = 0
    deserialized_pmap = deserialize_polygon_map(gt_pmap)

    for label_idx in sorted(deserialized_pmap.keys()):
        for polygon in deserialized_pmap[label_idx]:
            out_polygon = draw_polygon(polygon, pred_mask, label_idx)
            if undetected_accuracy(pred_mask, out_polygon) > ERROR_THRES:
                undetected_polygons.append(str(counter))
            counter += 1

    return polygon_ids_to_string(undetected_polygons)


def calculate_undetected_polygons_batch(
    preds: torch.Tensor,
    gt_polygon_maps: List[Dict],
) -> List[str]:
    """Calculates a set of undetected polygon ids from the ground truth
    that are not detected by the model for images in a batch

    Args:
        preds(torch.tensor): argmax of the prediction probabilities
        gt_polygon_maps(List[Dict[int, List[Polygon]]]):
            ground truth polygon map that we examine for undetected objects
    Returns
        List[List[str]]: list of lists of undetected objects by their
            polygon id for each image in a batch

    """
    undetected_polygons_batch = []
    for idx in range(len(preds)):
        pred_mask = preds[idx].numpy()
        gt_pmap = gt_polygon_maps[idx]
        # Calculate classes present in predictions and ground truth
        undetected_polygon_ids = calculate_undetected_polygons(pred_mask, gt_pmap)
        undetected_polygons_batch.append(undetected_polygon_ids)

    return undetected_polygons_batch
