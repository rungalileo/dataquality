from typing import Dict, List

import numpy as np
import torch

from dataquality.schemas.semantic_segmentation import PolygonMap
from dataquality.utils.semantic_segmentation.polygons import (
    deserialize_polygon_map,
    draw_polygon,
)


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

    Args:
        polygon_ids(List[int]): list of polygon ids

    Returns:
        str: comma separated string of polygon ids
    """
    return ",".join([str(polygon_id) for polygon_id in polygon_ids])


def misclassified_polygons(
    gt_mask: np.ndarray,
    deserialized_pred_polygon_map: Dict[int, List[List[np.ndarray]]],
) -> str:
    """Calculates a set of misclassified polygon ids for one image

    Args:
        gt_mask(np.ndarray): ground truth mask
        deserialized_pred_polygon_map(Dict[int, List[List[np.ndarray]]]):
            predicted polygon map for one image

    Returns:
        List[str]: list of undetected objects by their polygon id
    """
    all_missing_polygons = []
    counter = 0
    for key in sorted(deserialized_pred_polygon_map.keys()):
        for polygon in deserialized_pred_polygon_map[key]:
            out_polygon = draw_polygon(polygon, gt_mask, key)
            if polygon_accuracy(gt_mask, out_polygon) < 0.5:
                all_missing_polygons.append(str(counter))
            counter += 1
    out_string = polygon_ids_to_string(all_missing_polygons)
    return out_string


def calculate_misclassified_polygons(
    gt_mask: torch.Tensor,
    pred_polygon_maps: List[PolygonMap],
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
    misclassified_polygons_per_image = []
    for image in range(len(gt_mask)):
        pred_mask = gt_mask[image]
        deserialized_polygon_map = deserialize_polygon_map(pred_polygon_maps[image])
        # Calculate classes present in predictions and ground truth
        missing_segments = misclassified_polygons(
            pred_mask.numpy(), deserialized_polygon_map
        )
        misclassified_polygons_per_image.append(missing_segments)
    return misclassified_polygons_per_image


def undetected_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the amount of background predicted on a polygon
    calculated as (number of background pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape =  height, width)
    returns: accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    return (preds == 0)[relevant_region].sum() / relevant_region.sum()


def image_undetected_polygon(
    pred_mask: np.ndarray, deserialized_polygon_map: Dict[int, List[List[np.ndarray]]]
) -> str:
    """Calculates a set of undetected object polygon ids for one image

    Args:
        pred_mask(np.ndarray): argmax of the prediction probabilities
        deserialized_polygon_map(Dict[int, List[List[np.ndarray]]]):
            ground truth polygons that we examine for undetected objects

    Returns:
        List[str]: list of undetected objects by their polygon id for an image
    """
    all_undetected_objects = []
    counter = 0
    for key in sorted(deserialized_polygon_map.keys()):
        for polygon in deserialized_polygon_map[key]:
            out_polygon = draw_polygon(polygon, pred_mask, key)
            if undetected_accuracy(pred_mask, out_polygon) > 0.5:
                all_undetected_objects.append(str(counter))
            counter += 1
    out_string = polygon_ids_to_string(all_undetected_objects)
    return out_string


def calculate_undetected_polygons(
    preds: torch.Tensor,
    gt_polygon_maps: List[PolygonMap],
) -> List[str]:
    """Calculates a set of undetected polygon ids from the ground truth
    that are not detected by the model for images in a batch

    Args:
        preds(torch.tensor): argmax of the prediction probabilities
        deserialize_gt_polygon_maps(List[Dict[int, List[List[np.ndarray]]]]):
            ground truth polygon map that we examine for undetected objects
    Returns
        List[List[str]]: list of lists of undetected objects by their
            polygon id for each image in a batch

    """
    undetected_polygons_per_image = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        deserialized_polygon_map = deserialize_polygon_map(gt_polygon_maps[image])
        # Calculate classes present in predictions and ground truth
        undetected_segments = image_undetected_polygon(
            pred_mask.numpy(), deserialized_polygon_map
        )
        undetected_polygons_per_image.append(undetected_segments)

    return undetected_polygons_per_image
