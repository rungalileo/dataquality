from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from dataquality.schemas.semantic_segmentation import (
    ErrorType,
    MisclassifiedClassLabel,
    Polygon,
)
from dataquality.utils.semantic_segmentation.polygons import draw_polygon

MISCLASSIFIED_THRESHOLD = 0.5
UNDETECTED_THRESHOLD = 0.5
GHOST_THRESHOLD = 0.5


def polygon_accuracy(
    preds: np.ndarray, gold_mask: np.ndarray, correct_class: int
) -> Tuple[float, Tuple[int, float]]:
    """Calculates the accuracy of one ground truth polygon
    accuracy = (number of correct pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gold_masks: ground truth masks
        shape =  height, width)
    :param correct_class: the correct class of the polygon

    returns: pixel accuracy of the predictions
    """
    relevant_region = gold_mask != 0
    relevant_pred_region = preds != 0
    # use the relevant region to only select the pixels in the polygon
    # use the relevant_pred_region to only select the pixels in the pred polygon
    # that are not background pixels as classification errors are only
    # counted for non-background pixels
    pointwise_accuracy = (preds == gold_mask)[relevant_region & relevant_pred_region]

    misclassified_class = calculate_misclassified_class(
        preds, correct_class, relevant_region
    )
    return pointwise_accuracy.sum() / relevant_region.sum(), misclassified_class


def calculate_misclassified_class(
    pred_mask: np.ndarray,
    correct_class: int,
    relevant_region: np.ndarray,
) -> Tuple[int, float]:
    """Checks to see if the polygon is misclassified if over 50% of the pixels
    are another class and if so sets Polygon.misclassified as to the class

    Args:
        pred_mask (np.ndarray): predicted mask
        gold_mask (np.ndarray): ground truth mask
        correct_class (int): the correct class of the polygon
        relevant_region (np.ndarray): boolean mask of the relevant region of the polygon
    """
    area = relevant_region.sum()
    region_pixels = pred_mask[relevant_region]
    region_boolean = region_pixels != correct_class
    incorrect_pixels = region_pixels[region_boolean]
    # count the number of pixels in the pred mask relevant region that are
    # not the correct class
    areas = np.bincount(incorrect_pixels)
    top_candidate = (-1, 0)
    for i, incorrect_area in enumerate(areas):
        if incorrect_area > top_candidate[1]:
            top_candidate = (i, incorrect_area)
    top_candidate = (top_candidate[0], top_candidate[1] / area)
    return top_candidate if top_candidate[0] >= 0 else (-1, -1)


def calculate_misclassified_polygons(
    pred_mask: np.ndarray,
    gold_polygons: List[Polygon],
) -> None:
    """Checks for polygon misclassifications and sets the Polygon error_type field

    A misclassified polygon is a polygon from the predicted mask
    that has a pixel accuracy less than ERROR_THRES (0.5)

    In other words, if less than 50% of the pixels in the predicted polygon
    were correct, then the polygon is misclassified.

    Args:
        gold_mask (np.ndarray): ground truth mask
        pred_polygons (List[Polygon]):
            predicted polygon map for one image
    """
    for polygon in gold_polygons:
        out_polygon = draw_polygon(polygon, pred_mask.shape[-2:])
        accuracy, misclassified_label = polygon_accuracy(
            pred_mask, out_polygon, polygon.label_idx
        )
        polygon.accuracy = accuracy
        polygon.misclassified_class_label = MisclassifiedClassLabel(
            label=misclassified_label[0],
            pct=misclassified_label[1],
        )
        if accuracy < MISCLASSIFIED_THRESHOLD:
            polygon.error_type = ErrorType.classification


def calculate_misclassified_polygons_batch(
    pred_masks: torch.Tensor,
    gold_polygons_batch: List[List[Polygon]],
) -> None:
    """Calculates a set of misclassified polygon ids from the
    predicted mask for each image in a batch

    Also sets the error type field on the bad polygons to "misclassified"

    Args:
        gold_mask(torch.tensor): ground truth mask
        pred_polygon_maps(List[List[Polygon]]):
            list of predicted polygons for each image in a batch
    """
    for idx in range(len(pred_masks)):
        gold_mask = pred_masks[idx].numpy()
        pred_polygons = gold_polygons_batch[idx]
        calculate_misclassified_polygons(gold_mask, pred_polygons)


def calculate_missed_percentage(preds: np.ndarray, gold_mask: np.ndarray) -> float:
    """Calculates the amount of background predicted on a polygon
    calculated as (number of background pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gold_masks: ground truth masks
        shape = (height, width)

    returns: accuracy of the predictions
    """
    relevant_region = gold_mask != 0
    return (preds == 0)[relevant_region].sum() / relevant_region.sum()


def calculate_missed_polygons(
    pred_mask: np.ndarray, gold_polygons: List[Polygon]
) -> None:
    """Checks for polygon misclassifications and sets the Polygon error_type field

    Args:
        pred_mask(np.ndarray): argmax of the prediction probabilities
        deserialized_polygon_map(Dict[int, List[List[np.ndarray]]]):
            ground truth polygons that we examine for undetected objects
    """
    for polygon in gold_polygons:
        polygon_img = draw_polygon(polygon, pred_mask.shape[-2:])
        missed_percentage = calculate_missed_percentage(pred_mask, polygon_img)
        polygon.missed_percentage = missed_percentage
        if missed_percentage > UNDETECTED_THRESHOLD:
            polygon.error_type = ErrorType.undetected


def calculate_missed_polygons_batch(
    pred_masks: torch.Tensor,
    gold_polygons_batch: List[List[Polygon]],
) -> None:
    """Calculates a set of undetected polygon ids from the ground truth
    that are not detected by the model for images in a batch

    Args:
        preds(torch.tensor): argmax of the prediction probabilities
        gold_polygons_batch(List[List[Polygon]]):
            ground truth polygons that we examine for undetected objects
    """
    for idx in range(len(pred_masks)):
        pred_mask = pred_masks[idx].numpy()
        gold_polygons = gold_polygons_batch[idx]
        calculate_missed_polygons(pred_mask, gold_polygons)


def calculate_dep_polygon(
    dep_map: np.ndarray,
    polygon_img: np.ndarray,
) -> float:
    """Calculate the mean dep score for one polygon drawn onto an image of all
    zero's. We can then take the polygon's dep score by only selecting those pixels
    with a value greater than 0 and averageing them.

    Args:
        dep_map (np.ndarray): heatmap of dep scores for an image
        polygon_img (np.ndarray): image of all zeros with a polygon drawn on it

    Returns:
        dep_score (float): mean dep score for the polygon
    """
    relevant_region = polygon_img != 0
    dep_score = dep_map[relevant_region].mean()
    return dep_score


def calculate_dep_polygons_batch(
    polygons_batch: List[List[Polygon]],
    dep_heatmaps: np.ndarray,
    height: List[int],
    width: List[int],
) -> None:
    """Takes the mean dep score within a polygon and sets the polygon's
    dep score to the mean dep score

    Args:
        polygons_batch (List[List[[Polygon]]): list of the polygons
            for an image
        dep_heatmaps (np.ndarray): heatmaps of DEP scores for an image
        height (int): height of original image to resize the dep map to the correct
            dims
        width (int): width of original image to resize the dep map to the correct
            dims
    """
    resized_dep_maps = []
    for i, dep_map in enumerate(dep_heatmaps):
        resized_image = Image.fromarray(dep_map).resize((width[i], height[i]))
        resized_dep_maps.append(np.array(resized_image))

    for idx in range(len(resized_dep_maps)):
        dep_map = resized_dep_maps[idx]
        gold_polygons = polygons_batch[idx]
        for polygon in gold_polygons:
            polygon_img = draw_polygon(polygon, dep_map.shape)
            polygon.data_error_potential = calculate_dep_polygon(dep_map, polygon_img)


def calculate_ghost_polygons_batch(
    gold_masks: torch.Tensor,
    pred_polygons_batch: List[List[Polygon]],
) -> None:
    """Finds pred polygons predicted on background space that
    should be flagged as ghost objects. Algorithm is if 50 of the predicted
    pixels are background in the gold mask then it is a ghost object.

    Args:
        gold_masks (torch.Tensor): gold masks for an image
        pred_polygons_batch (List[List[Polygon]]): pred_polgons to be examined
    """
    for idx in range(len(gold_masks)):
        gold_mask = gold_masks[idx].numpy()
        pred_polygons = pred_polygons_batch[idx]
        calculate_ghost_polygons(gold_mask, pred_polygons)


def calculate_ghost_polygons(
    gold_mask: np.ndarray,
    pred_polygons: List[Polygon],
) -> None:
    """Finds pred polygons predicted on background space that
    should be flagged as ghost objects. Algorithm is if 50 of the predicted
    pixels are background in the gold mask then it is a ghost object.

    Args:
        gold_mask (np.ndarray): gold masks for an image
        pred_polygons (List[Polygon]): pred_polgons to be examined
    """
    for polygon in pred_polygons:
        polygon_img = draw_polygon(polygon, gold_mask.shape[-2:])
        if calculate_amount_ghosted(polygon_img, gold_mask) > GHOST_THRESHOLD:
            polygon.error_type = ErrorType.ghost


def calculate_amount_ghosted(polygon_im: np.ndarray, gold_mask: torch.Tensor) -> float:
    """Calculates the amount of background in the gt of a polygon

    Args:
        polygon_im (np.ndarray): np array of the polygon drawn onto an image
        gold_mask (torch.Tensor): gold mask to compare to

    Returns:
        float: percentage of pixels in pred polygon that have background in gt
    """
    relevant_region = polygon_im != 0
    return (gold_mask == 0)[relevant_region].sum() / relevant_region.sum()


def calculate_lm_polygons_batch(
    mislabeled_pixels: torch.Tensor, gold_polygons_batch: List[List[Polygon]]
) -> None:
    """Calculate and attach the LM percentage per polygon in a batch

    Args:
        mislabeled_pixels (torch.Tensor): map of bs, h, w of mislabled pixels
        gold_polygons_batch (List[List[Polygon]]): gold polygons for each image
    """
    for idx in range(len(mislabeled_pixels)):
        mislabeled_pixel_map = mislabeled_pixels[idx].numpy()
        gold_polygons = gold_polygons_batch[idx]
        calculate_lm_polygons(mislabeled_pixel_map, gold_polygons)


def calculate_lm_polygons(
    mislabelled_pixel_map: torch.Tensor, gold_polygons: List[Polygon]
) -> None:
    """Calculates and attaches the LM percentage to each polygon

    Args:
        mislabelled_pixel_map (torch.Tensor): map of bs, h, w of mislabled pixels
        gold_polygons (List[Polygon]): list of all gold polygons for an image
    """
    for polygon in gold_polygons:
        polygon_img = draw_polygon(polygon, mislabelled_pixel_map.shape[-2:])
        polygon.lm_percentage = calculate_lm_polygon(mislabelled_pixel_map, polygon_img)


def calculate_lm_polygon(
    mislabelled_pixel_map: torch.Tensor, polygon_img: np.ndarray
) -> float:
    """Calculates the percentage of mislabelled pixels in a polygon

    Args:
        mislabelled_pixel_map (torch.Tensor): map of bs, h, w of mislabled pixels
        polygon_img (np.ndarray): np array of the polygon drawn onto an image

    Returns:
        float: percentage of mislabelled pixels in a polygon
    """
    relevant_region = polygon_img != 0
    return (mislabelled_pixel_map != 0)[relevant_region].sum() / relevant_region.sum()
