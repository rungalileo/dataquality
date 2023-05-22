from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dataquality.schemas.semantic_segmentation import ErrorType, Polygon
from dataquality.utils.semantic_segmentation.constants import (
    BACKGROUND_CLASS,
    ERROR_THRESHOLD,
)
from dataquality.utils.semantic_segmentation.polygons import draw_polygon


def calculate_misclassified_class(
    pred_mask: np.ndarray,
    gold_mask: np.ndarray,
    correct_class: int,
    relevant_region: np.ndarray,
) -> Optional[int]:
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
    for i, incorrect_area in enumerate(areas):
        if incorrect_area / area > ERROR_THRESHOLD:
            return i
    return None


def polygon_accuracy(
    preds: np.ndarray, gold_mask: np.ndarray, correct_class: int
) -> Tuple[float, Optional[int]]:
    """Calculates the accuracy of one ground truth polygon
    accuracy = (number of correct pixels) / (number of pixels in polygon)

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gold_masks: ground truth masks
        shape =  height, width)
    :param correct_class: the correct class of the polygon

    returns: pixel accuracy of the predictions
    """
    relevant_region = gold_mask != BACKGROUND_CLASS
    relevant_pred_region = preds != BACKGROUND_CLASS
    # use the relevant region to only select the pixels in the polygon
    # use the relevant_pred_region to only select the pixels in the pred polygon
    # that are not background pixels as classification errors are only
    # counted for non-background pixels
    pointwise_accuracy = (preds == gold_mask)[relevant_region & relevant_pred_region]

    misclassified_class = calculate_misclassified_class(
        preds, gold_mask, correct_class, relevant_region
    )
    return pointwise_accuracy.sum() / relevant_region.sum(), misclassified_class


def add_classification_error_to_polygons(
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
        if accuracy < ERROR_THRESHOLD:
            polygon.error_type = ErrorType.classification
            if misclassified_label is not None:
                polygon.misclassified_class_label = misclassified_label


def add_classification_error_to_polygons_batch(
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
        add_classification_error_to_polygons(gold_mask, pred_polygons)


def background_accuracy(img_mask: np.ndarray, polygon_mask: np.ndarray) -> float:
    """Calculates the amount of background predicted on a polygon
    calculated as (number of background pixels) / (number of pixels in polygon)

    :param img_mask: mask of image with class (pred or gold)
        shape = (height, width)
    :param polygon_mask: mask of image with only the polygon (pred or gold)
        shape = (height, width)

    returns: accuracy of the predictions
    """
    relevant_region = polygon_mask != BACKGROUND_CLASS
    return (img_mask == BACKGROUND_CLASS)[relevant_region].sum() / relevant_region.sum()


def add_background_errors_to_polygons(
    img_mask: np.ndarray,
    img_polygons: List[Polygon],
    polygon_type: str,
) -> None:
    """Sets error type and pct for missed or background polygons

    For pred polygons we have 'background' errors
    For gold polygons we have 'missed' errors

    Otherwise, the logic is the exact same

    Args:
        img_mask(torch.tensor): argmax of the probabilities per image (pred or gold)
        img_polygons([List[Polygon]):
            Polygons per image (pred or gold)
        polygon_type (str): either "pred" or "gold"
    """
    for polygon in img_polygons:
        polygon_mask = draw_polygon(polygon, img_mask.shape[-2:])
        acc = background_accuracy(img_mask, polygon_mask)
        polygon.background_error_pct = acc
        if polygon_type == "pred" and acc > ERROR_THRESHOLD:
            polygon.error_type = ErrorType.background
        elif polygon_type == "gold" and acc > ERROR_THRESHOLD:
            polygon.error_type = ErrorType.missed


def add_background_errors_to_polygons_batch(
    masks: torch.Tensor,
    polygons_batch: List[List[Polygon]],
    polygon_type: str,
) -> None:
    """Add error type and error pct to polygons

    For missed errors:
      - The pred masks and gold polygons will be passed in

    For background errors:
      - The gold masks and pred polygons will be passed in

    Args:
        masks(torch.tensor): argmax of the probabilities in batch (pred or gold)
        polygons_batch(List[List[Polygon]]):
            Polygons per image in the batch (pred or gold)
        polygon_type (str): either "pred" or "gold"
    """
    for idx in range(len(masks)):
        img_mask = masks[idx].numpy()
        img_polygons = polygons_batch[idx]
        add_background_errors_to_polygons(img_mask, img_polygons, polygon_type)


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
    relevant_region = polygon_img != BACKGROUND_CLASS
    dep_score = dep_map[relevant_region].mean()
    return dep_score


def add_dep_to_polygons_batch(
    polygons_batch: List[List[Polygon]],
    dep_heatmaps: np.ndarray,
    height: List[int],
    width: List[int],
) -> None:
    """Takes the mean dep score within a polygon and sets the polygon's
    dep score to the mean dep score

    Args:
        polygons_batch (List[List[[Polygon]]): list of the polygons (gold or pred)
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
        polygons = polygons_batch[idx]
        for polygon in polygons:
            polygon_img = draw_polygon(polygon, dep_map.shape)
            polygon.data_error_potential = calculate_dep_polygon(dep_map, polygon_img)


def add_lm_polygons_batch(
    mislabeled_pixels: torch.Tensor,
    gold_polygons_batch: List[List[Polygon]],
    pred_polygons_batch: List[List[Polygon]],
    heights: List[int],
    widths: List[int],
) -> None:
    """Calculate and attach the LM percentage per polygon in a batch
    Args:
        mislabeled_pixels (torch.Tensor): map of bs, h, w of mislabled pixels
        polygons_batch (List[List[Polygon]]): polygons for each image
        shape (Tuple[int, int]): shape of the image to be resized to
    """

    for idx in range(len(mislabeled_pixels)):
        mislabeled_pixels = interpolate_lm_map(
            mislabeled_pixels[idx], heights[idx], widths[idx]
        )
        mislabeled_pixel_map = mislabeled_pixels[idx].numpy()

        gold_polygons = gold_polygons_batch[idx]
        pred_polygons = pred_polygons_batch[idx]
        add_lm_polygons(mislabeled_pixel_map, gold_polygons)
        add_lm_polygons(mislabeled_pixel_map, pred_polygons)


def add_lm_polygons(
    mislabelled_pixel_map: torch.Tensor, polygons: List[Polygon]
) -> None:
    """Calculates and attaches the LM percentage to each polygon
    Args:
        mislabelled_pixel_map (torch.Tensor): map of bs, h, w of mislabled pixels
        gold_polygons (List[Polygon]): list of all gold polygons for an image
    """
    for polygon in polygons:
        polygon_img = draw_polygon(polygon, mislabelled_pixel_map.shape[-2:])
        polygon.lm_percentage = add_lm_polygon(mislabelled_pixel_map, polygon_img)


def add_lm_polygon(
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
    if relevant_region.sum() == 0:
        return 0
    return (mislabelled_pixel_map != 0)[relevant_region].sum() / relevant_region.sum()


def interpolate_lm_map(
    mislabelled_pixel_map: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Interpolates the mislabelled pixel map to the same size as the gold mask
    Args:
        mislabelled_pixel_maps (torch.Tensor): map of bs, h, w of mislabled pixels
    Returns:
        np.ndarray: interpolated map of bs, h, w of mislabled pixels
    """
    # for interpolate need to be in format bs, c, h, w and right now we only have h, w
    mislabelled_pixel_map = mislabelled_pixel_map.unsqueeze(0).unsqueeze(0)
    mislabelled_pixel_map = F.interpolate(
        mislabelled_pixel_map, size=(height, width), mode="nearest"
    )

    return mislabelled_pixel_map.squeeze(0).squeeze(0)
