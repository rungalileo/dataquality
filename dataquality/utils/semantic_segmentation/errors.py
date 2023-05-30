from typing import List

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dataquality.schemas.semantic_segmentation import (
    ClassificationErrorData,
    ErrorType,
    Polygon,
)
from dataquality.utils.semantic_segmentation.constants import (
    BACKGROUND_CLASS,
    ERROR_THRESHOLD,
)
from dataquality.utils.semantic_segmentation.polygons import draw_polygon


def calculate_classification_error(
    candidate_mask: np.ndarray,
    comparison_mask: np.ndarray,
    correct_class: int,
    number_classes: int,
) -> ClassificationErrorData:
    """Calculates the accuracy of one ground truth polygon
    accuracy = (number of correct pixels) / (number of pixels in polygon)
    as well as the class with the most incorrect pixels in that respective polyon
    and the proportion of pixels in the polygon that are that class

    :param candidate_mask: mask we are using to calulcate the accuracy
        ie. if we are computing accuracy for gold polygons this is the pred
        mask and vice versa
    :param comparison_mask: mask we are using to compare the candidate mask to
        ie. if we are computing accuracy for gold polygons this is the gold
        mask and vice versa
    :param correct_class: the correct class of the polygon
    :param number_classes: number of classes

    returns: pixel accuracy of the predictions
    """
    relevant_region = comparison_mask != BACKGROUND_CLASS
    relevant_pred_region = candidate_mask != BACKGROUND_CLASS
    # use the relevant region to only select the pixels in the polygon
    # use the relevant_pred_region to only select the pixels in the pred polygon
    # that are not background pixels as classification errors are only
    # counted for non-background pixels
    pointwise_accuracy = (candidate_mask == comparison_mask)[
        relevant_region & relevant_pred_region
    ]
    float_accuracy = pointwise_accuracy.sum() / relevant_region.sum()

    area = relevant_region.sum()
    region_pixels = candidate_mask[relevant_region]
    region_boolean = region_pixels != correct_class
    incorrect_pixels = region_pixels[region_boolean]
    # count the number of pixels in the pred mask relevant region that are
    # not the correct class
    areas = np.bincount(incorrect_pixels, minlength=number_classes)
    argmax = np.argmax(areas)
    return ClassificationErrorData(
        accuracy=float_accuracy,
        mislabeled_class=argmax,
        mislabeled_class_pct=areas[argmax] / area,
    )


def add_classification_error_to_polygons(
    mask: np.ndarray,
    polygons: List[Polygon],
    number_classes: int,
) -> None:
    """Checks for polygon misclassifications and sets the Polygon error_type field

    A misclassified polygon is a polygon from the predicted mask
    that has a pixel accuracy less than ERROR_THRES (0.5)

    In other words, if less than 50% of the pixels in the predicted polygon
    were correct, then the polygon is misclassified.

    Args:
        mask (np.ndarray): mask of the image either gold or pred
            depending on which polygons are being checked
        polygons (List[Polygon]): list of polygons to check
        number_classes (int): number of classes
    """
    for polygon in polygons:
        out_polygon_im = draw_polygon(polygon, mask.shape[-2:])
        polygon.cls_error_data = calculate_classification_error(
            mask, out_polygon_im, polygon.label_idx, number_classes
        )
        if polygon.cls_error_data.accuracy < ERROR_THRESHOLD:
            polygon.error_type = ErrorType.classification


def add_classification_error_to_polygons_batch(
    masks: torch.Tensor,
    polygons_batch: List[List[Polygon]],
    number_classes: int,
) -> None:
    """Calculates a set of misclassified polygon ids from the
    predicted mask for each image in a batch

    Also sets the error type field on the bad polygons to "misclassified"

    Args:
        masks(torch.tensor): mask of the image either gold or pred
            depending on which polygons are being checked
        polygons_batch(List[List[Polygon]]):
            list of polygons for each image in a batch
        number_classes(int): number of classes
    """
    for idx in range(len(masks)):
        gold_mask = masks[idx].numpy()
        pred_polygons = polygons_batch[idx]
        add_classification_error_to_polygons(gold_mask, pred_polygons, number_classes)


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


def calculate_lm_pct(
    mislabeled_pixel_map: torch.Tensor, polygon_img: np.ndarray
) -> float:
    """Calculates the percentage of mislabeled pixels in a polygon

    Args:
        mislabeled_pixel_map (torch.Tensor): map of bs, h, w of mislabled pixels
            with value 1 if LM, 0 otherwise
        polygon_img (np.ndarray): np array of the polygon drawn onto an image
    Returns:
        float: percentage of mislabeled pixels in a polygon
    """
    relevant_region = polygon_img != BACKGROUND_CLASS
    if relevant_region.sum() == 0:
        return 0

    return (mislabeled_pixel_map != 0)[relevant_region].sum() / relevant_region.sum()


def add_lm_to_polygons(
    mislabeled_pixel_map: torch.Tensor, polygons: List[Polygon]
) -> None:
    """Calculates and attaches the LM percentage to each polygon
    Args:
        mislabeled_pixel_map (torch.Tensor): map of bs, h, w of mislabled pixels
        gold_polygons (List[Polygon]): list of all gold polygons for an image
    """
    for polygon in polygons:
        polygon_img = draw_polygon(polygon, mislabeled_pixel_map.shape[-2:])
        polygon.likely_mislabeled_pct = calculate_lm_pct(
            mislabeled_pixel_map, polygon_img
        )


def add_lm_to_polygons_batch(
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
        mislabeled_pixel_map = interpolate_lm_map(
            mislabeled_pixels[idx], heights[idx], widths[idx]
        ).numpy()

        gold_polygons = gold_polygons_batch[idx]
        pred_polygons = pred_polygons_batch[idx]
        add_lm_to_polygons(mislabeled_pixel_map, gold_polygons)
        add_lm_to_polygons(mislabeled_pixel_map, pred_polygons)


def interpolate_lm_map(
    mislabeled_pixel_map: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Interpolates the mislabeled pixel map to the same size as the gold mask
    Args:
        mislabeled_pixel_maps (torch.Tensor): map of bs, h, w of mislabled pixels
    Returns:
        np.ndarray: interpolated map of bs, h, w of mislabled pixels
    """
    # for interpolate need to be in format bs, c, h, w and right now we only have h, w
    # this results in mislabeled_pixel_map.shape = (1, 1, h, w)
    mislabeled_pixel_map_unsqueezed = mislabeled_pixel_map.unsqueeze(0).unsqueeze(0)
    mislabeled_pixel_map_interpolated = F.interpolate(
        mislabeled_pixel_map_unsqueezed, size=(height, width), mode="nearest"
    )
    # squeeze the extra dimensions back to (h, w)
    return mislabeled_pixel_map_interpolated.squeeze(0).squeeze(0)
