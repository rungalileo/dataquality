from typing import List

import numpy as np
import torch
from torch.nn import functional as F

from dataquality.schemas.semantic_segmentation import (
    ClassificationErrorData,
    ErrorType,
    Polygon,
    PolygonType,
)
from dataquality.utils.semantic_segmentation.constants import (
    BACKGROUND_CLASS,
    ERROR_THRESHOLD,
)
from dataquality.utils.semantic_segmentation.polygons import draw_polygon


def resize_maps(
    mislabeled_pixel_map: torch.Tensor, height: int, width: int
) -> np.ndarray:
    """Resize the lm/dep map to the correct size (size of the mask)
    for interpolate need to be in format bs, c, h, w and right now we only have bs, h, w
    this results in mislabeled_pixel_map.shape = (1, 1, h, w)
    """
    mislabeled_pixel_map_unsqueezed = mislabeled_pixel_map.unsqueeze(1)
    mislabeled_pixel_map_interpolated = F.interpolate(
        mislabeled_pixel_map_unsqueezed, size=(height, width), mode="nearest"
    )
    # squeeze the extra dimensions back to (h, w)
    return mislabeled_pixel_map_interpolated.squeeze(1).numpy()


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


def add_background_error_to_polygon(
    mask: np.ndarray, polygon_img: np.ndarray, polygon: Polygon, polygon_type: str
) -> None:
    """Adds background error to a polygon

    Args:
        mask (np.ndarray): mask of the image either gold or pred
        polygon_img (np.ndarray): img with polygon drawn on it
        polygon (Polygon): actual polygon object
        polygon_type (str): polygon type either gold or pred
    """
    acc = background_accuracy(
        mask,
        polygon_img,
    )
    polygon.background_error_pct = acc
    if polygon_type == PolygonType.pred and acc > ERROR_THRESHOLD:
        polygon.error_type = ErrorType.background
    elif polygon_type == PolygonType.gold and acc > ERROR_THRESHOLD:
        polygon.error_type = ErrorType.missed


def add_dep_to_polygon(
    dep_map: np.ndarray, polygon_img: np.ndarray, polygon: Polygon
) -> None:
    """Adds dep of the polygon to a specific polygon

    Args:
        dep_map (np.ndarray): dep heatmap with correct values
        polygon_img (np.ndarray): polygon drawn onto an image
        polygon (Polygon): polygon object
    """
    relevant_region = polygon_img != BACKGROUND_CLASS
    dep_score = dep_map[relevant_region].mean()
    polygon.data_error_potential = dep_score


def add_area_to_polygon(polygon_img: np.ndarray, polygon: Polygon) -> None:
    """Adds area of the polygon to a specific polygon

    Args:
        polygon_img (np.ndarray): polygon drawn onto an image
        polygon (Polygon): polygon object
    """
    polygon.area = (polygon_img != 0).sum()


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
    combined_relevant_region = relevant_region & relevant_pred_region
    # use the relevant region to only select the pixels in the polygon
    # use the relevant_pred_region to only select the pixels in the pred polygon
    # that are not background pixels as classification errors are only
    # counted for non-background pixels
    pointwise_accuracy = (candidate_mask == comparison_mask)[combined_relevant_region]
    area = relevant_region.sum()
    float_accuracy = pointwise_accuracy.sum() / area

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


def add_class_errors_to_polygon(
    mask: np.ndarray,
    polygon_img: np.ndarray,
    polygon: Polygon,
    polygon_type: str,
    number_classes: int,
) -> None:
    """Adds class error to a polygon

    Args:
        mask (np.ndarray): mask of the image either gold or pred
        polygon_img (np.ndarray): img with polygon drawn on it
        polygon (Polygon): actual polygon object
        polygon_type (str): polygon type either gold or pred
        number_classes (int): number of classes in the mask
    """
    polygon.cls_error_data = calculate_classification_error(
        mask, polygon_img, polygon.label_idx, number_classes
    )
    acc = polygon.cls_error_data.accuracy
    if polygon_type == PolygonType.pred and acc < ERROR_THRESHOLD:
        polygon.error_type = ErrorType.classification
    if polygon_type == PolygonType.gold and acc < ERROR_THRESHOLD:
        polygon.error_type = ErrorType.class_confusion


def calculate_lm_pct(
    mislabeled_pixel_map: np.ndarray, polygon_img: np.ndarray
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


def add_lm_to_polygon(
    mislabeled_pixels: np.ndarray, polygon_img: np.ndarray, polygon: Polygon
) -> None:
    """Adds lm of the polygon to a specific polygon

    Args:
        lm_map (np.ndarray): lm heatmap with correct values
        polygon_img (np.ndarray): polygon drawn onto an image
        polygon (Polygon): polygon object
    """
    polygon.likely_mislabeled_pct = calculate_lm_pct(mislabeled_pixels, polygon_img)


def add_all_errors_and_metrics_for_image(
    mask: np.ndarray,
    polygons: List[Polygon],
    number_classes: int,
    polygon_type: str,
    dep_heatmap: np.ndarray,
    mislabeled_pixels: np.ndarray,
) -> None:
    """Calculates and attaches the error types to each polygon

    Args:
        mask (np.ndarray): mask of the image
        polygons (List[Polygon]): list of all polygons for an image
        number_classes (int): number of classes in the mask
        polygon_type (PolygonType): whether the polygons are gold or pred
        dep_heatmap (np.ndarray): heatmap of the depth
        mislabled_pixels (np.ndarray): map of h, w of mislabled pixels
    """
    for polygon in polygons:
        if polygon.polygon_type == PolygonType.dummy:
            # We don't need to calculate errors for dummy polygons
            continue
        out_polygon_im = draw_polygon(polygon, mask.shape[-2:])
        add_class_errors_to_polygon(
            mask, out_polygon_im, polygon, polygon_type, number_classes
        )
        add_background_error_to_polygon(mask, out_polygon_im, polygon, polygon_type)
        add_dep_to_polygon(dep_heatmap, out_polygon_im, polygon)
        add_area_to_polygon(out_polygon_im, polygon)
        add_lm_to_polygon(mislabeled_pixels, out_polygon_im, polygon)


def add_errors_and_metrics_to_polygons_batch(
    masks: torch.Tensor,
    polygons: List[List[Polygon]],
    number_classes: int,
    polygon_type: str,
    dep_heatmaps: torch.Tensor,
    mislabeled_pixels: torch.Tensor,
    height: int,
    width: int,
) -> None:
    """Calculates and attaches the error types to each polygon for a whole batch

    Args:
        mask (np.ndarray): masks of the images
        polygons (List[Polygon]): list of all polygons for all images
        number_classes (int): number of classes in the mask
        polygon_type (PolygonType): whether the polygons are gold or pred
        dep_heatmaps (torch.Tensor): dep heatmaps for each image
        mislabeled_pixels (torch.Tensor): map of bs, h, w of mislabled pixels
        heights (List[int]): height of each image
        widths (List[int]): width of each image

    """
    lm_maps_resized = resize_maps(mislabeled_pixels, height, width)
    dep_maps_resized = resize_maps(dep_heatmaps, height, width)
    for idx in range(len(masks)):
        mask = masks[idx].numpy()
        polygons_for_image = polygons[idx]
        add_all_errors_and_metrics_for_image(
            mask,
            polygons_for_image,
            number_classes,
            polygon_type,
            dep_maps_resized[idx],
            lm_maps_resized[idx],
        )
