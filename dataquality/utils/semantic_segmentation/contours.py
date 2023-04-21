import json
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME
from dataquality.utils.semantic_segmentation.type import (
    Polygon,
    Contour,
    Polygon_Map,
    Pixel,
)

object_store = ObjectStore()


def find_polygon_maps(
    image_ids: List[int], pred_masks: torch.Tensor
) -> List[Polygon_Map]:
    """Creates polygon maps for a given batch

    Args:
        image_ids: List of image ids
        pred_masks: Tensor of predicted masks
            torch.Tensor of shape (batch_size, height, width)
    
    Returns:
        List of polygon maps
    """
    pred_masks_np = pred_masks.numpy()
    contour_list = []
    for i in range(len(image_ids)):
        pred_mask = pred_masks_np[i]
        contour_map = find_contours(pred_mask)
        contour_list.append(contour_map)
    return contour_list


def find_and_upload_contours(
    image_ids: List[int], pred_masks: torch.Tensor, obj_prefix: str
) -> List[Dict[int, List[List[np.ndarray]]]]:
    """Creates and uploads contours to the cloud for a given batch

    Args:
        image_ids: List of image ids
        pred_masks: Tensor of predicted masks
            torch.Tensor of shape (batch_size, height, width)
        obj_prefix: Prefix for the object store
            example: proj-id/run-id/split/contours/
    """
    pred_masks_np = pred_masks.numpy()
    paths = []
    contour_list = []
    for i in range(len(image_ids)):
        image_id = image_ids[i]
        pred_mask = pred_masks_np[i]
        contour_map = find_contours(pred_mask)
        contour_list.append(contour_map)
        obj_name = _upload_contour(image_id, contour_map, obj_prefix)
        paths.append(obj_name)
    return contour_list


def upload_contours(
    image_id: int,
    pred_polygon_map: Polygon_Map,
    gt_polygon_map: Polygon_Map,
    pred_obj_prefix: str,
    gt_obj_prefix: str,
    misclassified_obj: List[str],
    undetected_obj: List[str],
) -> None:
    pred_polygon_map = pred_polygon_map.unserialize_json()
    gt_polygon_map = gt_polygon_map.unserialize_json()
    
    # add the misclassifed to the pred_polygon_map
    int_misclassified_obj = []
    if len(misclassified_obj) >= 0:
        int_misclassified_obj = [int(obj) for obj in misclassified_obj.split(',')]
    for obj in int_misclassified_obj:
        pred_polygon_map[obj]['error_type'] = 'misclassified'
        
    # add the undetected to the gt_polygon_map
    int_undetected_obj = []
    if len(undetected_obj) >= 0:
        undetected_obj = [int(obj) for obj in undetected_obj.split(',')]
    for obj in int_undetected_obj:
        gt_polygon_map[obj]['error_type'] = 'undetected'
        
    _upload_contour(image_id, pred_polygon_map, pred_obj_prefix)
    _upload_contour(image_id, gt_polygon_map, gt_obj_prefix)
    


def _upload_contour(
    image_id: int, contour_map: Dict[int, List], obj_prefix: str
) -> None:
    """Uploads a contour to the cloud for a given image

    obj_prefix is the prefix of the object name. For example,
        - /p

    """
    # contour_map = contour_map.unserialize_json()
    obj_name = f"{obj_prefix}/{image_id}.json"
    with NamedTemporaryFile(mode="w+", delete=False) as f:
        json.dump(contour_map, f)

    object_store.create_object(
        object_name=obj_name,
        file_path=f.name,
        content_type="application/json",
        progress=False,
        bucket_name=GALILEO_DEFAULT_RESULT_BUCKET_NAME,
    )


def find_contours(mask: np.ndarray) -> Polygon_Map:
    """Returns a list of blob lists of contours

    A contour is a list of points that make up the boundary of a shape.
    Each image can be represented as a dictionary mapping a GT class to
        its corresponding contours.


    cv2.findContours returns a Tuple of contours, where each contour is a
        numpy array of shape (num_points, 1, 2)


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
    for label in np.unique(mask).astype(int).tolist():
        if label == 0:
            continue
        contour_mask = mask == label
        contour_mask = contour_mask.astype(np.uint8)  # maybe don't need this
        # contours is a tuple of numpy arrays
        contours, hierarchy = cv2.findContours(
            contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_map[label] = find_blobs(contours, hierarchy)

    return Polygon_Map(map=contours_map)


def find_blobs(contours: Tuple[np.ndarray], hierarchy: np.ndarray) -> List[Polygon]:
    """
    :param contours: a tuple of numpy arrays where each array is a contour
    :param hierarchy: a numpy array of shape (num_contours, 4) where each row is a contour

    :return: a list of blobs where each blob is a list of contours
    """
    all_blobs = defaultdict(list)
    for i, contour in enumerate(contours):
        parent = hierarchy[0, i, -1]
        # indicates that it does not have any parent so give it its own entry
        if parent != -1:
            current_parent = parent
            next_parent = parent
            while next_parent != -1:
                current_parent = next_parent
                next_parent = hierarchy[0, next_parent, -1]
        else:
            current_parent = i
        # process the contour
        new_contours = []
        for j, point in enumerate(contour):
            new_contours.append(Pixel(coord=[[point[0][0], point[0][1]]]))
        all_blobs[current_parent].append(Contour(pixels=new_contours))

    # serialize the blobs for easy storage
    final_blobs = []
    for key in all_blobs:
        final_blobs.append(Polygon(contours=all_blobs[key]))
    return final_blobs

def draw_one_blob(blob: List[np.ndarray], img: np.ndarray, key: int) -> np.ndarray:
    """Draws one blob on an image

    Args:
        blob (List[np.ndarray]): list of contours
        img (np.ndarray): image to draw contours on
        key (int): key to draw contours with

    Returns:
        np.ndarray: image with contours drawn on it
    """
    blank = np.zeros(img.shape[-2:])
    cv2.drawContours(blank, blob, -1, key, -1)
    return blank
