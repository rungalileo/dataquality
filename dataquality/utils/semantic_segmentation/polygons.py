import json
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME
from dataquality.schemas.semantic_segmentation import Contour, ErrorType, Pixel, Polygon

object_store = ObjectStore()


def find_polygon_maps(masks: torch.Tensor) -> List[Dict]:
    """Creates polygon maps for a given batch

    Args:
        masks: Tensor of ground truth or predicted masks
            torch.Tensor of shape (batch_size, height, width)

    Returns:
        List of polygon maps for the mask. Background doesn't get a polygons
    """
    masks_np = masks.numpy()
    bs = masks_np.shape[0]
    polygon_maps = []

    for i in range(bs):
        mask = masks_np[i]
        polygon_maps.append(build_polygon_map(mask))

    return polygon_maps


def build_polygon_map(mask: np.ndarray) -> Dict[int, List[Polygon]]:
    """Returns a polygon map for a given mask

    Args:
        mask: numpy array of shape (height, width) either gt or pred

    Returns:
        Dict: a mapping of labels to its corresponding polygons

    Key is the class and value is a list of Polygon objects

    A polygon is a list of CV2 contours, where each contour is a list of
    pixel coordinates that make up the boundary of a shape.
    """
    polygon_map = {}
    for label in np.unique(mask).astype(int).tolist():
        if label == 0:  # Background pixels don't get polygons
            continue

        class_mask = (mask == label).astype(np.uint8)  # maybe don't need this
        # contours is a tuple of numpy arrays
        contours, hierarchy = cv2.findContours(
            class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon_map[label] = build_polygons(contours, hierarchy)

    return polygon_map


def build_polygons(contours: Tuple[np.ndarray], hierarchy: np.ndarray) -> List[Polygon]:
    """
    :param contours: a tuple of numpy arrays where each array is a contour
    :param hierarchy: a numpy array of shape (num_contours, 4)
        where each row is a contour

    :return: a list of polygons where each polygon is a list of contours
    """
    all_polygons = defaultdict(list)
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
        # process the contour by creating a list of Pixel objects
        new_contours = []
        for j, point in enumerate(contour):
            new_contours.append(Pixel(x=point[0][0], y=point[0][1]))
        all_polygons[current_parent].append(Contour(pixels=new_contours))

    # serialize the polygons for easy storage
    final_polygons = []
    for key in all_polygons:
        final_polygons.append(Polygon(contours=all_polygons[key]))
    return final_polygons


def upload_polygon_map(
    polygon_map: Dict[int, List[Polygon]],
    image_id: int,
    prefix: str,
    error_ids: str,
    error_type: ErrorType,
) -> None:
    """Uploads a polygon map to the cloud for a given image

    Args:
        polygon_map(Dict): polygon map for one image
        image_id(int): image id to be used in the object name
        prefix(str): prefix of the object name in storage
        error_ids(str): comma separated string of error ids
        error_type(ErrorType): type of error defined in the schema
    """
    pmap_json = deserialize_polygon_map_json(polygon_map)

    # add the misclassifed to the pred_polygon_map
    error_ids_list = []

    if len(error_ids) > 0:
        error_ids_list = [int(error_id) for error_id in error_ids.split(",")]
    for polygon_id in error_ids_list:
        pmap_json[polygon_id]["error_type"] = error_type.value

    _upload_polygon_map(image_id, pmap_json, prefix)


def _upload_polygon_map(
    image_id: int,
    polygon_map: List[Dict[str, object]],
    obj_prefix: str,
) -> None:
    """Uploads a contour to the cloud for a given image

    Args:
        image_id(int): image id to be used in the object name
        polygon_map(Dict[int, List]): polygon map for one image
        obj_prefix(str): the prefix of the object name. For example,
            - /proj-id/run-id/training/masks/pred/1.json
    """
    obj_name = f"{obj_prefix}/{image_id}.json"
    with NamedTemporaryFile(mode="w+", delete=False) as f:
        json.dump(polygon_map, f)

    object_store.create_object(
        object_name=obj_name,
        file_path=f.name,
        content_type="application/json",
        progress=False,
        bucket_name=GALILEO_DEFAULT_RESULT_BUCKET_NAME,
    )


def draw_polygon(polygon: List[np.ndarray], img: np.ndarray, key: int) -> np.ndarray:
    """Draws one polygon on an image

    Args:
        polygon (List[np.ndarray]): list of contours
        img (np.ndarray): image to draw contours on
        key (int): key to draw contours with

    Returns:
        np.ndarray: image with contours drawn on it
    """
    blank = np.zeros(img.shape[-2:])
    cv2.drawContours(blank, polygon, -1, key, -1)
    return blank


def deserialize_polygon_map(
    pmap: Dict[int, List[Polygon]]
) -> Dict[int, List[List[np.ndarray]]]:
    """Deserializes a polygon map to be in dictionary form

    Args:
        map (Dict): Mapping of indices to polygons

    Returns:
        Dict[int, List[List[np.ndarray]]]: Mapping of indices to polygons
            in base np.ndarray form
    """
    deserialized_map = {}
    for key, polygons in pmap.items():
        deserialized_polygons = []
        for polygon in polygons:
            deserialized_polygon = polygon.deserialize_opencv()
            deserialized_polygons.append(deserialized_polygon)
        deserialized_map[key] = deserialized_polygons
    return deserialized_map


def deserialize_polygon_map_json(
    pmap: Dict[int, List[Polygon]],
) -> List[Dict[str, object]]:
    """Deserialize the polygon map for json consumptions

    Args:
        map (Dict): Mapping of class to list of polygons

    Returns:
        List[Dict]: Mapping of indices to polygons in json form
    """
    deserialized_map = []
    counter = 0
    for lbl, polygons in pmap.items():
        for polygon in polygons:
            deserialized_polygon = polygon.deserialize_json()
            polygon_object = {
                "id": counter,
                "label_idx": lbl,
                "error_type": "none",
                "polygon": deserialized_polygon,
            }
            counter += 1
            deserialized_map.append(polygon_object)
    return deserialized_map
