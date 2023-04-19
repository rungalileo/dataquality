import json
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME

object_store = ObjectStore()


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
        s_contour_map = serialize_contours(contour_map)
        contour_list.append(contour_map)
        obj_name = _upload_contour(image_id, s_contour_map, obj_prefix)
        paths.append(obj_name)
    return contour_list

def _upload_contour(
    image_id: int, contour_map: Dict[int, List], obj_prefix: str
) -> None:
    """Uploads a contour to the cloud for a given image

    obj_prefix is the prefix of the object name. For example,
        - /p

    """
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

def find_and_return_contours(
    gt_masks: torch.Tensor
) -> List[Dict[int, List[List[np.ndarray]]]]:
    """Creates and serialize contours for a given batch
    
    Args:
        gt_masks: Tensor of predicted masks
            torch.Tensor of shape (batch_size, height, width)
    Returns:
        List of contour maps
    """
    gt_masks_np = gt_masks.numpy()
    contour_maps = []
    if gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)
    for i in range(gt_masks.shape[0]):
        gt_mask = gt_masks_np[i]
        contour_map = find_contours(gt_mask)
        contour_maps.append(contour_map)
    return contour_maps



def find_contours(mask: np.ndarray) -> Dict[int, List[List[np.ndarray]]]:
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
        contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_map[label] = find_blobs(contours, hierarchy)

    return contours_map

def find_blobs(contours: Tuple[np.ndarray], hierarchy: np.ndarray) -> List[List[np.ndarray]]:
    """
    :param contours: a tuple of numpy arrays where each array is a contour
    :param hierarchy: a numpy array of shape (num_contours, 4) where each row is a contour
    
    :return: a list of blobs where each blob is a list of contours
    """
    all_blobs = defaultdict(list)
    for i, contour in enumerate(contours):
        parent = hierarchy[0, i, -1]
         # indicates that it does not have any parent so give it its own entry
        if parent == -1:
            all_blobs[i].append(contour)
        else:
            current_parent = parent
            next_parent = parent
            while next_parent != -1:
                current_parent = next_parent
                next_parent = hierarchy[0, next_parent, -1]
            all_blobs[current_parent].append(contour)

    # serialize the blobs for easy storage
    final_blobs = []
    for key in all_blobs:
        final_blobs.append(all_blobs[key])
    return final_blobs
    


def serialize_contours(contour_map: Dict[int, Tuple]
                       ) -> Dict[int, List[List[List[List[int]]]]]:
    """
    Converts a contour from a numpy array to a list of pixel coordinates

    Input:
    contours - a dictionary where the keys are integers representing object labels and
               the values are numpy arrays of shape (num_points, 1, 2) representing the
               contours of the corresponding objects.

    Output:
    A list of tuples representing the pixel coordinates of each contour.

    Example input:
    contours = {
        7: (
            np.array([[[13, 17]], [[19, 25]], [[22, 21]], [[13, 17]]]),
            np.array([[[12, 5]], [[11, 7]], [[10, 9]], [[12, 15]]]),
        ),
        15: (
            np.array([[[0, 3]], [[2, 5]], [[4, 6]], [[2, 2]], [[0, 3]]])
        )
    }
    print(contours[0].shape)  # Output: (4, 1, 2)

    Example output:
    {
        7: [  # 2 contours for class 7
            [[13, 17], [19, 25], [22, 21], [13, 17]],
            [[12, 5], [11, 7], [10, 9], [12, 15]],
        ],
        15: [
            [[0, 3], [2, 5], [4, 6], [2, 2], [0, 3]]
        ]
    }
    """
    serialized_contour_map = {}
    for label, contours in contour_map.items():
        # Remove the extra dimension in the numpy array and convert to a list of tuples
        serialized_blobs = []
        for blob in contours:
            blob_squeezed = [contour.squeeze(1).tolist() for contour in blob]
            serialized_blobs.append(blob_squeezed)
        serialized_contour_map[label] = serialized_blobs
    return serialized_contour_map


def unserialize_contours(serialized_contour_map: Dict[int, List]
) -> Dict[int, List[List[np.ndarray]]]:
    """
    Function to convert a serialized contour map back to a contour map for 
        plotting and use in error types
        
    Args:
        serialized_contour_map Dict[int, List]: the serialized contour map
            given by the function above

    Returns:
        Dict[int, List[np.ndarray]]: the contour map with the contours
    """
    unserialized_contour_map = {}
    for label, contours in serialized_contour_map.items():
        unserialized_blobs = []
        for blob in contours:
            unserialized_blob = [np.array(contour) for contour in blob]
            # unserialized_blob = np.array(blob).reshape(-1, 1, 2)
            unserialized_blobs.append(unserialized_blob)
        unserialized_contour_map[label] = unserialized_blobs

    return unserialized_contour_map

def draw_contours(serialized_contour_map: Dict[int, List], 
                  img: np.ndarray) -> np.ndarray:
    """Draws the contours from our serialized contour map

    Args:
        serialized_contour_map (Dict[int, List]): Map from key
            to blobs where blobs are a list of contours
        img (np.ndarray): image to draw contours on

    Returns:
        np.ndarray: image with contours drawn on it
    """
    blank = np.zeros(img.shape[-2:])
    unserialized_contours = unserialize(serialized_contour_map)  
    for key in unserialized_contours:
        for blob in unserialized_contours[key]:
            cv2.drawContours(blank, blob, -1, key, -1)
    return blank


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
