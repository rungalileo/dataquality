import json
import os
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch

from dataquality.clients.objectstore import ObjectStore


object_store = ObjectStore()


def find_and_upload_contours(
    image_ids: List[int], pred_masks: torch.Tensor, obj_prefix: str
) -> None:
    """Creates and uploads contours to the cloud for a given batch
    
    Args:
        image_ids: List of image ids
        pred_masks: Tensor of predicted masks
            torch.Tensor of shape (batch_size, height, width)
        obj_prefix: Prefix for the object store
            example: proj-id/run-id/split/contours/
    """
    pred_masks_np = pred_masks.numpy()
    for i in range(len(image_ids)):
        image_id = image_ids[i]
        pred_mask = pred_masks_np[i]
        contours = find_contours(pred_mask)
        s_contours = serialize_contours(contours)
        # _upload_contour(image_id, s_contours, obj_prefix)


def find_contours(pred_mask: np.ndarray) -> Dict[int, Tuple]:
    """Returns a list of GT contours from the pred mask

    A contour is a list of points that make up the boundary of a shape.
    Each image can be represented as a dictionary mapping a GT class to
        its corresponding contours.

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
    for label in np.unique(pred_mask).astype(int).tolist():
        if label == 0:
            continue

        mask = pred_mask == label
        mask = mask.astype(np.uint8)  # maybe don't need this
        # contours is a tuple of numpy arrays
        # erroring when shape is (1, H, W)
        if mask.shape[0] > 1:
            raise ValueError(f"Mask shape is {mask.shape}, expected (H, W)")
        contours, _ = cv2.findContours(mask[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_map[label] = contours

    return contours_map


def serialize_contours(contours: Dict[int, Tuple]) -> List[Tuple[int]]:
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
        0: np.array([[[13, 17]], [[19, 25]], [[22, 21]], [[13, 17]]]),
        1: np.array([[[0, 3]], [[2, 5]], [[4, 6]], [[2, 2]], [[0, 3]]])
    }
    print(contours[0].shape)  # Output: (4, 1, 2)

    Example output:
    [
        [(13, 17), (19, 25), (22, 21), (13, 17)],
        [(0, 3), (2, 5), (4, 6), (2, 2), (0, 3)]
    ]
    """
    serialized_contours = defaultdict(list)
    for label, contour in contours.items():
        for contour_item in contour:
            # Remove the extra dimension in the numpy array and convert to a list of tuples
            serialized_contours[label].append(list(map(tuple, contour_item.squeeze(1).tolist())))

    return list(serialized_contours.values())


def _upload_contour(image_id: int, contour: Dict[int, List], obj_prefix: str) -> None:
    """Uploads a contour to the cloud for a given image
    
    obj_prefix is the prefix of the object name. For example,
        - /p

    """
    obj_name = f"{obj_prefix}/{image_id}.json"
    with NamedTemporaryFile(suffix=".json", mode="w+") as f:
        json.dump(contour, f)
        # object_store.create_object(
        #     object_name=obj_name,
        #     file_path=f,
        #     content_type="application/json",
        #     progress=False,
        # )
