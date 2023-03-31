import json
import os
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

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
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_map[label] = contours

    return contours_map


def serialize_contours(contours: Dict[int, Tuple]) -> List[Tuple[int]]:
    """Converts a contour from a numpy array to a list of pixel coordinates

    Example input:
    contour = np.array([[[13, 17]], [[19, 25]], [[22, 21]], [[13, 17]]])
    print(contour.shape)
        => (4, 1, 2)

    Example output:
    [
        (13, 17),
        (19, 25),
        (22, 21),
        (13, 17),
    ]
    """
    serialized_contours = {}
    for label, contour in contours.items():
        serialized_contours[label] = list(map(tuple, contour.squeeze(1).tolist()))
    return serialized_contours

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
