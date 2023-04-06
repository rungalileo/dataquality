"""Utils for Object Detection"""

import numpy as np


def convert_cxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts center point xywh boxes to xyxy. Can be in either int coords or 0-1"""
    if bboxes.shape[0] == 0:
        return bboxes
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1, x2 = x - w / 2, x + w / 2
    y1, y2 = y - h / 2, y + h / 2
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x1, y1, x2, y2
    return bboxes


def convert_tlxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts top left xywh boxes to xyxy. Can be in either integer coords or 0-1"""
    if bboxes.shape[0] == 0:
        return bboxes
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x2, y2 = x + w, y + h
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x, y, x2, y2
    return bboxes
