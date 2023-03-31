from typing import List, Set

import numpy as np


def calculate_false_positives(
    preds: np.ndarray, gt_masks: np.ndarray
) -> List[Set[int]]:
    """Calculates a set of False Positive classes for each image in the batch

    For each image, returns a set of classes that were predicted but not
        present in the ground truth.

    :param preds: argmax of the prediction probabilities
    :param gt_masks: ground truth masks
    returns: list of sets of false positive classes for each image in the batch
    """
    false_positives: List[Set[int]] = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        gt_mask = gt_masks[image]

        # Calculate classes present in predictions and ground truth
        pred_classes = set(np.unique(pred_mask).astype(int))
        gt_classes = set(np.unique(gt_mask).astype(int))

        # Calculate classes present in predictions but not ground truth
        fp_classes = pred_classes.difference(gt_classes)
        false_positives.append(fp_classes)

    return false_positives


def calculate_missing_segments(
    preds: np.ndarray, gt_masks: np.ndarray
) -> List[Set[int]]:
    """Calculates a set of Missing Segment classes for each image in the batch

    For each image, returns a set of classes that were in the ground truth but not
        present in the predictions.

    :param preds: argmax of the prediction probabilities
    :param gt_masks: ground truth masks
    returns: list of sets of missing segment classes for each image in the batch
    """
    missing_segments: List[Set[int]] = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        gt_mask = gt_masks[image]

        # Calculate classes present in predictions and ground truth
        pred_classes = set(np.unique(pred_mask).astype(int))
        gt_classes = set(np.unique(gt_mask).astype(int))

        # Calculate classes present in predictions but not ground truth
        ms_classes = gt_classes.difference(pred_classes)
        missing_segments.append(ms_classes)

    return missing_segments
