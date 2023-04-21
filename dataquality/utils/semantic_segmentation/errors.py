from typing import List, Dict

import numpy as np
import torch
from dataquality.utils.semantic_segmentation.contours import draw_one_blob

def calculate_false_positives(preds: torch.Tensor, gt_masks: torch.Tensor) -> List[str]:
    """Calculates a set of False Positive classes for each image in the batch

    For each image, returns a set of classes that were predicted but not
        present in the ground truth.

    :param preds: argmax of the prediction probabilities
        shape = (bs, height, width)
    :param gt_masks: ground truth masks
        shape = (bs, height, width)
    returns: list of sets of false positive classes for each image in the batch
    """
    false_positives: List[str] = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        gt_mask = gt_masks[image]

        # Calculate classes present in predictions and ground truth
        pred_classes = np.unique(pred_mask).astype(int)
        gt_classes = np.unique(gt_mask).astype(int)

        # Calculate classes present in predictions but not ground truth
        fp_classes = np.setdiff1d(pred_classes, gt_classes).tolist()
        false_positives.append(",".join([str(fp) for fp in fp_classes]))

    return false_positives


def calculate_missing_segments(
    preds: torch.Tensor, gt_masks: torch.Tensor
) -> List[str]:
    """Calculates a set of Missing Segment classes for each image in the batch

    For each image, returns a set of classes that were in the ground truth but not
        present in the predictions.

    :param preds: argmax of the prediction probabilities
        shape = (bs, height, width)
    :param gt_masks: ground truth masks

    returns: list of sets of missing segment classes for each image in the batch
    """
    missing_segments: List[str] = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        gt_mask = gt_masks[image]

        # Calculate classes present in predictions and ground truth
        pred_classes = np.unique(pred_mask).astype(int)
        gt_classes = np.unique(gt_mask).astype(int)

        # Calculate classes present in predictions but not ground truth
        ms_classes = np.setdiff1d(gt_classes, pred_classes).tolist()
        missing_segments.append(",".join([str(ms) for ms in ms_classes]))

    return missing_segments


def blob_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the accuracy of one ground truth blob

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape =  height, width)
    returns: accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    pointwise_accuracy = (preds == gt_mask)[relevant_region]
    return pointwise_accuracy.sum() / relevant_region.sum()

def image_miscls_segments(pred_mask: np.ndarray, 
                           unserialized_contours_map: Dict[int, List[List[np.ndarray]]]
) -> List[str]:
    """Calculates a set of Missing Segment classes for one image"""
    all_missing_segments = []
    counter = 0
    for key in sorted(unserialized_contours_map.keys()):
        for blob in unserialized_contours_map[key]:
            out_blob = draw_one_blob(blob, pred_mask, key)
            if blob_accuracy(pred_mask, out_blob) < .5:
                all_missing_segments.append(str(counter))
            counter += 1
    out_string = ",".join(all_missing_segments)
    return out_string


def calculate_miscls_segments_blob(preds: torch.tensor, 
                              unserialized_contours_maps: List[Dict[int, List[List[np.ndarray]]]]
) -> List[str]:
    """Calculates a set of Misclassified Segment classes for each image in the batch"""
    segments_per_image = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        unserialized_contours_map = unserialized_contours_maps[image].unserialize()
        # Calculate classes present in predictions and ground truth
        missing_segments = image_miscls_segments(pred_mask.numpy(), unserialized_contours_map)
        segments_per_image.append(missing_segments)
    return segments_per_image

def undetected_accuracy(preds: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates the amount of background predicted on a blob 

    :param preds: argmax of the prediction probabilities
        shape = (height, width)
    :param gt_masks: ground truth masks
        shape =  height, width)
    returns: accuracy of the predictions
    """
    relevant_region = gt_mask != 0
    return (preds == 0)[relevant_region].sum() / relevant_region.sum()
    


def image_undetected_object(pred_mask: np.ndarray, 
                           unserialized_contours_map: Dict[int, List[List[np.ndarray]]]
) -> List[str]:
    """Calculates a set of Missing Segment classes for one image"""
    all_undetected_objects = []
    counter = 0
    for key in sorted(unserialized_contours_map.keys()):
        for blob in unserialized_contours_map[key]:
            out_blob = draw_one_blob(blob, pred_mask, key)
            if undetected_accuracy(pred_mask, out_blob) > .5:
                all_undetected_objects.append(str(counter))
            counter += 1
    out_string = ",".join(all_undetected_objects)
    return out_string


def calculate_undetected_object(preds: torch.Tensor, 
                                unserialized_contours_maps: List[Dict[int, List[List[np.ndarray]]]]
) -> List[str]:
    """Calculates a set of Undetected Object classes for each image in the batch"""
    undetected_objects = []
    for image in range(len(preds)):
        pred_mask = preds[image]
        unserialized_contours_map = unserialized_contours_maps[image].unserialize()
        # Calculate classes present in predictions and ground truth
        undetected_segments = image_undetected_object(pred_mask.numpy(), unserialized_contours_map)
        undetected_objects.append(undetected_segments)
        
    return undetected_objects
