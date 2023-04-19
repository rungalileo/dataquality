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


def blob_accuracy(preds: torch.Tensor, gt_mask: torch.Tensor) -> float:
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

def image_missing_segments(pred_mask: torch.Tensor, 
                           unserialized_contours_map: Dict[int, List[List[np.ndarray]]]
                           ) -> List[str]:
    all_missing_segments = []
    for key in unserialized_contours_map:
        for blob in unserialized_contours_map[key]:
            out_blob = draw_one_blob(blob, pred_mask)
            if blob_accuracy(pred_mask, out_blob) < .5:
                all_missing_segments.append([blob])


def calculate_missing_segments_blob(preds: torch.tensor, 
                              unserialized_contours_maps: List[Dict[int, List[List[np.ndarray]]]]
                              ) -> List[str]:
    for image in range(len(preds)):
        pred_mask = preds[image]
        unserialized_contours_map = unserialized_contours_maps[image]
        # Calculate classes present in predictions and ground truth
        missing_segments = image_missing_segments(pred_mask, unserialized_contours_map)
        import pdb; pdb.set_trace()
                    
                
