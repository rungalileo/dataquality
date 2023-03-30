from typing import List, Set

import evaluate
import numpy as np
import torch


def probs_to_preds(probs: torch.Tensor) -> torch.Tensor:
    """Takes pixel-wise arg-max to return preds"""
    return torch.argmax(probs, axis=-1)


def calculate_dep_heatmap(probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Calculates the Data Error Potential (DEP) for each image in the batch"""
    # probs: float, (bs, n_pixels, n_classes) or (bs, n_rows, n_cols, n_classes)
    # y: int, (bs, n_pixels,) or (bs, n_rows, n_cols)
    # returns: (bs, n_pixels)
    n_classes = probs.shape[-1]
    bs = probs.shape[0]
    # flatten the height and width dimensions
    probs = probs.view(bs, -1, n_classes)
    size = gt_masks.shape
    gt_masks = gt_masks.view(bs, -1, 1)

    gt_indices = (
        gt_masks.reshape((bs, -1, 1)).expand(-1, -1, probs.shape[2]).type(torch.int64)
    )
    value_at_ground_truth = torch.gather(probs, 2, gt_indices)[:, :, 0]

    next_highest = probs.clone()
    # Takes GT indices and puts 0 at that index so we don't use it as next highest value
    next_highest.scatter_(2, gt_indices, 0)
    next_highest = next_highest.max(dim=2).values

    dep_masks = 1 - ((value_at_ground_truth - next_highest) / 2)
    dep_masks = dep_masks.view(size)

    return dep_masks


def calculate_image_dep(dep_heatmap: torch.Tensor) -> List[float]:
    """Calculates the Data Error Potential (DEP) for each image in the batch"""
    return dep_heatmap.mean(axis=1).tolist()


def calculate_mean_iou(
    pred_boundary_masks: torch.Tensor, gold_boundary_masks: torch.Tensor, nc: int = 21
) -> List[float]:
    """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch"""
    metric = evaluate.load("mean_iou")
    ious = []
    for i in range(len(pred_boundary_masks)):
        iou = metric._compute(
            pred_boundary_masks[i : i + 1],  # tensor 1, 64, 21
            gold_boundary_masks[i : i + 1],  # tensor 1, 64, 64
            num_labels=nc,
            ignore_index=255,
        )
        ious.append(iou["mean_iou"].item())
    return ious


def calculate_false_positives(
    preds: np.ndarray, gt_masks: np.ndarray
) -> List[Set[int]]:
    """Calculates a set of False Positive classes for each image in the batch

    For each image, returns a set of classes that were predicted but not
        present in the ground truth.
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
