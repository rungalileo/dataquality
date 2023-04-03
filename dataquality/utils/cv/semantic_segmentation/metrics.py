from typing import List

import evaluate
import torch


def calculate_dep_heatmap(probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Data Error Potential (DEP) for each image in the batch
    
    :param probs: np array of floats, size = (bs, height, width, n_classes)
    :param gt_masks: np array of gt masks as ints, size = (bs, height, width)
    :return: (bs, height, width)
    """
    n_classes = probs.shape[-1]
    bs = probs.shape[0]
    # flatten the height and width dimensions
    probs = probs.view(bs, -1, n_classes)  # (bs, n_pixels, n_classes)
    size = gt_masks.shape
    gt_masks = gt_masks.view(bs, -1, 1)  # (bs, n_pixels, 1)

    gt_indices = (
        gt_masks.reshape((bs, -1, 1)).expand(-1, -1, probs.shape[2]).type(torch.int64)
    )  # (bs, n_pixels, n_classes)
    value_at_ground_truth = torch.gather(probs, 2, gt_indices)[:, :, 0]  # (bs, n_pixels)

    next_highest = probs.clone()
    # Takes GT indices and puts 0 at that index so we don't use it as next highest value
    next_highest.scatter_(2, gt_indices, 0)
    next_highest = next_highest.max(dim=2).values

    margin = value_at_ground_truth - next_highest
    # Since margin is between -1 and 1, we normalize it to be between 0 and 1
    normalized_margin = ((1 + margin) / 2)
    dep_masks = 1 - normalized_margin
    dep_masks = dep_masks.view(size)

    return dep_masks


def calculate_image_dep(dep_heatmap: torch.Tensor) -> List[float]:
    """Calculates the Data Error Potential (DEP) for each image in the batch
    
    :param dep_heatmap: DEP heatmap for each image in the batch
        size = (bs, height, width)
    :return: list of DEP values for each image in the batch
        size = (bs,)
    """
    return dep_heatmap.mean(dim=(1, 2)).tolist()


def calculate_mean_iou(
    pred_masks: torch.Tensor, gold_masks: torch.Tensor, nc: int = 21
) -> List[float]:
    """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch
    
    :param pred_masks: argmax of the prediction probabilities
       shape = (bs, height, width)
    :param gold_masks: ground truth masks
       shape = (bs, height, width)
    :param nc: number of classes

    returns: list of mIoU values for each image in the batch
    """
    metric = evaluate.load("mean_iou")
    ious = []

    # for iou need shape (bs, 1, height, width) for some reason - 
    # unsure if that is actually true but it works
    for i in range(len(pred_masks)):
        iou = metric._compute(
            pred_masks[i : i + 1],  # tensor (1, height, width)
            gold_masks[i : i + 1],  # tensor (1, height, width)
            num_labels=nc,
            ignore_index=255,
        )
        ious.append(iou["mean_iou"].item())
    return ious
