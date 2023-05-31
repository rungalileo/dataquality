from typing import List, Tuple

import numpy as np
from dataquality.schemas.semantic_segmentation import (
    IoUType,
    IouData,
)




def calculate_batch_iou(
    pred_masks: np.ndarray, gold_masks: np.ndarray, iou_type: str, nc: int
) -> List[IouData]:
    """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch

    If boundary masks are passed into this function, we return the
    boundary IoU (bIoU).

    :param pred_masks: argmax of the prediction probabilities
       shape = (bs, height, width)
    :param gold_masks: ground truth masks
       shape = (bs, height, width)
    :param iou_type: mean or boundary
    :param nc: number of classes
    :return: list of IoU values for each image in the batch
       shape = (bs,)
    """
    iou_data = []

    # for iou need shape (bs, 1, height, width) to get per mask iou
    for i in range(len(pred_masks)):
        iou, area_per_class = compute_iou(
            pred_masks[i : i + 1],  # tensor (1, height, width)
            gold_masks[i : i + 1],  # tensor (1, height, width)
            num_labels=nc,
            ignore_index=255,
        )

        iou_data.append(
            IouData(
                iou=np.nanmean(iou),
                iou_per_class=iou.tolist(),
                area_per_class=area_per_class.tolist(),
                iou_type=iou_type,
            )
        )
    return iou_data


def compute_iou(
    pred_mask: np.ndarray,
    gold_mask: np.ndarray,
    num_labels: int,
    ignore_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    intersection_bool = pred_mask == gold_mask
    
    intersection_pixels = np.histogram(pred_mask[intersection_bool], bins=num_labels, range=(0, num_labels))[0]
    pred_pixels = np.histogram(pred_mask, bins=num_labels, range=(0, num_labels))[0]
    gold_pixels = np.histogram(gold_mask, bins=num_labels, range=(0, num_labels))[0]
    
    union_pixels_per_class = pred_pixels + gold_pixels - intersection_pixels
    iou_per_class = intersection_pixels / union_pixels_per_class
    
    # fill the nans with 0s
    union_pixels_per_class = np.nan_to_num(union_pixels_per_class)
    
    return iou_per_class, union_pixels_per_class
    
    