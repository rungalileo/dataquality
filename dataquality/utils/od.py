"""Utils for Object Detection"""
from typing import Tuple

import numpy as np
import scipy
import torch


IOU_THRESH = 0.3

def scale_boxes(bboxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Normalizes boxes to image size"""
    bboxes[:, 0] *= img_size[0]
    bboxes[:, 1] *= img_size[1]
    bboxes[:, 2] *= img_size[0]
    bboxes[:, 3] *= img_size[1]
    return bboxes


def convert_cxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts center point xywh boxes to xyxy can be in either integer coords or 0-1"""
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1, x2 = x - w / 2, x + w / 2
    y1, y2 = y - h / 2, y + h / 2
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x1, y1, x2, y2
    return bboxes


def convert_tlxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts top left xywh boxes to xyxy can be in either integer coords or 0-1"""
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x2, y2 = x + w, y + h
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x, y, x2, y2
    return bboxes


def bbox_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """ "
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    ^^ corrected.
    FROM https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    Seems like boxes are given (top_left_x, top_left_y, bot_right_x, bot_right_y)
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <= 0 or interH <= 0:
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_bboxes(
    bbox_gold: np.ndarray, bbox_pred: np.ndarray, iou_thresh: float = IOU_THRESH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gold, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gold and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    """
    n_true = bbox_gold.shape[0]
    n_pred = bbox_pred.shape[0]
    MIN_IOU = 0.0

    # NUM_gold x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gold[i, :], bbox_pred[j, :])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate(
            (iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0
        )

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate(
            (iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1
        )

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        np.array([])
    else:
        iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gold_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gold_actual, idx_pred_actual]
    sel_valid = ious_actual > iou_thresh

    return (
        idx_gold_actual[sel_valid],
        idx_pred_actual[sel_valid],
        ious_actual[sel_valid],
    )


def dep_match(iou: float, probs: np.ndarray, gold_label: int) -> float:
    """Computes DEP for a predicted box that has a matching ground truth box

    DEP is computed as (1 - IOU) * margin
    """
    # pred score should be logits and we should do margin
    gold_logit = probs[gold_label.long()]
    # get the highest logit that is not equal to the gold logit
    if len(probs) > 1:
        top_idx_prob, second_idx = np.argsort(probs)[:2]
        if probs[top_idx_prob] == gold_logit:
            pred_logit = probs[second_idx]
        else:
            pred_logit = probs[top_idx_prob]
        margin = gold_logit - pred_logit
        dep_cls = (1 - margin) / 2
    else:
        dep_cls = 1 - gold_logit
    return float((1 - iou) * dep_cls)


def dep_no_match_gold() -> float:
    """If the ground truth box has no matching prediction box, its DEP is 1"""
    return 1.0


def dep_no_match_pred(probs: np.ndarray) -> float:
    """ "If a predited box has no ground truth, DEP is the confidence"""
    # need just the highest confidence score
    return float(probs.max())


def dep_and_boxes(
    gold_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    pred_embs: np.ndarray,
    gold_embs: np.ndarray,
    matching: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculates box-level and image-lev

    :param gold_boxes: List of gold boxes per image
        gold_boxes[idx].shape == (m, 4), m is # gold boxes in image
    :param pred_boxes: Pred boxes for an image
        pred_boxes[idx].shape == (n, 4), n is # predicted boxes in image
    :param labels: gold label for gold boxes
        labels.shape == (m,)
    :param prob: Model prob predictions for an image
        prob.shape == (n, 4), where n is # gold boxes per sample
    :param pred_embs: The embeddings associated to each pred box

    Return:
    (deps, all_boxes, embeddings, gold_or_dep, image_dep)
        deps: (n, 1) where n is the number of boxes
        all_boxes: (n, 8) where first four coords are pred and last 4 are matched gt
        gold_or_dep: (n, 1) flag indicating whether we have a prediction or only gold
        image_dep: float of the image level dep


    Takes in gold boxes, pred_boxes, gold_labels, probs and returns a dep per box, matching,
    embeddings per matched pair, flag indicating pred or gold, and image dep
    """
    with torch.no_grad():
        pred_deps, gold_deps, deps = (
            np.ones(len(pred_boxes)),
            np.ones(len(gold_boxes)),
            [],
        )
        all_boxes, gold_or_pred = [], []
        embeddings = []

        pred_idxs, gold_idxs, ious = matching[0], matching[1], matching[2]

        # empty box to be used for no match
        EMPTY_BOX = np.array([-1, -1, -1, -1])

        # dep for matches
        for gold_idx, pred_idx, iou in zip(gold_idxs, pred_idxs, ious):
            dep_match_score = dep_match(iou, probs[pred_idx], labels[gold_idx])
            deps.append(dep_match_score)
            pred_deps[pred_idx] = dep_match_score
            gold_deps[gold_idx] = dep_match_score

            # add embeddings and boxes
            embeddings.append(np.array(pred_embs[pred_idx]))
            arr = np.array([pred_boxes[pred_idx], gold_boxes[gold_idx]])
            all_boxes.append(arr)
            gold_or_pred.append(True)

        # dep for extra gold
        n_golds = gold_boxes.shape[0]
        for gold_extra_index in set(range(n_golds)).difference(gold_idxs):
            dep_no_match_gold_score = dep_no_match_gold()
            deps.append(dep_no_match_gold_score)
            gold_deps[gold_extra_index] = dep_no_match_gold_score

            # add embeddings and boxes
            embeddings.append(np.array(gold_embs[gold_extra_index]))
            # create a numpy array of the empty box and the gold box
            arr = np.array([EMPTY_BOX, gold_boxes[gold_extra_index]])
            all_boxes.append(arr)
            gold_or_pred.append(False)

        # dep for extra Pred
        n_preds = pred_boxes.shape[0]
        for pred_extra_index in set(range(n_preds)).difference(pred_idxs):
            # take the max of the pred scores - works for both seq and scalars
            dep_no_match_pred_score = dep_no_match_pred(max(probs[pred_extra_index]))
            deps.append(dep_no_match_pred_score)
            pred_deps[pred_extra_index] = dep_no_match_pred_score

            # add embeddings and boxes
            embeddings.append(np.array(pred_embs[pred_extra_index]))
            arr = np.array([pred_boxes[pred_extra_index], EMPTY_BOX])
            all_boxes.append(arr)
            gold_or_pred.append(True)

        image_dep = float(sum(deps) / len(deps))

        return deps, all_boxes, embeddings, gold_or_pred, image_dep
