from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.optimize
import torch

from dataquality.loggers.logger_config.object_detection import (
    BoxFormat,
    object_detection_logger_config
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger

""" Dereks stuff
'gt_labels', 
'bbox_pred', 
'prob', 
'bbox_gold', 
'all_bboxes',
'embeddings', 
'dep', 
'gt_or_pred', 
'image_dep', 
'img_size']



Derek's dict (key is image id):
{
    "1": {
        "gt_labels": ,
        "bbox_pred": ,
        "prob": ,
        "bbox_gold": ,
        "all_bboxe": ,
        "embeddings": ,
        "dep": ,
        "gt_or_pred": ,
        "image_dep": ,
        "img_size": ,
    },
    "2": {
        "gt_labels",
        "bbox_pred",
        "prob",
        "bbox_gold",
        "all_bboxe",
        "embeddings",
        "dep",
        "gt_or_pred",
        "image_dep",
        "img_size",
    },
}

pred_boxes
pred_logits
gold_boxes
gold_labels

"""


IOU_THRESH = 0.3


class ObjectDetectionModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "object_detection"
    logger_config = object_detection_logger_config

    def __init__(
        self,
        pred_boxes: List[np.ndarray],
        gold_boxes: List[np.ndarray],
        labels: List[np.ndarray],
        pred_embs: List[np.ndarray],
        gold_embs: List[np.ndarray],
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        img_size: Optional[Tuple[int, int]] = None,
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        """Takes in OD inputs as a list of batches

        :param pred_boxes: List of pred boxes per image
            len(pred_boxes) == bs, pred_boxes[idx].shape == (n, 4),
            where n is # predicted boxes per sample
        :param gold_boxes: List of gold boxes per image
            len(gold_boxes) == bs, gold_boxes[idx].shape == (n, 4),
            where n is # gold boxes per sample
        :param labels: List of box labels per image
            labels.shape == (bs, n, 4), where n is # gold boxes per sample
        """
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )
        self.pred_boxes = pred_boxes
        self.gold_boxes = gold_boxes
        self.labels = labels
        self.pred_embs = pred_embs
        self.gold_embs = gold_embs
        self.all_boxes = []
        self.deps = []
        self.is_gold = []
        self.is_pred = []
        self.image_dep = []

    def validate_and_format(self) -> None:

         # check for box format
        if self.logger_config.box_format == BoxFormat.tlxywh:
            self.gold_boxes = convert_tlxywh_xyxy(
                self.gold_boxes
            )  # func to convert tlxywh to xyxy
        elif self.logger_config.box_format == BoxFormat.cxywh:
            self.gold_boxes = convert_cxywh_xyxy(self.gold_boxes)

        # scale boxes if all less than 1
        if np.all(self.gold_boxes < 1):
            self.gold_boxes = scale_boxes(self.gold_boxes, self.img_size[0])
        
        for id in self.ids:
            matching = match_bboxes(self.pred_boxes[id], self.gold_boxes[id])

            # stuff below here may not be vectorizable
            deps, all_boxes, embs, gt_or_pred, image_dep = dep_and_boxes(
                self.gold_boxes[id], self.pred_boxes[id], self.labels[id], self.probs[id], self.pred_embs[id], self.gold_embs[id], matching
            )

            self.all_boxes.append(np.array(all_boxes))
            self.deps.append(np.array(deps))
            self.image_dep.append(np.array(image_dep))
            self.is_gold.append(~np.array(gt_or_pred))
            self.is_pred.append(np.array(gt_or_pred))
            self.embs.append(np.array(embs))


    def log():
        pass

def scale_boxes(bboxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    # scale boxes to image size
    bboxes[:, 0] *= img_size[0]
    bboxes[:, 1] *= img_size[1]
    bboxes[:, 2] *= img_size[0]
    bboxes[:, 3] *= img_size[1]
    return bboxes

def convert_cxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    # converts center point xywh boxes to xyxy can be in either integer coords or 0-1
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1, x2 = x - w / 2, x + w / 2
    y1, y2 = y - h / 2, y + h / 2
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x1, y1, x2, y2
    return bboxes

def convert_tlxywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    # convert top left xywh boxes to xyxy can be in either integer coords or 0-1
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x2, y2 = x + w, y + h
    bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = x, y, x2, y2
    return bboxes

def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.
    # FROM https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    # Seems like boxes are given (top_left_x, top_left_y, bot_right_x, bot_right_y)

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
    bbox_gt: np.ndarray, bbox_pred: np.ndarray, iou_thresh: float = IOU_THRESH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    """
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i, :], bbox_pred[j, :])

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
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = ious_actual > iou_thresh

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid]


def dep_match(iou: float, probs: np.ndarray, gt_label: int) -> float:
    """Computes DEP for a predicted box that has a matching ground truth box

    DEP is computed as (1 - IOU) * margin
    """
    # pred score should be logits and we should do margin
    gt_logit = probs[gt_label.long()]
    # get the highest logit that is not equal to the gt logit
    if len(probs) > 1:
        top_idx_prob, second_idx = np.argsort(probs)[:2]
        if probs[top_idx_prob] == gt_logit:
            pred_logit = probs[second_idx]
        else:
            pred_logit = probs[top_idx_prob]
        margin = gt_logit - pred_logit
        dep_cls = (1 - margin) / 2
    else:
        dep_cls = 1 - gt_logit
    return float((1 - iou) * dep_cls)


def dep_no_match_gt() -> float:
    """If the ground truth box has no matching prediction box, its DEP is 1"""
    return 1.0


def dep_no_match_pred(probs: np.ndarray) -> float:
    """ "If a predited box has no ground truth, DEP is the confidence"""
    # need just the highest confidence score
    return float(probs.max())


def dep_and_boxes(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    pred_embs: np.ndarray,
    gt_embs: np.ndarray,
    matching: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculates box-level and image-lev

    :param pred_boxes: Pred boxes for an image
        pred_boxes[idx].shape == (n, 4), n is # predicted boxes in image
    :param gold_boxes: List of gold boxes per image
        gold_boxes[idx].shape == (m, 4), m is # gold boxes in image
    :param labels: GT label for gold boxes
        labels.shape == (m,)
    :param prob: Model prob predictions for an image
        prob.shape == (n, 4), where n is # gold boxes per sample

    Return:
    (deps, pred_deps, gold_deps, dep)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections


    Takes in gt boxes, pred_boxes, gt_labels, probs and returns a per box dep and image dep
    """
    with torch.no_grad():
        pred_deps, gold_deps, deps = np.ones(len(pred_boxes)), np.ones(len(gt_boxes)), []
        all_boxes, gt_or_pred = [], []
        embeddings = []

        pred_idxs, gt_idxs, ious = matching[0], matching[1], matching[2]

        # empty box to be used for no match
        EMPTY_BOX = np.array([-1, -1, -1, -1])

        # dep for matches
        for gt_idx, pred_idx, iou in zip(gt_idxs, pred_idxs, ious):
            dep_match_score = dep_match(iou, probs[pred_idx], labels[gt_idx])
            deps.append(dep_match_score)
            pred_deps[pred_idx] = dep_match_score
            gold_deps[gt_idx] = dep_match_score

            #add embeddings and boxes
            embeddings.append(np.array(pred_embs[pred_idx]))
            arr = np.array([pred_boxes[pred_idx], gt_boxes[gt_idx]])
            all_boxes.append(arr)
            gt_or_pred.append(True)

        # dep for extra GT
        n_gts = gt_boxes.shape[0]
        for gt_extra_index in set(range(n_gts)).difference(gt_idxs):
            dep_no_match_gt_score = dep_no_match_gt()
            deps.append(dep_no_match_gt_score)
            gold_deps[gt_extra_index] = dep_no_match_gt_score

            #add embeddings and boxes
            embeddings.append(np.array(gt_embs[gt_extra_index]))
            # create a numpy array of the empty box and the gt box
            arr = np.array([EMPTY_BOX, gt_boxes[gt_extra_index]])
            all_boxes.append(arr)
            gt_or_pred.append(False)

        # dep for extra Pred
        n_preds = pred_boxes.shape[0]
        for pred_extra_index in set(range(n_preds)).difference(pred_idxs):
            # take the max of the pred scores - works for both seq and scalars
            dep_no_match_pred_score = dep_no_match_pred(max(probs[pred_extra_index]))
            deps.append(dep_no_match_pred_score)
            pred_deps[pred_extra_index] = dep_no_match_pred_score

            #add embeddings and boxes
            embeddings.append(np.array(pred_embs[pred_extra_index]))
            arr = np.array([pred_boxes[pred_extra_index], EMPTY_BOX])
            all_boxes.append(arr)
            gt_or_pred.append(True)

        image_dep = float(sum(deps) / len(deps))

        return deps, all_boxes, embeddings, gt_or_pred, image_dep

