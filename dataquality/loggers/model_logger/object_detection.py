from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from dataquality.loggers.logger_config.object_detection import (
    BoxFormat,
    object_detection_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.utils.od import convert_cxywh_xyxy, convert_tlxywh_xyxy, scale_boxes

""" Dereks stuff
'gold_labels',
'bbox_pred',
'prob',
'bbox_gold',
'all_bboxes',
'embeddings',
'dep',
'gold_or_pred',
'image_dep',
'img_size']



Derek's dict (key is image id):
{
    "1": {
        "gold_labels": ,
        "bbox_pred": ,
        "prob": ,
        "bbox_gold": ,
        "all_bboxe": ,
        "embeddings": ,
        "dep": ,
        "gold_or_pred": ,
        "image_dep": ,
        "img_size": ,
    },
    "2": {
        "gold_labels",
        "bbox_pred",
        "prob",
        "bbox_gold",
        "all_bboxe",
        "embeddings",
        "dep",
        "gold_or_pred",
        "image_dep",
        "img_size",
    },
}

pred_boxes
pred_logits
gold_boxes
gold_labels

"""


class ObjectDetectionModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "object_detection"
    logger_config = object_detection_logger_config

    def __init__(
        self,
        ids: Union[List, np.ndarray],
        pred_boxes: List[np.ndarray],
        gold_boxes: List[np.ndarray],
        labels: List[np.ndarray],
        pred_embs: List[np.ndarray],
        gold_embs: List[np.ndarray],
        image_size: Optional[Tuple[int, int]],
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
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


        self.all_boxes: (bs, n, 2, 4)) n = boxes first four are pred,
            last four are gold [-1] * 4 for empty boxes
        self.deps: (bs, n) n = boxes, all boxes have a dep
        self.image_dep: (bs, 1) image dep aggregated
        self.is_gold: (bs, n) n = boxes True if gold, False if pred
        self.is_pred: (bs, n) n = boxes True if pred, False if gold
        self.embs: (bs, n, dim) n = boxes embedding for each box
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
        assert ids is not None
        self.image_ids = ids
        self.pred_boxes = pred_boxes
        self.gold_boxes = gold_boxes
        self.labels = labels
        self.pred_embs = pred_embs
        self.gold_embs = gold_embs
        # self.all_boxes = []
        # self.deps = []
        # self.is_gold = []
        # self.is_pred = []
        # self.image_dep = []
        self.image_size = image_size

    def validate_and_format(self) -> None:
        assert (
            len(self.pred_embs) == len(self.gold_embs) == len(self.image_ids)
        ), """There must be 1 entry in pred_embs and gold_embs for every image"""
        assert (
            len(self.pred_boxes) == len(self.gold_boxes) == len(self.image_ids)
        ), """There must be 1 entry in pred_boxes and gold_boxes for every image"""
        for idx, image_id in enumerate(self.image_ids):
            # check for box format
            if self.logger_config.box_format == BoxFormat.tlxywh:
                self.gold_boxes[idx] = convert_tlxywh_xyxy(self.gold_boxes[idx])
            elif self.logger_config.box_format == BoxFormat.cxywh:
                self.gold_boxes[idx] = convert_cxywh_xyxy(self.gold_boxes[idx])

            # scale boxes if they are normalized
            # (ie the bounding boxes are between 0 and 1)
            # TODO: Scaling boxes is broken, doesn't consider padding (@Franz)
            if np.all(self.gold_boxes[idx] <= 1) and self.image_size:
                self.gold_boxes[idx] = scale_boxes(
                    self.gold_boxes[idx], self.image_size
                )
            if np.all(self.pred_boxes[idx] <= 1) and self.image_size:
                self.pred_boxes[idx] = scale_boxes(
                    self.pred_boxes[idx], self.image_size
                )

            # matching = match_bboxes(self.pred_boxes[idx], self.gold_boxes[idx])

            # stuff below here may not be vectorizable
            """deps, all_boxes, embs, gold_or_pred, image_dep = dep_and_boxes(
                self.gold_boxes[idx],
                self.pred_boxes[idx],
                self.labels[idx],
                self.probs[idx],
                self.pred_embs[idx],
                self.gold_embs[idx],
                matching,
            )"""

            # pred boxes + gt boxes
            # for each box - label (none if pred) probs (np.zero if gold)
            # embeddings
            # flag on whether predicted or gold

            # self.all_boxes.append(np.array(all_boxes))
            # self.deps.append(np.array(deps))
            # self.image_dep.append(np.array(image_dep))
            # self.is_gold.append(~np.array(gold_or_pred))
            # self.is_pred.append(np.array(gold_or_pred))
            # self.embs.append(np.array(embs))

    def construct_image_ids(self) -> List[int]:
        """Creates a list of image ids equal to the number of boxes

        The ids passed in for batch represent the ids of the images they map to
        Since we store the box data as 1 row per box, we need to duplicate the
        image id for each box of the same corresponding image.

        When constructing the data for the batch, we store all preds first, then
        all golds. So we do the same here to map the IDs properly
        """
        pred_box_ids = []
        gold_box_ids = []
        # If the particular image has no boxes, it's shape[0] will be
        # 0, so no ids will be added, which is what we want
        for idx, image_id in enumerate(self.image_ids):
            num_preds_for_image = self.pred_embs[idx].shape[0]
            pred_box_ids.extend([image_id] * num_preds_for_image)

            num_gold_for_image = self.gold_embs[idx].shape[0]
            gold_box_ids.extend([image_id] * num_gold_for_image)
        return pred_box_ids + gold_box_ids

    def _get_data_dict(self) -> Dict:
        """Filters out the pred/gold arrays that are actually empty

        For each image, we pass in a List[np.ndarray] to represent the gold/pred
        boxes for that image. In the event that an image has no gold or no pred boxes,
        they are passed in as empty numpy arrays like np.array([]). We need to filter
        those out properly before adding them to the data dict, otherwise we won't have
        a well formed numpy array. We do this here by checking the shape != (0,) which
        is the shape of an empty numpy array. We similarly construct the image ids
        in `construct_image_ids` to have the same length,
        """
        image_ids = np.array(self.construct_image_ids(), dtype=np.int32)
        pred_emb_arrays = np.concatenate(
            [arr for arr in self.pred_embs if arr.shape[0] != 0]
        )
        gold_emb_arrays = np.concatenate(
            [arr for arr in self.gold_embs if arr.shape[0] != 0]
        )
        pred_prob_arrays = np.concatenate(
            [arr for arr in self.probs if arr.shape[0] != 0]
        )
        gold_box_arrays = np.concatenate(
            [arr for arr in self.gold_boxes if arr.shape[0] != 0]
        )
        pred_box_arrays = np.concatenate(
            [arr for arr in self.pred_boxes if arr.shape[0] != 0]
        )
        # We pad gold probabilities with 0s to be able to fit it into a numpy matrix
        # Shape is (len(gold_embs), num_classes)
        num_pred = pred_emb_arrays.shape[0]
        num_gold = gold_emb_arrays.shape[0]
        gold_prob_shape = (num_gold, len(self.logger_config.labels))
        # -1 for preds because a pred box won't have a label
        golds = np.concatenate([[-1] * num_pred, np.concatenate(self.labels)]).astype(
            np.int32
        )
        obj = {
            "image_id": image_ids,
            "emb": np.concatenate([pred_emb_arrays, gold_emb_arrays]),
            "prob": np.concatenate([pred_prob_arrays, np.zeros(gold_prob_shape)]),
            "bbox": np.concatenate([pred_box_arrays, gold_box_arrays]),
            "is_pred": np.array([True] * num_pred + [False] * num_gold),
            "is_gold": np.array([False] * num_pred + [True] * num_gold),
            "split": [self.split] * len(image_ids),
            "epoch": [0] * len(image_ids),
            "gold": golds,
        }
        return obj
