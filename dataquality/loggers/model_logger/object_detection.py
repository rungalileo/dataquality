
from typing import List, Optional, Tuple, Union, Dict

import numpy as np

from dataquality.loggers.logger_config.object_detection import (
    BoxFormat,
    object_detection_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.utils.od import (
    convert_cxywh_xyxy,
    convert_tlxywh_xyxy,
    dep_and_boxes,
    match_bboxes,
    scale_boxes,
)

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
        image_size: Optional[Tuple[int, int]] = None,
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


        self.all_boxes: (bs, n, 2, 4)) n = boxes first four are pred, last four are gold [-1] * 4 for empty boxes
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
        self.image_ids = ids
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
        self.image_size = image_size

    def validate_and_format(self) -> None:
        for image_id in self.image_ids:
            # check for box format
            if self.logger_config.box_format == BoxFormat.tlxywh:
                self.gold_boxes[image_id] = convert_tlxywh_xyxy(self.gold_boxes[image_id])
            elif self.logger_config.box_format == BoxFormat.cxywh:
                self.gold_boxes[image_id] = convert_cxywh_xyxy(self.gold_boxes[image_id])

            # scale boxes if they are normalized (ie the bounding boxes are between 0 and 1)
            if np.all(self.gold_boxes[image_id] <= 1):
                self.gold_boxes[image_id] = scale_boxes(
                    self.gold_boxes[image_id], self.image_size
                )
            if np.all(self.pred_boxes[image_id] <= 1):
                self.pred_boxes[image_id] = scale_boxes(
                    self.pred_boxes[image_id], self.image_size
                )
                
            # matching = match_bboxes(self.pred_boxes[idx], self.gold_boxes[idx])

            # stuff below here may not be vectorizable
            '''deps, all_boxes, embs, gold_or_pred, image_dep = dep_and_boxes(
                self.gold_boxes[idx],
                self.pred_boxes[idx],
                self.labels[idx],
                self.probs[idx],
                self.pred_embs[idx],
                self.gold_embs[idx],
                matching,
            )'''

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
        ids = []
        for image_id in self.image_ids:
            prob_shape = self.probs[image_id].shape[0]
            ids.extend([image_id ]* prob_shape)
        for image_id in self.image_ids:
            gold_shape = self.gold_embs[image_id].shape[0]
            ids.extend([image_id] * gold_shape)

        return ids
        
        
    
    def _get_data_dict(self) -> Dict:
        # We pad gold probabilities with 0s to be able to fit it into a numpy matrix
        # Shape is (len(gold_embs), num_classes)
        gold_prob_shape = (len(self.gold_embs), self.probs.shape[1])
        num_pred = len(self.pred_embs)
        num_gold = len(self.gold_embs)
        image_ids = self.construct_image_ids()
        return {
            "image_id": image_ids,
            "emb": np.concatenate([self.pred_embs, self.gold_embs]),
            "prob": np.concatenate([self.probs, np.zeros(gold_prob_shape)]),
            "is_pred": np.array([True]*num_pred + [False]*num_gold),
            "is_gold": np.array([False]*num_pred + [True]*num_gold),
        }
