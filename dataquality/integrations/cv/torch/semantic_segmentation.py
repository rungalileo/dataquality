from typing import Any

import cv2
import numpy as np
import torch

from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)


# edited mask to boundary function with np.where to avoid unexpected behaviour
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    if mask.shape[1] == 1:
        mask = mask.squeeze(1)
    mask = mask.astype(np.uint8)
    n, h, w = mask.shape
    for im in range(n):
        img_diag = np.sqrt(h**2 + w**2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
        new_mask = cv2.copyMakeBorder(
            mask[im], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # if the number does not equal either the old mask or 0 then set to 0
        mask_erode = np.where(mask_erode != mask[im], 0, mask_erode)
        boundary_mask = mask[im] - mask_erode
        # G_d intersects G in the paper.
        mask[im] = boundary_mask
    return mask


class StoreHook:
    def on_finish(*args, **kwargs):
        pass

    def hook(self, model, model_input, model_output):
        self.model = model
        self.model_input = model_input
        self.model_output = model_output["out"]
        self.on_finish(model_input, model_output)


class BatchLogger:
    def __call__(self, batch):
        self.batch = batch
        return self.batch


class Manager:
    def __init__(self, model, num_classes: int = 10):
        self.step_pred = StoreHook()
        self.step_pred.h = model.register_forward_hook(self.step_pred.hook)
        self.bl = BatchLogger()
        self.hooked = True
        self.step_pred.on_finish = self._after_pred_step
        self.split = "Train"  # hard coded for now
        self.number_classes = num_classes

    def _after_pred_step(self, *args, **kwargs):
        with torch.no_grad():
            logging_data = self.bl.batch
            preds = self.step_pred.model_output

            # checks whether the model is (n, classes, w, h), or (n, w, h, classes)
            if preds.shape[1] == self.number_classes:
                preds = preds.permute(0, 2, 3, 1)

            argmax = torch.argmax(preds.permute(0, 2, 3, 1).clone(), dim=-1).unsqueeze(
                1
            )
            self.logits = preds.cpu().numpy()  # (bs, w, h, classes)
            self.boundary_gold_masks = mask_to_boundary(
                logging_data["mask"].clone().cpu().numpy()
            )  # (bs, w, h)
            self.boundary_pred_masks = mask_to_boundary(
                argmax.clone().cpu().numpy()
            )  # (bs, w, h)
            if logging_data["mask"].shape[1] == 1:
                logging_data["mask"] = logging_data["mask"].squeeze(1)  # (bs, w, h)
            self.gold_mask = logging_data["mask"].cpu().numpy()  # (bs, w, h)
            self.img_ids = logging_data["idx"].cpu().numpy()  # np.ndarray (bs,)
            # dq log model output
            logger = SemanticSegmentationModelLogger(
                self.img_ids,
                self.gold_mask,
                self.boundary_gold_masks,
                self.boundary_pred_masks,
                logits=self.logits,
            )
            logger.log()

    def register_hooks(self, model):
        self.step_embs.h = model.register_forward_hook(self.step_embs.hook)


def watch(model: Any) -> None:
    """
    Watches a model and logs the model outputs to the Galileo server
    :param model: Model to watch
    :return:
    """
