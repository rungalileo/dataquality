from typing import Any

import torch

from dataquality.loggers.logger_config.semantic_segmentation import semantic_segmentation_logger_config
from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)
from dataquality.utils.cv.semantic_segmentation.utils import mask_to_boundary


class StoreHook:
    def on_finish(*args: Any, **kwargs: Any) -> None:
        import pdb; pdb.set_trace()
        pass

    def hook(self, model, model_input, model_output) -> None:
        import pdb; pdb.set_trace()
        self.model = model
        self.model_input = model_input
        self.model_output = model_output["out"]
        self.on_finish(model_input, model_output)


class BatchLogger:
    def __call__(self, batch) -> Any:
        import pdb; pdb.set_trace()
        self.batch = batch
        return self.batch


class Manager:
    def __init__(self, model, num_classes: int = 10) -> None:
        import pdb; pdb.set_trace()
        self.step_pred = StoreHook()
        self.step_pred.h = model.register_forward_hook(self.step_pred.hook)
        self.bl = BatchLogger()
        self.hooked = True
        self.step_pred.on_finish = self._after_pred_step
        self.split = "Train"  # hard coded for now
        self.number_classes = num_classes

    def _after_pred_step(self, *args: Any, **kwargs: Any) -> None:
        import pdb; pdb.set_trace()
        with torch.no_grad():
            logging_data = self.bl.batch
            preds = self.step_pred.model_output

            # checks whether the model is (n, classes, w, h), or (n, w, h, classes)
            if preds.shape[1] == self.number_classes:
                preds = preds.permute(0, 2, 3, 1)

            argmax = torch.argmax(preds.clone(), dim=-1)
            logits = preds.cpu()  # (bs, w, h, classes)
            gold_boundary_masks = mask_to_boundary(
                logging_data["mask"].clone().cpu().numpy()
            )  # (bs, w, h)
            pred_boundary_masks = mask_to_boundary(
                argmax.clone().cpu().numpy()
            )  # (bs, w, h)
            if logging_data["mask"].shape[1] == 1:
                logging_data["mask"] = logging_data["mask"].squeeze(1)  # (bs, w, h)
            gold_mask = logging_data["mask"].cpu()  # (bs, w, h)
            img_ids = logging_data["idx"].cpu()  # np.ndarray (bs,)

            probs = torch.nn.Softmax(dim=1)(logits).cpu()  # (bs, w, h, classes)

            # dq log model output
            logger = SemanticSegmentationModelLogger(
                image_ids=img_ids.tolist(),
                gt_masks=gold_mask,  # Torch tensor (bs, w, h)
                pred_mask=argmax,  # Torch tensor (bs, w, h)
                gold_boundary_masks=torch.tensor(
                    gold_boundary_masks
                ),  # Torch tensor (bs, w, h)
                pred_boundary_masks=torch.tensor(
                    pred_boundary_masks
                ),  # Torch tensor (bs, w, h)
                output_probs=probs,  # Torch tensor (bs, w, h, classes)
            )
            logger._get_data_dict()
            # logger.log()

    def register_hooks(self, model) -> None:
        self.step_embs.h = model.register_forward_hook(self.step_embs.hook)


def watch(model: Any, n_classes: int) -> None:
    """
    Watches a model and logs the model outputs to the Galileo server
    :param model: Model to watch
    :return:
    """
    m = Manager(model, num_classes=n_classes)
    semantic_segmentation_logger_config.helper_data["manager"] = m
    model.