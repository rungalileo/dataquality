from typing import Any, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from dataquality.utils.helpers import map_indices_to_ids, wrap_fn

from dataquality.loggers.logger_config.semantic_segmentation import (
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)
from dataquality.utils.cv.semantic_segmentation.utils import mask_to_boundary
from dataquality.utils.torch import (
    patch_dataloaders,
    patch_iterator_with_store,
)
from dataquality.integrations.torch import (
    TorchLogger
)
from dataquality.schemas.torch import HelperData


class StoreHook:
    def on_finish(*args: Any, **kwargs: Any) -> None:
        pass

    def hook(self, model, model_input, model_output) -> None:
        self.model = model
        self.model_input = model_input
        self.model_output = model_output["out"]
        self.on_finish(model_input, model_output)


class BatchLogger:
    def __call__(self, batch) -> Any:
        self.batch = batch
        return self.batch


class Manager:
    def __init__(self, model, num_classes: int = 10) -> None:
        self.step_pred = StoreHook()
        self.step_pred.h = model.register_forward_hook(self.step_pred.hook)
        self.bl = {}
        self.batch_idx = {}
        self.hooked = True
        self.step_pred.on_finish = self._after_pred_step
        self.split = "Train"  # hard coded for now
        self.number_classes = num_classes
        self.helper_data = {}

    def _after_pred_step(self, *args: Any, **kwargs: Any) -> None:
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            logging_data = self.bl['batch']
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
            # logger._get_data_dict()
            logger.log()

    def register_hooks(self, model) -> None:
        self.step_embs.h = model.register_forward_hook(self.step_embs.hook)

from typing import Any, Callable, Dict, List, Tuple

# store the batch
def store_batch(store: Dict[str, List[int]]) -> Callable:
    def process_batch(
        next_batch_func: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        """Stores the batch"""
        batch = next_batch_func(*args, **kwargs)
        if batch:
            store['batch'] = batch
        return batch

    return process_batch


# add patch to the dataloader iterator
def patch_iterator_with_batch_store(store: Dict[str, List[int]]) -> Callable:
    """Patches the iterator of the dataloader to return the indices"""

    def patch_iterator(
        orig_iterator: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        iteraror = orig_iterator(*args, **kwargs)
        iteraror._next_data = wrap_fn(iteraror._next_data, store_batch(store))
        return iteraror

    return patch_iterator

def watch(model: Any, dataloader: DataLoader, n_classes: int) -> None:
    """
    Watches a model and logs the model outputs to the Galileo server
    :param model: Model to watch
    :return:
    """
    tl = Manager(model, num_classes=n_classes)
    semantic_segmentation_logger_config.helper_data["manager"] = tl

    dataloader._get_iterator = wrap_fn(  # type: ignore
                dataloader._get_iterator,
                patch_iterator_with_batch_store(
                    tl.bl
                ),
            )
    
