from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
import numpy as np
from dataquality.utils.helpers import wrap_fn
from dataquality.utils.torch import store_batch_indices
from dataquality.loggers.logger_config.semantic_segmentation import (
    semantic_segmentation_logger_config,
)
from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)
from dataquality.utils.cv.semantic_segmentation.utils import mask_to_boundary
from dataquality.utils.helpers import wrap_fn


class StoreHook:
    def __init__(self) -> None:
        self.h: Optional[RemovableHandle] = None

    def on_finish(self, *args: Any, **kwargs: Any) -> None:
        pass

    def hook(
        self, 
        model: torch.nn.Module, 
        model_input: torch.Tensor, 
        model_output: Dict[str, torch.Tensor]
    ) -> None:
        """"
        Hook to store the model input (tensor) and extract the output from a dictionary and store

        :param model: torch.nn.Module segmentation model
        :param model_input: torch.Tensor input to the model - an image (bs, 3, h, w)
        :param model_output: torch.Tensor output of the model - a mask same dim as input (bs, h, w)
        """
        self.model = model
        self.model_input = model_input
        # model_output['out'] is common for torch segmentation models as they use resizing and return a dict
        # will have to adjust for transformer models / check if output is a dict
        self.model_output = model_output["out"]
        self.on_finish(model_input, model_output)


class Manager:
    """"
    Hook manager to log the necessary data for semantic segmentation
    Contains preprocessing to convert output to a format that can be logged
    """
    def __init__(self, model: torch.nn.Module, num_classes: int = 10) -> None:
        """
        :param model: torch.nn.Module segmentation model
        :param num_classes: int number of classes in the model (possible we can extract)
        """
        self.step_pred = StoreHook()
        self.step_pred.h = model.register_forward_hook(self.step_pred.hook)
        self.bl: Dict = {}  # So this will now have batch and ids.
        self.hooked = True
        self.step_pred.on_finish = self._after_pred_step  # type: ignore
        self.split = "Train"  # hard coded for now
        self.number_classes = num_classes
        self.helper_data: Dict = {}

    def _after_pred_step(self, *args: Any, **kwargs: Any) -> None:
        """
        Method called after each prediction step by the model 
        Then processes the output along with the mask to log the appropriate
        predictions and masks that can be used to calculate data quality metrics
        """
        with torch.no_grad():
            logging_data = self.bl["batch"]
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


# store the batch
def store_batch(store: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]) -> Callable:
    """
    Stores the batch in the passed store
    :param store: Dict[str, torch.Tensor] location to store the batch
    """
    def process_batch(
        next_batch_func: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        """
        Patches the next_batch function to store the batch as well as returning
        :param next_batch_func: Callable original next_batch function of the dataloader
        """
        batch = next_batch_func(*args, **kwargs)
        if batch:
            store["batch"] = batch
        return batch

    return process_batch


# add patch to the dataloader iterator
def patch_iterator_and_batch(store: Dict[str, Any]) -> Callable:
    """
    Patches the iterator of the dataloader to return the indices and the batch
    :param store: Dict[str, Any] location to store the indices and the batch
    """

    def patch_iterator(
        orig_iterator: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        """
        Patches the iterator and wraps the next_index and next_data methods
        :param orig_iterator: Callable original iterator of the dataloader
        """
        iteraror = orig_iterator(*args, **kwargs)
        iteraror._next_index = wrap_fn(iteraror._next_index, store_batch_indices(store))
        iteraror._next_data = wrap_fn(iteraror._next_data, store_batch(store))
        return iteraror

    return patch_iterator


def watch(model: Any, dataloader: DataLoader, n_classes: int) -> None:
    """
    Watches a model and logs the model outputs to the Galileo server
    :param model: Model to watch
    """
    tl = Manager(model, num_classes=n_classes)
    semantic_segmentation_logger_config.helper_data["manager"] = tl
    dataloader._get_iterator = wrap_fn(  # type: ignore
                dataloader._get_iterator,
                patch_iterator_and_batch(
                    tl.bl
                ),
            )    
