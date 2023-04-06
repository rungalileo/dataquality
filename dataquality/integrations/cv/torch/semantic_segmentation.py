from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from queue import Queue
import warnings
from warnings import warn

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
import dataquality as dq
from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)
from dataquality.schemas.torch import HelperData
from dataquality.utils.helpers import wrap_fn
from dataquality.utils.semantic_segmentation.utils import mask_to_boundary
from dataquality.utils.torch import store_batch_indices
from dataquality.exceptions import GalileoException
from dataquality.utils.torch import (
    ModelHookManager, 
    unpatch, 
    remove_all_forward_hooks
)

from dataquality.integrations.torch import TorchLogger
from dataquality.schemas.task_type import TaskType

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("torch")

class StoreHook:
    def __init__(self) -> None:
        self.h: Optional[RemovableHandle] = None

    def on_finish(self, *args: Any, **kwargs: Any) -> None:
        pass

    def hook(
        self,
        model: torch.nn.Module,
        model_input: torch.Tensor,
        model_output: Dict[str, torch.Tensor],
    ) -> None:
        """ "
        Hook to store the model input (tensor) and extract the output
        from a dictionary and store

        :param model: torch.nn.Module segmentation model
        :param model_input: torch.Tensor input to the model - an image (bs, 3, h, w)
        :param model_output: torch.Tensor output of the model
            shape = (bs, h, w)
        """
        self.model = model
        self.model_input = model_input
        # model_output['out'] is common for torch segmentation models as they use
        # resizing and return a dict will have to adjust for transformer
        # models / check if output is a dict
        self.model_output = model_output["out"]
        self.on_finish(model_input, model_output)

class StoreInputHook:
    def __init__(self, store: Dict[str, Any]) -> None:
        self.h: Optional[RemovableHandle] = None
        self.store = store

    def on_finish(self, *args: Any, **kwargs: Any) -> None:
        pass

    def hook(
        self,
        model: torch.nn.Module,
        model_input: torch.Tensor,
    ) -> None:
        """ "
        Hook to store the model input (tensor) and extract the output
        from a dictionary and store

        :param model: torch.nn.Module segmentation model
        :param model_input: torch.Tensor input to the model - an image (bs, 3, h, w)
        :param model_output: torch.Tensor output of the model
            shape = (bs, h, w)
        """
        self.store["model_input"] = model_input[0]


class SemanticTorchLogger(TorchLogger):
    def __init__(self, 
                 model: torch.nn.Module, 
                 num_classes: int = 10,
                 mask_col_name: Optional[str] = None,
                helper_data: Dict[str, Any] = {},
                task: Union[TaskType, None] = TaskType.text_classification,
                 *args: Any,
                **kwargs: Any) -> None:
        
        super().__init__(model=model, *args, **kwargs)
        self._init_helper_data(self.hook_manager, model)
        self.attach_model_input_hook(model)
        self.number_classes = num_classes
        self.mask_col_name = mask_col_name

    def find_mask_category(self, batch: Dict[str, Any]) -> None:
        """
        Finds the mask category and stores it in the helper data
        :param batch: Dict[str, Any] batch from the dataloader
        """
        if not self.mask_col_name:
            for key in batch:
                if "mask" in key or 'label' in key or 'target' in key:
                    self.mask_col_name = key
            if not self.mask_col_name:
                raise ValueError("No mask column found in the batch please specify in watch method")
        print(f"Mask column name is {self.mask_col_name}")
        return
    
    def attach_model_input_hook(self, model: torch.nn.Module) -> None:
        """
        Attaches a hook to the model to store the input so we get the correct shape
        :param model: torch.nn.Module
        """
        store_hook = StoreInputHook(self.helper_data)
        h = model.register_forward_pre_hook(store_hook.hook)
        self.helper_data[HelperData.patches].append(h)

    
    def _init_helper_data(self, hm: ModelHookManager, model: Module) -> None:
        """
        Initialize the helper data with ids from the dataloader indices,
        patches for applied monkey patched functions and the hook manager.
        :param hm: Hook manager
        :param model: torch.nn.Module model that we are hooking
        """
        self.helper_data.clear()
        self.helper_data.update(
            {
                HelperData.dl_next_idx_ids: [],
                HelperData.last_action: "init",
                HelperData.patches: [],
                HelperData.model_outputs_store: {},
                HelperData.hook_manager: hm,
                HelperData.model: model,
                HelperData.batch: {},
                HelperData.model_input: {},
            }
        )

    def _on_step_end(self) -> None:
        # find the column corresponding to the mask on the first iteration else throw error in func
        if not self.mask_col_name:
            self.find_mask_category(self.helper_data['batch']['data'])
        with torch.no_grad():
            logging_data = self.helper_data['batch']['data']
            img_ids =  self.helper_data['batch']['ids'] # np.ndarray (bs,)
            
            # resize the logits to the input size based on hooks
            preds = self.helper_data['model_outputs_store']['logits']
            input_shape = self.helper_data['model_input'].shape[-2:]
            preds = F.interpolate(preds, size=input_shape, mode="bilinear", align_corners=False)

            # checks whether the model is (n, classes, w, h), or (n, w, h, classes)
            if preds.shape[1] == self.number_classes:
                preds = preds.permute(0, 2, 3, 1)

            argmax = torch.argmax(preds.clone(), dim=-1)
            logits = preds.cpu()  # (bs, w, h, classes)
            gold_boundary_masks = mask_to_boundary(
                logging_data[self.mask_col_name].clone().cpu().numpy()
            )  # (bs, w, h)
            pred_boundary_masks = mask_to_boundary(
                argmax.clone().cpu().numpy()
            )  # (bs, w, h)
            if logging_data[self.mask_col_name].shape[1] == 1:
                logging_data[self.mask_col_name] = logging_data["mask"].squeeze(1)  # (bs, w, h)
            gold_mask = logging_data[self.mask_col_name].cpu()  # (bs, w, h)

            probs = torch.nn.Softmax(dim=1)(logits).cpu()  # (bs, w, h, classes)

            # dq log model output
            logger = SemanticSegmentationModelLogger(
                image_ids=img_ids,
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



class Manager:
    """ "
    Hook manager to log the necessary data for semantic segmentation
    Contains preprocessing to convert output to a format that can be logged
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 num_classes: int = 10,
                 mask_col_name: Optional[str] = None) -> None:
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
        self.mask_col_name = mask_col_name

    def find_mask_category(self, batch: Dict[str, Any]) -> None:
        """
        Finds the mask category and stores it in the helper data
        :param batch: Dict[str, Any] batch from the dataloader
        """
        if not self.mask_col_name:
            for key in batch:
                if "mask" in key or 'label' in key or 'target' in key:
                    self.helper_data['mask_col'] = key
            if not self.mask_col:
                raise ValueError("No mask column found in the batch please specify in watch method")
        print(f"Mask column name is {self.helper_data['mask_col']}")
        return
        

    def _after_pred_step(self, *args: Any, **kwargs: Any) -> None:
        """
        Method called after each prediction step by the model
        Then processes the output along with the mask to log the appropriate
        predictions and masks that can be used to calculate data quality metrics
        """

        # find the column corresponding to the mask on the first iteration else throw error in func
        if not self.mask_col_name:
            self.find_mask_category(self.helper_data['Batch']['batch'])
        with torch.no_grad():
            logging_data = self.helper_data['Batch']['batch']
            img_ids =  self.helper_data['Batch']['ids'] # np.ndarray (bs,)
            import pdb; pdb.set_trace()
            preds = self.step_pred.model_output

            # checks whether the model is (n, classes, w, h), or (n, w, h, classes)
            if preds.shape[1] == self.number_classes:
                preds = preds.permute(0, 2, 3, 1)

            argmax = torch.argmax(preds.clone(), dim=-1)
            logits = preds.cpu()  # (bs, w, h, classes)
            gold_boundary_masks = mask_to_boundary(
                logging_data[self.helper_data['mask_col']].clone().cpu().numpy()
            )  # (bs, w, h)
            pred_boundary_masks = mask_to_boundary(
                argmax.clone().cpu().numpy()
            )  # (bs, w, h)
            if logging_data[self.helper_data['mask_col']].shape[1] == 1:
                logging_data[self.helper_data['mask_col']] = logging_data["mask"].squeeze(1)  # (bs, w, h)
            gold_mask = logging_data[self.helper_data['mask_col']].cpu()  # (bs, w, h)

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
def store_batch(
    store: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]
) -> Callable:
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
            store["data"] = batch
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

def watch(
    model: Module,
    n_classes: int,
    dataloaders: Optional[List[DataLoader]] = [],
    classifier_layer: Optional[Union[str, Module]] = None,
    mask_col_name: Optional[str] = None,
    unpatch_on_start: bool = False,
) -> None:
    """
    wraps a PyTorch model and optionally dataloaders to log the
    embeddings and logits to [Galileo](https://www.rungalileo.io/).

    .. code-block:: python

        dq.log_dataset(train_dataset, split="train")
        train_dataloader = torch.utils.data.DataLoader()
        model = TextClassificationModel(num_labels=len(train_dataset.list_of_labels))
        watch(model, [train_dataloader, test_dataloader])
        for epoch in range(NUM_EPOCHS):
            dq.set_epoch_and_split(epoch,"training")
            train()
            dq.set_split("validation")
            validate()
        dq.finish()

    :param model: Pytorch Model to be wrapped
    :param dataloaders: List of dataloaders to be wrapped
    :param classifier_layer: Layer to hook into (usually 'classifier' or 'fc').
        Inputs are the embeddings and outputs are the logits.
    """
    a.log_function("torch/watch")
    assert dq.config.task_type, GalileoException(
        "dq client must be initialized. " "For example: dq.init('text_classification')"
    )
    if unpatch_on_start:
        unwatch(model, force=True)
    if not getattr(model, "_dq", False):
        setattr(model, "_dq", True)
    else:
        raise GalileoException(
            "Model is already being watched, run unwatch(model) first"
        )

    # throwing an error as get_model_logger() needs parameters
    # but from original code so leaving it in to talk with Franz about
    # helper_data = dq.get_model_logger().logger_config.helper_data
    print("Attaching dataquality to model and dataloaders")
    tl = SemanticTorchLogger(model, 
                             num_classes=n_classes,
                             mask_col_name=mask_col_name)
    # Patch the dataloader class if no dataloaders are passed
    # or if the dataloaders have num_workers > 0
    if dataloaders is None:
        dataloaders = []
    is_single_process_dataloader = all(
        [getattr(d, "num_workers", 0) == 0 for d in dataloaders]
    )
    if len(dataloaders) > 0 and is_single_process_dataloader:
        for dataloader in dataloaders:
            assert isinstance(dataloader, DataLoader), GalileoException(
                "Invalid dataloader. Must be a pytorch dataloader"
                "from torch.utils.data import DataLoader..."
                "train_dataloader = DataLoader(dataset)"
            )
            assert (
                getattr(dataloader, "num_workers", 0) == 0
            ), "Dataloaders with num_workers > 0 are not supported"
            dataloader._get_iterator = wrap_fn(  # type: ignore
                dataloader._get_iterator,
                patch_iterator_and_batch(tl.helper_data['batch']),
            )
    else:
        # Patch the dataloader class globally
        # Can be unpatched with unwatch()
        raise NotImplementedError("Dataloaders with num_workers > 0 are not supported")
        # patch_dataloaders(tl.helper_data)


# UNWATCH ERRORS ON DQ.GET_MODEL_LOGGER() BECAUSE IT NEEDS PARAMETERS
def unwatch(model: Optional[Module] = None, force: bool = True) -> None:
    """Unwatches the model. Run after the run is finished.
    :param force: Force unwatch even if the model is not watched"""

    helper_data = dq.get_model_logger().logger_config.helper_data
    model = model or helper_data.get(HelperData.model, None)
    if not getattr(model or {}, "_dq", False):
        warn("Model is not watched, run watch(model) first")
        if not force:
            return

    # Unpatch the dataloaders
    unpatch(helper_data.get(HelperData.patches, []))
    # Detach hooks the model. in the future use the model passed
    # https://discuss.pytorch.org/t/how-to-check-where-the-hooks-are-in-the-model/120120/2
    hook_manager = helper_data.get(HelperData.hook_manager, None)
    if hook_manager:
        hook_manager.detach_hooks()
    # Remove the model from the helper data
    if isinstance(model, Module):
        remove_all_forward_hooks(model)
    else:
        warnings.warn("model is not a Module")
    if "model" in helper_data:
        del helper_data[HelperData.model]
    if model and hasattr(model, "_dq"):
        del model._dq



'''def watch(model: Any,
          dataloader: DataLoader,
          n_classes: int,
          mask_col_name: Optional[str] = None) -> None:
    """
    Watches a model and logs the model outputs to the Galileo server
    :param model: Model to watch
    :param dataloader: Dataloader to watch
    :param n_classes: Number of classes in the model
    :param mask_col_name: Name of the mask column in the batch
    """
    tl = SemanticTorchLogger(model, num_classes=n_classes, mask_col_name=mask_col_name)

    semantic_segmentation_logger_config.helper_data["manager"] = tl
    dataloader._get_iterator = wrap_fn(  # type: ignore
        dataloader._get_iterator,
        patch_iterator_and_batch(tl.helper_data['batch']),
    )'''
