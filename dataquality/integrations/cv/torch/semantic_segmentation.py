from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from queue import Queue
import warnings
from warnings import warn
import os

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
from dataquality.loggers.data_logger.semantic_segmentation import SemSegCols
from dataquality.utils.torch import store_batch_indices
from dataquality.exceptions import GalileoException
from dataquality.utils.torch import (
    ModelHookManager, 
    patch_dataloaders,
    unpatch, 
    remove_all_forward_hooks
)

from dataquality.integrations.torch import TorchLogger
from dataquality.schemas.task_type import TaskType
from dataquality.integrations.torch import unwatch

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("torch")


class SemanticTorchLogger(TorchLogger):
    def __init__(self, 
                 model: torch.nn.Module, 
                 bucket_name: str,
                 dataset_path: str,
                 mask_col_name: Optional[str] = None,
                 helper_data: Dict[str, Any] = {},
                 dataloaders: List[torch.utils.data.DataLoader] = [],
                 task: Union[TaskType, None] = TaskType.text_classification,
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Class to log semantic segmentation models to Galileo

        :param model: model to log
        :param bucket_name: name of the bucket that currently stores images in cloud
        :param dataset_path: path to the parent dataset folder
        :param mask_col_name: name of the column that contains the mask
        :param helper_data: helper data to be logged
        :param dataloaders: dataloaders to be logged
        :param task: task type
        """
        super().__init__(model=model, helper_data=helper_data, *args, **kwargs)
        
        self._init_helper_data(self.hook_manager, model)
        self.mask_col_name = mask_col_name
        self.dataset_path = os.path.abspath(dataset_path)

        # There is a hook on dataloader so must convert before attaching hook
        self.file_map = {}
        self.mask_file_map = {}
        self.bucket_name = bucket_name
        self.dataloaders = dataloaders
        self.datasets = [self.convert_dataset(dataloader.dataset) for dataloader in dataloaders]

        # capture the model input
        self.hook_manager.attach_hook(model, self._dq_input_hook)

        # try to infer just from the model architecture if not do it on first step
        try:
            self.number_classes = model.classifier[-1].out_channels
        except AttributeError:
            self.number_classes = None
            
        self.image_col = 'image'
        self.label_col = 'label'
        
        


    def convert_dataset(self, dataset: Any) -> List:
        """Convert the dataset to the format expected by the dataquality client"""
        
        # we wouldn't need any of this if we could map ids to the cloud images
        assert len(dataset) > 0
        processed_dataset = []
        for i, data in enumerate(dataset):
            if 'image_path' not in data:
                raise GalileoException("Missing image_path in data .\
                                        Please have both specified in your dataset.\
                                        Ie. for batch in dataloader: batch['image_path'] = 'path/to/image'")

            self.file_map[data['image_path']] = i

            # cut the dataset path from the image path so we can use relative path
            # within the bucket to each image
            image_path = os.path.abspath(data['image_path'])
            image_path = image_path.replace(self.dataset_path, '')

            processed_dataset.append({
                SemSegCols.image_path: image_path,
                SemSegCols.id: i
            })
        # I have assumed we can collect the masks from the hooks in the dataloader
        return processed_dataset

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
    
    def _dq_classifier_hook_with_step_end(
        self,
        model: Module,
        model_input: torch.Tensor,
        model_output: Dict[str, torch.Tensor],
    ) -> None:
        """
        Hook to extract the logits, embeddings from the model.
            Overrides the superclass method and moves self._on_step_end()
            to the input hook as that hook is called second.
        :param model: Model pytorch model
        :param model_input: Model input
        :param model_output: Model output
        """
        self._classifier_hook(model, model_input, model_output)
    
    def _dq_input_hook(self,
                       model: torch.nn.Module,
                       model_input: torch.Tensor,
                       model_output: Dict[str, torch.Tensor]) -> None:
        """
        Hook to store the model input (tensor) and extract the output
        from a dictionary and store
        
        :param model: torch.nn.Module segmentation model
        :param model_input: torch.Tensor input to the model - an image (bs, 3, h, w)
        :param model_output: torch.Tensor output of the model

        """
        self.helper_data[HelperData.model_input] = model_input[0].detach().cpu().numpy()
        self._on_step_end()
    
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
        """Funciton to be called at the end of each step to log the inputs and outputs"""
        if not self.mask_col_name:
            self.find_mask_category(self.helper_data['batch']['data'])

        # if we have not inferred the number of classes from the model architecture
        if not self.number_classes:
            self.number_classes = self.helper_data[HelperData.model_outputs_store]['logits'].shape[1]

        with torch.no_grad():
            logging_data = self.helper_data['batch']['data']
            img_ids =  self.helper_data['batch']['ids'] # np.ndarray (bs,)
            image_paths = logging_data['image_path']
            # convert the img_ids to absolute ids from file map
            img_ids = [self.file_map[path] for path in image_paths]
            
            # resize the logits to the input size based on hooks
            preds = self.helper_data[HelperData.model_outputs_store]['logits']
            input_shape = self.helper_data[HelperData.model_input].shape[-2:]
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
            
    def finish(self) -> None:
        # finish function that runs our inference at the end of training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.splits = ['training', 'validation', 'test']
        for i, dataloader in enumerate(self.dataloaders):
            print('Running dataquality on dataloader: ', self.splits[i])
            dq.set_epoch_and_split(0, self.splits[i])
            self.run_one_epoch(dataloader, device)
        return
                
    def run_one_epoch(self, dataloader: DataLoader, device: str):
        
        if device == "cuda":
            torch.cuda.empty_cache()
        print(device)
        with torch.autocast('cuda'):
            for i, batch in enumerate(dataloader):
                img = batch[self.image_col]
                img  = img.to(device)
                self.model(img)
                if i == 10:
                    break
        return
                    
                    


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
    bucket_name: str,
    dataset_path: str,
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
    :param bucket_name: Name of the bucket from which the images come
    :param dataset_path: Path to the dataset
    :param dataloaders: List of dataloaders to be wrapped
    :param classifier_layer: Layer to hook into (usually 'classifier' or 'fc').
        Inputs are the embeddings and outputs are the logits.
    """
    print("We assume the dataloaders passed only have transforms that Tensor, Resize, and Normalize the image and mask\n"
      "‼ Any other transforms passed will lead to unexpected results\n"
      "See docs at https://dq.readthedocs.io/en/latest/ (placeholder) for more info \n \n")

    
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
    helper_data = dq.get_model_logger().logger_config.helper_data
    print("Attaching dataquality to model and dataloaders")
    tl = SemanticTorchLogger(model, 
                             bucket_name=bucket_name,
                             dataset_path=dataset_path,
                             mask_col_name=mask_col_name,
                             helper_data=helper_data,
                             dataloaders=dataloaders,)
    
    # we can override the num workers as we are the ones using the dataloader
    # would be better to use as many as possible but this is a quick fix
    # have to check with Franz about this
    for dataloader in dataloaders:
        if getattr(dataloader, "num_workers", 0) > 0:
            dataloader.num_workers = 0
            
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
        # how can we add our dataloader watch to this portion
        patch_dataloaders(tl.helper_data['batch'])
        # patch_dataloaders(tl.helper_data)
        
    return tl

