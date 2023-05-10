import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException
from dataquality.integrations.torch import TorchLogger, unwatch
from dataquality.loggers.model_logger.semantic_segmentation import (
    SemanticSegmentationModelLogger,
)
from dataquality.schemas.semantic_segmentation import SemSegCols
from dataquality.schemas.split import Split
from dataquality.schemas.torch import HelperData
from dataquality.utils.helpers import wrap_fn
from dataquality.utils.semantic_segmentation.lm import (
    calculate_lm_for_batch,
    calculate_self_confidence_threshold,
    semseg_calculate_self_confidence,
    semseg_fill_confident_counts,
)
from dataquality.utils.semantic_segmentation.overlapping_classes import (
    add_batch_to_graph,
    compute_community_probability_mass,
    compute_louvain_communities,
    convert_to_undirected,
    normalize_graph,
    save_community_scores,
    upload_graph_obj,
)
from dataquality.utils.semantic_segmentation.utils import mask_to_boundary
from dataquality.utils.torch import ModelHookManager, store_batch_indices

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("torch")
object_store = ObjectStore()


# Heuristic used to calculate Likely Mislabeled for Semantic Segmentation
# A larger queue size corresponds to a more accurate estimate of LM.
# We keep a queue size to overcome memory issues with large SemSeg datasets.
LIKELY_MISLABELED_QUEUE_SIZE = 100
# Overlapping classes heurisitic values to use
MIN_K = 4
RESOLUTION = 1
OVERLAPPING_RANDOM_STATE = 42


class SemanticTorchLogger(TorchLogger):
    def __init__(
        self,
        bucket_name: str,
        dataset_path: str,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        mask_col_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Class to log semantic segmentation models to Galileo

        :param bucket_name: name of the bucket that currently stores images in cloud
        :param dataset_path: path to the parent dataset folder
        :param mask_col_name: name of the column that contains the mask
        :param dataloaders: dataloaders to be logged
        """
        super().__init__(*args, **kwargs)

        self._init_helper_data(self.hook_manager, self.model)
        self.mask_col_name = mask_col_name
        self.dataset_path = os.path.abspath(dataset_path)

        # There is a hook on dataloader so must convert before attaching hook
        self.dataloader_path_to_id: Dict[str, Any] = {
            split: {} for split in dataloaders.keys()
        }
        self.id_to_relative_path: Dict[str, Any] = {
            split: {} for split in dataloaders.keys()
        }
        self.bucket_name = bucket_name
        self.dataloaders = dataloaders
        self.image_col = "image"
        self.converted_datasets = []
        for split, dataloader in self.dataloaders.items():
            convert_dataset = self.convert_dataset(dataloader.dataset, split)
            self.converted_datasets.append(convert_dataset)
        # capture the model input
        self.hook_manager.attach_hook(self.model, self._dq_input_hook)

        self.called_finish = False
        self.queue_size = LIKELY_MISLABELED_QUEUE_SIZE

        self.overlapping_classes_graph = nx.DiGraph()
        self.labels_counter: Counter = Counter()

    def convert_dataset(self, dataset: Any, split: str) -> List:
        """Convert the dataset to the format expected by the dataquality client

        Args:
            dataset (Any): dataset to convert
            start_int (int): starting unique id for each example in the dataset
                as we need a unique identifier for each example. Defaults to 0.
        """
        # we wouldn't need any of this if we could map ids to the cloud images
        assert len(dataset) > 0
        assert (
            self.image_col in dataset[0].keys()
        ), 'The dataset must have a column named "image" that contains the image data'
        processed_dataset = []
        for i, data in enumerate(dataset):
            if "image_path" not in data:
                raise GalileoException(
                    "Missing image_path in data .\
                    Please have both specified in your dataset.\
                    Ie. for batch in dataloader: batch['image_path'] = 'path/to/image'\
                        This should get us to the image in the cloud by concatenating\
                            bucker_name + image_path"
                )

            self.dataloader_path_to_id[split][data["image_path"]] = i

            # cut the dataset path from the image path so we can use relative path
            # within the bucket to each image
            image_path = os.path.abspath(data["image_path"])
            image_path = image_path.replace(self.dataset_path, "")
            self.id_to_relative_path[split][i] = image_path

            processed_dataset.append(
                {SemSegCols.image_path: image_path, SemSegCols.id: i}
            )
        # need to add 1 to the last index to get the next index for the following
        # datasets as i is 0 indexed so 0 + start_int would equal the ending index
        # of the previous dataset
        return processed_dataset

    def find_mask_category(self, batch: Dict[str, Any]) -> None:
        """
        Finds the mask category and stores it in the helper data
        :param batch: Dict[str, Any] batch from the dataloader
        """
        if not self.mask_col_name:
            for key in batch:
                if "mask" in key or "label" in key or "target" in key:
                    self.mask_col_name = key
            if not self.mask_col_name:
                raise ValueError(
                    "No mask column found in the batch please specify in watch method"
                )

    def _dq_classifier_hook_with_step_end(
        self,
        model: Module,
        model_input: torch.Tensor,
        model_output: Union[Any, torch.Tensor],
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

    def _dq_input_hook(
        self,
        model: torch.nn.Module,
        model_input: torch.Tensor,
        model_output: Dict[str, torch.Tensor],
    ) -> None:
        """
        Hook to store the model input (tensor) and extract the output
        from a dictionary and store

        :param model: torch.nn.Module segmentation model
        :param model_input: torch.Tensor input to the model - an image (bs, 3, h, w)
        :param model_output: torch.Tensor output of the model

        """
        # model input comes as a tuple of length 1
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

    def queue_gt_and_pred(self, probs: torch.Tensor, gt: torch.Tensor) -> None:
        """Enqueue the ground truth and predicted masks for the batch

        Args:
            probs (torch.Tensor): probability vectors to queue for LM
            gt (torch.Tensor): gt masks resized to queue for LM
        """
        bs = probs.shape[0]
        # interpolate expects N, C, H, W so have to reshuffle probs
        probs = probs.permute(0, 3, 1, 2)
        # resize the tensors to be 64, 64 for compressed storage
        probs = F.interpolate(probs, size=(64, 64), mode="bicubic")
        probs = probs.permute(0, 2, 3, 1)
        gt = gt.unsqueeze(1)
        gt = F.interpolate(gt, size=(64, 64), mode="nearest").long()
        gt = gt.squeeze(1)

        # stack on the end of the queue and remove front to keep only most recent
        self.prob_queue: torch.Tensor = torch.cat((self.prob_queue, probs), dim=0)
        self.gt_queue: torch.Tensor = torch.cat((self.gt_queue, gt), dim=0)
        if self.prob_queue.shape[0] > self.queue_size:
            self.prob_queue = self.prob_queue[bs:]
            self.gt_queue = self.gt_queue[bs:]

    def _init_lm_labels(self) -> None:
        # initialize variables for likely mislabelled
        self.confident_count = torch.zeros(
            (self.number_classes, self.number_classes), dtype=torch.int64
        )
        self.counts_per_class = torch.zeros(self.number_classes, dtype=torch.int64)
        self.thresholds = torch.zeros(self.number_classes, dtype=torch.float32)
        self.thresholds += 0.5

        # create a queue to store the last 100 probs and gt for computing LM
        self.prob_queue = torch.empty((self.queue_size, 64, 64, self.number_classes))
        self.gt_queue = torch.empty((self.queue_size, 64, 64))

    def _on_step_end(self) -> None:
        """Function to be called at the end of step to log the inputs and outputs"""
        if not self.mask_col_name:
            self.find_mask_category(self.helper_data["batch"]["data"])

        # if we have not inferred the number of classes from the model architecture
        self.number_classes = self.helper_data[HelperData.model_outputs_store][
            "logits"
        ].shape[1]
        self._init_lm_labels()
        split = self.logger_config.cur_split.lower()  # type: ignore

        with torch.no_grad():
            logging_data = self.helper_data["batch"]["data"]
            img_ids = self.helper_data["batch"]["ids"]  # np.ndarray (bs,)
            # convert the img_ids to absolute ids from file map
            img_ids = [
                self.dataloader_path_to_id[split][path]
                for path in logging_data["image_path"]
            ]
            log_image_paths = [self.id_to_relative_path[split][id] for id in img_ids]
            image_paths = [pth.lstrip("./") for pth in log_image_paths]

            # resize the logits to the input size based on hooks
            preds = self.helper_data[HelperData.model_outputs_store]["logits"].cpu()
            input_shape = self.helper_data[HelperData.model_input].shape[-2:]
            preds = F.interpolate(preds, size=input_shape, mode="bilinear")

            # checks whether the model is (n, classes, w, h), or (n, w, h, classes)
            if preds.shape[1] == self.number_classes:
                preds = preds.permute(0, 2, 3, 1)
            assert (
                preds.shape[-1] == self.number_classes
            ), "The model output shape is not as expected. \
                    Expected classes to be in last dimension"

            argmax = preds.clone().argmax(dim=-1)
            logits = preds  # (bs, w, h, classes)
            gold_boundary_masks = mask_to_boundary(
                logging_data[self.mask_col_name].clone().cpu().numpy()
            )  # (bs, w, h)
            pred_boundary_masks = mask_to_boundary(
                argmax.clone().cpu().numpy()
            )  # (bs, w, h)
            if logging_data[self.mask_col_name].shape[1] == 1:
                logging_data[self.mask_col_name] = logging_data["mask"].squeeze(
                    1
                )  # (bs, w, h)
            gold_mask = logging_data[self.mask_col_name].cpu()  # (bs, w, h)

            probs = torch.nn.Softmax(dim=-1)(logits).cpu()  # (bs, w, h, classes)

            # update the necessary variable in order to caluclate likely mislabled
            self.queue_gt_and_pred(probs, gold_mask)
            out_threshold = calculate_self_confidence_threshold(
                self.prob_queue, self.gt_queue
            )
            for cls in torch.unique(gold_mask):
                self.thresholds[cls] = (
                    self.thresholds[cls] * 0.999 + out_threshold[cls] * 0.001
                )
            for class_idx in range(self.number_classes):
                self.confident_count = semseg_fill_confident_counts(
                    probs[..., class_idx],
                    gold_mask,
                    class_idx,
                    per_class_threshold=self.thresholds[class_idx],
                    confident_counts=self.confident_count,
                )
            self.counts_per_class += torch.bincount(
                gold_mask.view(-1), minlength=probs.shape[-1]
            )
            self_confidence = semseg_calculate_self_confidence(
                self.prob_queue, self.gt_queue
            )
            mislabeled_pixels = calculate_lm_for_batch(
                self_confidence,
                self.confident_count,
                self.counts_per_class,
                self.gt_queue,
                self.number_classes,
                self.prob_queue,
            )
            # if we have not reached our queue size, we do not report mislabeled
            if self.prob_queue.shape[0] < self.queue_size:
                mislabeled_pixels = torch.zeros_like(mislabeled_pixels)
            bs = probs.shape[0]
            mislabeled_pixels = mislabeled_pixels[-bs:]
            # do not log if we are not in the final inference loop
            if not self.called_finish:
                return

            # compute overlapping classes here as we only want the prob
            # from the inference loop instead of all iterations as graph
            # will likely get to big to compute
            self.update_overlapping_classes_graph(probs, gold_mask)

            logger = SemanticSegmentationModelLogger(
                bucket_name=self.bucket_name,
                image_paths=image_paths,
                image_ids=img_ids,
                gt_masks=gold_mask,  # Torch tensor (bs, w, h)
                pred_masks=argmax,  # Torch tensor (bs, w, h)
                gt_boundary_masks=torch.tensor(
                    gold_boundary_masks
                ),  # Torch tensor (bs, w, h)
                pred_boundary_masks=torch.tensor(
                    pred_boundary_masks
                ),  # Torch tensor (bs, w, h)
                output_probs=probs,  # Torch tensor (bs, w, h, classes)
                mislabeled_pixels=mislabeled_pixels,  # torch tensor (bs, w, h)
            )
            logger.log()

    def finish(self) -> None:
        self.called_finish = True
        # finish function that runs our inference at the end of training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for split, dataloader in self.dataloaders.items():
            # For sem seg the final inference loop is always considered epoch 0
            dq.set_epoch_and_split(0, Split[split])
            with torch.no_grad():
                self.run_one_epoch(dataloader, device)

            self.normalize_and_compute_overlapping_classes()

    def run_one_epoch(self, dataloader: DataLoader, device: torch.device) -> None:
        """Runs one epoch for our inference loop to collect the necessary data

        Args:
            dataloader (DataLoader): dataloader to run inference on
            device (torch.device): device to run inference on
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.autocast("cuda"):
            for i, batch in enumerate(dataloader):
                img = batch[self.image_col]
                img = img.to(device)
                self.model(img)

    def update_overlapping_classes_graph(
        self, probs: torch.Tensor, gt_masks: torch.Tensor
    ) -> None:
        """Adds the current batch to our overlapping classes graph

        Args:
            probs (torch.Tensor): (bs, w, h, classes)
            gt_masks (torch.Tensor): (bs, w, h)
        """
        if not hasattr(self, 'overlapping_classes_top_k'):
            self.overlapping_classes_top_k = min(self.number_classes, MIN_K)
        # compute the overlapping classes
        # (bs, w, h, classes) -> (bs, w, h, classes, 1)
        np_probs = probs.view(-1, self.number_classes).numpy()
        np_masks = gt_masks.view(-1).numpy()
        self.overlapping_classes_graph = add_batch_to_graph(
            self.overlapping_classes_graph,
            np_probs,
            np_masks,
            self.overlapping_classes_top_k,
        )

        bincount = np.bincount(np_masks, minlength=self.number_classes)
        for i, count in enumerate(bincount):
            self.labels_counter[i] += count

    def normalize_and_compute_overlapping_classes(self) -> None:
        split = self.logger_config.cur_split.lower()  # type: ignore
        project_run = dq.get_model_logger().proj_run  # type: ignore

        self.overlapping_classes_graph = normalize_graph(
            self.overlapping_classes_graph, self.labels_counter
        )

        undirected_graph = convert_to_undirected(self.overlapping_classes_graph)
        upload_graph_obj(undirected_graph, project_run, split)

        communities = compute_louvain_communities(
            undirected_graph, RESOLUTION, OVERLAPPING_RANDOM_STATE
        )
        scores, size = compute_community_probability_mass(
            communities,
            self.prob_queue,
            self.gt_queue,
        )
        save_community_scores(communities, scores, size, project_run, split)


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
    dataloaders: Dict[str, DataLoader],
    mask_col_name: Optional[str] = None,
    unpatch_on_start: bool = False,
) -> None:
    """
    wraps a PyTorch model and optionally dataloaders to log the
    embeddings and logits to [Galileo](https://www.rungalileo.io/).

        train_dataloader = torch.utils.data.DataLoader()
        model = SemSegModel()
        watch(model, bucket_name, dataset_path, [train_dataloader, test_dataloader])
        for epoch in range(NUM_EPOCHS):
            dq.set_epoch_and_split(epoch,"training")
            train()
            dq.set_split("validation")
            validate()
        dq.finish()

    :param model: Pytorch Model to be wrapped
    :param bucket_name: Name of the bucket from which the images come
    :param dataset_path: Path to the dataset which we can remove from the image path
    :param dataloaders: List of dataloaders to be wrapped
    :param mask_col_name: Name of the column in the dataloader that contains the mask
    :param unpatch_on_start: Whether to unpatch the model before patching it
    """
    print(
        "We assume the dataloaders passed only have transforms that Tensor, Resize, \
        and Normalize the image and mask\n"
        "‼ Any cropping or shearing transforms passed will lead to unexpected \
        results\n"
        "See docs at https://dq.readthedocs.io/en/latest/ (placeholder) for more info \
        \n \n"
    )

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

    dataloaders = dataloaders or {}
    if not dataloaders:
        raise GalileoException(
            "No dataloaders passed. Please pass a list of dataloaders to watch"
        )
    for key, dataloader in dataloaders.items():
        assert key in Split.__members__, GalileoException(
            f"Dataloader key {key} is not a valid split"
        )
        assert isinstance(dataloader, DataLoader), GalileoException(
            "Invalid dataloader. Must be a pytorch dataloader"
            "from torch.utils.data import DataLoader..."
            "train_dataloader = DataLoader(dataset)"
        )
        # We override the dataloader to have 0 workers since multi-processing
        # is not supported for the way we do the final inference run over
        # data in the logging step
        if getattr(dataloader, "num_workers", 0) > 0:
            dataloader.num_workers = 0

    helper_data = dq.get_model_logger().logger_config.helper_data

    # we assume that the image_path they pass to us is relative to the bucket / dataset
    # ie if the path they give to us should be the same path we can use in their bucket
    # to find the data (ie bucket_name/image_path == dataset_path/image_path)

    tl = SemanticTorchLogger(
        bucket_name=bucket_name,
        dataset_path=dataset_path,
        dataloaders=dataloaders,
        mask_col_name=mask_col_name,
        helper_data=helper_data,
        model=model,
    )

    dq.get_model_logger().logger_config.finish = tl.finish
    for key, dataloader in dataloaders.items():
        dataloader._get_iterator = wrap_fn(  # type: ignore
            dataloader._get_iterator,
            patch_iterator_and_batch(tl.helper_data["batch"]),
        )
