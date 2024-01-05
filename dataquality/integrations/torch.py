import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers.modeling_outputs import TokenClassifierOutput

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType
from dataquality.schemas.torch import DimensionSlice, InputDim, Layer
from dataquality.utils.helpers import map_indices_to_ids
from dataquality.utils.torch import (
    ModelHookManager,
    PatchDataloadersGlobally,
    PatchSingleDataloaderIterator,
    TorchBaseInstance,
    TorchHelper,
    remove_all_forward_hooks,
    unpatch,
)

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("torch")


class TorchLogger(TorchBaseInstance):
    """
    [`TorchLogger`] that sends the logs to [Galileo](https://www.rungalileo.io/)
    for each training training step.
    """

    embedding_dim: Optional[DimensionSlice]
    logits_dim: Optional[DimensionSlice]
    model: Module

    def __init__(
        self,
        model: Module,
        last_hidden_state_layer: Optional[Layer] = None,
        embedding_dim: Optional[Union[str, DimensionSlice]] = None,
        logits_dim: Optional[Union[str, DimensionSlice]] = None,
        classifier_layer: Optional[Layer] = None,
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
        helper_data: Optional[Dict[str, Any]] = None,
        task: Union[TaskType, None] = TaskType.text_classification,
    ):
        task_type = task or dq.config.task_type
        assert task_type is not None, GalileoException(
            "Dataquality task cannot be None."
            "Setup with dq.init(task_type='text_classification')"
        )
        self.task = task_type
        self.model = model
        self.last_hidden_state_layer = last_hidden_state_layer
        self.classifier_layer = classifier_layer
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn

        self._init_dimension(embedding_dim, logits_dim)
        self.hook_manager = ModelHookManager()
        self._attach_hooks_to_model(model, classifier_layer, last_hidden_state_layer)
        if helper_data is None:
            helper_data = {}
        helper_data["torch_helper"] = TorchHelper(model, self.hook_manager)
        self.torch_helper_data = helper_data["torch_helper"]
        self._init_helper_data(self.hook_manager, self.model)
        self.logger_config = dq.get_data_logger().logger_config

    def _init_helper_data(self, hm: ModelHookManager, model: Module) -> None:
        """
        Initialize the helper data with ids from the dataloader indices,
        patches for applied monkey patched functions and the hook manager.
        :param hm: Hook manager
        """
        self.torch_helper_data.clear()
        self.torch_helper_data = TorchHelper(model, hm)

    def _attach_hooks_to_model(
        self, model: Module, classifier_layer: Layer, last_hidden_state_layer: Layer
    ) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        """
        try:
            self.hook_manager.attach_classifier_hook(
                model, self._dq_classifier_hook_with_step_end, classifier_layer
            )
        except Exception as e:
            if dq.config.task_type != TaskType.semantic_segmentation:
                warn(
                    "Warning: Could not attach function to model layer."
                    f" {e}. Please check that the classifier layer name:"
                    f" {classifier_layer} exists in the model. Common layers"
                    " to extract logits and the last hidden state are 'classifier'"
                    "and 'fc'. To fix this, pass the correct layer name to the "
                    "'classifier_layer' parameter in the 'watch' function. "
                    "For example: 'watch(model, classifier_layer='fc')'."
                    "You can view the model layers by using the 'model.named_children'"
                    "function or by printing the model."
                )
            self.hook_manager.attach_hooks_to_model(
                model, self._dq_embedding_hook, last_hidden_state_layer
            )
            self.hook_manager.attach_hook(model, self._dq_logit_hook_with_step_end)

    def _dq_classifier_hook_with_step_end(
        self,
        model: Module,
        model_input: Tensor,
        model_output: Union[TokenClassifierOutput, Tensor],
    ) -> None:
        """
        Hook to extract the logits, embeddings from the model.
        :param model: Model pytorch model
        :param model_input: Model input
        :param model_output: Model output
        """
        self._classifier_hook(model, model_input, model_output)
        self._on_step_end()

    def _dq_logit_hook_with_step_end(
        self,
        model: Module,
        model_input: Tensor,
        model_output: Union[TokenClassifierOutput, Tensor],
    ) -> None:
        """
        Hook to extract the logits from the model.
        :param model: Model pytorch model
        :param model_input: Model input
        :param model_output: Model output
        """
        self._dq_logit_hook(model, model_input, model_output)
        self._on_step_end()

    def _on_step_end(self) -> None:
        """Log the embeddings, ids and logits.

        We save the embeddings and logits in a dict called model_output_store
        in the helper data. This is because the embeddings and logits are
        extracted in the hooks and we need to log them in the on_step_end
        method.
        """

        model_outputs_store = self.torch_helper_data.model_outputs_store
        # Workaround for multiprocessing
        if model_outputs_store.ids is None and len(
            self.torch_helper_data.dl_next_idx_ids
        ):
            model_outputs_store.ids = self.torch_helper_data.dl_next_idx_ids.pop(0)

        # Log only if embedding exists
        assert model_outputs_store.embs is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert model_outputs_store.logits is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert model_outputs_store.ids is not None, GalileoException(
            "id column missing in dataset (needed to map rows to the indices/ids)"
        )
        # Convert the indices to ids
        cur_split = self.logger_config.cur_split
        assert cur_split is not None, GalileoException(
            "Current split must be set before logging"
        )
        cur_split = cur_split.lower()  # type: ignore
        model_outputs_store.ids = map_indices_to_ids(
            self.logger_config.idx_to_id_map[cur_split], model_outputs_store.ids
        )
        dq.log_model_outputs(**model_outputs_store.to_dict())
        model_outputs_store.clear()


def watch(
    model: Module,
    dataloaders: Optional[List[DataLoader]] = [],
    classifier_layer: Optional[Union[str, Module]] = None,
    embedding_dim: Optional[InputDim] = None,
    logits_dim: Optional[InputDim] = None,
    embedding_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
    last_hidden_state_layer: Union[Module, str, None] = None,
    unpatch_on_start: bool = False,
    dataloader_random_sampling: bool = False,
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
    :param embedding_dim: Dimension of the embeddings for example `"[:, 0]"`
        to remove the cls token
    :param logits_dim: Dimension of the logits from layer input and
        logits from layer output. For example in NER `"[:,1:,:]"`.
        If the layer is not found, the last_hidden_state_layer will be used
    :param embedding_fn: Function to process embeddings from the model
    :param logits_fn: Function to process logits from the model f.e.
        `lambda x: x[0]`
    :param last_hidden_state_layer: Layer to extract the embeddings from
    :param unpatch_on_start: Force unpatching of dataloaders
        instead of global patching
    :param dataloader_random_sampling: Whether a RandomSampler
        or WeightedRandomSampler is being used. If random sampling
        is being used, you must set this to True, otherwise logging
        will fail at the end of training.

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

    logger_config = get_data_logger().logger_config
    helper_data = logger_config.helper_data

    print("Attaching dataquality to model and dataloaders")
    tl = TorchLogger(
        model=model,
        last_hidden_state_layer=last_hidden_state_layer,
        embedding_dim=embedding_dim,
        logits_dim=logits_dim,
        classifier_layer=classifier_layer,
        embedding_fn=embedding_fn,
        logits_fn=logits_fn,
        task=dq.config.task_type,
        helper_data=helper_data,
    )
    # Patch the dataloader class if no dataloaders are passed
    # or if the dataloaders have num_workers > 0
    dataloaders = dataloaders or []
    is_single_process_dataloader = all([d.num_workers == 0 for d in dataloaders])
    if len(dataloaders) > 0 and is_single_process_dataloader:
        for dataloader in dataloaders:
            assert isinstance(dataloader, DataLoader), GalileoException(
                "Invalid dataloader. Must be a pytorch dataloader"
                "from torch.utils.data import DataLoader..."
                "train_dataloader = DataLoader(dataset)"
            )
            assert (
                dataloader.num_workers == 0
            ), "Dataloaders with num_workers > 0 are not supported"

            if not isinstance(dataloader.sampler, SequentialSampler):
                logger_config.dataloader_random_sampling = True

            # Patch the dataloader class
            PatchSingleDataloaderIterator(
                dataloader, tl.torch_helper_data.model_outputs_store
            )
    else:
        # Patch the dataloader class globally
        # Can be unpatched with unwatch()
        PatchDataloadersGlobally(tl.torch_helper_data)

    if dataloader_random_sampling:
        logger_config.dataloader_random_sampling = True
    # cleanup_manager = RefManager(lambda: unwatch(model))
    # helper_data["cleaner"] = Cleanup(cleanup_manager)


def unwatch(model: Optional[Module] = None, force: bool = True) -> None:
    """Unwatches the model. Run after the run is finished.
    :param force: Force unwatch even if the model is not watched"""
    torch_helper_data: TorchHelper = (
        dq.get_model_logger().logger_config.helper_data.get(
            "torch_helper", TorchHelper()
        )
    )

    model = model or torch_helper_data.model
    if not getattr(model or {}, "_dq", False):
        warn("Model is not watched, run watch(model) first")
        if not force:
            return

    # Unpatch the dataloaders

    unpatch(torch_helper_data.patches)
    # Detach hooks the model. in the future use the model passed
    # https://discuss.pytorch.org/t/how-to-check-where-the-hooks-are-in-the-model/120120/2
    hook_manager = torch_helper_data.hook_manager
    if hook_manager:
        hook_manager.detach_hooks()
    # Remove the model from the helper data
    if isinstance(model, Module):
        remove_all_forward_hooks(model)
    else:
        warnings.warn("model is not a Module")
    torch_helper_data.model = None
    if model and hasattr(model, "_dq"):
        del model._dq
