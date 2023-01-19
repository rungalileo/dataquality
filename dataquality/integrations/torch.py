from typing import Any, Callable, Dict, List, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers.modeling_outputs import TokenClassifierOutput

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType
from dataquality.schemas.torch import DimensionSlice, InputDim, Layer
from dataquality.utils.helpers import map_indices_to_ids, wrap_fn
from dataquality.utils.torch import (
    ModelHookManager,
    TorchBaseInstance,
    patch_dataloaders,
    patch_iterator_with_store,
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
        last_hidden_state_layer: Layer = None,
        embedding_dim: Optional[Union[str, DimensionSlice]] = None,
        logits_dim: Optional[Union[str, DimensionSlice]] = None,
        classifier_layer: Layer = "classifier",
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
        helper_data: Dict[str, Any] = {},
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
        self.helper_data = helper_data
        self._init_helper_data(self.hook_manager, self.model)
        self.logger_config = dq.get_data_logger().logger_config

    def _init_helper_data(self, hm: ModelHookManager, model: Module) -> None:
        """
        Initialize the helper data with ids from the dataloader indices,
        patches for applied monkey patched functions and the hook manager.
        :param hm: Hook manager
        :return: None
        """
        self.helper_data.clear()
        self.helper_data.update(
            {
                "ids": [],
                "last_action": "init",
                "patches": [],
                "model_outputs": {},
                "hook_manager": hm,
                "model": model,
            }
        )

    def _attach_hooks_to_model(
        self, model: Module, classifier_layer: Layer, last_hidden_state_layer: Layer
    ) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        :return: None
        """
        try:
            self.hook_manager.attach_classifier_hook(
                model, self.classifier_hook_with_step_end, classifier_layer
            )
        except Exception as e:
            print(
                f"Could not attach classifier hook to model. "
                f"Error: {e}. "
                f"Please check the classifier layer name: {classifier_layer}"
            )
            self.hook_manager.attach_hooks_to_model(
                model, self._embedding_hook, last_hidden_state_layer
            )
            self.hook_manager.attach_hook(model, self._logit_hook_with_step_end)

    def classifier_hook_with_step_end(
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
        :return: None
        """
        self._classifier_hook(model, model_input, model_output)
        self._on_step_end()

    def _logit_hook_with_step_end(
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
        :return: None
        """
        self._logit_hook(model, model_input, model_output)
        self._on_step_end()

    def _on_step_end(self) -> None:
        """
        Log the embeddings, ids and logits.
        :return: None
        """
        model_outputs = self.helper_data["model_outputs"]
        # Workaround for multiprocessing
        if model_outputs.get("ids") is None and len(self.helper_data["ids"]):
            model_outputs["ids"] = self.helper_data["ids"].pop(0)
            self.helper_data["last_action"] = "pop"

        # Log only if embedding exists
        assert model_outputs.get("embs") is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert model_outputs.get("logits") is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert model_outputs.get("ids") is not None, GalileoException(
            "id column missing in dataset (needed to map rows to the indices/ids)"
        )
        # Convert the indices to ids
        cur_split = dq.get_data_logger().logger_config.cur_split.lower()  # type: ignore
        model_outputs["ids"] = map_indices_to_ids(
            self.logger_config.idx_to_id_map[cur_split], model_outputs["ids"]
        )
        dq.log_model_outputs(**model_outputs)
        model_outputs.clear()


def watch(
    model: Module,
    dataloaders: Optional[List[DataLoader]] = [],
    last_hidden_state_layer: Union[Module, str, None] = None,
    embedding_dim: InputDim = None,
    logits_dim: InputDim = None,
    classifier_layer: Union[str, Module] = "classifier",
    embedding_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
    unpatch_on_start: bool = True,
) -> None:
    """
    [`watch`] is a function that wraps the model and dataloaders to log the
    embeddings and logits to [Galileo](https://www.rungalileo.io/).
    :param model: Pytorch Model to be wrapped
    :param dataloaders: List of dataloaders to be wrapped
    :param last_hidden_state_layer: Layer to extract the embeddings from
    :param embedding_dim: Dimension of the embeddings for example "[:, 0]"
    to remove the cls token
    :param logits_dim: Dimension to extract the logits for example in NER
    "[:,1:,:]"
    :param logits_dim: Dimension of the logits
    :param classifier_layer: Layer to hook into. This will extract embeddings
    from layer input and logits from layer output. If the layer is not found,
    the last_hidden_state_layer will be used
    :param embedding_fn: Function to process embeddings from the model
    :param logits_fn: Function to process logits from the model f.e. lambda x[0]
    :param unpatch_on_start: Force unpatching of dataloaders
    instead of global patching
    :return: None
    ```
    dq.log_dataset(train_dataset, split="train")
    train_dataloader = torch.utils.data.DataLoader()
    model = TextClassificationModel(num_labels=len(train_dataset.list_of_labels))
    watch(model, classifier_layer = "classifier")
    for epoch in range(NUM_EPOCHS):
        dq.set_epoch_and_split(epoch,"training")
        train()
        dq.set_split("validate")
        validate()
    dq.finish()

    ```
    """
    a.log_function("torch/watch")
    assert dq.config.task_type, GalileoException(
        "dq client must be initialized. " "For example: dq.init('text_classification')"
    )
    if unpatch_on_start:
        unwatch(force=True)
    if not getattr(model, "_dq", False):
        setattr(model, "_dq", True)
    else:
        raise GalileoException(
            "Model is already being watched, run unwatch(model) first"
        )

    helper_data = dq.get_model_logger().logger_config.helper_data
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
                patch_iterator_with_store(tl.helper_data["model_outputs"]),
            )
    else:
        # Patch the dataloader class globally
        # Can be unpatched with unwatch()
        patch_dataloaders(tl.helper_data)


def unwatch(model: Optional[Module] = None, force: bool = True) -> None:
    """Unwatches the model. Run after the run is finished.
    :param force: Force unwatch even if the model is not watched"""

    helper_data = dq.get_model_logger().logger_config.helper_data
    model = model or helper_data.get("model", None)
    if not getattr(model or {}, "_dq", False) and not force:
        raise GalileoException("Model is not watched, run watch(model) first")
    # Unpatch the dataloaders
    unpatch(helper_data.get("patches", []))
    # Detach hooks the model. in the future use the model passed
    # https://discuss.pytorch.org/t/how-to-check-where-the-hooks-are-in-the-model/120120/2
    hook_manager = helper_data.get("hook_manager", None)
    if hook_manager:
        hook_manager.detach_hooks()
    # Remove the model from the helper data
    if "model" in helper_data:
        del helper_data["model"]
    if model and hasattr(model, "_dq"):
        del model._dq
