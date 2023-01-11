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
        self._init_helper_data(helper_data, self.hook_manager, self.model)
        self.logger_config = dq.get_data_logger().logger_config

    def _init_helper_data(
        self, helper_data: Dict[str, Any], hm: ModelHookManager, model: Module
    ) -> Dict[str, Any]:
        """
        Initialize the helper data with ids from the dataloader indices,
        patches for applied monkey patched functions and the hook manager.
        :param hm: Hook manager
        :return: None
        """
        helper_data.clear()
        helper_data.update(
            {
                "ids": [],
                "last_action": "init",
                "patches": [],
                "model_outputs": {},
                "hook_manager": hm,
                "model": model,
            }
        )
        return helper_data

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
    dataloaders: List[DataLoader] = [],
    last_hidden_state_layer: Union[Module, str, None] = None,
    embedding_dim: InputDim = None,
    logits_dim: InputDim = None,
    classifier_layer: Union[str, Module] = "classifier",
    embedding_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
    force_local_patching: bool = False,
) -> None:
    """
    [`watch`] is a function that wraps the model and dataloaders to log the
    embeddings and logits to [Galileo](https://www.rungalileo.io/).
    :param model: Pytorch model
    :param dataloaders: List of dataloaders
    :param layer: Layer to extract the embeddings from
    :param embedding_dim: Embedding dimension to for example "[:, 0]"
    to remove the cls token
    :param logits_dim: Dimension to extract the logits for example in NER
    "[:,1:,:]"
    :return: None
    ```
    dq.log_dataset(train_dataset, split="train")
    train_dataloader = torch.utils.data.DataLoader()
    model = TextClassificationModel(num_labels=len(train_dataset.list_of_labels))
    watch(model, [train_dataloader,test_dataloader])
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

    if force_local_patching:
        if len(dataloaders) == 0:
            raise GalileoException("No dataloaders passed to watch")

        for dataloader in dataloaders:
            assert isinstance(dataloader, DataLoader), GalileoException(
                "Invalid dataloader. Must be a pytorch dataloader"
                "from torch.utils.data import DataLoader..."
                "train_dataloader = DataLoader(dataset)"
            )
            assert dataloader.num_workers == 0, GalileoException(
                "Dataloaders passed to watch must have num_workers=0."
                "Parralelization is not yet supported"
            )
            dataloader._get_iterator = wrap_fn(  # type: ignore
                dataloader._get_iterator,
                patch_iterator_with_store(helper_data["model_outputs"]),
            )
    else:
        patch_dataloaders(tl.helper_data)


def unwatch() -> None:
    """Unpatches the model. Run after the run is finished
    :param model: The model to unpatch"""
    helper_data = dq.get_model_logger().logger_config.helper_data

    if not getattr(helper_data["model"], "_dq", False):
        raise GalileoException("Model is not watched, run watch(model) first")

    unpatch(helper_data["patches"])
    helper_data["hook_manager"].detach_hooks()
    model = helper_data["model"]
    if hasattr(model, "_dq"):
        del model._dq
