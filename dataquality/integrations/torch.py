from typing import Any, Dict, List, Optional, Union

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
    patch_iterator_with_store,
)

a = Analytics(ApiClient, dq.config)
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
        model_layer: Layer = None,
        embedding_dim: Optional[Union[str, DimensionSlice]] = None,
        logits_dim: Optional[Union[str, DimensionSlice]] = None,
        task: Union[TaskType, None] = TaskType.text_classification,
    ):
        task_type = task or dq.config.task_type
        assert task_type is not None, GalileoException(
            "Dataquality task cannot be None."
            "Setup with dq.init(task_type='text_classification')"
        )
        self.task = task_type
        self.model = model
        self.model_layer = model_layer
        self._init_dimension(embedding_dim, logits_dim)
        self.hook_manager = ModelHookManager()
        self.hook_manager.attach_embedding_hook(
            model, self._embedding_hook, model_layer
        )
        self.hook_manager.attach_hook(model, self._logit_hook_step_end)
        self.helper_data: Dict[str, Any] = {}
        self.logger_config = dq.get_data_logger().logger_config

    def _logit_hook_step_end(
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
        # Log only if embedding exists
        assert self.helper_data.get("embs") is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert self.helper_data.get("logits") is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert self.helper_data.get("ids") is not None, GalileoException(
            "id column missing in dataset (needed to map rows to the indices/ids)"
        )

        # Convert the indices to ids
        cur_split = dq.get_data_logger().logger_config.cur_split.lower()  # type: ignore
        self.helper_data["ids"] = map_indices_to_ids(
            self.logger_config.idx_to_id_map[cur_split], self.helper_data["ids"]
        )
        dq.log_model_outputs(**self.helper_data)
        self.helper_data.clear()


def watch(
    model: Module,
    dataloaders: List[DataLoader],
    layer: Union[Module, str, None] = None,
    embedding_dim: InputDim = None,
    logits_dim: InputDim = None,
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

    print("Attaching dataquality to model and dataloaders")
    tl = TorchLogger(
        model, layer, embedding_dim, logits_dim=logits_dim, task=dq.config.task_type
    )
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
            dataloader._get_iterator, patch_iterator_with_store(tl.helper_data)
        )
