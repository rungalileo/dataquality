from typing import Any, Dict, List, Union

from torch.nn import Module

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.transformers_trainer import Layer
from dataquality.schemas.task_type import TaskType
from dataquality.schemas.torch import EmbeddingDim
from dataquality.utils.helpers import hook
from dataquality.utils.torch import (
    HookManager,
    convert_fancy_idx_str_to_slice,
    patch_iterator_with_store,
)


class TorchLogger:
    embedding_dim: Union[None, EmbeddingDim]
    logits_dim: Union[None, EmbeddingDim]
    task: TaskType
    model: Module

    def __init__(
        self,
        model: Module,
        model_layer: Layer = None,
        embedding_dim: EmbeddingDim = None,
        logits_dim: EmbeddingDim = None,
        task: Union[TaskType, None] = TaskType.text_classification,
    ):
        assert task is not None, GalileoException(
            "Dataquality task cannot be None."
            "Setup with dq.init(task_type='text_classification')"
        )
        self.task = task
        self.model = model
        self.model_layer = model_layer
        self._init_dimension(embedding_dim, logits_dim)
        self.hook_manager = HookManager()
        self.hook_manager.attach_embedding_hook(
            model, model_layer, self._embedding_hook
        )
        self.hook_manager.attach_hook(model, self._logit_hook)
        self.helper_data: Dict[str, Any] = {}
        self.logger_config = dq.get_data_logger().logger_config

    def _init_dimension(
        self, embedding_dim: EmbeddingDim, logits_dim: EmbeddingDim
    ) -> None:
        """
        Initialize the dimensions of the embeddings and logits
        :param embedding_dim: Dimension of the embedding
        :param logits_dim: Dimension of the logits
        :return: None
        """
        # If embedding_dim is a string, convert it to a slice
        # else assume it is a slice or None
        if isinstance(embedding_dim, str):
            self.embedding_dim = convert_fancy_idx_str_to_slice(embedding_dim)
        elif embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = None

        # If logits_dim is a string, convert it to a slice
        # else assume it is a slice or None
        if isinstance(logits_dim, str):
            self.logits_dim = convert_fancy_idx_str_to_slice(logits_dim)
        elif logits_dim is not None:
            self.logits_dim = logits_dim
        else:
            self.logits_dim = None

    def _embedding_hook(
        self, model: Module, model_input: Any, model_output: Any
    ) -> None:
        """
        Hook to extract the embeddings from the model
        :param model: Model pytorch model
        :param model_input: Model input
        :param model_output: Model output
        :return: None
        """
        if hasattr(model_output, "last_hidden_state"):
            output_detached = model_output.last_hidden_state.detach()
        else:
            output_detached = model_output.detach()
        # If embedding has the CLS token, remove it
        if self.embedding_dim is not None:
            output_detached = output_detached[self.embedding_dim]
        elif len(output_detached.shape) == 3 and (
            self.task in [TaskType.text_classification, TaskType.text_multi_label]
        ):
            # It is assumed that the CLS token is removed through this dimension
            # for text classification tasks and multi label tasks
            output_detached = output_detached[:, 0]
        elif len(output_detached.shape) == 3 and self.task == TaskType.text_ner:
            # It is assumed that the CLS token is removed through this dimension
            # for NER tasks
            output_detached = output_detached[:, 1:, :]
        self.helper_data["embs"] = output_detached

    def _logit_hook(self, model: Module, model_input: Any, model_output: Any) -> None:
        """
        Hook to extract the logits from the model.
        :param model: Model pytorch model
        :param model_input: Model input
        :param model_output: Model output
        :return: None
        """
        if hasattr(model_output, "logits"):
            logits = model_output.logits
        else:
            logits = model_output

        logits = logits.detach()
        if self.logits_dim is not None:
            logits = logits[self.logits_dim]
        elif len(logits.shape) == 3 and self.task == TaskType.text_ner:
            # It is assumed that the CLS token is removed
            # through this dimension for NER tasks
            logits = logits[:, 1:, :]
        self.helper_data["logits"] = logits
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

        # 🔭🌕 Galileo logging
        mapped_idx_to_id = []
        cur_split = dq.get_data_logger().logger_config.cur_split
        for idx in self.helper_data["ids"]:
            ID = self.logger_config.idx_to_id_map[str(cur_split)][idx]
            mapped_idx_to_id.append(ID)
        self.helper_data["ids"] = mapped_idx_to_id

        dq.log_model_outputs(**self.helper_data)


def watch(
    model: Module,
    dataloaders: List[Any] = [],
    layer: Union[Module, str, None] = None,
    embedding_dim: Any = None,
    logits_dim: Any = None,
) -> None:
    """
    [`watch`] is used to hook into to the trainer
    to log to [Galileo](https://www.rungalileo.io/)
    :param trainer: Trainer object
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
    print("Attaching dataquality to model and dataloaders")
    tl = TorchLogger(
        model, layer, embedding_dim, logits_dim=logits_dim, task=dq.config.task_type
    )
    for dataloader in dataloaders:
        dataloader._get_iterator = hook(
            dataloader._get_iterator, patch_iterator_with_store(tl.helper_data)
        )
    tl
