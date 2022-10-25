# Imports for the hook manager
from typing import Any, Dict, Optional, Union

from datasets import Dataset
from torch import Tensor
from torch.nn import Module
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.torch import HookManager, convert_fancy_idx_str_to_slice
from dataquality.utils.transformers import (
    add_id_to_signature_columns,
    remove_id_collate_fn_wrapper,
)

EmbeddingDim = Any
Layer = Optional[Union[Module, str]]


# Trainer callback for Huggingface transformers Trainer library
class DQCallback(TrainerCallback):
    """
    [`TrainerCallback`] that sends the logs to [Galileo](https://www.rungalileo.io/)
    for each training training step.
    """

    embedding_dim: EmbeddingDim
    logits_dim: EmbeddingDim
    hook_manager: HookManager

    def __init__(
        self,
        layer: Layer = None,
        embedding_dim: EmbeddingDim = None,
        logits_dim: EmbeddingDim = None,
    ) -> None:
        # Access the dq logger helper data
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self._initialized = False
        # Hook manager for attaching hooks to the model
        self.hook_manager = HookManager()
        self.layer = layer
        self._init_dimension(embedding_dim, logits_dim)

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

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data.clear()

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        self._clear_logger_config_helper_data()

    def setup(
        self,
        args: TrainingArguments,
        state: TrainerState,
        kwargs: Dict,
    ) -> None:
        """Setup the callback
        :param args: Training arguments
        :param state: Trainer state
        :param model: Model
        :param kwargs: Keyword arguments (eval_dataloader, train_dataloader, tokenizer)
        :return: None"""
        # Attach hooks to the model
        # assert dq.config.task_type == TaskType.text_classification, GalileoException(
        #     "dq client must be initialized for text classification. "
        #     "For example: dq.init('text_classification')"
        # )
        self.task = dq.config.task_type
        model: Module = kwargs["model"]
        self._attach_hooks_to_model(model, self.layer)
        train_dataloader = kwargs["train_dataloader"]
        train_dataloader_ds = train_dataloader.dataset
        if isinstance(train_dataloader_ds, Dataset):
            assert "id" in train_dataloader_ds.column_names, GalileoException(
                "Did you map IDs to your dataset before watching the model?\n"
                "To add the id column with datasets. You can run:\n"
                """`ds= dataset.map(lambda x, idx: {"id": idx},"""
                " with_indices=True)`. The id (index) column is needed in "
                "the dataset for logging"
            )
        else:
            raise GalileoException(f"Unknown dataset type {type(train_dataloader_ds)}")
        self._initialized = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Event called at the beginning of training. Attaches hooks to model.
        :param args: Training arguments
        :param state: Trainer state
        :param control: Trainer control
        :param kwargs: Keyword arguments (model, eval_dataloader, tokenizer...)
        :return: None
        """
        if not self._initialized:
            self.setup(args, state, kwargs)
        dq.set_split(Split.training)  # 🔭🌕 Galileo logging

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        state_epoch = state.epoch
        if state_epoch is not None:
            state_epoch = int(state_epoch)
            dq.set_epoch(state_epoch)  # 🔭🌕 Galileo logging

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        dq.set_split(Split.validation)  # 🔭🌕 Galileo logging

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Perform a training step on a batch of inputs.
        Log the embeddings, ids and logits.
        :param args: Training arguments
        :param state: Trainer state
        :param control: Trainer control
        :param kwargs: Keyword arguments (including the model, inputs, outputs)
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
            "Did you map IDs to your dataset before watching the model? You can run:\n"
            "`ds= dataset.map(lambda x, idx: {'id': idx}, with_indices=True)`\n"
            "id (index) column is needed in the dataset for logging"
        )

        # 🔭🌕 Galileo logging
        dq.log_model_outputs(**self.helper_data)

    def _attach_hooks_to_model(self, model: Module, layer: Layer) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        :return: None
        """
        self.hook_manager.attach_embedding_hook(model, layer, self._embedding_hook)
        self.hook_manager.attach_hook(model, self._logit_hook)

    def _embedding_hook(
        self,
        model: Module,
        model_input: Tensor,
        model_output: Union[BaseModelOutput, Tensor],
    ) -> None:
        """
        Hook to extract the embeddings from the model
        Keyword arguments won't be passed to the hooks and only to the ``forward``.
        The hook can modify the input. User can either return a tuple or a
        single modified value in the hook. We will wrap the value into a tuple
        if a single value is returned(unless that value is already a tuple).
        The hook will be called every time after :func:`forward` has computed an output.

        :param model: Model pytorch model / layer
        :param model_input: Input of the current layer
        :param model_output: Output of the current layer
        :return: None
        """
        if isinstance(model_output, Tensor):
            output = model_output
        elif hasattr(model_output, "last_hidden_state"):
            output = model_output.last_hidden_state
        output_detached = output.detach()
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

    def _logit_hook(
        self,
        model: Module,
        model_input: Tensor,
        model_output: Union[TokenClassifierOutput, Tensor],
    ) -> None:
        """
        Hook to extract the logits from the model.
        :param model: Model pytorch model
        :param model_input: Model input of the current layer
        :param model_output: Model output of the current layer
        :return: None
        """
        if isinstance(model_output, Tensor):
            logits = model_output
        elif hasattr(model_output, "logits"):
            logits = model_output.logits
        logits = logits.detach()
        if self.logits_dim is not None:
            logits = logits[self.logits_dim]
        elif len(logits.shape) == 3 and self.task == TaskType.text_ner:
            # It is assumed that the CLS token is removed
            # through this dimension for NER tasks
            logits = logits[:, 1:, :]
        self.helper_data["logits"] = logits


@check_noop
def watch(
    trainer: Trainer,
    layer: Layer = None,
    embedding_dim: EmbeddingDim = None,
    logits_dim: LogitsDim = None,
) -> None:
    """
    [`watch`] is used to hook into to the trainer
    to log to [Galileo](https://www.rungalileo.io/)
    :param trainer: Trainer object
    :return: None
    """
    # Callback which we add to the trainer
    dqcallback = DQCallback(
        layer=layer, embedding_dim=embedding_dim, logits_dim=logits_dim
    )
    # The columns needed for the forward process
    signature_cols = add_id_to_signature_columns(trainer)
    # We wrap the data collator to remove the id column
    trainer.data_collator = remove_id_collate_fn_wrapper(
        trainer.data_collator, signature_cols, dqcallback.helper_data
    )
    trainer.add_callback(dqcallback)
