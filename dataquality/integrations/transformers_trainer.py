# Imports for the hook manager
from queue import Queue
from typing import Callable, Dict, Iterable, List, Optional, Union

from datasets import Dataset
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import check_noop
from dataquality.utils.transformers import (
    add_id_to_signature_columns,
    remove_id_collate_fn_wrapper,
)

Layer = Optional[Union[Module, str]]
EmbeddingDim = Optional[Iterable[int]]


# Trainer callback for Huggingface transformers Trainer library
class DQCallback(TrainerCallback):
    """
    [`TrainerCallback`] that sends the logs to [Galileo](https://www.rungalileo.io/)
    for each training training step.
    """

    def __init__(
        self,
        layer: Layer = None,
        embedding_dim: EmbeddingDim = None,
    ) -> None:
        # Access the dq logger helper data
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self._initialized = False
        # Hook manager for attaching hooks to the model
        self.hook_manager = HookManager()
        self.layer = layer
        self.embedding_dim = embedding_dim

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
        assert dq.config.task_type == TaskType.text_classification, GalileoException(
            "dq client must be initialized for text classification. "
            "For example: dq.init('text_classification')"
        )
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
            dim, index = self.embedding_dim
            output_detached = output_detached.select(dim, index)
        elif len(output_detached.shape) == 3:
            # It is assumed that the CLS token is removed through this dimension
            output_detached = output_detached[:, 0]
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
        self.helper_data["logits"] = logits.detach()


class HookManager:
    """
    Manages hooks for models. Has the ability to find the layer automatically.
    Otherwise the layer or the layer name needs to be provided.
    """

    # Stores all hooks to remove them from the model later.
    hooks: List[RemovableHandle] = []

    def get_embedding_layer_auto(self, model: Module) -> Module:
        """
        Use a scoring algorithm to find the embedding layer automatically
        given a model. The higher the score the more likely it is the embedding layer.
        """
        name, layer = next(model.named_children())
        print(f"Selected layer for the last hidden state embedding {name}")
        return layer

    def get_embedding_layer_by_name(self, model: Module, name: str) -> Module:
        """
        Iterate over each layer and stop once the the layer name matches
        :param model: Model
        :parm name: string
        """
        queue: Queue = Queue()
        start = model.named_children()
        queue.put(start)
        layer_names = []
        layer_names_str: str = ""
        while not queue.empty():
            named_children = queue.get()
            for layer_name, layer_model in named_children:
                layer_names.append(layer_name)
                layer_names_str = ", ".join(layer_names)
                if layer_name == name:
                    print(
                        f"Found layer {layer_name} in model layers: {layer_names_str}"
                    )
                    return layer_model
                layer_model._get_name()
                queue.put(layer_model.named_children())
        raise GalileoException(
            f"Layer could not be found in {layer_names_str}, "
            "make sure to check capitalization"
        )

    def attach_embedding_hook(
        self,
        model: Module,
        model_layer: Optional[Union[Module, str]] = None,
        embedding_hook: Callable = print,
    ) -> RemovableHandle:
        """Attach hook and save it in our hook list"""
        if model_layer is None:
            selected_layer = self.get_embedding_layer_auto(model)
        elif isinstance(model_layer, str):
            selected_layer = self.get_embedding_layer_by_name(model, model_layer)
        else:
            selected_layer = model_layer
        return self.attach_hook(selected_layer, embedding_hook)

    def attach_hook(self, selected_layer: Module, hook: Callable) -> RemovableHandle:
        """Register a hook and save it in our hook list"""
        h = selected_layer.register_forward_hook(hook)
        self.hooks.append(h)
        return h

    def remove_hook(self) -> None:
        """Remove all hooks from the model"""
        for h in self.hooks:
            h.remove()


@check_noop
def watch(
    trainer: Trainer, layer: Layer = None, embedding_dim: EmbeddingDim = None
) -> None:
    """
    [`watch`] is used to hook into to the trainer
    to log to [Galileo](https://www.rungalileo.io/)
    :param trainer: Trainer object
    :return: None
    """
    # Callback which we add to the trainer
    dqcallback = DQCallback(layer=layer, embedding_dim=embedding_dim)
    # The columns needed for the forward process
    signature_cols = add_id_to_signature_columns(trainer)
    # We wrap the data collator to remove the id column
    trainer.data_collator = remove_id_collate_fn_wrapper(
        trainer.data_collator, signature_cols, dqcallback.helper_data
    )
    trainer.add_callback(dqcallback)
