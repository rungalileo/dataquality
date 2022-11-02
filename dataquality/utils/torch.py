import re
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # noqa: F401
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput

from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType
from dataquality.schemas.torch import DimensionSlice, Layer
from dataquality.utils.helpers import wrap_fn


class TorchBaseInstance:
    embedding_dim: Optional[DimensionSlice]
    logits_dim: Optional[DimensionSlice]
    task: TaskType
    helper_data: Dict[str, Any]

    def _init_dimension(
        self,
        embedding_dim: Optional[Union[str, DimensionSlice]],
        logits_dim: Optional[Union[str, DimensionSlice]],
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


# store indices
def store_batch_indices(store: Dict[str, List[int]]) -> Callable:
    def process_batch_indices(
        next_index_func: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        """Stores the indices of the batch"""
        indices = next_index_func(*args, **kwargs)
        if indices:
            store["ids"] = indices
        return indices

    return process_batch_indices


# add patch to the dataloader iterator
def patch_iterator_with_store(store: Dict[str, List[int]]) -> Callable:
    """Patches the iterator of the dataloader to return the indices"""

    def patch_iterator(
        orig_iterator: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        iteraror = orig_iterator(*args, **kwargs)
        iteraror._next_index = wrap_fn(iteraror._next_index, store_batch_indices(store))
        return iteraror

    return patch_iterator


def validate_fancy_index_str(input_str: str = "[:, 1:, :]") -> bool:
    """Validates a fancy index string.
    :param input_str: The string to validate for example "[:, 1:, :]"
    :return: True if the string is valid, False otherwise
    """
    valid = re.fullmatch(r"[\s,\[\]\d:\(\)]+", input_str)
    if valid:
        if input_str.count("[") != input_str.count("]"):
            return False
        return True
    return False


def convert_fancy_idx_str_to_slice(
    fstr: str = "[:, 1:, :]",
) -> Union[Tuple, slice, int]:
    """Converts a fancy index string to a slice.
    :param fstr: The fancy index string to convert for example "[:, 1:, :]"
    :return: The slice for example:
    (slice(None, None, None), slice(1, None, None), slice(None, None, None))
    """

    clean_str = fstr
    # Remove outer brackets
    if fstr[0] == "[" and fstr[-1] == "]":
        clean_str = clean_str[1:-1]
    # Validate the string before we eval the fancy index
    assert validate_fancy_index_str(
        clean_str
    ), 'Fancy index string is not valid. Valid format: "[:, 1:, :]"'
    return eval("np.s_[{}]".format(clean_str))


class ModelHookManager:
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
        embedding_hook: Callable,
        model_layer: Layer = None,
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
