from queue import Queue
from typing import Any, Callable, List, Union

from datasets import Dataset
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback  # noqa: E402
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments  # noqa: E402

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.helpers import check_noop
from dataquality.utils.torch import remove_id_collate_fn_wrapper


class HookManager:
    """
    Manages hooks for models. Has the ability to find the layer automatically.
    Otherwise the layer or the layer name needs to be provided.
    """

    # Stores all hooks to remove them from the model later.
    hooks: List[RemovableHandle] = []

    def get_embedding_layer_auto(self, model: Any) -> Any:
        """
        Use a scoring algorithm to find the embedding layer automatically
        given a model. The higher the score the more likely it is the embedding layer.
        """
        print("selecting embedding layer", next(model.named_children())[0])

        return next(model.children())

    def get_embedding_layer_by_name(self, model: Any, name: str) -> Any:
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
            f"Layer could not be found in { layer_names_str }, "  # noqa: E501
            "make sure to check capitalization"
        )

    def attach_embedding_hook(
        self, model: Any, model_layer: Any = None, embedding_hook: Callable = print
    ) -> RemovableHandle:
        """attach hook and save it in our hook list"""
        if model_layer is None:
            selected_layer = self.get_embedding_layer_auto(model)
        elif type(model_layer) is str:
            selected_layer = self.get_embedding_layer_by_name(model, model_layer)
        else:
            selected_layer = model_layer
        h = self.attach_hook(selected_layer, embedding_hook)
        return h

    def attach_hook(self, selected_layer: Any, hook: Callable) -> RemovableHandle:
        h = selected_layer.register_forward_hook(hook)
        self.hooks.append(h)
        return h

    def remove_hook(self) -> None:
        for h in self.hooks:
            h.remove()


class TorchLogger:
    def __init__(self, model: Any, model_layer: Any = None, embedding_dim: Any = None):
        self._dq = dq
        self.embedding_dim = embedding_dim
        self.model = model
        self.model_layer = model_layer
        self.hook_manager = HookManager()
        self.hook_manager.attach_embedding_hook(model, model_layer,
            self._embedding_hook,embedding_dim)
        self.hook_manager.attach_hook(model, self._logit_hook)


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
            output_detached = output_detached.select(*self.embedding_dim)
        elif len(output_detached.shape) == 3:
            # It is assumed that the CLS token is removed through this dimension
            output_detached = output_detached[:, 0]
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
        self.helper_data["logits"] = logits.detach()

        self._on_step_end()

    def _on_step_end(
        self
    ) -> None:
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
        self._dq.log_model_outputs(**self.helper_data)


def watch(
    model: Trainer, dataloaders= [], layer: Union[Module, str, None] = None, embedding_dim: Any = None
) -> None:
    """
    [`watch`] is used to hook into to the trainer
    to log to [Galileo](https://www.rungalileo.io/)
    :param trainer: Trainer object
    :return: None
    """
    print("Attaching dataquality to model and dataloaders")
    tl = TorchLogger(model, layer, embedding_dim)
    for dataloader in dataloaders:
        dataloader.collate_fn = remove_id_collate_fn_wrapper(dataloader.collate_fn)