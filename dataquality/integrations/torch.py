from typing import Any, Dict, List, Union

from torch.nn import Module
from transformers import Trainer

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.transformers_trainer import Layer
from dataquality.utils.torch import HookManager, remove_id_collate_fn_wrapper


class TorchLogger:
    def __init__(
        self, model: Module, model_layer: Layer = None, embedding_dim: Any = None
    ):
        self.embedding_dim = embedding_dim
        self.model = model
        self.model_layer = model_layer
        self.hook_manager = HookManager()
        self.hook_manager.attach_embedding_hook(
            model, model_layer, self._embedding_hook
        )
        self.hook_manager.attach_hook(model, self._logit_hook)
        self.helper_data: Dict[str, Any] = {}

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
        dq.log_model_outputs(**self.helper_data)


def watch(
    model: Module,
    dataloaders: List[Any] = [],
    layer: Union[Module, str, None] = None,
    embedding_dim: Any = None,
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
        dataloader.collate_fn = remove_id_collate_fn_wrapper(
            dataloader.collate_fn, tl.helper_data
        )
    tl
