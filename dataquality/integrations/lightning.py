# Imports for the hook manager
from typing import Any, Callable, Optional, Union
from transformers.modeling_outputs import TokenClassifierOutput

from torch import Tensor
from torch.nn import Module

from dataquality.exceptions import GalileoException
from dataquality.utils.helpers import map_indices_to_ids
from dataquality.utils.patcher import PatchManager

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.schemas.torch import InputDim, Layer
from dataquality.utils.torch import (
    ModelHookManager,
    PatchDataloadersGlobally,
    TorchBaseInstance,
    TorchHelper,
)


class DQCallback(Callback, TorchBaseInstance, PatchManager):
    hook_manager: ModelHookManager

    def __init__(
        self,
        last_hidden_state_layer: Optional[Layer] = None,
        embedding_dim: Optional[InputDim] = None,
        logits_dim: Optional[InputDim] = None,
        classifier_layer: Layer = "classifier",
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
    ):
        self.last_hidden_state_layer = last_hidden_state_layer
        self.embedding_dim = embedding_dim
        self.logits_dim = logits_dim
        self.classifier_layer = classifier_layer
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn
        self.hook_manager = ModelHookManager()

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when fit begins."""
        helper_data = dq.get_model_logger().logger_config.helper_data
        self.torch_helper_data = TorchHelper()
        helper_data["torch_helper"] = self.torch_helper_data
        PatchDataloadersGlobally(self.torch_helper_data)
        self._init_dimension(self.embedding_dim, self.logits_dim)
        self.model = trainer.model
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
            self.hook_manager.attach_hooks_to_model(
                model, self._dq_embedding_hook, last_hidden_state_layer
            )
            self.hook_manager.attach_hook(model, self._dq_logit_hook_with_step_end)

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

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when fit ends."""
        pass

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the train epoch begins."""
        dq.set_split(Split.training)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the val epoch begins."""
        dq.set_split(Split.validation)

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the test epoch begins."""
        dq.set_split(Split.test)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """
        pass
