# Imports for the hook manager
from typing import Any, Callable, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.schemas.torch import InputDim, Layer
from dataquality.utils.torch import (
    ModelHookManager,
    TorchHelper,
)


class DQCallback:
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
                torch_helper: TorchHelper,

        torch_helper.clear()
        self.torch_helper_data = torch_helper
        self.model_outputs_store = self.torch_helper_data.model_outputs_store

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when fit begins."""
        trainer.model

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
