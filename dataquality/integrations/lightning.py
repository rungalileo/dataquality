# Imports for the hook manager
from typing import Callable, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

import dataquality as dq
from dataquality.integrations.torch import TorchLogger
from dataquality.schemas.split import Split
from dataquality.schemas.torch import DimensionSlice, Layer
from dataquality.utils.patcher import PatchManager
from dataquality.utils.torch import (
    ModelHookManager,
    PatchDataloadersGlobally,
    TorchHelper,
)


class DQCallback(Callback, TorchLogger, PatchManager):
    hook_manager: ModelHookManager

    def __init__(
        self,
        last_hidden_state_layer: Optional[Layer] = None,
        embedding_dim: Optional[Union[str, DimensionSlice]] = None,
        logits_dim: Optional[Union[str, DimensionSlice]] = None,
        classifier_layer: Layer = "classifier",
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
    ):
        self.last_hidden_state_layer = last_hidden_state_layer
        self._init_dimension(embedding_dim, logits_dim)
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
        self.model = pl_module.model  # type: ignore
        self._init_helper_data(self.hook_manager, self.model)
        PatchDataloadersGlobally(self.torch_helper_data)
        self.logger_config = dq.get_data_logger().logger_config
        self._attach_hooks_to_model(
            model=self.model,
            classifier_layer=self.classifier_layer,
            last_hidden_state_layer=self.last_hidden_state_layer,
        )
        dq.set_epoch(0)

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
        dq.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the val epoch begins."""
        dq.set_split(Split.validation)
        dq.set_epoch(trainer.current_epoch)

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the test epoch begins."""
        dq.set_split(Split.test)
        dq.set_epoch(trainer.current_epoch)
