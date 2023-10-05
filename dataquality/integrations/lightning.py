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


class LightningDQCallback(Callback, TorchLogger, PatchManager):
    hook_manager: ModelHookManager

    def __init__(
        self,
        classifier_layer: Layer = "classifier",
        embedding_dim: Optional[Union[str, DimensionSlice]] = None,
        logits_dim: Optional[Union[str, DimensionSlice]] = None,
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
        last_hidden_state_layer: Optional[Layer] = None,
    ):
        """
        PyTorch Lightning callback for logging model outputs to DataQuality.
        :param classifier_layer: The layer to extract the logits from
            (the output is taken as the logits and the input to the layer
            as the hidden state layer).
        :param embedding_dim: The dimension to extract from the last hidden state.
        :param logits_dim: The dimension to extract from the logits.
        :param embedding_fn: A function to apply to the embedding.
        :param logits_fn: A function to apply to the logits.
        :param last_hidden_state_layer: Optional the layer to extract
            the last hidden state from. This will overwrite the
            input of the classifier_layer regarding the hidden state.

        Example usage:

        .. code-block:: python

            train_dataset = datasets.ImageFolder("train_images",
                                                transform=load_transforms)
            train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)

            # 🔭🌕 Galileo logging
            dq.init("test_project", "test_run", task_type="image_classification")
            dq.set_labels_for_run(["labelA", "labelB"])
            dq.log_image_dataset(train_dataset, split="train")
            callback = DQCallback(classifier_layer=model.model[2])
            trainer = pl.Trainer(max_epochs=1, callbacks=[callback])
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader
            )

        """
        self.last_hidden_state_layer = last_hidden_state_layer
        self._init_dimension(embedding_dim, logits_dim)
        self.classifier_layer = classifier_layer
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn
        self.hook_manager = ModelHookManager()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit ends."""
        pass

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the train epoch begins."""
        dq.set_split(Split.training)
        dq.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the val epoch begins."""
        dq.set_split(Split.validation)
        dq.set_epoch(trainer.current_epoch)

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the test epoch begins."""
        dq.set_split(Split.test)
        dq.set_epoch(0)
