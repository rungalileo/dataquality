import warnings
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedDataset
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import dataquality
from dataquality import config

from .config import (
    GalileoDataConfig,
    GalileoModelConfig,
    get_dataconfig_attr,
    get_modelconfig_attr,
)


class DataQualityCallback(Callback):
    """
    The PyTorch Lightning Callback for Galileo's dataquality module. This module
    handles the logging of input data and model configs to Galileo. It makes the
    following assumptions:
    * Your model class has an attribute containing a valid GalileoModelConfig
    * You have a DataSet that extends PyTorch's DataSet and has an attribute containing
    a valid GalileoDataConfig
    """

    def __init__(self) -> None:
        self.checkpoint_data = {"epoch_start": False, "epoch": 0}

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        callback_state: Any,
    ) -> None:
        self.checkpoint_data = callback_state

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Any
    ) -> Dict[str, Any]:
        return self.checkpoint_data.copy()

    def _log_input_data(
        self, split: str, dataloader: Optional[Union[DataLoader, Sequence[DataLoader]]]
    ) -> None:
        #
        # 🔭 Logging Inputs with Galileo!
        #
        if dataloader is None:
            warnings.warn(f"No {split} dataset available. Cannot log to Galileo")
            return
        loaders = dataloader if isinstance(dataloader, Sequence) else [dataloader]
        for loader in loaders:
            dataset = loader.dataset
            if isinstance(dataset, CombinedDataset):
                datasets = dataset.datasets
                if isinstance(datasets, Dataset):
                    dataset = datasets
                elif isinstance(datasets, Sequence):
                    warnings.warn(
                        "Galileo currently supports logging one dataset. "
                        "The first dataset in your CombinedDataset will be "
                        "logged"
                    )
                    dataset = datasets[0]
                else:
                    warnings.warn(
                        f"Your {split} dataset is of an unsupported type "
                        f"{type(datasets)}. Currently a single dataset is "
                        f"supported for logging."
                    )
                    return
            try:
                config_attr = get_dataconfig_attr(dataset)
            except AttributeError:
                warnings.warn(
                    "No GalileoDataConfig found in your DataSet. Logging of input "
                    "data to Galileo will be skipped"
                )
                return

            data_config: GalileoDataConfig = getattr(dataset, config_attr)
            try:
                data_config.validate()
            except AssertionError as e:
                warnings.warn(
                    f"The provided GalileoDataConfig is invalid. Logging to "
                    f"Galileo will be skipped. Config Error: {str(e)}"
                )
                return

            ids = data_config.ids if data_config.ids else range(len(data_config.text))
            for idx, text, label in zip(ids, data_config.text, data_config.labels):
                dataquality.log_input_data(
                    {
                        "id": idx,
                        "text": text,
                        "gold": str(label),
                        "split": split,
                    }
                )

    def _log_model_outputs(self, trainer: pl.Trainer, split: str) -> None:

        try:
            config_attr = get_modelconfig_attr(trainer.model)
        except AttributeError:
            warnings.warn(
                "No GalileoModelConfig found for this model, logging of model "
                "config to Galileo will be skipped."
            )
            return

        model_config: GalileoModelConfig = getattr(trainer.model, config_attr)
        try:
            model_config.validate()
        except AssertionError as e:
            warnings.warn(
                f"The provided GalileoModelConfig is invalid. Logging to "
                f"Galileo will be skipped. Config Error: {str(e)}"
            )
            return

        for id, prob, emb in zip(
            model_config.ids, model_config.probs, model_config.emb
        ):

            #
            # 🔭 Logging outputs with Galileo!
            #
            dataquality.log_model_output(
                {
                    "id": id,
                    "epoch": self.checkpoint_data["epoch"],
                    "split": split,
                    "emb": emb,
                    "prob": prob,
                    "pred": str(int(np.argmax(prob))),
                }
            )

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert (
            config.current_project_id
        ), "You must initialize dataquality before invoking a callback!"
        assert (
            config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"
        self._log_input_data("training", trainer.train_dataloader)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert (
            config.current_project_id
        ), "You must initialize dataquality before invoking a callback!"
        assert (
            config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"
        self._log_input_data("test", trainer.test_dataloaders)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert (
            config.current_project_id
        ), "You must initialize dataquality before invoking a callback!"
        assert (
            config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"
        self._log_input_data("validation", trainer.val_dataloaders)

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.checkpoint_data["start_of_new_epoch"] = True

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # We need to implement epoch counting like this because there is a
        # bug in pytorch where at the end of each epoch The on_epoch_end
        # callback is called twice. This makes sure we only count it once
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
        if self.checkpoint_data["start_of_new_epoch"]:
            self.checkpoint_data["epoch"] += 1
        self.checkpoint_data["start_of_new_epoch"] = False

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, "training")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, "validation")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, "test")

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        # Don't cleanup because might call test after fit many times. We'd want to
        # append in that case
        dataquality.finish(cleanup=False)
