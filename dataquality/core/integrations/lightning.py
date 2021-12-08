import warnings
from typing import Any, Dict, Optional, Sequence, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedDataset
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager


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
        self,
        split: Split,
        dataloader: Optional[Union[DataLoader, Sequence[DataLoader]]],
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
                config_attr = BaseGalileoDataLogger.get_datalogger_attr(dataset)
            except AttributeError:
                warnings.warn(
                    "No GalileoDataConfig found in your DataSet. Logging of input "
                    "data to Galileo will be skipped"
                )
                return

            data_config: BaseGalileoDataLogger = getattr(dataset, config_attr)
            data_config.split = split
            try:
                print("here trying to log")
                data_config.log()
                print("done logging")
            except GalileoException as e:
                warnings.warn(
                    f"Logging data inputs to Galileo could not be completed. See "
                    f"exception: {str(e)}"
                )
                return

    def _log_model_outputs(self, trainer: pl.Trainer, split: Split) -> None:
        try:
            config_attr = BaseGalileoModelLogger.get_modellogger_attr(trainer.model)
        except AttributeError:
            warnings.warn(
                "No GalileoModelConfig found for this model, logging of model "
                "config to Galileo will be skipped."
            )
            return

        model_config: BaseGalileoModelLogger = getattr(trainer.model, config_attr)
        model_config.epoch = self.checkpoint_data["epoch"]
        model_config.split = split.value

        try:
            print("here logging model outputs")
            model_config.log()
            print("here done model outputs")
        except GalileoException as e:
            warnings.warn(
                f"Logging model outputs to Galileo could not occur. "
                f"See exception: {str(e)}"
            )
            return

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        assert (
            config.current_project_id and config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_input_data(Split.training, trainer.train_dataloader)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_input_data(Split.test, trainer.test_dataloaders)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_input_data(Split.validation, trainer.val_dataloaders)

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_input_data(Split.inference, trainer.predict_dataloaders)

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
        self._log_model_outputs(trainer, Split.training)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, Split.validation)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, Split.test)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, Split.inference)

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        # So we know everything has been fully written and finished
        ThreadPoolManager.wait_for_threads()
