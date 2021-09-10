from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data.dataloader import DataLoader

import dataquality
from dataquality import config


class DataQualityCallback(Callback):
    def __init__(self) -> None:  # dataloader_config: Dict[str, str] = None
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

    def _log_dataquality_input_data(
        self, split: str, dataloader: Union[DataLoader, Sequence[DataLoader]]
    ) -> None:
        #
        # 🔭 Logging Inputs with Galileo!
        #
        print(f"Logging data input for split {split} of epoch {self.epoch}")
        loaders = dataloader if isinstance(dataloader, Sequence) else [dataloader]
        for loader in loaders:
            dataset = (
                loader.dataset
                if split in ("test", "validation")
                else loader.dataset.datasets
            )
            assert hasattr(dataset, "dataset") or hasattr(dataset, "data"), (
                "Your Dataloader's Dataset must have a 'data' or "
                "'dataset' attribute with your data!"
            )
            data = dataset.dataset if hasattr(dataset, "dataset") else dataset.data
            for i in range(len(data)):
                dataquality.log_input_data(
                    {
                        "id": i,
                        "text": data["text"][i],
                        "gold": str(data["label"][i]),
                        "split": split,
                    }
                )

    def _log_model_outputs(self, trainer: pl.Trainer, batch: Any, split: str) -> None:
        x_idxs, x, attention_mask, y = batch
        out = trainer.model.model(x, attention_mask=attention_mask)
        probs = F.softmax(out.logits, dim=1)
        encoded_layers = trainer.model.feature_extractor(x, return_dict=False)[0]
        epoch = self.checkpoint_data["epoch"]
        print(f"Logging model outputs for split {split} epoch {epoch}")
        if x_idxs is not None:
            for i in range(len(x_idxs)):
                index = int(x_idxs[i])
                prob = probs[i].detach().cpu().numpy().tolist()
                emb = encoded_layers[i, 0].detach().cpu().numpy().tolist()

                #
                # 🔭 Logging outputs with Galileo!
                #
                dataquality.log_model_output(
                    {
                        "id": index,
                        "epoch": epoch,
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
        print("Starting train!")
        self._log_dataquality_input_data("training", trainer.train_dataloader)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert (
            config.current_project_id
        ), "You must initialize dataquality before invoking a callback!"
        assert (
            config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"
        print("Starting test!")
        self._log_dataquality_input_data("test", trainer.test_dataloaders)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert (
            config.current_project_id
        ), "You must initialize dataquality before invoking a callback!"
        assert (
            config.current_run_id
        ), "You must initialize dataquality before invoking a callback!"
        print("Starting validation!")
        self._log_dataquality_input_data("validation", trainer.val_dataloaders)

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.checkpoint_data["epoch_start"] = True

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # We need to implement epoch counting like this because there is a
        # bug in pytorch where at the end of each epoch The on_epoch_end
        # callback is called twice. This makes sure we only count it once
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
        if self.checkpoint_data["epoch_start"]:
            self.checkpoint_data["epoch"] += 1
        self.checkpoint_data["epoch_start"] = False

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, batch, "training")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, batch, "validation")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_model_outputs(trainer, batch, "test")

    # TODO: Is this okay? This will be called whenever training validation or testing
    #  end. Theres no callback for ONLY after ALL 3 end We could change this to
    #  on_test_end which we can assume occurs after train/val and is the last action
    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        print("done!")
        dataquality.finish()

    """
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('done!')
        dataquality.finish()
    """
