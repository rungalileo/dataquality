from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import lightning.pytorch as pl
import vaex
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# %%
from torchvision import datasets, transforms

import dataquality as dq
from dataquality.integrations.lightning import LightningDQCallback
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import TestSessionVariables

load_transforms = transforms.Compose(
    [
        transforms.Resize(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Define the PyTorch Lightning model
class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Output two values for binary classification
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)  # Use cross-entropy loss for classification
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)


model = LightningModel()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_lightning_cv_end_to_end(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """Base case: Tests creating a new project and run"""

    train_dataset = datasets.ImageFolder(
        "tests/assets/train_images", transform=load_transforms
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    # train the model (hint: here are some helpful
    # Trainer arguments for rapid idea iteration)

    set_test_config(task_type=TaskType.image_classification)
    # 🔭🌕 Galileo logging
    dq.set_labels_for_run(["labelA", "labelB"])

    dq.log_image_dataset(train_dataset, split="train")
    dq.log_image_dataset(train_dataset, split="validation")
    dq.log_image_dataset(train_dataset, split="test")
    callback = LightningDQCallback(classifier_layer=model.model[2])
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, callbacks=[callback])
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=train_dataloader,
    )
    trainer.test(dataloaders=train_dataloader)

    ThreadPoolManager.wait_for_threads()
    dq.get_data_logger().upload()
    # All data for splits should be logged
    assert len(vaex.open(f"{test_session_vars.LOCATION}/training/0/*.hdf5")) == len(
        train_dataset
    )
    assert len(vaex.open(f"{test_session_vars.LOCATION}/validation/0/*.hdf5")) == len(
        train_dataset
    )

    vaex.open(f"{test_session_vars.TEST_PATH}/training/0/data/data.*")
    dq.finish()
