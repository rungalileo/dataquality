import pytorch_lightning as pl
import torch

import dataquality
from dataquality.core.integrations.lightning import DataQualityCallback
from dataquality.core.integrations.torch import log_input_data, watch
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.test_dataquality import validate_uploaded_data
from tests.utils.data_utils import validate_cleanup_data
from tests.utils.lightning_model import (
    NUM_RECORDS,
    NewsgroupDataset,
    model,
    torch_model,
)

dataquality.config.task_type = "text_classification"


def test_lightning_autolog(cleanup_after_use) -> None:
    """
    Tests the lightning autolog config and that data is properly stored / logged
    """
    train_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("training"), batch_size=NUM_RECORDS, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("test"), batch_size=NUM_RECORDS, shuffle=True
    )
    trainer = pl.Trainer(
        max_epochs=1, num_sanity_val_steps=0, callbacks=[(DataQualityCallback())]
    )

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)
    ThreadPoolManager.wait_for_threads()
    # Mock call to finish
    logger = dataquality.get_data_logger()
    logger.upload()
    validate_uploaded_data(expected_num_records=NUM_RECORDS)
    logger._cleanup()
    validate_cleanup_data()


def test_torch_autolog(cleanup_after_use) -> None:
    """Tests our watch(model) functionality for pytorch"""
    train_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("training"), batch_size=NUM_RECORDS, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("test"), batch_size=NUM_RECORDS, shuffle=True
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch_model.to(device)

    log_input_data(train_dataloader, "training")
    log_input_data(test_dataloader, "test")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, torch_model.parameters()), lr=1e-5
    )

    watch(torch_model)

    for epoch in range(2):
        torch_model.train()
        for data in train_dataloader:
            x_idxs, x, attention_mask, y = data
            x = x.to(device)
            attention_mask = attention_mask.to(device)
            y = torch.tensor(y, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            _ = torch_model(
                x, attention_mask, x_idxs=x_idxs, epoch=epoch, split="training"
            )

        with torch.no_grad():
            for data in test_dataloader:
                x_idxs, x, attention_mask, y = data

                x = x.to(device)
                attention_mask = attention_mask.to(device)
                _ = torch_model(
                    x, attention_mask, epoch=epoch, split="test", x_idxs=x_idxs
                )

    ThreadPoolManager.wait_for_threads()
    # Mock call to finish
    logger = dataquality.get_data_logger()
    logger.upload()
    validate_uploaded_data(expected_num_records=NUM_RECORDS)
    logger._cleanup()
    validate_cleanup_data()
