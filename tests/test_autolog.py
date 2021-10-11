import pytorch_lightning as pl
import torch

from dataquality.core.integrations.lightning import DataQualityCallback
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.lightning_model import NUM_RECORDS, NewsgroupDataset, model
from tests.test_dataquality import validate_uploaded_data


def test_lightning_autolog(cleanup_after_use) -> None:
    """
    Tests the lightning autolog config and that data is properly stored / logged
    """

    train_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("training"), batch_size=64, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        NewsgroupDataset("test"), batch_size=64, shuffle=True
    )
    trainer = pl.Trainer(
        max_epochs=1, num_sanity_val_steps=0, callbacks=[(DataQualityCallback())]
    )

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)
    ThreadPoolManager.wait_for_threads()
    validate_uploaded_data(expected_num_records=NUM_RECORDS)
