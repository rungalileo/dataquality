from datasets import DatasetDict
from transformers import Trainer

import dataquality as dq
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split


def do_train(
    trainer: Trainer,
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: bool = False,
) -> None:
    watch(trainer)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore
    dq.finish(wait=wait, create_data_embs=create_data_embs)
