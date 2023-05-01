from typing import Optional

from datasets import DatasetDict
from transformers import Trainer

import dataquality as dq
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split


def do_train(
    trainer: Trainer,
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> None:
    watch(trainer)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore

    inf_names = [k for k in encoded_data if k not in Split.get_valid_keys()]
    for inf_name in inf_names:
        dq.set_split(Split.inference, inference_name=inf_name)
        trainer.predict(test_dataset=encoded_data[inf_name])
    dq.finish(wait=wait, create_data_embs=create_data_embs)
