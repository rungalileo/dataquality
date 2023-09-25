from typing import Any, Callable, Generator
from unittest.mock import MagicMock, patch

import evaluate
import numpy as np
import torch
import vaex
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pytorch_lightning as pl
from torch.optim import AdamW


import dataquality as dq
from dataquality import config
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import HF_TEST_BERT_PATH, TestSessionVariables, model, tokenizer
from tests.test_utils.hf_datasets_mock import mock_hf_dataset, mock_hf_dataset_repeat
from tests.test_utils.mock_request import (
    mocked_create_project_run,
    mocked_get_project_run,
)
from tests.test_utils.pt_datasets_mock import CustomDatasetWithTokenizer

metric = evaluate.load("accuracy")


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=201,
        truncation=True,
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


# 🔭🌕 Galileo logging
mock_dataset_with_ids = mock_hf_dataset.map(
    lambda x, idx: {"id": idx}, with_indices=True
)
mock_dataset_with_ids_repeat = mock_hf_dataset_repeat.map(
    lambda x, idx: {"id": idx}, with_indices=True
)

encoded_train_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)
encoded_test_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

encoded_train_dataset_repeat = mock_dataset_with_ids_repeat.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)
encoded_test_dataset_repeat = mock_dataset_with_ids_repeat.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)


# Define the PyTorch Lightning model
class LightningModel(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super(LightningModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        predictions = outputs.logits.argmax(axis=1)
        labels = batch["labels"]
        acc = (predictions == labels).float().mean()
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=3e-4)


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_hf_watch_with_pt_dataset_e2e(
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
    set_test_config(task_type=TaskType.text_classification)
    # 🔭🌕 Galileo logging
    dq.set_labels_for_run(["pos", "neg"])
    train_dataset = CustomDatasetWithTokenizer(tokenizer)
    val_dataset = CustomDatasetWithTokenizer(tokenizer)
    test_dataset = CustomDatasetWithTokenizer(tokenizer)
    dq.log_dataset(mock_dataset_with_ids, split="train")
    dq.log_dataset(mock_dataset_with_ids, split="validation")
    dq.log_dataset(mock_dataset_with_ids, split="test")

    lightning_model = LightningModel(model_seq, tokenizer)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    # 🔭🌕 Galileo logging
    watch(trainer)
    trainer.train()
    trainer.predict(val_dataset)
    ThreadPoolManager.wait_for_threads()
    # All data for splits should be logged
    assert len(vaex.open(f"{test_session_vars.LOCATION}/training/0/*.hdf5")) == len(
        train_dataset
    )
    assert len(vaex.open(f"{test_session_vars.LOCATION}/validation/0/*.hdf5")) == len(
        val_dataset
    )
    assert len(vaex.open(f"{test_session_vars.LOCATION}/test/0/*.hdf5")) == len(
        test_dataset
    )

    # Should upload without failing on data validation or otherwise
    dq.finish()
