import copy
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

from datasets import load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality import config
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.task_type import TaskType
from tests.utils.mock_request import mocked_create_project_run, mocked_get_project_run

from .utils.hf_datasets_mock import mock_dataset

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-distilbert")
model = AutoModelForSequenceClassification.from_pretrained(
    "hf-internal-testing/tiny-random-distilbert"
)
metric = load_metric("accuracy")


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", max_length=201, truncation=True
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 🔭🌕 Galileo logging
mock_dataset_with_ids = mock_dataset.map(lambda x,idx: {"id":idx}, with_indices=True)


encoded_train_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)
encoded_test_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

# Training arguments and training part
metric_name = "accuracy"
batch_size = 16

args_default = TrainingArguments(
    output_dir="tmp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end_to_end_without_callback(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Training on a dataset"""
    args = copy.deepcopy(args_default)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_hf_watch_e2e(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    """Base case: Tests creating a new project and run"""
    global encoded_train_dataset, encoded_test_dataset
    # dq.init(task_type="text_classification")
    set_test_config(task_type=TaskType.text_classification)
    # 🔭🌕 Galileo logging
    dq.set_labels_for_run(mock_dataset.features["label"].names)
    train_dataset = mock_dataset_with_ids
    test_dataset = mock_dataset_with_ids
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(test_dataset, split="test")

    args = copy.deepcopy(args_default)
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # 🔭🌕 Galileo logging
    watch(trainer)
    trainer.train()
    dq.finish()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_remove_unused_columns(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests watch function to pass"""
    dq.init(task_type="text_classification")
    train_dataset = mock_dataset_with_ids
    test_dataset = mock_dataset_with_ids
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(test_dataset, split="test")
    dq.set_labels_for_run(mock_dataset.features["label"].names)
    assert config.current_run_id
    assert config.current_project_id

    args = copy.deepcopy(args_default)

    trainer = Trainer(
        model,
        args,
        train_dataset= encoded_train_dataset,
        eval_dataset=  encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    watch(trainer)
