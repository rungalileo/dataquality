import copy
from typing import Callable
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
from dataquality.utils.transformers import add_id_col_to_dataset, pre_process_dataset
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


encoded_train_dataset = mock_dataset.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

encoded_test_dataset = mock_dataset.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

# Training arguments and training part
metric_name = "accuracy"
batch_size = 16

args_default = TrainingArguments(
    f"finetuned",
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


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end_to_end_with_callback(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    global encoded_train_dataset, encoded_test_dataset
    dq.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id
    # 🔭🌕 Galileo logging
    dq.init(task_type="text_classification")
    dq.get_data_logger()
    args = copy.deepcopy(args_default)
    args.remove_unused_columns = False
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
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
    train_dataset = add_id_col_to_dataset(mock_dataset)
    test_dataset = add_id_col_to_dataset(mock_dataset)
    dq.log_dataset(pre_process_dataset(train_dataset), split="train")
    dq.log_dataset(pre_process_dataset(test_dataset), split="test")
    dq.set_labels_for_run(mock_dataset.features["label"].names)
    assert config.current_run_id
    assert config.current_project_id
    dq.get_data_logger()
    args = copy.deepcopy(args_default)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    watch(trainer)
