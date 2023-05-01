from typing import Callable, Generator
from unittest.mock import MagicMock, patch

from datasets import load_metric
from transformers import Trainer, TrainingArguments

import dataquality
from dataquality.clients.api import ApiClient
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, model, tokenizer
from tests.test_utils.hf_datasets_mock import mock_hf_dataset, mock_hf_dataset_repeat

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

# Training arguments and training part
metric_name = "accuracy"
batch_size = 4

args_default = TrainingArguments(
    output_dir="tmp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)


@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dataquality.core.finish, "_version_check")
@patch.object(dataquality.core.finish, "_reset_run")
@patch.object(dataquality.core.finish, "upload_dq_log_file")
@patch.object(dataquality.clients.api.ApiClient, "make_request")
@patch.object(dataquality.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dataquality.clients.api.ApiClient,
    "get_healthcheck_dq",
    return_value={
        "bucket_names": {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        },
        "minio_fqdn": "127.0.0.1:9000",
    },
)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_start_hf_tc(
    mock_valid_user: MagicMock,
    mock_get_healthcheck_dq: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    train_dataset = mock_dataset_with_ids
    val_dataset = mock_dataset_with_ids
    test_dataset = mock_dataset_with_ids

    trainer = Trainer(
        model,
        args_default,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    labels = mock_hf_dataset.features["label"].names
    with dataquality(
        trainer,
        labels=labels,
        train_data=train_dataset,
        test_data=test_dataset,
        val_dataset=val_dataset,
        task="text_classification",
    ):
        trainer.train()
        trainer.evaluate()
