from typing import Any, Callable, Generator
import vaex
from unittest.mock import MagicMock, patch
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import tensorflow as tf
import dataquality as dq
from dataquality import config
from dataquality.integrations.experimental.keras import watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager

# from tests.conftest import LOCATION
from tests.test_utils.hf_datasets_mock import mock_dataset, mock_dataset_numbered
from tests.test_utils.mock_request import (
    mocked_create_project_run,
    mocked_get_project_run,
)
from tests.conftest import LOCATION

tmp_checkpoint = "tmp/tiny-distillbert"
checkpoint = "hf-internal-testing/tiny-bert-for-token-classification"
checkpoint = "distilbert-base-uncased"
# Load models locally
try:
    tokenizer = AutoTokenizer.from_pretrained(tmp_checkpoint)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.save_pretrained(tmp_checkpoint)
try:
    model = TFAutoModelForSequenceClassification.from_pretrained(tmp_checkpoint)
except Exception:
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.save_pretrained(tmp_checkpoint)


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", max_length=201, truncation=True
    )


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


# 🔭🌕 Galileo logging
mock_dataset_with_ids = mock_dataset.map(lambda x, idx: {"id": idx}, with_indices=True)
mock_dataset_with_ids_number = mock_dataset_numbered.map(
    lambda x, idx: {"id": idx}, with_indices=True
)

encoded_train_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)
encoded_test_dataset = mock_dataset_with_ids.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

encoded_train_dataset_number = mock_dataset_with_ids_number.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)
encoded_test_dataset_number = mock_dataset_with_ids_number.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

# Training arguments and training part
metric_name = "accuracy"
batch_size = 4


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_tf_watch_e2e_numbered(
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
    set_test_config(task_type=TaskType.text_classification)
    # 🔭🌕 Galileo logging
    dq.set_labels_for_run(mock_dataset_numbered.features["label"].names)
    train_dataset = mock_dataset_with_ids_number
    val_dataset = mock_dataset_with_ids_number
    test_dataset = mock_dataset_with_ids_number
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(val_dataset, split="validation")
    dq.log_dataset(test_dataset, split="test")
    train_dataset = encoded_train_dataset_number.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids", "id"],
        label_cols=["label"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    test_dataset = encoded_test_dataset_number.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids", "id"],
        label_cols=["label"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    num_epochs = 1
    # model.compile(
    # metrics=["accuracy"], optimizer="adam", loss="sparse_categorical_crossentropy"
    # )
    # model.fit(train_dataset, epochs=num_epochs)
    model_h = TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(mock_dataset_numbered.features["label"].names)
    )

    num_epochs = 2
    model_h.compile(
        metrics=["accuracy"],
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        run_eagerly=True,
    )
    watch(model_h)

    model_h.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs)

    ThreadPoolManager.wait_for_threads()
    # All data for splits should be logged
    assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(train_dataset)
    assert len(vaex.open(f"{LOCATION}/validation/0/*.hdf5")) == len(val_dataset)
    assert len(vaex.open(f"{LOCATION}/test/0/*.hdf5")) == len(test_dataset)
    # Should upload without failing on data validation or otherwise
    dq.finish()
