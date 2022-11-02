from typing import Any, Callable, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import vaex
from datasets import load_metric
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality import config
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCATION
from tests.test_utils.hf_datasets_mock import mock_dataset, mock_dataset_repeat
from tests.test_utils.mock_request import (
    mocked_create_project_run,
    mocked_get_project_run,
)

# Load models locally
try:
    tokenizer = AutoTokenizer.from_pretrained("tmp/testing-random-distilbert-tokenizer")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-distilbert"
    )
    tokenizer.save_pretrained("tmp/testing-random-distilbert-tokenizer")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        "tmp/testing-random-distilbert-sq"
    )
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-distilbert"
    )
    model.save_pretrained("tmp/testing-random-distilbert-sq")


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
mock_dataset_with_ids = mock_dataset.map(lambda x, idx: {"id": idx}, with_indices=True)
mock_dataset_with_ids_repeat = mock_dataset_repeat.map(
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

    trainer = Trainer(
        model,
        args_default,
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
    set_test_config(task_type=TaskType.text_classification)
    # 🔭🌕 Galileo logging
    dq.set_labels_for_run(mock_dataset.features["label"].names)
    train_dataset = mock_dataset_with_ids
    val_dataset = mock_dataset_with_ids
    test_dataset = mock_dataset_with_ids
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(val_dataset, split="validation")
    dq.log_dataset(test_dataset, split="test")

    trainer = Trainer(
        model,
        args_default,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # 🔭🌕 Galileo logging
    watch(trainer)
    trainer.train()
    trainer.predict(encoded_test_dataset)
    ThreadPoolManager.wait_for_threads()
    # All data for splits should be logged
    assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(train_dataset)
    assert len(vaex.open(f"{LOCATION}/validation/0/*.hdf5")) == len(val_dataset)
    assert len(vaex.open(f"{LOCATION}/test/0/*.hdf5")) == len(test_dataset)
    # Should upload without failing on data validation or otherwise
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
    set_test_config(task_type=TaskType.text_classification)

    train_dataset = mock_dataset_with_ids
    test_dataset = mock_dataset_with_ids
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(test_dataset, split="test")
    dq.set_labels_for_run(mock_dataset.features["label"].names)
    assert config.current_run_id
    assert config.current_project_id
    t_args = TrainingArguments(
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
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        t_args,
        train_dataset=encoded_train_dataset.with_format(
            "torch", columns=["id", "attention_mask", "input_ids", "label"]
        ),
        eval_dataset=encoded_test_dataset.with_format(
            "torch", columns=["id", "attention_mask", "input_ids", "label"]
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    assert trainer._signature_columns is None, "Signature columns should be None"
    watch(trainer)
    assert trainer._signature_columns is None, "Signature columns should be None"

    trainer.train()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_training_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests watch function to pass"""

    trainer = Trainer(
        model,
        args_default,
        train_dataset=encoded_train_dataset_repeat,
        eval_dataset=encoded_test_dataset_repeat,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def test_embedding_layer_indexing():
    """Tests that the embedding layer can be indexed with select"""
    arr = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    tensor = torch.tensor(arr)
    arr_sliced = arr[:, 0]
    arr_sliced.shape
    tensor_sliced = tensor.select(*(1, 0))
    assert arr.shape == tensor.shape, "shape must match"
    assert np.array_equal(arr, tensor.numpy()), "values must match"
    assert tensor_sliced.shape == arr_sliced.shape, "sliced shape must match"
    assert np.array_equal(arr_sliced, tensor_sliced.numpy()), "shape must be same"


historic_embeddings = None


def _embedding_hook(model: Module, model_input: Any, model_output: Any) -> None:
    """
    Hook to extract the embeddings from the model
    :param model: Model pytorch model
    :param model_input: Model input
    :param model_output: Model output
    :return: None
    """
    global historic_embeddings
    if hasattr(model_output, "last_hidden_state"):
        output_detached = model_output.last_hidden_state.detach()
        if historic_embeddings is None:
            historic_embeddings = output_detached


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_forward_hook(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    """Tests that the embedding layer is correctly logged"""
    model_seq = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-distilbert"
    )
    model_base = AutoModel.from_pretrained("hf-internal-testing/tiny-random-distilbert")

    train_sample = mock_dataset_with_ids_repeat.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    ).with_format("torch", columns=["attention_mask", "input_ids", "id"])

    dataloader = DataLoader(
        train_sample,
        batch_size=4,
    )
    for batch in dataloader:
        batch.pop("id").detach().numpy()
        pred = model_base(**batch)
        next(model_seq.children()).register_forward_hook(_embedding_hook)
        model_seq(**batch)
        embeddings = pred[0][:, 0].detach().numpy()[0]
        embeddings_cls = historic_embeddings[:, 0].detach().numpy()[0]
        break

    assert np.array_equal(embeddings, embeddings_cls), "Embeddings must be same"
