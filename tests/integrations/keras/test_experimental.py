from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import tensorflow as tf
import vaex
from keras.engine import data_adapter
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TFAutoModelForSequenceClassification,
)

import dataquality as dq
from dataquality.integrations.experimental.keras import unwatch, watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCATION

# from tests.conftest import LOCATION
from tests.test_utils.hf_datasets_mock import mock_dataset, mock_dataset_numbered

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
mock_dataset_with_ids = mock_dataset.map(
    lambda x, idx: {"id": idx}, with_indices=True
).select(range(7))
mock_dataset_with_ids_number = mock_dataset_numbered.map(
    lambda x, idx: {"id": idx}, with_indices=True
).select(range(7))

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


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_hf_watch_e2e_numbered(
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
    batch_size = 5
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
    assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(
        encoded_train_dataset_number
    )
    assert len(vaex.open(f"{LOCATION}/validation/0/*.hdf5")) == len(
        encoded_test_dataset_number
    )
    # Should upload without failing on data validation or otherwise
    dq.finish()


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
    set_test_config(task_type=TaskType.text_classification)
    dataset_len = 13
    val_dataset_len = 14
    dq.set_labels_for_run(tf.range(10).numpy())
    train_dataset = pd.DataFrame(
        {
            "text": ["x"] * dataset_len,
            "label": [1] * dataset_len,
            "id": tf.range(dataset_len).numpy(),
        }
    )
    val_dataset = pd.DataFrame(
        {
            "text": ["x"] * val_dataset_len,
            "label": [1] * val_dataset_len,
            "id": tf.range(val_dataset_len).numpy(),
        }
    )
    dq.log_dataset(train_dataset, split="train")
    dq.log_dataset(val_dataset, split="validation")
    dq.log_dataset(val_dataset, split="test")

    batch_size = 8
    input_data = (dataset_len, 28, 28, 1)
    input_shape = (28, 28, 1)
    num_classes = 10
    # for the model read further in the tensorflow tests
    # reuters_mlp_benchmark_test.py
    # mnist_conv_benchmark_test.py
    model_s = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier"),
        ]
    )
    model_s.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    x = tf.ones(input_data)
    y_ = tf.ones(dataset_len, dtype="int32")
    y = tf.one_hot(y_, depth=10)
    val_input_data = (val_dataset_len, 28, 28, 1)
    val_x = tf.ones(val_input_data)
    val_y_ = tf.ones(val_dataset_len, dtype="int32")
    val_y = tf.one_hot(val_y_, depth=10)
    watch(model_s, "classifier")
    model_s.fit(
        x=x,
        y=y,
        validation_data=(val_x, val_y),
        batch_size=batch_size,
        epochs=2,
    )

    model_s.predict(x=x, batch_size=batch_size)

    ThreadPoolManager.wait_for_threads()
    assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(x)
    assert len(vaex.open(f"{LOCATION}/validation/0/*.hdf5")) == len(val_x)
    assert len(vaex.open(f"{LOCATION}/test/0/*.hdf5")) == len(x)
    unwatch(model_s)
    dq.finish()
    model_s.fit(
        x=x,
        y=y,
        validation_data=(val_x, val_y),
        batch_size=batch_size,
        epochs=1,
    )


def test_create_epoch_data() -> None:
    dh_kwargs = {"x": tf.range(13), "batch_size": 4, "epochs": 2}
    e_model = tf.keras.Sequential([])
    e_model.compile(loss="mse", run_eagerly=True)
    data_handler = data_adapter.DataHandler(
        model=e_model,
        **dh_kwargs,
    )
    for epoch, iterator in data_handler.enumerate_epochs():
        for step in data_handler.steps():
            assert isinstance(epoch, int)
            assert isinstance(step, int)
            assert len(next(iterator))


def test_model() -> None:
    layer1 = tf.keras.layers.Embedding(output_dim=2, input_dim=7)
    # create a classifier layer
    layer2 = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, name="classifier"
    )
    # create a sequential model
    model = tf.keras.models.Sequential([layer1, layer2])
    layer1.set_weights(
        [tf.constant([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])]
    )
    model.run_eagerly = True
    input_list = tf.constant([0, 1, 2, 3, 4, 5, 6], dtype="int32")
    # set the weights in classifier layer (layer2) so it will always predict 1
    layer2.set_weights([tf.constant([[0.5], [0.5]])])
    model.predict(input_list)
