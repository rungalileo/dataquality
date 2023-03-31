from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import tensorflow as tf
import vaex

import dataquality as dq
from dataquality import DataQuality
from dataquality.clients.api import ApiClient
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dq.clients.api.ApiClient,
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
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_text_keras(
    mock_valid_user: MagicMock,
    mock_bucket_names: MagicMock,
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
    labels = tf.range(10).numpy()

    with DataQuality(
        model_s,
        labels=labels,
        train_data=train_dataset,
        val_dataset=val_dataset,
        test_data=val_dataset,
        task="text_classification",
    ):
        model_s.fit(
            x=x,
            y=y,
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            epochs=2,
        )
        ThreadPoolManager.wait_for_threads()
        assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(train_dataset)
        assert len(vaex.open(f"{LOCATION}/test/0/*.hdf5")) == len(val_dataset)
