from random import random
from typing import Callable

import numpy as np
import pytest
import vaex

import dataquality
import dataquality.core._config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import TEST_PATH
from tests.utils.data_utils import (
    NUM_LOGS,
    NUM_RECORDS,
    _log_text_classification_data,
    validate_cleanup_data,
    validate_uploaded_data,
)

MAX_META_COLS = BaseGalileoDataLogger.MAX_META_COLS
MAX_STR_LEN = BaseGalileoDataLogger.MAX_STR_LEN


def test_threaded_logging_and_upload(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    num_records = 32
    num_logs = 20
    num_emb = 50
    _log_text_classification_data(
        num_records=num_records, num_logs=num_logs, num_emb=num_emb
    )
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger("text_classification")
        c.validate_labels()
        c.upload()
        validate_uploaded_data(num_records * num_logs)
        c._cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_multi_label_logging(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    set_test_config(task_type=TaskType.text_multi_label)
    num_records = 32
    num_logs = 20
    num_emb = 50
    _log_text_classification_data(
        num_records=num_records, num_logs=num_logs, num_emb=num_emb, multi_label=True
    )
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger()
        c.validate_labels()
        c.upload()
        validate_uploaded_data(num_records * num_logs, multi_label=True)
        c._cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_metadata_logging(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests that logging metadata columns persist
    """
    meta_cols = ["test1", "meta2"]
    meta = {}
    for i in meta_cols:
        meta[i] = [random() for _ in range(NUM_RECORDS * NUM_LOGS)]
    _log_text_classification_data(meta=meta)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger()
        c.upload()
        validate_uploaded_data(meta_cols=meta_cols)
        c._cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_metadata_logging_different_splits(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests that logging metadata columns only attach to the splits we log them for
    """
    dataquality.set_labels_for_run(["APPLE", "ORANGE"])
    input_data = {
        "text": ["sentence_1", "sentence_2"],
        "labels": ["APPLE", "ORANGE"],
        "split": "training",
        "meta": {"train_meta_1": [1, 2]},
        "ids": [1, 2],
    }
    dataquality.log_input_data(**input_data)
    input_data["split"] = "test"
    input_data["meta"] = {"test_meta_1": ["foo", "bar"]}
    dataquality.log_input_data(**input_data)

    output_data = {
        "emb": np.random.rand(2, 100),
        "logits": np.random.rand(2, 5),
        "ids": [1, 2],
        "split": "training",
        "epoch": 0,
    }
    dataquality.log_model_outputs(**output_data)
    output_data["split"] = "test"
    dataquality.log_model_outputs(**output_data)
    dataquality.get_data_logger().upload()

    train_data = vaex.open(f"{TEST_PATH}/training/0/data/data.hdf5")
    test_data = vaex.open(f"{TEST_PATH}/test/0/data/data.hdf5")

    assert "train_meta_1" in train_data.get_column_names()
    assert "train_meta_1" not in test_data.get_column_names()

    assert "test_meta_1" in test_data.get_column_names()
    assert "test_meta_1" not in train_data.get_column_names()

    assert sorted(train_data["train_meta_1"].tolist()) == [1, 2]
    assert sorted(test_data["test_meta_1"].tolist()) == ["bar", "foo"]


def test_metadata_logging_invalid(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests our metadata logging validation
    """
    meta = {
        "test1": [random() for _ in range(NUM_RECORDS * NUM_LOGS)],
        "meta2": [random() for _ in range(NUM_RECORDS * NUM_LOGS)],
        "bad_attr": [
            "te" * MAX_STR_LEN for _ in range(NUM_RECORDS * NUM_LOGS)
        ],  # String too long
        "another_bad_attr": ["test", "test", "test"],  # Wrong number of values
        # Right length, but can't contain a list
        "bad_attr_3": [[1]] + [random() for _ in range(NUM_RECORDS * NUM_LOGS - 1)],
        "gold": [random() for _ in range(NUM_RECORDS * NUM_LOGS)],  # Reserved key
    }

    # Too many metadata columns
    for i in range(MAX_META_COLS):
        meta[f"attr_{i}"] = [random() for _ in range(NUM_RECORDS * NUM_LOGS)]

    _log_text_classification_data(meta=meta)
    valid_meta_cols = ["test1", "meta2"]
    valid_meta_cols += [f"attr_{i}" for i in range(44)]
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        c = dataquality.get_data_logger("text_classification")
        c.upload()
        validate_uploaded_data(meta_cols=valid_meta_cols)
        c._cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_logging_duplicate_ids(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    Tests that logging duplicate ids triggers a failure
    """
    num_records = 50
    _log_text_classification_data(num_records=num_records, unique_ids=False)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger("text_classification")
        with pytest.raises(GalileoException):
            c.upload()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()
