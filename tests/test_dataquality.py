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
    cleanup_after_use: Callable, set_test_config: Callable, input_data: Callable
) -> None:
    """
    Tests that logging metadata columns only attach to the splits we log them for
    """
    training_data = input_data(meta={"training_meta": [1, 2]})
    dataquality.set_labels_for_run(training_data["labels"])
    dataquality.log_input_data(**training_data)
    test_data = input_data(split="test", meta={"test_meta": ["foo", "bar"]})
    dataquality.log_input_data(**test_data)

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

    assert "training_meta" in train_data.get_column_names()
    assert "training_meta" not in test_data.get_column_names()

    assert "test_meta" in test_data.get_column_names()
    assert "test_meta" not in train_data.get_column_names()

    assert sorted(train_data["training_meta"].tolist()) == [1, 2]
    assert sorted(test_data["test_meta"].tolist()) == ["bar", "foo"]


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
        with pytest.raises(GalileoException) as e:
            c.upload()

        assert str(e.value).startswith(
            "It seems as though you do not have unique ids in this split."
        )
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_logging_inference_run(
    cleanup_after_use: Callable, set_test_config: Callable, input_data: Callable
) -> None:
    """
    Tests that logging metadata columns only attach to the splits we log them for
    """
    inference_data = input_data(
        split="inference", meta={"inference_meta_1": [3.14, 42]}
    )
    dataquality.log_input_data(**inference_data)
    inference_data = input_data(split="inference", inference_name="last-week-customers")
    dataquality.log_input_data(**inference_data)

    dataquality.set_split("inference", inference_name="all-customers")
    emb_1 = np.random.rand(2, 100)
    logits_1 = np.random.rand(2, 5)
    output_data = {
        "emb": emb_1,
        "logits": logits_1,
        "ids": [1, 2],
    }
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("inference", inference_name="last-week-customers")
    emb_2 = np.random.rand(2, 100)
    logits_2 = np.random.rand(2, 5)
    output_data = {
        "emb": emb_2,
        "logits": logits_2,
        "ids": [1, 2],
    }
    dataquality.log_model_outputs(**output_data)

    ThreadPoolManager.wait_for_threads()
    dataquality.get_data_logger().upload()

    inference_data_1 = vaex.open(f"{TEST_PATH}/inference/all-customers/data/data.hdf5")
    inference_data_2 = vaex.open(
        f"{TEST_PATH}/inference/last-week-customers/data/data.hdf5"
    )

    assert "inference_meta_1" in inference_data_1.get_column_names()
    assert "inference_meta_1" not in inference_data_2.get_column_names()
    assert sorted(inference_data_1["inference_meta_1"].tolist()) == [3.14, 42]

    inference_emb_1 = vaex.open(f"{TEST_PATH}/inference/all-customers/emb/emb.hdf5")
    inference_emb_2 = vaex.open(
        f"{TEST_PATH}/inference/last-week-customers/emb/emb.hdf5"
    )

    assert (inference_emb_1.emb.to_numpy() == emb_1).all()
    assert (inference_emb_2.emb.to_numpy() == emb_2).all()

    inference_prob_1 = vaex.open(f"{TEST_PATH}/inference/all-customers/prob/prob.hdf5")
    inference_prob_2 = vaex.open(
        f"{TEST_PATH}/inference/last-week-customers/prob/prob.hdf5"
    )

    assert "logits" not in inference_prob_1.get_column_names()
    assert "logits" not in inference_prob_2.get_column_names()
    assert "prob" in inference_prob_1.get_column_names()
    assert "prob" in inference_prob_2.get_column_names()

    assert (
        inference_prob_1.prob.to_numpy()
        == dataquality.get_model_logger()().convert_logits_to_probs(logits_1)
    ).all()
    assert (
        inference_prob_2.prob.to_numpy()
        == dataquality.get_model_logger()().convert_logits_to_probs(logits_2)
    ).all()


def test_logging_train_test_inference(
    cleanup_after_use: Callable, set_test_config: Callable, input_data: Callable
) -> None:
    """
    Tests that logging metadata columns only attach to the splits we log them for
    """
    dataquality.set_labels_for_run(["APPLE", "ORANGE"])
    training_data = input_data(meta={"training_meta": [1.414, 123]})
    dataquality.log_input_data(**training_data)
    test_data = input_data(split="test", meta={"test_meta": [3.14, 42]})
    dataquality.log_input_data(**test_data)
    inference_data = input_data(split="inference")
    dataquality.log_input_data(**inference_data)

    dataquality.set_split("training")
    train_emb = np.random.rand(2, 100)
    train_logits = np.random.rand(2, 5)
    output_data = {"emb": train_emb, "logits": train_logits, "ids": [1, 2], "epoch": 0}
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("test")
    test_emb = np.random.rand(2, 100)
    test_logits = np.random.rand(2, 5)
    output_data = {"emb": test_emb, "logits": test_logits, "ids": [1, 2], "epoch": 0}
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("inference", inference_name="all-customers")
    inf_emb = np.random.rand(2, 100)
    inf_logits = np.random.rand(2, 5)
    output_data = {
        "emb": inf_emb,
        "logits": inf_logits,
        "ids": [1, 2],
    }
    dataquality.log_model_outputs(**output_data)

    ThreadPoolManager.wait_for_threads()
    dataquality.get_data_logger().upload()

    train_data = vaex.open(f"{TEST_PATH}/training/0/data/data.hdf5")
    test_data = vaex.open(f"{TEST_PATH}/test/0/data/data.hdf5")
    inference_data = vaex.open(f"{TEST_PATH}/inference/all-customers/data/data.hdf5")

    assert "training_meta" in train_data.get_column_names()
    assert sorted(train_data["training_meta"].tolist()) == [1.414, 123]
    assert "test_meta" in test_data.get_column_names()
    assert sorted(test_data["test_meta"].tolist()) == [3.14, 42]

    train_emb_data = vaex.open(f"{TEST_PATH}/training/0/emb/emb.hdf5")
    test_emb_data = vaex.open(f"{TEST_PATH}/test/0/emb/emb.hdf5")
    inference_emb_data = vaex.open(f"{TEST_PATH}/inference/all-customers/emb/emb.hdf5")

    assert (train_emb_data.emb.to_numpy() == train_emb).all()
    assert (test_emb_data.emb.to_numpy() == test_emb).all()
    assert (inference_emb_data.emb.to_numpy() == inf_emb).all()

    train_prob_data = vaex.open(f"{TEST_PATH}/training/0/prob/prob.hdf5")
    test_prob_data = vaex.open(f"{TEST_PATH}/test/0/prob/prob.hdf5")
    inference_prob_data = vaex.open(
        f"{TEST_PATH}/inference/all-customers/prob/prob.hdf5"
    )

    assert "logits" not in train_prob_data.get_column_names()
    assert "logits" not in test_prob_data.get_column_names()
    assert "logits" not in inference_prob_data.get_column_names()

    assert (
        train_prob_data.prob.to_numpy()
        == dataquality.get_model_logger()().convert_logits_to_probs(train_logits)
    ).all()
    assert (
        test_prob_data.prob.to_numpy()
        == dataquality.get_model_logger()().convert_logits_to_probs(test_logits)
    ).all()
    assert (
        inference_prob_data.prob.to_numpy()
        == dataquality.get_model_logger()().convert_logits_to_probs(inf_logits)
    ).all()
