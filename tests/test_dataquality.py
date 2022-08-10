import os
import time
from random import random
from typing import Callable
from unittest import mock
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import vaex

import dataquality
import dataquality as dq
import dataquality.clients.api
import dataquality.core._config
import dataquality.core.finish
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)
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
    num_embs = 50
    _log_text_classification_data(
        num_records=num_records, num_logs=num_logs, num_embs=num_embs
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
    num_embs = 50
    _log_text_classification_data(
        num_records=num_records, num_logs=num_logs, num_embs=num_embs, multi_label=True
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
    dataquality.log_data_samples(**training_data)
    test_data = input_data(split="test", meta={"test_meta": ["foo", "bar"]})
    dataquality.log_data_samples(**test_data)

    output_data = {
        "embs": np.random.rand(2, 100),
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
    dataquality.log_data_samples(**inference_data)
    inference_data = input_data(split="inference", inference_name="last-week-customers")
    dataquality.log_data_samples(**inference_data)

    dataquality.set_split("inference", inference_name="all-customers")
    embs_1 = np.random.rand(2, 100)
    logits_1 = np.random.rand(2, 5)
    output_data = {
        "embs": embs_1,
        "logits": logits_1,
        "ids": [1, 2],
    }
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("inference", inference_name="last-week-customers")
    embs_2 = np.random.rand(2, 100)
    logits_2 = np.random.rand(2, 5)
    output_data = {
        "embs": embs_2,
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

    assert (inference_emb_1.emb.to_numpy() == embs_1).all()
    assert (inference_emb_2.emb.to_numpy() == embs_2).all()

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
    dataquality.log_data_samples(**training_data)
    test_data = input_data(split="test", meta={"test_meta": [3.14, 42]})
    dataquality.log_data_samples(**test_data)
    inference_data = input_data(split="inference")
    dataquality.log_data_samples(**inference_data)

    dataquality.set_split("training")
    train_embs = np.random.rand(2, 100)
    train_logits = np.random.rand(2, 5)
    output_data = {
        "embs": train_embs,
        "logits": train_logits,
        "ids": [1, 2],
        "epoch": 0,
    }
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("test")
    test_embs = np.random.rand(2, 100)
    test_logits = np.random.rand(2, 5)
    output_data = {"embs": test_embs, "logits": test_logits, "ids": [1, 2], "epoch": 0}
    dataquality.log_model_outputs(**output_data)
    dataquality.set_split("inference", inference_name="all-customers")
    inf_embs = np.random.rand(2, 100)
    inf_logits = np.random.rand(2, 5)
    output_data = {
        "embs": inf_embs,
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

    assert (train_emb_data.emb.to_numpy() == train_embs).all()
    assert (test_emb_data.emb.to_numpy() == test_embs).all()
    assert (inference_emb_data.emb.to_numpy() == inf_embs).all()

    train_prob_data = vaex.open(f"{TEST_PATH}/training/0/prob/prob.hdf5")
    test_prob_data = vaex.open(f"{TEST_PATH}/test/0/prob/prob.hdf5")
    inference_prob_data = vaex.open(
        f"{TEST_PATH}/inference/all-customers/prob/prob.hdf5"
    )

    assert "training_meta" in train_data.get_column_names()
    assert "training_meta" not in test_data.get_column_names()
    assert "training_meta" not in inference_data.get_column_names()

    assert "test_meta" not in train_data.get_column_names()
    assert "test_meta" in test_data.get_column_names()
    assert "test_meta" not in inference_data.get_column_names()

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


def test_prob_only(set_test_config) -> None:
    """Tests that the last_epoch is determined at the split level"""
    logger = dataquality.get_data_logger()
    logger.logger_config.last_epoch = 5
    train_split_runs = ["0", "1", "5", "3", "4"]
    val_split_runs = ["0", "1", "3", "4"]
    test_split_runs = ["0"]

    assert logger.prob_only(train_split_runs, "training", "0")
    assert not logger.prob_only(train_split_runs, "training", "5")
    assert logger.prob_only(val_split_runs, "validation", "0")
    assert not logger.prob_only(val_split_runs, "validation", "5")
    assert not logger.prob_only(test_split_runs, "test", "0")


@mock.patch.object(dataquality.clients.api.ApiClient, "get_presigned_url")
@mock.patch.object(
    dataquality.clients.objectstore.ObjectStore, "_upload_file_from_local"
)
def test_log_invalid_model_outputs(
    mock_upload_from_local: mock.MagicMock,
    mock_presigned_url: mock.MagicMock,
    cleanup_after_use: Callable,
    set_test_config: Callable,
    input_data: Callable,
) -> None:
    """Validate that we interrupt the main process if issues occur while logging"""
    dataquality.set_labels_for_run(["APPLE", "ORANGE"])
    training_data = input_data(meta={"training_meta": [1.414, 123]})
    dataquality.log_data_samples(**training_data)

    dataquality.set_split("training")
    train_embs = np.random.rand(1, 100)  # Not enough embeddings
    train_logits = np.random.rand(2, 5)
    output_data = {
        "embs": train_embs,
        "logits": train_logits,
        "ids": [1, 2],
        "epoch": 0,
    }
    with pytest.raises(GalileoException) as e:
        mock_presigned_url.return_value = "https://google.com"
        mock_upload_from_local.return_value = None
        dataquality.log_model_outputs(**output_data)
        time.sleep(1)  # ensure the first one records a failure
        dataquality.log_model_outputs(**output_data)

    assert dataquality.get_model_logger().logger_config.exception != ""
    assert str(e.value).startswith("An issue occurred while logging model outputs.")


@mock.patch.object(dataquality.clients.api.ApiClient, "get_presigned_url")
@mock.patch.object(
    dataquality.clients.objectstore.ObjectStore, "_upload_file_from_local"
)
def test_log_invalid_model_outputs_final_thread(
    mock_upload_from_local: mock.MagicMock,
    mock_presigned_url: mock.MagicMock,
    cleanup_after_use: Callable,
    set_test_config: Callable,
    input_data: Callable,
) -> None:
    """Validate that we error on finish if issues occur while logging"""
    assert dataquality.get_model_logger().logger_config.exception == ""
    dataquality.set_labels_for_run(["APPLE", "ORANGE"])
    training_data = input_data(meta={"training_meta": [1.414, 123]})
    dataquality.log_data_samples(**training_data)

    dataquality.set_split("training")
    train_embs = np.random.rand(1, 100)  # Not enough embeddings
    train_logits = np.random.rand(2, 5)
    output_data = {
        "embs": train_embs,
        "logits": train_logits,
        "ids": [1, 2],
        "epoch": 0,
    }

    mock_presigned_url.return_value = "https://google.com"
    dataquality.log_model_outputs(**output_data)
    with pytest.raises(GalileoException) as e:
        mock_upload_from_local.return_value = None
        dataquality.get_data_logger().upload()

    assert dataquality.get_model_logger().logger_config.exception != ""
    assert str(e.value).startswith("An issue occurred while logging model outputs.")


def test_log_outputs_binary(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    """Tests that binary logging of logits works"""
    set_test_config(task_type="text_classification")

    logger = TextClassificationModelLogger(
        embs=np.random.rand(10, 100),  # 10 samples, 100 emb per sample
        logits=np.random.rand(10, 1),  # 10 samples
        ids=list(range(10)),
        split="training",
        epoch=0,
    )
    logger._log()

    assert not hasattr(logger, "logits")
    assert logger.probs.shape == (10, 2)  # samples X tasks X classes per task
    for sample_probs in logger.probs:
        assert np.isclose(np.sum(sample_probs), 1.0)


def test_calls_noop() -> None:
    os.environ["GALILEO_DISABLED"] = "True"
    c = dataquality.core._config.set_config()
    assert c.api_url == "http://"
    with mock.patch("dataquality.core.log.log_data_samples") as mock_log:
        dataquality.log_data_samples(texts=["test"], labels=["1"], ids=[1])
        mock_log.assert_not_called()
    del os.environ["GALILEO_DISABLED"]


@mock.patch.object(dataquality.core._config.Config, "update_file_config")
@mock.patch.object(dataquality.core.finish.os, "rename")
@mock.patch.object(dataquality.clients.api.ApiClient, "get_project_run")
@mock.patch.object(dataquality.clients.api.ApiClient, "get_project")
@mock.patch.object(dataquality.clients.api.ApiClient, "delete_run")
@mock.patch.object(dataquality.clients.api.ApiClient, "create_run")
def test_reset_run_new_task_type(
    mock_create_run: MagicMock,
    mock_delete_run: MagicMock,
    mock_get_project: MagicMock,
    mock_get_project_run: MagicMock,
    mock_rename: MagicMock,
    mock_update_file_config: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Callable,
) -> None:
    """Tests that resetting a run with a new task type changes the task type"""
    set_test_config(task_type=TaskType.text_classification)
    mock_get_project.return_value = {"name": "project_name"}
    # task_type 0 from API is text_classification
    mock_get_project_run.return_value = {"name": "run_name", "task_type": 0}
    pid = uuid4()
    rid = uuid4()
    mock_create_run.return_value = {"id": rid}
    dataquality.core.finish._reset_run(pid, rid, TaskType.text_multi_label)

    assert mock_create_run.called_once_with(pid, rid, TaskType.text_multi_label)
    assert dataquality.config.current_run_id == rid


@pytest.mark.parametrize("last_epoch", [1, 3, None, 10])
@mock.patch.object(dataquality.core.finish, "_version_check")
@mock.patch.object(dataquality.core.finish, "_reset_run")
@mock.patch.object(dataquality.core.finish, "upload_dq_log_file")
@mock.patch.object(dataquality.clients.api.ApiClient, "make_request")
@mock.patch.object(dataquality.loggers.base_logger.BaseGalileoLogger, "_cleanup")
@mock.patch.object(
    dataquality.loggers.data_logger.text_classification.TextClassificationDataLogger,
    "upload_in_out_frames",
)
def test_finish_last_epoch(
    mock_upload_frames: MagicMock,
    mock_cleanup: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Callable,
    last_epoch: int,
) -> None:
    text_inputs = ["sample1", "sample2", "sample3"] * 30
    gold = ["A", "C", "B"] * 30
    ids = list(range(90))

    dq.set_labels_for_run(["A", "B", "C"])
    dq.log_data_samples(texts=text_inputs, labels=gold, split="validation", ids=ids)
    dq.log_data_samples(texts=text_inputs, labels=gold, split="training", ids=ids)

    embs = np.random.rand(3, 100)
    logits = np.random.rand(3, 3)
    last_logged_epoch = 0
    for i in range(0, 15, 3):
        ids = list(range(i, i + 3))
        dq.log_model_outputs(
            embs=embs, logits=logits, ids=ids, split="training", epoch=i // 3
        )
        last_logged_epoch += 1
    dq.finish(last_epoch=last_epoch, wait=False)
    # if max epoch is None, all should be uploaded.
    # if max epoch is greater than the max logged, all should be uploaded
    # if max epoch is less than max logged, only log up to and including max_epoch
    max_uploaded = (
        min(last_logged_epoch, last_epoch + 1) if last_epoch else last_logged_epoch
    )
    for i, call in enumerate(mock_upload_frames.call_args_list):
        # python 3.7 vs 3.8+ compatibility
        try:  # python 3.8+
            assert int(call.args[-1]) == i
        except TypeError:  # python 3.7
            assert int(call[0][-1]) == i
    assert mock_upload_frames.call_count == max_uploaded


@mock.patch("dataquality.core.log.np.random.rand")
def test_log_model_outputs_exclude_embs(
    mock_numpy: mock.MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Callable,
) -> None:
    """
    Tests that logging model outputs with missing embeddings and exclude embs
    generates random embs
    """
    # Use random.randn to avoid clash with numpy.random.rand mock
    mock_numpy.return_value = np.random.randn(10, 2)
    output_data = {
        "embs": None,
        "logits": np.random.randn(10, 5),
        "ids": list(range(1, 11)),
        "split": "training",
        "epoch": 0,
        "exclude_embs": True,
    }
    dataquality.log_model_outputs(**output_data)
    # Assert that random embeddings were created with dim 2
    mock_numpy.assert_called_once_with(10, 2)


def test_log_model_outputs_missing_embs_exclude_emb_false(
    set_test_config: Callable,
) -> None:
    """
    Tests that logging model outputs with missing embeddings raises an error
    """
    output_data = {
        "embs": None,
        "logits": np.random.rand(2, 5),
        "ids": [1, 2],
        "split": "training",
        "epoch": 0,
    }

    with pytest.raises(AssertionError) as e:
        dataquality.log_model_outputs(**output_data)
    assert str(e.value) == "embs can be None if and only if exclude_embs is True"


def test_log_model_outputs_with_embs_exclude_emb_true(
    set_test_config: Callable,
) -> None:
    """
    Tests that logging model outputs embeddings with exclude_emb raises an error
    """
    output_data = {
        "embs": np.random.rand(2, 100),
        "logits": np.random.rand(2, 5),
        "ids": [1, 2],
        "split": "training",
        "epoch": 0,
        "exclude_embs": True,
    }
    with pytest.raises(AssertionError) as e:
        dataquality.log_model_outputs(**output_data)

    assert str(e.value) == "embs can be None if and only if exclude_embs is True"
