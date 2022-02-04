import os
from importlib import reload
from random import random

import pytest

import dataquality
import dataquality.core._config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.utils.data_utils import (
    NUM_LOGS,
    NUM_RECORDS,
    _log_data,
    validate_cleanup_data,
    validate_uploaded_data,
)

MAX_META_COLS = BaseGalileoDataLogger.MAX_META_COLS
MAX_STR_LEN = BaseGalileoDataLogger.MAX_STR_LEN


def test_threaded_logging_and_upload(cleanup_after_use) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    dataquality.config.task_type = "text_classification"
    num_records = 32
    num_logs = 200
    num_emb = 50
    _log_data(num_records=num_records, num_logs=num_logs, num_emb=num_emb)
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


def test_multi_label_logging(cleanup_after_use) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    dataquality.config.task_type = "text_multi_label"
    num_records = 32
    num_logs = 200
    num_emb = 50
    _log_data(
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


def test_metadata_logging(cleanup_after_use) -> None:
    """
    Tests that logging metadata columns persist
    """
    dataquality.config.task_type = "text_classification"
    meta_cols = ["test1", "meta2"]
    meta = {}
    for i in meta_cols:
        meta[i] = [random() for _ in range(NUM_RECORDS * NUM_LOGS)]
    _log_data(meta=meta)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger("text_classification")
        c.upload()
        validate_uploaded_data(meta_cols=meta_cols)
        c._cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_metadata_logging_invalid(cleanup_after_use) -> None:
    """
    Tests our metadata logging validation
    """
    dataquality.config.task_type = "text_classification"
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

    _log_data(meta=meta)
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


def test_logging_duplicate_ids(cleanup_after_use) -> None:
    """
    Tests that logging duplicate ids triggers a failure
    """
    dataquality.config.task_type = "text_classification"
    num_records = 50
    _log_data(num_records=num_records, unique_ids=False)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        c = dataquality.get_data_logger("text_classification")
        with pytest.raises(GalileoException):
            c.upload()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_config_no_vars(monkeypatch):
    """Should throw a nice error if we don't set our env vars"""
    x = os.getenv("GALILEO_API_URL")
    os.environ["GALILEO_API_URL"] = ""
    if os.path.isfile(".galileo/config.json"):
        os.remove(".galileo/config.json")

    monkeypatch.setattr("builtins.input", lambda inp: "" if "region" in inp else "test")
    monkeypatch.setattr("getpass.getpass", lambda _: "test_pass")

    reload(dataquality.core._config)
    assert dataquality.core._config.config.api_url == "http://test"
    assert dataquality.core._config.config.minio_region == "us-east-1"

    os.environ["GALILEO_API_URL"] = x
