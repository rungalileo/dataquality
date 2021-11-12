import os
from importlib import reload

import pytest

import dataquality
import dataquality.core.config
from dataquality.core.finish import _cleanup, _upload
from dataquality.schemas.jsonl_logger import JsonlInputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.utils.data_utils import (
    NUM_LOGS,
    _log_data,
    validate_cleanup_data,
    validate_uploaded_data,
)


def test_threaded_logging_and_upload(cleanup_after_use) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    num_records = 50
    _log_data(num_records=num_records)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        _upload()
        validate_uploaded_data(num_records * NUM_LOGS)
        _cleanup()
        validate_cleanup_data()
    finally:
        # Mock finish() call without calling the API
        ThreadPoolManager.wait_for_threads()


def test_set_data_version_fail():
    """
    You should not be able to set the data_schema_version of your logged data.
    An error should be thrown
    """
    with pytest.raises(ValueError):
        JsonlInputLogItem(
            id=1, split=Split.training, text="test", data_schema_version=5
        )


def test_config_no_vars(monkeypatch):
    """Should throw a nice error if we don't set our env vars"""
    x = os.getenv("GALILEO_API_URL")
    os.environ["GALILEO_API_URL"] = ""
    if os.path.isfile(".galileo/config.json"):
        os.remove(".galileo/config.json")

    monkeypatch.setattr("builtins.input", lambda inp: "" if "region" in inp else "test")
    monkeypatch.setattr("getpass.getpass", lambda _: "test_pass")

    reload(dataquality.core.config)
    assert dataquality.core.config.config.api_url == "http://test"
    assert dataquality.core.config.config.minio_secret_key == "test_pass"
    assert dataquality.core.config.config.minio_region == "us-east-1"

    os.environ["GALILEO_API_URL"] = x
