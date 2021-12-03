import os
from importlib import reload

import numpy as np
import pandas as pd
import pytest
from random import random
import vaex

import dataquality
import dataquality.core._config
from dataquality.core.finish import _cleanup, _upload
from dataquality.exceptions import GalileoException
from dataquality.schemas.jsonl_logger import JsonlInputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import expand_df
from tests.utils.data_utils import (
    _log_data,
    validate_cleanup_data,
    validate_uploaded_data, NUM_RECORDS,
)


def test_threaded_logging_and_upload(cleanup_after_use) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    num_records = 32
    num_logs = 200
    num_emb = 700
    _log_data(num_records=num_records, num_logs=num_logs, num_emb=num_emb)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        _upload()
        validate_uploaded_data(num_records * num_logs, expected_num_emb=num_emb)
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


def test_logging_duplicate_ids(cleanup_after_use) -> None:
    """
    Tests that logging duplicate ids triggers a failure
    """
    num_records = 50
    _log_data(num_records=num_records, unique_ids=False)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        with pytest.raises(GalileoException):
            _upload()
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
    assert dataquality.core._config.config.minio_secret_key == "test_pass"
    assert dataquality.core._config.config.minio_region == "us-east-1"

    os.environ["GALILEO_API_URL"] = x


def test_expand_df():
    embeds_a = [np.array([random() for _ in range(100)]) for _ in range(NUM_RECORDS)]
    ids = list(range(NUM_RECORDS))
    data = {
        'embeds_a': embeds_a,
        'id': ids
    }
    df = vaex.from_pandas(pd.DataFrame(data))
    emb = np.array(list(df["embeds_a"][:1].values))
    df = expand_df(df, "embeds_a")
    cols = [i for i in df.get_column_names() if i != "id"]
    emb_exp = df[cols][:1].values
    assert (np.isclose(np.array(emb), np.array(emb_exp))).all()
