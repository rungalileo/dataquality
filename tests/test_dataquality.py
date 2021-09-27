import os
from random import random

import pandas as pd
import pytest
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

import dataquality
from dataquality import config
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager


def patch_upload(wait=True):
    """
    Override the upload function. Waits for threads to complete conditionally
    """
    if wait:
        ThreadPoolManager.wait_for_threads()
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    in_frame = pd.read_json(f"{location}/{JsonlLogger.INPUT_FILENAME}", lines=True)
    out_frame = pd.read_json(f"{location}/{JsonlLogger.OUTPUT_FILENAME}", lines=True)
    in_out = in_frame.merge(
        out_frame, on=["split", "id", "data_schema_version"], how="left"
    )
    for row in in_out["emb"].values.tolist():
        assert isinstance(row, list), (
            f"Row should be a list, " f"but was {row} of type {type(row)}"
        )
        assert len(row), f"Row should have data, but got {row}"


def _log_data(num_records=100) -> None:
    """
    Logs some mock data to disk
    """
    dataquality.config.current_project_id = "asdf"
    dataquality.config.current_run_id = "asdf"
    write_output_dir = (
        f"{JsonlLogger.LOG_FILE_DIR}/"
        f"{dataquality.config.current_project_id}/"
        f"{dataquality.config.current_run_id}"
    )
    if not os.path.exists(write_output_dir):
        os.makedirs(write_output_dir)

    # Log train data
    for split in [Split.test, Split.training]:
        print("logging data for split", split)
        newsgroups_train = fetch_20newsgroups(
            subset="train" if split == Split.training else split.value,
            remove=("headers", "footers", "quotes"),
        )
        dataset = pd.DataFrame()
        dataset["text"] = newsgroups_train.data
        dataset["label"] = newsgroups_train.target
        dataset = dataset[:23]
        gconfig_train = GalileoDataConfig(
            text=dataset["text"], labels=dataset["label"], split=split.value
        )
        dataquality.log_batch_input_data(gconfig_train)

    for split in [Split.training, Split.test]:
        print("Logging model data for split ", split)
        for _ in tqdm(range(num_records)):
            emb = [[random() for _ in range(700)] for _ in range(len(dataset))]
            probs = [[random() for _ in range(8)] for _ in range(len(dataset))]
            epoch = 0
            ids = [i for i in range(len(dataset))]

            model_config = GalileoModelConfig(
                emb=emb, probs=probs, split=split.value, epoch=epoch, ids=ids
            )
            dataquality.log_model_outputs(model_config)


def test_threaded_logging() -> None:
    """
    Tests that threaded logging does not miss any records

    :return: None
    """
    _log_data(num_records=101)  # We do 101 because max active threads is 100
    try:
        patch_upload()
    finally:
        ThreadPoolManager.wait_for_threads()
        dataquality._cleanup()


def test_threaded_logging_failure() -> None:
    """
    Tests that threaded logging does miss records when we don't wait for threads to
    complete

    :return: None
    """
    _log_data(num_records=50)
    try:
        with pytest.raises(AssertionError):
            patch_upload(wait=False)
    finally:
        ThreadPoolManager.wait_for_threads()
        dataquality._cleanup()
