import os
import shutil
from glob import glob
from random import random

import pandas as pd
from pydantic.types import UUID4
from sklearn.datasets import fetch_20newsgroups

import dataquality
from dataquality import config
from dataquality.clients import object_store
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import Serialization
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager

dataquality.config.current_project_id = "asdf"
dataquality.config.current_run_id = "asdf"

TEST_STORE_DIR = "TEST_STORE"
NUM_RECORDS = 23
NUM_LOGS = 100

LOCATION = (
    f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
    f"/{config.current_run_id}"
)
TEST_PATH = f"{LOCATION}/{TEST_STORE_DIR}"

config.serialization = Serialization.jsonl  # easier for testing purposes


def patch_object_upload(
    project_id: UUID4,
    run_id: UUID4,
    object_name: str,
    file_path: str,
    content_type: str = "application/octet-stream",
) -> None:
    """
    A patch for the object_store.create_project_run_object so we don't have to talk to
    minio for testing
    """
    test_dir = f"{LOCATION}/{TEST_STORE_DIR}"
    os.system(f"cp {file_path} {test_dir}")


# Patch the upload so we don't write to S3/minio
object_store.create_project_run_object = patch_object_upload


def validate_uploaded_data(expected_num_records: int) -> None:
    """
    Checks for testing
    """

    df = pd.DataFrame()
    for file in glob(f"{TEST_PATH}/*"):
        fname = file.split("/")[-1]
        assert not os.path.isfile(f"{LOCATION}/{fname}")
        df_subset = pd.read_json(file, lines=True)
        df = pd.concat([df, df_subset])

    print("Validating dataframe")
    for row in df["emb"].values.tolist():
        assert isinstance(
            row, list
        ), f"Row should be a list, but was {row} of type {type(row)}"
        assert len(row), f"Row should have data, but got {row}"
    assert len(df) == expected_num_records


def _log_data(num_records=NUM_RECORDS, num_logs=NUM_LOGS, with_upload=False) -> None:
    """
    Logs some mock data to disk
    """
    test_dir = f"{LOCATION}/{TEST_STORE_DIR}"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    write_output_dir = LOCATION
    if not os.path.exists(write_output_dir):
        os.makedirs(write_output_dir)

    # Log train/test data
    for split in [Split.test, Split.training]:
        newsgroups_train = fetch_20newsgroups(
            subset="train" if split == Split.training else split.value,
            remove=("headers", "footers", "quotes"),
        )
        dataset = pd.DataFrame()
        dataset["text"] = newsgroups_train.data
        dataset["label"] = newsgroups_train.target
        dataset = dataset[:num_records]
        gconfig = GalileoDataConfig(
            text=dataset["text"], labels=dataset["label"], split=split.value
        )
        dataquality.log_batch_input_data(gconfig)

    for split in [Split.training, Split.test]:
        for _ in range(num_logs):
            emb = [[random() for _ in range(700)] for _ in range(num_records)]
            probs = [[random() for _ in range(8)] for _ in range(num_records)]
            epoch = 0
            ids = list(range(num_records))

            model_config = GalileoModelConfig(
                emb=emb, probs=probs, split=split.value, epoch=epoch, ids=ids
            )
            dataquality.log_model_outputs(model_config, upload=with_upload)


def test_threaded_logging() -> None:
    """
    Tests that threaded logging does not miss any records

    :return: None
    """
    num_records = 23
    num_logs = 101  # We do 101 because max active threads is 100
    _log_data(num_records=num_records, num_logs=num_logs, with_upload=False)
    try:
        dataquality.upload()
        validate_uploaded_data(num_records * num_logs * 2)
    finally:
        ThreadPoolManager.wait_for_threads()
        dataquality._cleanup()
        shutil.rmtree(LOCATION)


def test_threaded_logging_and_upload() -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    num_records = 50
    _log_data(num_records=num_records, with_upload=True)
    try:
        ThreadPoolManager.wait_for_threads()  # Essentially the `finish` call from users
        validate_uploaded_data(num_records * NUM_LOGS * 2)
        for file in glob(f"{TEST_PATH}/*"):
            df = pd.read_json(file, lines=True)
            assert len(df["id"].unique()) == len(df)
            assert len(df["split"].unique()) == 1
            assert len(df["epoch"].unique()) == 1
    finally:
        ThreadPoolManager.wait_for_threads()
        dataquality._cleanup()
        shutil.rmtree(LOCATION)
