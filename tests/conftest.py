import os
import shutil
from typing import Any, Callable, Dict, Generator, List
from uuid import uuid4

import pytest
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients import objectstore
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType

config.current_project_id = uuid4()
config.current_run_id = uuid4()

DEFAULT_API_URL = os.environ["GALILEO_API_URL"]
DEFAULT_MINIO_URL = os.environ["GALILEO_MINIO_URL"]

LOCATION = (
    f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}"
    f"/{config.current_run_id}"
)
TEST_STORE_DIR = "TEST_STORE"
TEST_PATH = f"{LOCATION}/{TEST_STORE_DIR}"
SPLITS = ["training", "test"]
SUBDIRS = ["data", "emb", "prob"]


@pytest.fixture(scope="function")
def cleanup_after_use() -> Generator:
    try:
        if not os.path.isdir(TEST_PATH):
            for split in SPLITS:
                for subdir in SUBDIRS:
                    os.makedirs(f"{TEST_PATH}/{split}/{subdir}")
        yield
    finally:
        shutil.rmtree(LOCATION)


@pytest.fixture()
def set_test_config(
    default_token: str = "sometoken",
    default_task_type: TaskType = TaskType.text_classification,
    default_api_url: str = DEFAULT_API_URL,
    default_minio_url: str = DEFAULT_MINIO_URL,
) -> Callable:
    config.token = default_token
    config.task_type = default_task_type
    config.api_url = default_api_url
    config.minio_url = default_minio_url

    def curry(**kwargs: Dict[str, Any]) -> None:
        # Override test config with custom value by currying
        for k, v in kwargs.items():
            if k in config.dict():
                config.__setattr__(k, v)

    return curry


@pytest.fixture()
def statuses_response() -> Dict[str, List]:
    return {
        "statuses": [
            {"status": "started", "timestamp": "2022-02-20"},
            {"status": "finished", "timestamp": "2022-02-24"},
        ]
    }


def patch_object_upload(self: Any, df: DataFrame, object_name: str) -> None:
    """
    A patch for the object_store.create_project_run_object_from_df so we don't have to
    talk to minio for testing
    """
    # separate folder per split (test, train, val) and data type (emb, prob, data)
    split, epoch, data_type, file_name = object_name.split("/")[-4:]
    export_path = f"{TEST_PATH}/{split}/{data_type}"
    export_loc = f"{export_path}/{file_name}"

    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    df.export(export_loc)


# Patch the upload so we don't write to S3/minio
objectstore.ObjectStore.create_project_run_object_from_df = patch_object_upload
