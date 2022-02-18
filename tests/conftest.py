import os
import shutil
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import pytest
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients import objectstore
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType

config.current_project_id = uuid4()
config.current_run_id = uuid4()

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
def set_config_token(default_token: str = "sometoken") -> Callable:
    # Set default fixture token to "sometoken"
    config.token = default_token

    def curry(token: Optional[str] = None) -> None:
        # Override config token with custom value by currying
        config.token = token

    return curry


@pytest.fixture()
def set_config_task_type(
    default_task_type: TaskType = TaskType.text_classification,
) -> Callable:
    # Set default fixture token to "text_classification"
    config.task_type = default_task_type

    def curry(task_type: TaskType) -> None:
        # Override config task_type with custom value by currying
        config.task_type = task_type

    return curry


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
