import os
import shutil
from typing import Any, Generator
from uuid import uuid4

import pytest
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients import objectstore
from dataquality.loggers import BaseGalileoLogger

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
