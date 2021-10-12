import os
import shutil

import pytest
from pydantic.types import UUID4

import dataquality
from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.jsonl_logger import JsonlLogger

dataquality.config.current_project_id = "asdf"
dataquality.config.current_run_id = "asdf"

LOCATION = (
    f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
    f"/{config.current_run_id}"
)
TEST_STORE_DIR = "TEST_STORE"
TEST_PATH = f"{LOCATION}/{TEST_STORE_DIR}"
SPLITS = ["training", "test"]
SUBDIRS = ["data", "emb", "prob"]


@pytest.fixture(scope="function")
def cleanup_after_use():
    try:
        if not os.path.isdir(TEST_PATH):
            for split in SPLITS:
                for subdir in SUBDIRS:
                    os.makedirs(f"{TEST_PATH}/{split}/{subdir}")
        yield
    finally:
        shutil.rmtree(LOCATION)


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
    # separate folder per split (test, train, val) and data type (emb, prob, data)
    split, epoch, data_type, file_name = object_name.split("/")[-4:]
    print(f"HERE: {split, epoch, data_type, file_name}")
    os.system(f"cp {file_path} {TEST_PATH}/{split}/{data_type}/{file_name}")


# Patch the upload so we don't write to S3/minio
object_store.create_project_run_object = patch_object_upload
