import os
import shutil

import pytest

import dataquality
from dataquality import config
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
