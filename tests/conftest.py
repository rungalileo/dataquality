import os
import shutil
from typing import Any, Callable, Dict, Generator, List
from uuid import UUID

import pytest
import requests
import spacy
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients import objectstore
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType

DEFAULT_API_URL = "http://localhost:8088"
DEFAULT_MINIO_URL = "127.0.0.1:9000"
DEFAULT_PROJECT_ID = UUID("399057bc-b276-4027-a5cf-48893ac45388")
DEFAULT_RUN_ID = UUID("399057bc-b276-4027-a5cf-48893ac45388")

LOCATION = f"{BaseGalileoLogger.LOG_FILE_DIR}/{DEFAULT_PROJECT_ID}/{DEFAULT_RUN_ID}"
TEST_STORE_DIR = "TEST_STORE"
TEST_PATH = f"{LOCATION}/{TEST_STORE_DIR}"
SPLITS = ["training", "test"]
SUBDIRS = ["data", "emb", "prob"]

spacy.util.fix_random_seed()


@pytest.fixture(autouse=True)
def disable_network_calls(request, monkeypatch):
    # Tests that fetch datasets need network access
    if "noautofixt" in request.keywords:
        return

    def stunted_get():
        raise RuntimeError("Network access not allowed during testing!")

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())


@pytest.fixture(scope="function")
def cleanup_after_use() -> Generator:
    try:
        if os.path.isdir(BaseGalileoLogger.LOG_FILE_DIR):
            shutil.rmtree(BaseGalileoLogger.LOG_FILE_DIR)
        if not os.path.isdir(TEST_PATH):
            for split in SPLITS:
                for subdir in SUBDIRS:
                    os.makedirs(f"{TEST_PATH}/{split}/{subdir}")
        yield
    finally:
        shutil.rmtree(BaseGalileoLogger.LOG_FILE_DIR)


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
    config.current_run_id = DEFAULT_RUN_ID
    config.current_project_id = DEFAULT_PROJECT_ID

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


@pytest.fixture()
def input_data() -> Callable:
    def curry(
        split: str = "training",
        inference_name: str = "all-customers",
        meta: Dict = None,
    ) -> Dict:
        data = {
            "text": ["sentence_1", "sentence_2"],
            "split": split,
            "ids": [1, 2],
        }
        if split == "inference":
            data.update(inference_name=inference_name)
        else:
            data.update(labels=["APPLE", "ORANGE"])

        if meta:
            data.update(meta=meta)

        return data

    return curry


def patch_object_upload(self: Any, df: DataFrame, object_name: str) -> None:
    """
    A patch for the object_store.create_project_run_object_from_df so we don't have to
    talk to minio for testing
    """
    # separate folder per split (test, train, val) and data type (emb, prob, data)
    split, epoch, data_type, file_name = object_name.split("/")[-4:]
    export_path = f"{TEST_PATH}/{split}/{epoch}/{data_type}"
    export_loc = f"{export_path}/{file_name}"

    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    df.export(export_loc)


# Patch the upload so we don't write to S3/minio
objectstore.ObjectStore.create_project_run_object_from_df = patch_object_upload
