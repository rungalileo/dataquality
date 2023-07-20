import os
import shutil
import warnings
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from uuid import UUID

import pytest
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaex.dataframe import DataFrame

import dataquality
from dataquality import AggregateFunction, Condition, ConditionFilter, Operator, config
from dataquality.clients import objectstore
from dataquality.exceptions import GalileoWarning
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dq_logger import DQ_LOG_FILE_HOME
from tests.test_utils.mock_request import MockResponse

try:
    torch.set_default_device("cpu")
except AttributeError:
    warnings.warn("Torch default device not set to CPU", GalileoWarning)
DEFAULT_API_URL = "http://localhost:8088"
UUID_STR = "399057bc-b276-4027-a5cf-48893ac45388"
TEST_STORE_DIR = "TEST_STORE"
SPLITS = ["training", "test"]
SUBDIRS = ["data", "emb", "prob"]

# Load models locally
HF_TEST_BERT_PATH = "hf-internal-testing/tiny-random-distilbert"
LOCAL_MODEL_PATH = f"{os.getcwd()}/tmp/testing-random-distilbert-sq"
try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, device="cpu")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(HF_TEST_BERT_PATH, device="cpu")
    tokenizer.save_pretrained(LOCAL_MODEL_PATH)

try:
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH).to(
        "cpu"
    )
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained(HF_TEST_BERT_PATH).to(
        "cpu"
    )

    model.save_pretrained(LOCAL_MODEL_PATH)


class TestSessionVariables:
    def __init__(
        self,
        DEFAULT_PROJECT_ID: UUID,
        DEFAULT_RUN_ID: UUID,
        LOCATION: str,
        DQ_LOG_FILE_LOCATION: str,
        TEST_PATH: str,
    ) -> None:
        self.DEFAULT_PROJECT_ID = DEFAULT_PROJECT_ID
        self.DEFAULT_RUN_ID = DEFAULT_RUN_ID
        self.LOCATION = LOCATION
        self.DQ_LOG_FILE_LOCATION = DQ_LOG_FILE_LOCATION
        self.TEST_PATH = TEST_PATH


@pytest.fixture(scope="session")
def test_session_vars() -> TestSessionVariables:
    pid = str(os.getpid())
    uuid_with_pid = UUID_STR[: -len(pid)] + pid
    DEFAULT_PROJECT_ID = UUID(uuid_with_pid)
    DEFAULT_RUN_ID = UUID(uuid_with_pid)
    LOCATION = f"{BaseGalileoLogger.LOG_FILE_DIR}/{DEFAULT_PROJECT_ID}/{DEFAULT_RUN_ID}"
    DQ_LOG_FILE_LOCATION = f"{DQ_LOG_FILE_HOME}/{DEFAULT_RUN_ID}"
    TEST_PATH = f"{LOCATION}/{TEST_STORE_DIR}"
    return TestSessionVariables(
        DEFAULT_PROJECT_ID=DEFAULT_PROJECT_ID,
        DEFAULT_RUN_ID=DEFAULT_RUN_ID,
        LOCATION=LOCATION,
        DQ_LOG_FILE_LOCATION=DQ_LOG_FILE_LOCATION,
        TEST_PATH=TEST_PATH,
    )


@pytest.fixture(autouse=True)
def disable_network_calls(request, monkeypatch):
    # Tests that fetch datasets need network access
    if "noautofixt" in request.keywords:
        return

    def stunted_get(url: str) -> Union[Dict, MockResponse]:
        """Unless it's a mocked call to healthcheck, disable network access"""
        bucket_names = {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        }
        if "healthcheck" in url:
            return MockResponse(
                json_data={
                    "minimum_dq_version": "0.0.0",
                    "api_version": "100.0.0",
                    "bucket_names": bucket_names,
                },
                status_code=200,
            )
        raise RuntimeError("Network access not allowed during testing!")

    monkeypatch.setattr(requests, "get", lambda url, *args, **kwargs: stunted_get(url))


@pytest.fixture(scope="function")
def cleanup_after_use(test_session_vars: TestSessionVariables) -> Generator:
    for task_type in TaskType.get_valid_tasks():
        dataquality.get_data_logger(task_type).logger_config.reset()
    try:
        if os.path.isdir(test_session_vars.LOCATION):
            shutil.rmtree(test_session_vars.LOCATION)
        if not os.path.isdir(test_session_vars.TEST_PATH):
            for split in SPLITS:
                for subdir in SUBDIRS:
                    os.makedirs(f"{test_session_vars.TEST_PATH}/{split}/{subdir}")
        if not os.path.isdir(test_session_vars.DQ_LOG_FILE_LOCATION):
            os.makedirs(test_session_vars.DQ_LOG_FILE_LOCATION)
        yield
    finally:
        if os.path.exists(test_session_vars.LOCATION):
            shutil.rmtree(test_session_vars.LOCATION)
        if os.path.exists(test_session_vars.DQ_LOG_FILE_LOCATION):
            shutil.rmtree(test_session_vars.DQ_LOG_FILE_LOCATION)
        for task_type in TaskType.get_valid_tasks():
            dataquality.get_data_logger(task_type).logger_config.reset()


@pytest.fixture()
def set_test_config(
    test_session_vars: TestSessionVariables,
    default_token: str = "sometoken",
    default_task_type: TaskType = TaskType.text_classification,
    default_api_url: str = DEFAULT_API_URL,
) -> Callable:
    config.token = default_token
    config.task_type = default_task_type
    config.api_url = default_api_url
    config.current_run_id = test_session_vars.DEFAULT_RUN_ID
    config.current_project_id = test_session_vars.DEFAULT_PROJECT_ID

    def curry(**kwargs: Dict[str, Any]) -> None:
        # Override test config with custom value by currying
        for k, v in kwargs.items():
            if k in config.dict():
                config.__setattr__(k, v)
        dataquality.get_model_logger().logger_config.reset()

    return curry


@pytest.fixture()
def statuses_response() -> Dict[str, str]:
    return {"job_id": "2", "status": "completed", "created_at": "2022-02-24"}


@pytest.fixture()
def input_data() -> Callable:
    def curry(
        split: str = "training",
        inference_name: str = "all-customers",
        meta: Optional[Dict] = None,
    ) -> Dict:
        data = {
            "texts": ["sentence_1", "sentence_2"],
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


@pytest.fixture()
def test_condition() -> Callable:
    def curry(
        agg: AggregateFunction = AggregateFunction.avg,
        metric: Optional[str] = "confidence",
        operator: Operator = Operator.lt,
        threshold: float = 0.5,
        filters: Optional[List[ConditionFilter]] = [],
    ) -> Condition:
        return Condition(
            agg=agg,
            metric=metric,
            operator=operator,
            threshold=threshold,
            filters=filters,
        )

    return curry


def patch_object_upload(
    self: Any, df: DataFrame, object_name: str, bucket_name: Optional[str] = None
) -> None:
    """
    A patch for the object_store.create_project_run_object_from_df so we don't have to
    talk to minio for testing
    """
    # separate folder per split (test, train, val) and data type (emb, prob, data)
    split, epoch, data_type, file_name = object_name.split("/")[-4:]
    export_path = (
        f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
        f"{config.current_run_id}/{TEST_STORE_DIR}/{split}/{epoch}/{data_type}"
    )
    export_loc = f"{export_path}/{file_name}"

    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    df.export(export_loc)


# Patch the upload so we don't write to S3/minio
objectstore.ObjectStore.create_project_run_object_from_df = patch_object_upload
