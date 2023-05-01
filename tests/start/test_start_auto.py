from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import dataquality
import dataquality as dq
from dataquality.clients.api import ApiClient
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dq.clients.api.ApiClient,
    "get_healthcheck_dq",
    return_value={
        "bucket_names": {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        },
        "minio_fqdn": "127.0.0.1:9000",
    },
)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_auto(
    mock_valid_user: MagicMock,
    mock_bucket_names: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    df_train = pd.DataFrame({"text": ["hello"] * 20, "label": ["hello"] * 20})
    df_test = pd.DataFrame({"text": ["hello"] * 20, "label": ["hello"] * 20})
    df_train.to_csv("tmp/train.csv", index=False)
    df_test.to_csv("tmp/test.csv", index=False)
    # with dataquality(
    #    hf_data="rungalileo/emotion",
    #    task="text_classification",
    # ):
    #    dataquality.get_insights()

    # Load the newsgroups dataset from sklearn
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")

    df_train = pd.DataFrame(
        {"text": newsgroups_train.data, "label": newsgroups_train.target}
    ).head(4)
    df_test = pd.DataFrame(
        {"text": newsgroups_test.data, "label": newsgroups_test.target}
    ).head(4)

    dataquality(
        train_data=df_train, test_data=df_test, labels=newsgroups_train.target_names
    )
