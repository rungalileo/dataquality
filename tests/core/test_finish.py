from typing import Callable, Generator
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import dataquality
import dataquality.core.log
from dataquality.exceptions import GalileoWarning
from dataquality.schemas.task_type import TaskType


def test_finish_no_init() -> None:
    """
    Tests finish without an init call
    """
    dataquality.config.current_run_id = dataquality.config.current_project_id = None
    with pytest.raises(AssertionError):
        dataquality.finish()


@mock.patch.object(dataquality.core.init.ApiClient, "wait_for_run")
def test_wait_for_run(mock_client: MagicMock) -> None:
    """
    Tests that wait_for_run calls ApiClient
    """
    dataquality.wait_for_run(project_name="Carrots", run_name="Rhubarb")
    mock_client.assert_called_once_with(project_name="Carrots", run_name="Rhubarb")


@mock.patch.object(
    dataquality.core.init.ApiClient,
    "get_run_status",
    return_value={"status": "in_progress"},
)
def test_get_run_status(mock_client: MagicMock) -> None:
    """
    Tests that get_run_status calls ApiClient
    """
    status = dataquality.get_run_status(project_name="Carrots", run_name="Rhubarb")
    mock_client.assert_called_once_with(project_name="Carrots", run_name="Rhubarb")
    assert status.get("status") == "in_progress"


@mock.patch.object(dataquality.core.finish, "_version_check")
@mock.patch.object(dataquality.core.finish, "_reset_run")
@mock.patch.object(dataquality.core.finish, "upload_dq_log_file")
@mock.patch.object(dataquality.clients.api.ApiClient, "make_request")
@mock.patch.object(
    dataquality.core.finish.dataquality,
    "get_data_logger",
)
@mock.patch.object(dataquality.core.finish, "wait_for_run")
def test_finish_waits_default(
    mock_wait_for_run: MagicMock,
    mock_get_data_logger: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config,
) -> None:
    set_test_config(task_type=TaskType.text_classification)
    mock_get_data_logger.return_value = MagicMock()
    dataquality.finish()
    mock_wait_for_run.assert_called_once()


@mock.patch.object(dataquality.core.finish, "_version_check")
@mock.patch.object(dataquality.core.finish, "_reset_run")
@mock.patch.object(dataquality.core.finish, "upload_dq_log_file")
@mock.patch.object(dataquality.clients.api.ApiClient, "make_request")
@mock.patch.object(
    dataquality.core.finish.dataquality,
    "get_data_logger",
)
@mock.patch.object(dataquality.core.finish, "wait_for_run")
def test_finish_no_waits_when_false(
    mock_wait_for_run: MagicMock,
    mock_get_data_logger: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config,
) -> None:
    set_test_config(task_type=TaskType.text_classification)
    mock_get_data_logger.return_value = MagicMock()
    dataquality.finish(wait=False)
    mock_wait_for_run.assert_not_called()


@mock.patch.object(dataquality.core.finish, "_version_check")
@mock.patch.object(dataquality.core.finish, "_reset_run")
@mock.patch.object(dataquality.core.finish, "upload_dq_log_file")
@mock.patch.object(dataquality.clients.api.ApiClient, "make_request")
@mock.patch.object(dataquality.core.finish, "wait_for_run")
def test_finish_ignores_missing_inference_name_inframe(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    """Test case where no inference name dataset was logged"""
    set_test_config(task_type=TaskType.text_classification)
    data = {
        "id": list(range(100)),
        "text": ["hey"] * 100,
        "embeddings": [np.random.rand(384) for i in range(100)],
    }
    df = pd.DataFrame.from_dict(data)
    dataquality.log_dataset(df, split="inference", inference_name="inference_name")
    dataquality.set_split("inference", "new_name")
    dataquality.set_labels_for_run([0, 1])
    dataquality.log_model_outputs(
        embs=[np.random.rand(384) for i in range(100)],
        logits=[np.random.rand(2) for i in range(100)],
        ids=list(range(100)),
    )
    # This should return and NOT log data and NOT throw an exception
    with pytest.warns(GalileoWarning):
        dataquality.finish()
