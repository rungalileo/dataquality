from unittest import mock
from unittest.mock import MagicMock

import pytest

import dataquality
import dataquality.core.log
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
